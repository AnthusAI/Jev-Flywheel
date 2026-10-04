"""
Corpus storage and ingestion for Biblicus.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import quote, unquote, urlparse

import yaml
from pydantic import ValidationError

from .catalog_compat import load_corpus_catalog
from .constants import (
    ANALYSIS_DIR_NAME,
    CORPUS_DIR_NAME,
    DEFAULT_RAW_DIR,
    EXTRACTED_DIR_NAME,
    GRAPH_DIR_NAME,
    LEGACY_CORPUS_DIR_NAME,
    RETRIEVAL_DIR_NAME,
    SCHEMA_VERSION,
    SIDECAR_SUFFIX,
)
from .errors import IngestCollisionError
from .frontmatter import parse_front_matter, render_front_matter
from .hook_manager import HookManager
from .hooks import HookPoint
from .ignore import load_corpus_ignore_spec
from .models import (
    CatalogItem,
    CorpusCatalog,
    CorpusConfig,
    ExtractionSnapshotListEntry,
    ExtractionSnapshotReference,
    IngestResult,
    RemoteSourcePullResult,
    RetrievalSnapshot,
)
from .remote_sources import AzureBlobRemoteSource, S3RemoteSource
from .sources import _media_type_from_filename, load_source
from .time import utc_now_iso
from .uris import corpus_ref_to_path, normalize_corpus_uri
from .user_config import resolve_source_profile


def _sha256_bytes(data: bytes) -> str:
    """
    Compute a Secure Hash Algorithm 256 digest for byte content.

    :param data: Input bytes.
    :type data: bytes
    :return: Secure Hash Algorithm 256 hex digest.
    :rtype: str
    """
    return hashlib.sha256(data).hexdigest()


def _write_stream_and_hash(
    stream, destination_path: Path, *, chunk_size: int = 1024 * 1024
) -> Dict[str, object]:
    """
    Write a binary stream to disk while computing a digest.

    :param stream: Binary stream to read from.
    :type stream: object
    :param destination_path: Destination path to write to.
    :type destination_path: Path
    :param chunk_size: Chunk size for reads.
    :type chunk_size: int
    :return: Mapping containing sha256 and bytes_written.
    :rtype: dict[str, object]
    :raises OSError: If the destination cannot be written.
    """
    hasher = hashlib.sha256()
    bytes_written = 0
    with destination_path.open("wb") as destination_handle:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            destination_handle.write(chunk)
            bytes_written += len(chunk)
    return {"sha256": hasher.hexdigest(), "bytes_written": bytes_written}


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a filename into a portable, filesystem-friendly form.

    :param name: Raw filename.
    :type name: str
    :return: Sanitized filename.
    :rtype: str
    """
    allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._() ")
    sanitized_name = "".join(
        (character if character in allowed_characters else "_") for character in name
    ).strip()
    return sanitized_name or "file"


def _preferred_extension_for_media_type(media_type: str) -> Optional[str]:
    """
    Return a preferred filename extension for a media type.

    :param media_type: Internet Assigned Numbers Authority media type.
    :type media_type: str
    :return: Preferred extension or None.
    :rtype: str or None
    """
    media_type_overrides = {
        "image/jpeg": ".jpg",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    if media_type in media_type_overrides:
        return media_type_overrides[media_type]
    return mimetypes.guess_extension(media_type)


def _ensure_filename_extension(filename: str, *, media_type: str) -> str:
    """
    Ensure a usable filename extension for a media type.

    :param filename: Raw filename.
    :type filename: str
    :param media_type: Internet Assigned Numbers Authority media type.
    :type media_type: str
    :return: Filename with a compatible extension.
    :rtype: str
    """
    raw_name = filename.strip()

    if media_type == "text/markdown":
        if raw_name.lower().endswith((".md", ".markdown")):
            return raw_name
        return raw_name + ".md"

    if Path(raw_name).suffix:
        if "%2F" in raw_name or "%3A" in raw_name:
            decoded = unquote(raw_name)
            parsed = urlparse(decoded)
            decoded_path = parsed.path if parsed.scheme else decoded
            if not Path(decoded_path).suffix:
                pass
            else:
                return raw_name
        else:
            return raw_name

    ext = _preferred_extension_for_media_type(media_type)
    if not ext:
        return raw_name
    return raw_name + ext


def _encode_source_uri_for_filename(source_uri: str) -> str:
    """
    Percent-encode a source uniform resource identifier for filename use.

    :param source_uri: Source uniform resource identifier to encode.
    :type source_uri: str
    :return: Percent-encoded uniform resource identifier safe for filenames.
    :rtype: str
    """
    return quote(source_uri, safe="")


def _storage_filename_for_ingest(
    *, filename: Optional[str], media_type: str, source_uri: Optional[str]
) -> str:
    """
    Derive a collision-safe filename for corpus storage.

    If a source uniform resource identifier is provided, the full uniform resource identifier is
    percent-encoded to namespace the stored file, preventing collisions between identical basenames
    from different sources. When no uniform resource identifier is available, fall back to a
    sanitized filename.

    :param filename: Optional filename hint from the caller.
    :type filename: str or None
    :param media_type: Media type of the payload.
    :type media_type: str
    :param source_uri: Optional source uniform resource identifier for provenance.
    :type source_uri: str or None
    :return: Storage filename with an appropriate extension, or an empty string when no hint exists.
    :rtype: str
    """
    base_name = ""
    if source_uri:
        base_name = _encode_source_uri_for_filename(source_uri)
        if filename and not source_uri.startswith("file:"):
            sanitized = _sanitize_filename(filename)
            if sanitized:
                base_name = f"{base_name}--{sanitized}"
    if not base_name and filename:
        base_name = _sanitize_filename(filename)
    if not base_name:
        return ""
    if len(base_name) > 180:
        digest = hashlib.sha256(base_name.encode("utf-8")).hexdigest()
        base_name = f"hash-{digest}"
    return _ensure_filename_extension(base_name, media_type=media_type)


def _merge_tags(explicit: Sequence[str], from_frontmatter: Any) -> List[str]:
    """
    Merge tags from explicit input and front matter values.

    :param explicit: Explicit tags provided by callers.
    :type explicit: Sequence[str]
    :param from_frontmatter: Tags from front matter.
    :type from_frontmatter: Any
    :return: Deduplicated tag list preserving order.
    :rtype: list[str]
    """
    merged_tags: List[str] = []

    for explicit_tag in explicit:
        cleaned_tag = explicit_tag.strip()
        if cleaned_tag:
            merged_tags.append(cleaned_tag)

    if isinstance(from_frontmatter, str):
        merged_tags.append(from_frontmatter)
    elif isinstance(from_frontmatter, list):
        for item in from_frontmatter:
            if isinstance(item, str) and item.strip():
                merged_tags.append(item.strip())

    seen_tags = set()
    deduplicated_tags: List[str] = []
    for tag_value in merged_tags:
        if tag_value not in seen_tags:
            seen_tags.add(tag_value)
            deduplicated_tags.append(tag_value)
    return deduplicated_tags


def _sidecar_path_for(content_path: Path) -> Path:
    """
    Compute the sidecar metadata path for a content file.

    :param content_path: Path to the content file.
    :type content_path: Path
    :return: Sidecar path.
    :rtype: Path
    """
    return content_path.with_name(content_path.name + SIDECAR_SUFFIX)


def _load_sidecar(content_path: Path) -> Dict[str, Any]:
    """
    Load sidecar metadata for a content file.

    :param content_path: Path to the content file.
    :type content_path: Path
    :return: Parsed sidecar metadata.
    :rtype: dict[str, Any]
    :raises ValueError: If the sidecar content is not a mapping.
    """
    path = _sidecar_path_for(content_path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Sidecar metadata must be a mapping/object: {path}")
    return dict(data)


def _write_sidecar(content_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Write a sidecar metadata file.

    :param content_path: Path to the content file.
    :type content_path: Path
    :param metadata: Metadata to serialize.
    :type metadata: dict[str, Any]
    :return: None.
    :rtype: None
    """
    path = _sidecar_path_for(content_path)
    text = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    path.write_text(text + "\n", encoding="utf-8")


def _ensure_biblicus_block(
    metadata: Dict[str, Any], *, item_id: str, source_uri: str
) -> Dict[str, Any]:
    """
    Ensure the biblicus metadata block exists and is populated.

    :param metadata: Existing metadata.
    :type metadata: dict[str, Any]
    :param item_id: Item identifier to store.
    :type item_id: str
    :param source_uri: Source uniform resource identifier to store.
    :type source_uri: str
    :return: Updated metadata mapping.
    :rtype: dict[str, Any]
    """
    updated_metadata = dict(metadata)
    existing_biblicus = updated_metadata.get("biblicus")
    if not isinstance(existing_biblicus, dict):
        existing_biblicus = {}
    biblicus_block = dict(existing_biblicus)
    biblicus_block["id"] = item_id
    biblicus_block["source"] = source_uri
    updated_metadata["biblicus"] = biblicus_block
    return updated_metadata


def _update_biblicus_block(metadata: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    updated_metadata = dict(metadata)
    existing_biblicus = updated_metadata.get("biblicus")
    if not isinstance(existing_biblicus, dict):
        existing_biblicus = {}
    biblicus_block = dict(existing_biblicus)
    for key, value in updates.items():
        if value is not None:
            biblicus_block[key] = value
    updated_metadata["biblicus"] = biblicus_block
    return updated_metadata


def _parse_uuid_prefix(filename: str) -> Optional[str]:
    """
    Extract a universally unique identifier prefix from a filename, if present.

    :param filename: Filename to inspect.
    :type filename: str
    :return: Universally unique identifier string or None.
    :rtype: str or None
    """
    if len(filename) < 36:
        return None
    prefix = filename[:36]
    try:
        return str(uuid.UUID(prefix))
    except ValueError:
        return None


def _merge_metadata(front: Dict[str, Any], side: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge front matter and sidecar metadata.

    :param front: Front matter metadata.
    :type front: dict[str, Any]
    :param side: Sidecar metadata.
    :type side: dict[str, Any]
    :return: Merged metadata.
    :rtype: dict[str, Any]
    """
    merged_metadata: Dict[str, Any] = dict(front)

    front_biblicus = merged_metadata.get("biblicus")
    sidecar_biblicus = side.get("biblicus")
    if isinstance(front_biblicus, dict) or isinstance(sidecar_biblicus, dict):
        merged_biblicus: Dict[str, Any] = {}
        if isinstance(front_biblicus, dict):
            merged_biblicus.update(front_biblicus)
        if isinstance(sidecar_biblicus, dict):
            merged_biblicus.update(sidecar_biblicus)
        merged_metadata["biblicus"] = merged_biblicus

    merged_tags = _merge_tags(_merge_tags([], front.get("tags")), side.get("tags"))
    if merged_tags:
        merged_metadata["tags"] = merged_tags

    for metadata_key, metadata_value in side.items():
        if metadata_key in {"biblicus", "tags"}:
            continue
        if metadata_key in merged_metadata:
            continue
        merged_metadata[metadata_key] = metadata_value

    return merged_metadata


class Corpus:
    """
    Local corpus manager for Biblicus.

    :ivar root: Corpus root directory.
    :vartype root: Path
    :ivar meta_dir: Metadata directory under the corpus root.
    :vartype meta_dir: Path
    :ivar raw_dir: Raw item directory under the corpus root.
    :vartype raw_dir: Path
    :ivar config: Parsed corpus config, if present.
    :vartype config: CorpusConfig or None
    """

    def __init__(self, root: Path):
        """
        Initialize a corpus wrapper around a filesystem path.

        :param root: Corpus root directory.
        :type root: Path
        """
        self.root = root
        self.meta_dir = self.root / CORPUS_DIR_NAME
        self.config = self._load_config()
        self.raw_dir = self._resolve_raw_dir()
        self._hooks = self._load_hooks()

    def _ensure_local_ingest_allowed(self) -> None:
        if self.config is not None and self.config.source is not None:
            raise ValueError(
                "Local ingest is disabled because this corpus is backed by a remote source. "
                "Use `biblicus source pull` to refresh the corpus."
            )

    def _resolve_raw_dir(self) -> Path:
        """
        Resolve the raw directory path for the corpus.

        :return: Raw directory path.
        :rtype: Path
        """
        raw_dir = DEFAULT_RAW_DIR
        if self.config is not None and isinstance(self.config.raw_dir, str):
            raw_dir = self.config.raw_dir.strip() or DEFAULT_RAW_DIR
        if raw_dir == ".":
            return self.root
        return self.root / raw_dir

    @property
    def extracted_dir(self) -> Path:
        """
        Location of extraction artifacts for the corpus.

        :return: Extracted artifacts directory.
        :rtype: Path
        """
        return self.root / EXTRACTED_DIR_NAME

    @property
    def graph_dir(self) -> Path:
        """
        Location of graph artifacts for the corpus.

        :return: Graph artifacts directory.
        :rtype: Path
        """
        return self.root / GRAPH_DIR_NAME

    @property
    def retrieval_dir(self) -> Path:
        """
        Location of retrieval artifacts for the corpus.

        :return: Retrieval artifacts directory.
        :rtype: Path
        """
        return self.root / RETRIEVAL_DIR_NAME

    @property
    def analysis_dir(self) -> Path:
        """
        Location of analysis artifacts for the corpus.

        :return: Analysis artifacts directory.
        :rtype: Path
        """
        return self.root / ANALYSIS_DIR_NAME

    @property
    def uri(self) -> str:
        """
        Return the canonical uniform resource identifier for the corpus root.

        :return: Corpus uniform resource identifier.
        :rtype: str
        """
        return self.root.as_uri()

    def _load_config(self) -> Optional[CorpusConfig]:
        """
        Load the corpus config if it exists.

        :return: Parsed corpus config or None.
        :rtype: CorpusConfig or None
        :raises ValueError: If the config schema is invalid.
        """
        path = self.meta_dir / "config.json"
        if not path.is_file():
            legacy_path = self.root / LEGACY_CORPUS_DIR_NAME / "config.json"
            if not legacy_path.is_file():
                return None
            path = legacy_path
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            return CorpusConfig.model_validate(data)
        except ValidationError as exc:
            has_hook_error = any(
                isinstance(error.get("loc"), tuple)
                and error.get("loc")
                and error.get("loc")[0] == "hooks"
                for error in exc.errors()
            )
            if has_hook_error:
                raise ValueError(f"Invalid hook specification: {exc}") from exc
            raise ValueError(f"Invalid corpus config: {exc}") from exc

    def _load_hooks(self) -> Optional[HookManager]:
        """
        Load the hook manager from config if hooks are configured.

        :return: Hook manager or None.
        :rtype: HookManager or None
        :raises ValueError: If hook specifications are invalid.
        """
        if self.config is None or not self.config.hooks:
            return None
        return HookManager.from_config(
            corpus_root=self.root,
            corpus_uri=self.uri,
            hook_specs=self.config.hooks,
        )

    def _reserved_dir_names(self) -> List[str]:
        return [
            CORPUS_DIR_NAME,
            LEGACY_CORPUS_DIR_NAME,
            EXTRACTED_DIR_NAME,
            GRAPH_DIR_NAME,
            RETRIEVAL_DIR_NAME,
            ANALYSIS_DIR_NAME,
        ]

    def _is_reserved_path(self, path: Path) -> bool:
        if not path.is_absolute():
            candidate = (self.root / path).resolve()
        else:
            candidate = path.resolve()
        try:
            relative = candidate.relative_to(self.root)
        except ValueError:
            return False
        if not relative.parts:
            return False
        if relative.parts[0] in self._reserved_dir_names():
            return True
        return relative.name == ".biblicusignore"

    def _raw_relpath(self, *, output_name: str, storage_subdir: Optional[str]) -> str:
        relpath = Path(output_name)
        if storage_subdir:
            relpath = Path(storage_subdir) / relpath
        raw_dir_name = DEFAULT_RAW_DIR
        if self.config is not None and isinstance(self.config.raw_dir, str):
            raw_dir_name = self.config.raw_dir.strip() or DEFAULT_RAW_DIR
        if raw_dir_name and raw_dir_name != ".":
            relpath = Path(raw_dir_name) / relpath
        return str(relpath)

    def _raw_prefix_for_storage(self, storage_subdir: str) -> Path:
        raw_dir_name = DEFAULT_RAW_DIR
        if self.config is not None and isinstance(self.config.raw_dir, str):
            raw_dir_name = self.config.raw_dir.strip() or DEFAULT_RAW_DIR
        if raw_dir_name and raw_dir_name != ".":
            return Path(raw_dir_name) / storage_subdir
        return Path(storage_subdir)

    @classmethod
    def find(cls, start: Path) -> "Corpus":
        """
        Locate a corpus by searching upward from a path.

        :param start: Starting path to search.
        :type start: Path
        :return: Located corpus instance.
        :rtype: Corpus
        :raises FileNotFoundError: If no corpus config is found.
        """
        start = start.resolve()
        for candidate in [start, *start.parents]:
            if (candidate / CORPUS_DIR_NAME / "config.json").is_file():
                return cls(candidate)
            if (candidate / LEGACY_CORPUS_DIR_NAME / "config.json").is_file():
                return cls(candidate)
        raise FileNotFoundError(
            f"Not a Biblicus corpus (no metadata or legacy config found from {start})"
        )

    @classmethod
    def open(cls, ref: str | Path) -> "Corpus":
        """
        Open a corpus from a path or uniform resource identifier reference.

        :param ref: Filesystem path or file:// uniform resource identifier.
        :type ref: str or Path
        :return: Opened corpus instance.
        :rtype: Corpus
        """
        return cls.find(corpus_ref_to_path(ref))

    @classmethod
    def init(cls, root: Path, *, force: bool = False) -> "Corpus":
        """
        Initialize a new corpus on disk.

        :param root: Corpus root directory.
        :type root: Path
        :param force: Whether to overwrite existing config.
        :type force: bool
        :return: Initialized corpus instance.
        :rtype: Corpus
        :raises FileExistsError: If the corpus already exists and force is False.
        """
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        corpus = cls(root)

        corpus.meta_dir.mkdir(parents=True, exist_ok=True)
        if corpus.raw_dir != corpus.root:
            corpus.raw_dir.mkdir(parents=True, exist_ok=True)

        config_path = corpus.meta_dir / "config.json"
        if config_path.exists() and not force:
            raise FileExistsError(f"Corpus already exists at {root}")

        config = CorpusConfig(
            schema_version=SCHEMA_VERSION,
            created_at=utc_now_iso(),
            corpus_uri=normalize_corpus_uri(root),
            raw_dir=DEFAULT_RAW_DIR,
        )
        config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

        corpus._init_catalog()
        return corpus

    @property
    def catalog_path(self) -> Path:
        """
        Return the path to the corpus catalog file.

        :return: Catalog file path.
        :rtype: Path
        """
        return self.meta_dir / "catalog.json"

    def _init_catalog(self) -> None:
        """
        Initialize the catalog if it does not already exist.

        :return: None.
        :rtype: None
        """
        if self.catalog_path.exists():
            return
        catalog = CorpusCatalog(
            schema_version=SCHEMA_VERSION,
            generated_at=utc_now_iso(),
            corpus_uri=normalize_corpus_uri(self.root),
            raw_dir=DEFAULT_RAW_DIR,
            latest_snapshot_id=None,
            items={},
            order=[],
        )
        self._write_catalog(catalog)

    def _load_catalog(self) -> CorpusCatalog:
        """
        Read and validate the corpus catalog file.

        :return: Parsed corpus catalog.
        :rtype: CorpusCatalog
        :raises FileNotFoundError: If the catalog file does not exist.
        :raises ValueError: If the catalog schema is invalid.
        """
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Missing corpus catalog: {self.catalog_path}")
        catalog_data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return load_corpus_catalog(catalog_data)

    def load_catalog(self) -> CorpusCatalog:
        """
        Load the current corpus catalog.

        :return: Parsed corpus catalog.
        :rtype: CorpusCatalog
        :raises FileNotFoundError: If the catalog file does not exist.
        :raises ValueError: If the catalog schema is invalid.
        """
        return self._load_catalog()

    def has_items(self) -> bool:
        """
        Return whether the corpus catalog contains any items.

        :return: True when the catalog has at least one item.
        :rtype: bool
        """
        catalog = self._load_catalog()
        return bool(catalog.items)

    def catalog_generated_at(self) -> str:
        """
        Return the catalog generation timestamp.

        :return: International Organization for Standardization 8601 timestamp.
        :rtype: str
        """
        return self._load_catalog().generated_at

    def _write_catalog(self, catalog: CorpusCatalog) -> None:
        """
        Atomically write a corpus catalog to disk.

        :param catalog: Catalog to persist.
        :type catalog: CorpusCatalog
        :return: None.
        :rtype: None
        """
        temp_path = self.catalog_path.with_suffix(".json.tmp")
        temp_path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.catalog_path)

    def _find_item_by_source_uri(self, source_uri: str) -> Optional[CatalogItem]:
        """
        Locate an existing catalog item by source uniform resource identifier.

        :param source_uri: Source uniform resource identifier to search for.
        :type source_uri: str
        :return: Matching catalog item or None.
        :rtype: CatalogItem or None
        """
        if not source_uri:
            return None
        self._init_catalog()
        catalog = self._load_catalog()
        for item in catalog.items.values():
            if item.source_uri == source_uri:
                return item
        return None

    @property
    def snapshots_dir(self) -> Path:
        """
        Location of retrieval snapshot manifests.

        :return: Path to the snapshots directory.
        :rtype: Path
        """
        return self.retrieval_dir

    @property
    def extraction_snapshots_dir(self) -> Path:
        """
        Location of extraction snapshot artifacts.

        :return: Path to the extraction snapshots directory.
        :rtype: Path
        """
        return self.extracted_dir

    @property
    def analysis_runs_dir(self) -> Path:
        """
        Location of analysis snapshot artifacts.

        :return: Path to the analysis snapshots directory.
        :rtype: Path
        """
        return self.analysis_dir

    @property
    def graph_snapshots_dir(self) -> Path:
        """
        Location of graph snapshot artifacts.

        :return: Path to the graph snapshots directory.
        :rtype: Path
        """
        return self.graph_dir

    def extraction_snapshot_dir(self, *, extractor_id: str, snapshot_id: str) -> Path:
        """
        Resolve an extraction snapshot directory.

        :param extractor_id: Extractor plugin identifier.
        :type extractor_id: str
        :param snapshot_id: Extraction snapshot identifier.
        :type snapshot_id: str
        :return: Extraction snapshot directory.
        :rtype: Path
        """
        return self.extraction_snapshots_dir / extractor_id / snapshot_id

    def analysis_run_dir(self, *, analysis_id: str, snapshot_id: str) -> Path:
        """
        Resolve an analysis snapshot directory.

        :param analysis_id: Analysis backend identifier.
        :type analysis_id: str
        :param snapshot_id: Analysis snapshot identifier.
        :type snapshot_id: str
        :return: Analysis snapshot directory.
        :rtype: Path
        """
        return self.analysis_runs_dir / analysis_id / snapshot_id

    def graph_snapshot_dir(self, *, extractor_id: str, snapshot_id: str) -> Path:
        """
        Resolve a graph snapshot directory.

        :param extractor_id: Graph extractor identifier.
        :type extractor_id: str
        :param snapshot_id: Graph snapshot identifier.
        :type snapshot_id: str
        :return: Graph snapshot directory.
        :rtype: Path
        """
        return self.graph_snapshots_dir / extractor_id / snapshot_id

    def read_extracted_text(
        self, *, extractor_id: str, snapshot_id: str, item_id: str
    ) -> Optional[str]:
        """
        Read extracted text for an item from an extraction snapshot, when present.

        :param extractor_id: Extractor plugin identifier.
        :type extractor_id: str
        :param snapshot_id: Extraction snapshot identifier.
        :type snapshot_id: str
        :param item_id: Item identifier.
        :type item_id: str
        :return: Extracted text or None if the artifact does not exist.
        :rtype: str or None
        :raises OSError: If the file exists but cannot be read.
        """
        path = (
            self.extraction_snapshot_dir(extractor_id=extractor_id, snapshot_id=snapshot_id)
            / "text"
            / f"{item_id}.txt"
        )
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def load_extraction_snapshot_manifest(self, *, extractor_id: str, snapshot_id: str):
        """
        Load an extraction snapshot manifest from the corpus.

        :param extractor_id: Extractor plugin identifier.
        :type extractor_id: str
        :param snapshot_id: Extraction snapshot identifier.
        :type snapshot_id: str
        :return: Parsed extraction snapshot manifest.
        :rtype: biblicus.extraction.ExtractionSnapshotManifest
        :raises FileNotFoundError: If the manifest file does not exist.
        :raises ValueError: If the manifest data is invalid.
        """
        from .extraction import ExtractionSnapshotManifest

        manifest_path = (
            self.extraction_snapshot_dir(extractor_id=extractor_id, snapshot_id=snapshot_id)
            / "manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing extraction snapshot manifest: {manifest_path}")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ExtractionSnapshotManifest.model_validate(data)

    def list_extraction_snapshots(
        self, *, extractor_id: Optional[str] = None
    ) -> List[ExtractionSnapshotListEntry]:
        """
        List extraction snapshots stored under the corpus.

        :param extractor_id: Optional extractor identifier filter.
        :type extractor_id: str or None
        :return: Summary list entries for each snapshot.
        :rtype: list[biblicus.models.ExtractionSnapshotListEntry]
        """
        snapshots_root = self.extraction_snapshots_dir
        if not snapshots_root.is_dir():
            return []

        extractor_dirs: List[Path]
        if extractor_id is None:
            extractor_dirs = [path for path in sorted(snapshots_root.iterdir()) if path.is_dir()]
        else:
            extractor_path = snapshots_root / extractor_id
            extractor_dirs = [extractor_path] if extractor_path.is_dir() else []

        entries: List[ExtractionSnapshotListEntry] = []
        for extractor_dir in extractor_dirs:
            for snapshot_dir in sorted(extractor_dir.iterdir()):
                if not snapshot_dir.is_dir():
                    continue
                manifest_path = snapshot_dir / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifest = self.load_extraction_snapshot_manifest(
                        extractor_id=extractor_dir.name,
                        snapshot_id=snapshot_dir.name,
                    )
                except (FileNotFoundError, ValueError):
                    continue
                entries.append(
                    ExtractionSnapshotListEntry(
                        extractor_id=extractor_dir.name,
                        snapshot_id=snapshot_dir.name,
                        configuration_id=manifest.configuration.configuration_id,
                        configuration_name=manifest.configuration.name,
                        catalog_generated_at=manifest.catalog_generated_at,
                        created_at=manifest.created_at,
                        stats=dict(manifest.stats),
                    )
                )

        entries.sort(
            key=lambda entry: (entry.created_at, entry.extractor_id, entry.snapshot_id),
            reverse=True,
        )
        return entries

    def latest_extraction_snapshot_reference(
        self, *, extractor_id: Optional[str] = None
    ) -> Optional[ExtractionSnapshotReference]:
        """
        Return the most recent extraction snapshot reference.

        :param extractor_id: Optional extractor identifier filter.
        :type extractor_id: str or None
        :return: Latest extraction snapshot reference or None when no snapshots exist.
        :rtype: biblicus.models.ExtractionSnapshotReference or None
        """
        entries = self.list_extraction_snapshots(extractor_id=extractor_id)
        if not entries:
            return None
        latest = entries[0]
        return ExtractionSnapshotReference(
            extractor_id=latest.extractor_id, snapshot_id=latest.snapshot_id
        )

    def delete_extraction_snapshot(self, *, extractor_id: str, snapshot_id: str) -> None:
        """
        Delete an extraction snapshot directory and its derived artifacts.

        :param extractor_id: Extractor plugin identifier.
        :type extractor_id: str
        :param snapshot_id: Extraction snapshot identifier.
        :type snapshot_id: str
        :return: None.
        :rtype: None
        :raises FileNotFoundError: If the extraction snapshot directory does not exist.
        """
        snapshot_dir = self.extraction_snapshot_dir(
            extractor_id=extractor_id, snapshot_id=snapshot_id
        )
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"Missing extraction snapshot directory: {snapshot_dir}")
        shutil.rmtree(snapshot_dir)

    def _ensure_snapshots_dir(self) -> None:
        """
        Ensure the retrieval snapshots directory exists.

        :return: None.
        :rtype: None
        """
        self.retrieval_dir.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, snapshot: RetrievalSnapshot) -> None:
        """
        Persist a retrieval snapshot manifest and update the catalog pointer.

        :param snapshot: Snapshot manifest to persist.
        :type snapshot: RetrievalSnapshot
        :return: None.
        :rtype: None
        """
        self._ensure_snapshots_dir()
        retriever_id = snapshot.configuration.retriever_id
        snapshot_dir = self.retrieval_dir / retriever_id / snapshot.snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / "manifest.json"
        path.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
        latest_path = self.retrieval_dir / retriever_id / "latest.json"
        latest_path.write_text(
            json.dumps(
                {"snapshot_id": snapshot.snapshot_id, "created_at": snapshot.created_at},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        catalog = self._load_catalog()
        catalog.latest_snapshot_id = snapshot.snapshot_id
        self._write_catalog(catalog)

    def load_snapshot(self, snapshot_id: str) -> RetrievalSnapshot:
        """
        Load a retrieval snapshot manifest by identifier.

        :param snapshot_id: Snapshot identifier.
        :type snapshot_id: str
        :return: Parsed snapshot manifest.
        :rtype: RetrievalSnapshot
        :raises FileNotFoundError: If the snapshot manifest does not exist.
        """
        legacy_path = self.snapshots_dir / f"{snapshot_id}.json"
        if legacy_path.is_file():
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
            return RetrievalSnapshot.model_validate(data)
        if not self.retrieval_dir.is_dir():
            raise FileNotFoundError(f"Missing snapshot manifest for: {snapshot_id}")
        for retriever_dir in sorted(self.retrieval_dir.iterdir()):
            if not retriever_dir.is_dir():
                continue
            manifest_path = retriever_dir / snapshot_id / "manifest.json"
            if manifest_path.is_file():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                return RetrievalSnapshot.model_validate(data)
        raise FileNotFoundError(f"Missing snapshot manifest for: {snapshot_id}")

    @property
    def latest_snapshot_id(self) -> Optional[str]:
        """
        Latest retrieval snapshot identifier recorded in the catalog.

        :return: Latest snapshot identifier or None.
        :rtype: str or None
        """
        return self._load_catalog().latest_snapshot_id

    def _upsert_catalog_item(self, item: CatalogItem) -> None:
        """
        Upsert a catalog item and reset the latest run pointer.

        :param item: Catalog item to insert or update.
        :type item: CatalogItem
        :return: None.
        :rtype: None
        """
        self._init_catalog()
        catalog = self._load_catalog()
        catalog.items[item.id] = item

        ordered_ids = [item_id for item_id in catalog.order if item_id != item.id]
        ordered_ids.insert(0, item.id)
        catalog.order = ordered_ids
        catalog.generated_at = utc_now_iso()
        catalog.latest_snapshot_id = None

        self._write_catalog(catalog)

    def ingest_item(
        self,
        data: bytes,
        *,
        filename: Optional[str] = None,
        media_type: str = "application/octet-stream",
        title: Optional[str] = None,
        tags: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        source_uri: str = "unknown",
        storage_subdir: Optional[str] = "imports",
    ) -> IngestResult:
        """
        Ingest a single raw item into the corpus.

        This is the modality-neutral primitive: callers provide bytes + a media type.
        Higher-level conveniences (ingest_note, ingest_source, and related methods) build on top.

        :param data: Raw item bytes.
        :type data: bytes
        :param filename: Optional filename for the stored item.
        :type filename: str or None
        :param media_type: Internet Assigned Numbers Authority media type for the item.
        :type media_type: str
        :param title: Optional title metadata.
        :type title: str or None
        :param tags: Tags to associate with the item.
        :type tags: Sequence[str]
        :param metadata: Optional metadata mapping.
        :type metadata: dict[str, Any] or None
        :param source_uri: Source uniform resource identifier for provenance.
        :type source_uri: str
        :param storage_subdir: Optional subdirectory under the raw root.
        :type storage_subdir: str or None
        :return: Ingestion result summary.
        :rtype: IngestResult
        :raises ValueError: If markdown is not Unicode Transformation Format 8.
        :raises IngestCollisionError: If a source uniform resource identifier is already ingested.
        """
        self._ensure_local_ingest_allowed()
        existing_item = self._find_item_by_source_uri(source_uri)
        if existing_item is not None:
            raise IngestCollisionError(
                source_uri=source_uri,
                existing_item_id=existing_item.id,
                existing_relpath=existing_item.relpath,
            )

        item_id = str(uuid.uuid4())
        storage_filename = _storage_filename_for_ingest(
            filename=filename, media_type=media_type, source_uri=source_uri
        )

        if media_type == "text/markdown":
            output_name = f"{item_id}--{storage_filename}" if storage_filename else f"{item_id}.md"
        else:
            if storage_filename:
                output_name = f"{item_id}--{storage_filename}"
            else:
                extension = _preferred_extension_for_media_type(media_type) or ""
                output_name = f"{item_id}{extension}" if extension else f"{item_id}"

        relpath = self._raw_relpath(output_name=output_name, storage_subdir=storage_subdir)
        output_path = self.root / relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_title = title.strip() if isinstance(title, str) and title.strip() else None
        resolved_tags = list(tags)
        metadata_input: Dict[str, Any] = dict(metadata or {})
        if resolved_title and "title" not in metadata_input:
            metadata_input["title"] = resolved_title
        if resolved_tags and "tags" not in metadata_input:
            metadata_input["tags"] = list(resolved_tags)

        if self._hooks is not None:
            mutation = self._hooks.run_ingest_hooks(
                hook_point=HookPoint.before_ingest,
                filename=filename,
                media_type=media_type,
                title=resolved_title,
                tags=list(resolved_tags),
                metadata=dict(metadata_input),
                source_uri=source_uri,
            )
            if mutation.add_tags:
                for tag in mutation.add_tags:
                    if tag not in resolved_tags:
                        resolved_tags.append(tag)

        frontmatter: Dict[str, Any] = {}

        if media_type == "text/markdown":
            try:
                markdown_text = data.decode("utf-8")
            except UnicodeDecodeError as decode_error:
                raise ValueError(
                    "Markdown must be Unicode Transformation Format 8"
                ) from decode_error

            parsed_document = parse_front_matter(markdown_text)
            frontmatter = dict(parsed_document.metadata)

            merged_tags = _merge_tags(resolved_tags, frontmatter.get("tags"))
            if merged_tags:
                frontmatter["tags"] = merged_tags
            resolved_tags = merged_tags

            if resolved_title and not (
                isinstance(frontmatter.get("title"), str) and frontmatter.get("title").strip()
            ):
                frontmatter["title"] = resolved_title

            title_value = frontmatter.get("title")
            if isinstance(title_value, str) and title_value.strip():
                resolved_title = title_value.strip()

            frontmatter = _ensure_biblicus_block(
                frontmatter, item_id=item_id, source_uri=source_uri
            )
            rendered_document = render_front_matter(frontmatter, parsed_document.body)
            data_to_write = rendered_document.encode("utf-8")
        else:
            data_to_write = data

        sha256_digest = _sha256_bytes(data_to_write)
        output_path.write_bytes(data_to_write)

        if media_type != "text/markdown":
            sidecar: Dict[str, Any] = {}
            sidecar["media_type"] = media_type
            if resolved_tags:
                sidecar["tags"] = resolved_tags
            if metadata_input:
                for metadata_key, metadata_value in metadata_input.items():
                    if metadata_key in {"tags", "biblicus"}:
                        continue
                    sidecar[metadata_key] = metadata_value
            sidecar["biblicus"] = {"id": item_id, "source": source_uri}
            _write_sidecar(output_path, sidecar)
            frontmatter = sidecar

        if self._hooks is not None:
            mutation = self._hooks.run_ingest_hooks(
                hook_point=HookPoint.after_ingest,
                filename=filename,
                media_type=media_type,
                title=resolved_title,
                tags=list(resolved_tags),
                metadata=dict(metadata_input),
                source_uri=source_uri,
                item_id=item_id,
                relpath=relpath,
            )
            if mutation.add_tags:
                updated_tags = list(resolved_tags)
                for tag in mutation.add_tags:
                    if tag not in updated_tags:
                        updated_tags.append(tag)
                resolved_tags = updated_tags
                sidecar_metadata = _load_sidecar(output_path)
                sidecar_metadata["tags"] = resolved_tags
                if media_type != "text/markdown":
                    sidecar_metadata["media_type"] = media_type
                sidecar_metadata["biblicus"] = {"id": item_id, "source": source_uri}
                _write_sidecar(output_path, sidecar_metadata)
                frontmatter = _merge_metadata(
                    frontmatter if isinstance(frontmatter, dict) else {}, sidecar_metadata
                )

        created_at = utc_now_iso()
        item_record = CatalogItem(
            id=item_id,
            relpath=relpath,
            sha256=sha256_digest,
            bytes=len(data_to_write),
            media_type=media_type,
            title=resolved_title,
            tags=list(resolved_tags),
            metadata=dict(frontmatter or {}),
            created_at=created_at,
            source_uri=source_uri,
        )
        self._upsert_catalog_item(item_record)

        return IngestResult(item_id=item_id, relpath=relpath, sha256=sha256_digest)

    def ingest_item_stream(
        self,
        stream,
        *,
        filename: Optional[str] = None,
        media_type: str = "application/octet-stream",
        tags: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        source_uri: str = "unknown",
        storage_subdir: Optional[str] = "imports",
    ) -> IngestResult:
        """
        Ingest a binary item from a readable stream.

        This method is intended for large non-markdown items. It writes bytes to disk incrementally
        while computing a checksum.

        :param stream: Readable binary stream.
        :type stream: object
        :param filename: Optional filename for the stored item.
        :type filename: str or None
        :param media_type: Internet Assigned Numbers Authority media type for the item.
        :type media_type: str
        :param tags: Tags to associate with the item.
        :type tags: Sequence[str]
        :param metadata: Optional metadata mapping.
        :type metadata: dict[str, Any] or None
        :param source_uri: Source uniform resource identifier for provenance.
        :type source_uri: str
        :param storage_subdir: Optional subdirectory under the raw root.
        :type storage_subdir: str or None
        :return: Ingestion result summary.
        :rtype: IngestResult
        :raises ValueError: If the media_type is text/markdown.
        """
        self._ensure_local_ingest_allowed()
        if media_type == "text/markdown":
            raise ValueError("Stream ingestion is not supported for Markdown")

        existing_item = self._find_item_by_source_uri(source_uri)
        if existing_item is not None:
            raise IngestCollisionError(
                source_uri=source_uri,
                existing_item_id=existing_item.id,
                existing_relpath=existing_item.relpath,
            )

        item_id = str(uuid.uuid4())
        storage_filename = _storage_filename_for_ingest(
            filename=filename, media_type=media_type, source_uri=source_uri
        )

        if storage_filename:
            output_name = f"{item_id}--{storage_filename}"
        else:
            extension = _preferred_extension_for_media_type(media_type) or ""
            output_name = f"{item_id}{extension}" if extension else f"{item_id}"

        relpath = self._raw_relpath(output_name=output_name, storage_subdir=storage_subdir)
        output_path = self.root / relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_tags = list(tags)
        metadata_input: Dict[str, Any] = dict(metadata or {})
        if resolved_tags and "tags" not in metadata_input:
            metadata_input["tags"] = list(resolved_tags)

        if self._hooks is not None:
            mutation = self._hooks.run_ingest_hooks(
                hook_point=HookPoint.before_ingest,
                filename=filename,
                media_type=media_type,
                title=None,
                tags=list(resolved_tags),
                metadata=dict(metadata_input),
                source_uri=source_uri,
            )
            if mutation.add_tags:
                for tag in mutation.add_tags:
                    if tag not in resolved_tags:
                        resolved_tags.append(tag)

        write_result = _write_stream_and_hash(stream, output_path)
        sha256_digest = str(write_result["sha256"])
        bytes_written = int(write_result["bytes_written"])

        sidecar: Dict[str, Any] = {}
        sidecar["media_type"] = media_type
        if resolved_tags:
            sidecar["tags"] = resolved_tags
        if metadata_input:
            for metadata_key, metadata_value in metadata_input.items():
                if metadata_key in {"tags", "biblicus"}:
                    continue
                sidecar[metadata_key] = metadata_value
        sidecar["biblicus"] = {"id": item_id, "source": source_uri}
        _write_sidecar(output_path, sidecar)

        if self._hooks is not None:
            mutation = self._hooks.run_ingest_hooks(
                hook_point=HookPoint.after_ingest,
                filename=filename,
                media_type=media_type,
                title=None,
                tags=list(resolved_tags),
                metadata=dict(metadata_input),
                source_uri=source_uri,
                item_id=item_id,
                relpath=relpath,
            )
            if mutation.add_tags:
                updated_tags = list(resolved_tags)
                for tag in mutation.add_tags:
                    if tag not in updated_tags:
                        updated_tags.append(tag)
                resolved_tags = updated_tags
                sidecar["tags"] = resolved_tags
                _write_sidecar(output_path, sidecar)

        created_at = utc_now_iso()
        item_record = CatalogItem(
            id=item_id,
            relpath=relpath,
            sha256=sha256_digest,
            bytes=bytes_written,
            media_type=media_type,
            title=None,
            tags=list(resolved_tags),
            metadata=dict(sidecar or {}),
            created_at=created_at,
            source_uri=source_uri,
        )
        self._upsert_catalog_item(item_record)

        return IngestResult(item_id=item_id, relpath=relpath, sha256=sha256_digest)

    def ingest_note(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        tags: Sequence[str] = (),
        source_uri: Optional[str] = None,
    ) -> IngestResult:
        """
        Ingest a text note as Markdown.

        :param text: Note content.
        :type text: str
        :param title: Optional title metadata.
        :type title: str or None
        :param tags: Tags to associate with the note.
        :type tags: Sequence[str]
        :param source_uri: Optional source uniform resource identifier for provenance.
        :type source_uri: str or None
        :return: Ingestion result summary.
        :rtype: IngestResult
        """
        if source_uri is None:
            digest_source = (title or "") + "\n" + text
            digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
            source_uri = f"text:{digest}"
        data = text.encode("utf-8")
        return self.ingest_item(
            data,
            filename=None,
            media_type="text/markdown",
            title=title,
            tags=tags,
            metadata=None,
            source_uri=source_uri,
            storage_subdir="notes",
        )

    def _register_existing_file(
        self,
        *,
        path: Path,
        tags: Sequence[str],
        metadata: Optional[Dict[str, Any]],
        source_uri: str,
    ) -> IngestResult:
        sanitized_name = _sanitize_filename(path.name)
        if sanitized_name != path.name:
            destination = path.with_name(sanitized_name)
            if destination.exists():
                raise IngestCollisionError(
                    source_uri=source_uri,
                    existing_item_id="unknown",
                    existing_relpath=str(destination.relative_to(self.root)),
                )
            path = path.rename(destination)
        if self._is_reserved_path(path):
            raise ValueError("Cannot ingest files inside reserved corpus folders")
        existing_item = self._find_item_by_source_uri(source_uri)
        if existing_item is not None:
            raise IngestCollisionError(
                source_uri=source_uri,
                existing_item_id=existing_item.id,
                existing_relpath=existing_item.relpath,
            )

        data = path.read_bytes()
        relpath = str(path.relative_to(self.root))

        media_type, _ = mimetypes.guess_type(path.name)
        media_type = media_type or "application/octet-stream"
        if path.suffix.lower() in {".md", ".markdown"}:
            media_type = "text/markdown"

        frontmatter: Dict[str, Any] = {}
        markdown_body: Optional[str] = None
        if media_type == "text/markdown":
            try:
                decoded = data.decode("utf-8")
                parsed_document = parse_front_matter(decoded)
            except UnicodeDecodeError as decode_error:
                raise ValueError(
                    "Markdown file must be Unicode Transformation Format 8"
                ) from decode_error
            frontmatter = dict(parsed_document.metadata)
            markdown_body = parsed_document.body

        sidecar = _load_sidecar(path)
        merged_metadata = _merge_metadata(frontmatter, sidecar)
        resolved_tags = _merge_tags(_merge_tags([], tags), merged_metadata.get("tags"))
        if metadata:
            for metadata_key, metadata_value in metadata.items():
                if metadata_key in {"tags", "biblicus"}:
                    continue
                merged_metadata[metadata_key] = metadata_value

        biblicus_block = merged_metadata.get("biblicus")
        item_id: Optional[str] = None
        if isinstance(biblicus_block, dict):
            biblicus_id = biblicus_block.get("id")
            if isinstance(biblicus_id, str):
                try:
                    item_id = str(uuid.UUID(biblicus_id))
                except ValueError:
                    item_id = None

        if item_id is None:
            item_id = str(uuid.uuid4())

        if media_type == "text/markdown":
            updated_metadata: Dict[str, Any] = dict(merged_metadata)
            if resolved_tags:
                updated_metadata["tags"] = resolved_tags
            updated_metadata = _ensure_biblicus_block(
                updated_metadata, item_id=item_id, source_uri=source_uri
            )
            if markdown_body is None:
                markdown_body = ""
            rendered_document = render_front_matter(updated_metadata, markdown_body)
            path.write_text(rendered_document, encoding="utf-8")
            data = rendered_document.encode("utf-8")
            merged_metadata = updated_metadata
        else:
            sidecar_metadata: Dict[str, Any] = dict(merged_metadata)
            sidecar_metadata["biblicus"] = {"id": item_id, "source": source_uri}
            if resolved_tags:
                sidecar_metadata["tags"] = resolved_tags
            sidecar_metadata["media_type"] = media_type
            _write_sidecar(path, sidecar_metadata)
            merged_metadata = sidecar_metadata

        sha256_digest = _sha256_bytes(data)
        created_at = utc_now_iso()
        item_record = CatalogItem(
            id=item_id,
            relpath=relpath,
            sha256=sha256_digest,
            bytes=len(data),
            media_type=media_type,
            title=merged_metadata.get("title")
            if isinstance(merged_metadata.get("title"), str)
            else None,
            tags=list(resolved_tags),
            metadata=dict(merged_metadata),
            created_at=created_at,
            source_uri=source_uri,
        )
        self._upsert_catalog_item(item_record)
        return IngestResult(item_id=item_id, relpath=relpath, sha256=sha256_digest)

    def ingest_source(
        self,
        source: str | Path,
        *,
        tags: Sequence[str] = (),
        source_uri: Optional[str] = None,
        allow_external: bool = False,
    ) -> IngestResult:
        """
        Ingest a file path or uniform resource locator source.

        :param source: File path or uniform resource locator.
        :type source: str or Path
        :param tags: Tags to associate with the item.
        :type tags: Sequence[str]
        :param source_uri: Optional override for the source uniform resource identifier.
        :type source_uri: str or None
        :param allow_external: Whether to ingest files outside the corpus root by copying them into imports.
        :type allow_external: bool
        :return: Ingestion result summary.
        :rtype: IngestResult
        """
        candidate_path = Path(source) if isinstance(source, str) and "://" not in source else None
        if isinstance(source, Path) or (candidate_path is not None and candidate_path.exists()):
            path = source if isinstance(source, Path) else candidate_path
            assert isinstance(path, Path)
            path = path.resolve()
            if not allow_external and not path.is_relative_to(self.root):
                raise ValueError(
                    "Local ingest requires the file to be inside the corpus root. "
                    "Move the file into the corpus and run reindex."
                )
            resolved_source_uri = source_uri or path.as_uri()
            data = path.read_bytes()
            media_type, _ = mimetypes.guess_type(path.name)
            media_type = media_type or "application/octet-stream"
            if path.suffix.lower() in {".md", ".markdown"}:
                media_type = "text/markdown"
            return self.ingest_item(
                data,
                filename=path.name,
                media_type=media_type,
                title=None,
                tags=tags,
                metadata=None,
                source_uri=resolved_source_uri,
                storage_subdir="imports",
            )

        payload = load_source(source, source_uri=source_uri)
        return self.ingest_item(
            payload.data,
            filename=payload.filename,
            media_type=payload.media_type,
            title=None,
            tags=tags,
            metadata=None,
            source_uri=payload.source_uri,
            storage_subdir="imports",
        )

    def import_tree(self, source_root: Path, *, tags: Sequence[str] = ()) -> Dict[str, int]:
        """
        Import a folder tree into the corpus, preserving relative paths and provenance.

        Imported content must already live under the corpus root. The import registers files
        in-place and writes sidecars when needed.

        :param source_root: Root directory of the folder tree to import.
        :type source_root: Path
        :param tags: Tags to associate with imported items.
        :type tags: Sequence[str]
        :return: Import statistics.
        :rtype: dict[str, int]
        :raises FileNotFoundError: If the source_root does not exist.
        :raises ValueError: If the source root is outside the corpus root.
        """
        self._ensure_local_ingest_allowed()
        source_root = source_root.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Import source root does not exist: {source_root}")
        if not source_root.is_relative_to(self.root):
            raise ValueError(
                "Import requires the source folder to live inside the corpus root. "
                "Move it under the corpus and reindex."
            )

        ignore_spec = load_corpus_ignore_spec(self.root)
        stats = {"scanned": 0, "ignored": 0, "imported": 0}

        for source_path in sorted(source_root.rglob("*")):
            if not source_path.is_file():
                continue
            relative_root_path = source_path.relative_to(self.root)
            relative_source_path = relative_root_path.as_posix()
            stats["scanned"] += 1
            if ignore_spec.matches(relative_source_path):
                stats["ignored"] += 1
                continue
            if self._is_reserved_path(relative_root_path):
                stats["ignored"] += 1
                continue
            self._register_existing_file(
                path=source_path,
                tags=tags,
                metadata=None,
                source_uri=source_path.as_uri(),
            )
            stats["imported"] += 1

        return stats

    def _import_file(
        self,
        *,
        source_path: Path,
        import_id: str,
        relative_source_path: str,
        tags: Sequence[str],
    ) -> None:
        """
        Import a single file into the corpus under an import namespace.

        :param source_path: Source file path to import.
        :type source_path: Path
        :param import_id: Import identifier.
        :type import_id: str
        :param relative_source_path: Relative path within the imported tree.
        :type relative_source_path: str
        :param tags: Tags to apply.
        :type tags: Sequence[str]
        :return: None.
        :rtype: None
        :raises ValueError: If a markdown file cannot be decoded as Unicode Transformation Format 8.
        """
        item_id = str(uuid.uuid4())
        destination_relpath = self._raw_relpath(
            output_name=relative_source_path,
            storage_subdir=str(Path("imports") / import_id),
        )
        destination_path = (self.root / destination_relpath).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        raw_bytes = source_path.read_bytes()
        sha256_digest = _sha256_bytes(raw_bytes)

        media_type, _ = mimetypes.guess_type(source_path.name)
        media_type = media_type or "application/octet-stream"
        if source_path.suffix.lower() in {".md", ".markdown"}:
            media_type = "text/markdown"

        title: Optional[str] = None
        frontmatter_metadata: Dict[str, Any] = {}
        if media_type == "text/markdown":
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as decode_error:
                raise ValueError(
                    f"Markdown file must be Unicode Transformation Format 8: {relative_source_path}"
                ) from decode_error
            parsed_document = parse_front_matter(text)
            frontmatter_metadata = dict(parsed_document.metadata)
            title_value = frontmatter_metadata.get("title")
            if isinstance(title_value, str) and title_value.strip():
                title = title_value.strip()

        destination_path.write_bytes(raw_bytes)

        sidecar: Dict[str, Any] = {}
        if tags:
            sidecar["tags"] = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        if media_type != "text/markdown":
            sidecar["media_type"] = media_type
        sidecar["biblicus"] = {"id": item_id, "source": source_path.as_uri()}
        _write_sidecar(destination_path, sidecar)

        merged_metadata = _merge_metadata(frontmatter_metadata, sidecar)
        resolved_tags = _merge_tags([], merged_metadata.get("tags"))

        item_record = CatalogItem(
            id=item_id,
            relpath=destination_relpath,
            sha256=sha256_digest,
            bytes=len(raw_bytes),
            media_type=media_type,
            title=title,
            tags=list(resolved_tags),
            metadata=dict(merged_metadata or {}),
            created_at=utc_now_iso(),
            source_uri=source_path.as_uri(),
        )
        self._upsert_catalog_item(item_record)

    def list_items(self, *, limit: int = 50) -> List[CatalogItem]:
        """
        List items from the catalog.

        :param limit: Maximum number of items to return.
        :type limit: int
        :return: Catalog items ordered by recency.
        :rtype: list[CatalogItem]
        """
        catalog = self._load_catalog()
        ordered_ids = catalog.order[:limit] if catalog.order else list(catalog.items.keys())[:limit]
        collected_items: List[CatalogItem] = []
        for item_id in ordered_ids:
            item = catalog.items.get(item_id)
            if item is not None:
                collected_items.append(item)
        return collected_items

    def get_item(self, item_id: str) -> CatalogItem:
        """
        Fetch a catalog item by identifier.

        :param item_id: Item identifier.
        :type item_id: str
        :return: Catalog item.
        :rtype: CatalogItem
        :raises KeyError: If the item identifier is unknown.
        """
        catalog = self._load_catalog()
        item = catalog.items.get(item_id)
        if item is None:
            raise KeyError(f"Unknown item identifier: {item_id}")
        return item

    def create_crawl_id(self) -> str:
        """
        Create a new crawl identifier.

        :return: Crawl identifier.
        :rtype: str
        """
        return str(uuid.uuid4())

    def ingest_crawled_payload(
        self,
        *,
        crawl_id: str,
        relative_path: str,
        data: bytes,
        filename: str,
        media_type: str,
        source_uri: str,
        tags: Sequence[str],
    ) -> None:
        """
        Ingest a crawled payload under a crawl import namespace.

        :param crawl_id: Crawl identifier used to group crawled artifacts.
        :type crawl_id: str
        :param relative_path: Relative path within the crawl prefix.
        :type relative_path: str
        :param data: Raw payload bytes.
        :type data: bytes
        :param filename: Suggested filename from the payload metadata.
        :type filename: str
        :param media_type: Internet Assigned Numbers Authority media type.
        :type media_type: str
        :param source_uri: Source uniform resource identifier (typically an http or https uniform resource locator).
        :type source_uri: str
        :param tags: Tags to attach to the stored item.
        :type tags: Sequence[str]
        :return: None.
        :rtype: None
        """
        self._ensure_local_ingest_allowed()
        _ = filename
        item_id = str(uuid.uuid4())
        destination_relpath = self._raw_relpath(
            output_name=relative_path,
            storage_subdir=str(Path("imports") / "crawl" / crawl_id),
        )
        destination_path = (self.root / destination_relpath).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(data)

        sha256_digest = _sha256_bytes(data)

        sidecar: Dict[str, Any] = {}
        sidecar["tags"] = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
        sidecar["media_type"] = media_type
        sidecar["biblicus"] = {"id": item_id, "source": source_uri}
        _write_sidecar(destination_path, sidecar)

        merged_metadata = _merge_metadata({}, sidecar)
        resolved_tags = _merge_tags([], merged_metadata.get("tags"))

        item_record = CatalogItem(
            id=item_id,
            relpath=destination_relpath,
            sha256=sha256_digest,
            bytes=len(data),
            media_type=media_type,
            title=None,
            tags=list(resolved_tags),
            metadata=dict(merged_metadata or {}),
            created_at=utc_now_iso(),
            source_uri=source_uri,
        )
        self._upsert_catalog_item(item_record)

    def pull_source(
        self, *, tag_resolver: Optional[Callable[[str], List[str]]] = None
    ) -> RemoteSourcePullResult:
        """
        Mirror a remote source into the corpus.

        :return: Pull summary.
        :rtype: RemoteSourcePullResult
        :raises ValueError: If the corpus has no configured remote source.
        """
        if self.config is None or self.config.source is None:
            raise ValueError("Remote source is not configured for this corpus.")
        source_config = self.config.source
        source_name = self._resolve_remote_source_name(source_config)
        storage_subdir = str(Path("imports") / "remote" / source_name)
        ignore_spec = load_corpus_ignore_spec(self.root)

        profile = resolve_source_profile(source_config.profile)
        if source_config.kind != profile.kind:
            raise ValueError(
                "Remote source kind does not match source profile kind: "
                f"{source_config.kind} vs {profile.kind}"
            )
        if source_config.kind == "s3":
            source = S3RemoteSource(source_config, profile)
        elif source_config.kind == "azure-blob":
            source = AzureBlobRemoteSource(source_config, profile)
        else:
            raise ValueError(f"Unsupported remote source kind: {source_config.kind}")

        result = RemoteSourcePullResult()
        items = source.list_items()
        result.listed = len(items)
        remote_uris = set()

        for item in items:
            remote_uris.add(item.source_uri)
            relative_key = self._relative_remote_key(item.key, prefix=source_config.prefix)
            if not relative_key:
                result.skipped += 1
                continue
            if ignore_spec.matches(relative_key):
                result.skipped += 1
                continue
            existing_item = self._find_item_by_source_uri(item.source_uri)
            if existing_item is not None and self._remote_item_unchanged(existing_item, item):
                result.skipped += 1
                continue
            content, content_type = source.fetch_bytes(item)
            relpath = self._raw_relpath(output_name=relative_key, storage_subdir=storage_subdir)
            if existing_item is not None and existing_item.relpath != relpath:
                self._delete_item_files(existing_item.relpath)
            extra_tags = tag_resolver(relative_key) if tag_resolver is not None else []
            catalog_item = self._write_remote_item(
                data=content,
                relpath=relpath,
                source_uri=item.source_uri,
                source_etag=item.etag,
                source_last_modified=item.last_modified,
                content_type=content_type or item.content_type,
                item_id=existing_item.id if existing_item is not None else None,
                created_at=existing_item.created_at if existing_item is not None else None,
                extra_tags=extra_tags,
            )
            self._upsert_catalog_item(catalog_item)
            if existing_item is None:
                result.downloaded += 1
            else:
                result.updated += 1

        result.pruned = self._prune_remote_items(
            storage_subdir=storage_subdir, remote_uris=remote_uris
        )
        self.reindex()
        return result

    def _resolve_remote_source_name(self, source_config) -> str:
        if source_config.name and source_config.name.strip():
            return _sanitize_filename(source_config.name.strip())
        if source_config.kind == "s3" and source_config.bucket:
            return _sanitize_filename(source_config.bucket)
        if source_config.kind == "azure-blob" and source_config.container:
            return _sanitize_filename(source_config.container)
        return "remote"

    def _relative_remote_key(self, key: str, *, prefix: Optional[str]) -> str:
        relative_key = key
        if prefix and relative_key.startswith(prefix):
            relative_key = relative_key[len(prefix) :]
        return relative_key.lstrip("/")

    def _remote_item_unchanged(self, existing_item: CatalogItem, item) -> bool:
        metadata = existing_item.metadata or {}
        biblicus_block = metadata.get("biblicus") if isinstance(metadata, dict) else None
        if not isinstance(biblicus_block, dict):
            biblicus_block = {}
        existing_etag = biblicus_block.get("source_etag")
        existing_last_modified = biblicus_block.get("source_last_modified")
        if item.etag and existing_etag == item.etag:
            return True
        if not item.etag and item.last_modified and existing_last_modified == item.last_modified:
            return True
        return False

    def _write_remote_item(
        self,
        *,
        data: bytes,
        relpath: str,
        source_uri: str,
        source_etag: Optional[str],
        source_last_modified: Optional[str],
        content_type: Optional[str],
        item_id: Optional[str],
        created_at: Optional[str],
        extra_tags: Optional[Sequence[str]] = None,
    ) -> CatalogItem:
        output_path = self.root / relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_type = None
        if content_type:
            normalized_type = content_type.split(";", 1)[0].strip()
        media_type = normalized_type or _media_type_from_filename(output_path.name)
        if output_path.suffix.lower() in {".md", ".markdown"}:
            media_type = "text/markdown"

        resolved_item_id = item_id or str(uuid.uuid4())
        metadata: Dict[str, Any] = {}
        title: Optional[str] = None
        tags: List[str] = []

        if media_type == "text/markdown":
            try:
                markdown_text = data.decode("utf-8")
            except UnicodeDecodeError as decode_error:
                raise ValueError(
                    f"Markdown must be Unicode Transformation Format 8: {output_path.name}"
                ) from decode_error
            sidecar_path = _sidecar_path_for(output_path)
            if sidecar_path.exists():
                sidecar_path.unlink()
            parsed_document = parse_front_matter(markdown_text)
            frontmatter = dict(parsed_document.metadata)
            frontmatter = _ensure_biblicus_block(
                frontmatter, item_id=resolved_item_id, source_uri=source_uri
            )
            frontmatter = _update_biblicus_block(
                frontmatter,
                {
                    "source_etag": source_etag,
                    "source_last_modified": source_last_modified,
                },
            )
            if extra_tags:
                merged_tags = _merge_tags([], frontmatter.get("tags"))
                merged_tags = _merge_tags(merged_tags, extra_tags)
                frontmatter["tags"] = list(merged_tags)
            rendered_document = render_front_matter(frontmatter, parsed_document.body)
            data_to_write = rendered_document.encode("utf-8")
            metadata = frontmatter
        else:
            data_to_write = data
            sidecar: Dict[str, Any] = {}
            sidecar["media_type"] = media_type
            sidecar = _ensure_biblicus_block(
                sidecar, item_id=resolved_item_id, source_uri=source_uri
            )
            sidecar = _update_biblicus_block(
                sidecar,
                {
                    "source_etag": source_etag,
                    "source_last_modified": source_last_modified,
                },
            )
            if extra_tags:
                merged_tags = _merge_tags([], sidecar.get("tags"))
                merged_tags = _merge_tags(merged_tags, extra_tags)
                sidecar["tags"] = list(merged_tags)
            _write_sidecar(output_path, sidecar)
            metadata = sidecar

        title_value = metadata.get("title")
        if isinstance(title_value, str) and title_value.strip():
            title = title_value.strip()
        tags = _merge_tags([], metadata.get("tags"))

        sha256_digest = _sha256_bytes(data_to_write)
        output_path.write_bytes(data_to_write)

        return CatalogItem(
            id=resolved_item_id,
            relpath=relpath,
            sha256=sha256_digest,
            bytes=len(data_to_write),
            media_type=media_type,
            title=title,
            tags=list(tags),
            metadata=dict(metadata or {}),
            created_at=created_at or utc_now_iso(),
            source_uri=source_uri,
        )

    def _delete_item_files(self, relpath: str) -> None:
        content_path = self.root / relpath
        if content_path.exists():
            content_path.unlink()
        sidecar_path = _sidecar_path_for(content_path)
        if sidecar_path.exists():
            sidecar_path.unlink()

    def _prune_remote_items(self, *, storage_subdir: str, remote_uris: set[str]) -> int:
        catalog = self._load_catalog()
        prefix_path = self._raw_prefix_for_storage(storage_subdir)
        prefix_text = prefix_path.as_posix().rstrip("/") + "/"
        pruned = 0
        removed_ids = []
        for item_id, item in catalog.items.items():
            if not item.relpath.startswith(prefix_text):
                continue
            if item.source_uri in remote_uris:
                continue
            self._delete_item_files(item.relpath)
            removed_ids.append(item_id)
            pruned += 1
        if removed_ids:
            for item_id in removed_ids:
                catalog.items.pop(item_id, None)
            catalog.order = [item_id for item_id in catalog.order if item_id not in removed_ids]
            catalog.generated_at = utc_now_iso()
            catalog.latest_snapshot_id = None
            self._write_catalog(catalog)
        return pruned

    def reindex(self) -> Dict[str, int]:
        """
        Rebuild/refresh the corpus catalog from the current on-disk corpus contents.

        This is the core "mutable corpus with re-indexing" loop: edit raw files or sidecars,
        then reindex to refresh the derived catalog.

        :return: Reindex statistics.
        :rtype: dict[str, int]
        :raises ValueError: If a markdown file cannot be decoded as Unicode Transformation Format 8.
        """
        self._init_catalog()
        existing_catalog = self._load_catalog()
        stats = {"scanned": 0, "skipped": 0, "inserted": 0, "updated": 0}

        if self.raw_dir == self.root:
            content_files = [
                content_path
                for content_path in self.root.rglob("*")
                if content_path.is_file()
                and not content_path.name.endswith(SIDECAR_SUFFIX)
                and not self._is_reserved_path(content_path)
            ]
        else:
            content_files = [
                content_path
                for content_path in self.raw_dir.rglob("*")
                if content_path.is_file() and not content_path.name.endswith(SIDECAR_SUFFIX)
            ]

        new_items: Dict[str, CatalogItem] = {}

        for content_path in content_files:
            stats["scanned"] += 1
            relpath = str(content_path.relative_to(self.root))
            data = content_path.read_bytes()
            sha256 = _sha256_bytes(data)

            media_type, _ = mimetypes.guess_type(content_path.name)
            media_type = media_type or "application/octet-stream"

            sidecar = _load_sidecar(content_path)

            frontmatter: Dict[str, Any] = {}
            if content_path.suffix.lower() in {".md", ".markdown"}:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError as decode_error:
                    raise ValueError(
                        f"Markdown file must be Unicode Transformation Format 8: {relpath}"
                    ) from decode_error
                parsed_document = parse_front_matter(text)
                frontmatter = parsed_document.metadata
                media_type = "text/markdown"

            merged_metadata = _merge_metadata(frontmatter, sidecar)

            if media_type != "text/markdown":
                media_type_override = merged_metadata.get("media_type")
                if isinstance(media_type_override, str) and media_type_override.strip():
                    media_type = media_type_override.strip()

            item_id: Optional[str] = None
            biblicus_block = merged_metadata.get("biblicus")
            if isinstance(biblicus_block, dict):
                biblicus_id = biblicus_block.get("id")
                if isinstance(biblicus_id, str):
                    try:
                        item_id = str(uuid.UUID(biblicus_id))
                    except ValueError:
                        item_id = None

            if item_id is None:
                item_id = _parse_uuid_prefix(content_path.name)

            if item_id is None:
                stats["skipped"] += 1
                continue

            title: Optional[str] = None
            title_value = merged_metadata.get("title")
            if isinstance(title_value, str) and title_value.strip():
                title = title_value.strip()

            resolved_tags = _merge_tags([], merged_metadata.get("tags"))

            source_uri: Optional[str] = None
            if isinstance(biblicus_block, dict):
                source_value = biblicus_block.get("source")
                if isinstance(source_value, str) and source_value.strip():
                    source_uri = source_value.strip()

            previous_item = existing_catalog.items.get(item_id)
            created_at = previous_item.created_at if previous_item is not None else utc_now_iso()
            source_uri = source_uri or (
                previous_item.source_uri if previous_item is not None else None
            )

            if previous_item is None:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1

            new_items[item_id] = CatalogItem(
                id=item_id,
                relpath=relpath,
                sha256=sha256,
                bytes=len(data),
                media_type=media_type,
                title=title,
                tags=list(resolved_tags),
                metadata=dict(merged_metadata or {}),
                created_at=created_at,
                source_uri=source_uri,
            )

        order = sorted(
            new_items.keys(),
            key=lambda item_id: (new_items[item_id].created_at, item_id),
            reverse=True,
        )

        catalog = CorpusCatalog(
            schema_version=SCHEMA_VERSION,
            generated_at=utc_now_iso(),
            corpus_uri=normalize_corpus_uri(self.root),
            raw_dir=DEFAULT_RAW_DIR,
            latest_snapshot_id=None,
            items=new_items,
            order=order,
        )
        self._write_catalog(catalog)

        return stats

    @property
    def name(self) -> str:
        """
        Return the corpus name (directory basename).

        :return: Corpus name.
        :rtype: str
        """
        return self.root.name

    def purge(self, *, confirm: str) -> None:
        """
        Delete all ingested items and derived files, preserving corpus identity/config.

        :param confirm: Confirmation string matching the corpus name.
        :type confirm: str
        :return: None.
        :rtype: None
        :raises ValueError: If the confirmation does not match.
        """
        expected = self.name
        if confirm != expected:
            raise ValueError(
                f"Confirmation mismatch: pass --confirm {expected!r} to purge this corpus"
            )

        if self.raw_dir == self.root:
            for path in self.root.iterdir():
                if path.name in self._reserved_dir_names():
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        elif self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        for path in self.meta_dir.iterdir():
            if path.name == "config.json":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        for derived_dir in [
            self.extracted_dir,
            self.graph_dir,
            self.retrieval_dir,
            self.analysis_dir,
        ]:
            if derived_dir.exists():
                shutil.rmtree(derived_dir)
        self._init_catalog()
        self._write_catalog(
            CorpusCatalog(
                schema_version=SCHEMA_VERSION,
                generated_at=utc_now_iso(),
                corpus_uri=normalize_corpus_uri(self.root),
                raw_dir=DEFAULT_RAW_DIR,
                latest_snapshot_id=None,
                items={},
                order=[],
            )
        )
