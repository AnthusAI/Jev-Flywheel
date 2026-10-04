"""
Remote collection mirroring for Biblicus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List

from .corpus import Corpus
from .models import (
    CollectionMembership,
    CorpusConfig,
    RemoteCollectionPullResult,
    RemoteCorpusCollectionConfig,
    RemoteCorpusCollectionDiscovery,
    RemoteCorpusSourceConfig,
)
from .remote_sources import AzureBlobRemoteSource, S3RemoteSource
from .time import utc_now_iso
from .user_config import resolve_source_profile

COLLECTION_CONFIG_NAME = "config.json"
COLLECTION_DIR_NAME = "collections"
ARCHIVE_DIR_NAME = ".archived"


def collection_config_path(collection_root: Path) -> Path:
    """
    Return the path to the collection configuration file.

    :param collection_root: Collection root directory.
    :type collection_root: Path
    :return: Configuration file path.
    :rtype: Path
    """
    return collection_root / "metadata" / COLLECTION_CONFIG_NAME


def load_collection_config(collection_root: Path) -> RemoteCorpusCollectionConfig:
    """
    Load the collection configuration from disk.

    :param collection_root: Collection root directory.
    :type collection_root: Path
    :return: Parsed collection configuration.
    :rtype: RemoteCorpusCollectionConfig
    :raises FileNotFoundError: If the config file is missing.
    :raises ValueError: If the config file is invalid.
    """
    config_path = collection_config_path(collection_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing collection config: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return RemoteCorpusCollectionConfig.model_validate(raw)


def init_collection(collection_root: Path, config: RemoteCorpusCollectionConfig) -> None:
    """
    Initialize a collection directory on disk.

    :param collection_root: Collection root directory.
    :type collection_root: Path
    :param config: Collection configuration to persist.
    :type config: RemoteCorpusCollectionConfig
    :return: None.
    :rtype: None
    """
    metadata_dir = collection_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_path = metadata_dir / COLLECTION_CONFIG_NAME
    config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


def pull_collection(collection_root: Path) -> RemoteCollectionPullResult:
    """
    Mirror a remote collection into local corpora.

    :param collection_root: Collection root directory.
    :type collection_root: Path
    :return: Collection pull summary.
    :rtype: RemoteCollectionPullResult
    """
    collection_root = collection_root.resolve()
    config = load_collection_config(collection_root)
    profile = resolve_source_profile(config.source.profile)
    if config.source.kind != profile.kind:
        raise ValueError(
            "Collection source kind does not match source profile kind: "
            f"{config.source.kind} vs {profile.kind}"
        )

    if config.source.kind == "s3":
        source = S3RemoteSource(config.source, profile)
    elif config.source.kind == "azure-blob":
        source = AzureBlobRemoteSource(config.source, profile)
    else:
        raise ValueError(f"Unsupported remote source kind: {config.source.kind}")

    result = RemoteCollectionPullResult()
    discovery = config.discovery
    corpus_root = _resolve_corpus_root(collection_root, config)

    if discovery.mode == "partition":
        corpus = _ensure_partition_corpus(
            corpus_root,
            collection_name=config.collection_name,
            source_config=_build_partition_source_config(config.source),
        )
        tag_resolver = _partition_tag_resolver()
        mirror = corpus.pull_source(tag_resolver=tag_resolver)
        result.discovered = len(_discover_subfolders(source, config.source, discovery))
        result.mirrored = 1 if mirror.listed > 0 else 0
        return result

    subfolders = _discover_subfolders(source, config.source, discovery)
    result.discovered = len(subfolders)

    created = 0
    mirrored = 0
    for subfolder in subfolders:
        corpus_path = corpus_root / subfolder
        created += int(
            _ensure_collection_corpus(
                corpus_path,
                collection_name=config.collection_name,
                corpus_name=subfolder,
                source_config=_build_subfolder_source_config(config.source, subfolder),
                auto_create=config.auto_create,
            )
        )
        corpus = Corpus.open(corpus_path)
        corpus.pull_source()
        mirrored += 1

    result.created = created
    result.mirrored = mirrored
    result.archived = _archive_missing_corpora(
        corpus_root, subfolders, deletion_policy=config.deletion_policy
    )
    return result


def _resolve_corpus_root(collection_root: Path, config: RemoteCorpusCollectionConfig) -> Path:
    base = collection_root.parent
    if base.name == COLLECTION_DIR_NAME:
        base = base.parent
    corpus_root = Path(config.corpus_root)
    if corpus_root.is_absolute():
        return corpus_root
    return (base / corpus_root).resolve()


def _build_subfolder_source_config(
    source_config: RemoteCorpusSourceConfig, subfolder: str
) -> RemoteCorpusSourceConfig:
    prefix = _join_prefix(source_config.prefix, subfolder)
    return RemoteCorpusSourceConfig(
        kind=source_config.kind,
        profile=source_config.profile,
        name=subfolder,
        bucket=source_config.bucket,
        container=source_config.container,
        prefix=prefix,
    )


def _build_partition_source_config(
    source_config: RemoteCorpusSourceConfig,
) -> RemoteCorpusSourceConfig:
    return RemoteCorpusSourceConfig(
        kind=source_config.kind,
        profile=source_config.profile,
        name=source_config.name,
        bucket=source_config.bucket,
        container=source_config.container,
        prefix=source_config.prefix,
    )


def _ensure_collection_corpus(
    corpus_root: Path,
    *,
    collection_name: str,
    corpus_name: str,
    source_config: RemoteCorpusSourceConfig,
    auto_create: bool,
) -> bool:
    config_path = corpus_root / "metadata" / "config.json"
    created = False
    if not config_path.exists():
        if not auto_create:
            raise ValueError(f"Corpus missing for collection: {corpus_root}")
        Corpus.init(corpus_root, force=True)
        created = True
    else:
        _ = Corpus.open(corpus_root)

    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = CorpusConfig.model_validate(base_config)
    updated = config.model_copy(
        update={
            "source": source_config,
            "collection": CollectionMembership(
                collection_name=collection_name,
                corpus_name=corpus_name,
            ),
        }
    )
    config_path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return created


def _ensure_partition_corpus(
    corpus_root: Path,
    *,
    collection_name: str,
    source_config: RemoteCorpusSourceConfig,
) -> Corpus:
    config_path = corpus_root / "metadata" / "config.json"
    if not config_path.exists():
        Corpus.init(corpus_root, force=True)
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = CorpusConfig.model_validate(base_config)
    updated = config.model_copy(
        update={
            "source": source_config,
            "collection": CollectionMembership(
                collection_name=collection_name,
                corpus_name=collection_name,
            ),
        }
    )
    config_path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return Corpus.open(corpus_root)


def _discover_subfolders(
    source: object,
    source_config: RemoteCorpusSourceConfig,
    discovery: RemoteCorpusCollectionDiscovery,
) -> List[str]:
    items = source.list_items()
    subfolders = set()
    for item in items:
        relative = _relative_key(item.key, source_config.prefix)
        if not relative:
            continue
        parts = [segment for segment in relative.split("/") if segment]
        if len(parts) < 2:
            continue
        subfolders.add(parts[0])
    return sorted(subfolders)


def _partition_tag_resolver():
    def _resolver(relative_key: str) -> List[str]:
        if not relative_key:
            return []
        parts = [segment for segment in relative_key.split("/") if segment]
        if len(parts) < 2:
            return []
        return [f"table:{parts[0]}"]

    return _resolver


def _archive_missing_corpora(
    corpus_root: Path, discovered: Iterable[str], *, deletion_policy: str
) -> int:
    discovered_set = set(discovered)
    if not corpus_root.exists():
        return 0
    archived = 0
    for child in corpus_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == ARCHIVE_DIR_NAME:
            continue
        if child.name in discovered_set:
            continue
        if deletion_policy == "delete":
            shutil.rmtree(child)
            archived += 1
        else:
            archive_root = corpus_root / ARCHIVE_DIR_NAME
            archive_root.mkdir(parents=True, exist_ok=True)
            destination = archive_root / f"{child.name}-{_archive_suffix()}"
            child.rename(destination)
            archived += 1
    return archived


def _archive_suffix() -> str:
    return utc_now_iso().replace(":", "").replace("-", "")


def _relative_key(key: str, prefix: str) -> str:
    relative = key
    if prefix and relative.startswith(prefix):
        relative = relative[len(prefix) :]
    return relative.lstrip("/")


def _join_prefix(prefix: str, subfolder: str) -> str:
    if prefix:
        return f"{prefix.rstrip('/')}/{subfolder.strip('/')}/"
    return f"{subfolder.strip('/')}/"
