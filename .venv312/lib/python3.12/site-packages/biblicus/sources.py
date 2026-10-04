"""
Source loading helpers for Biblicus ingestion.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


def _looks_like_uri(value: str) -> bool:
    """
    Check whether a string resembles a uniform resource identifier.

    :param value: Candidate string.
    :type value: str
    :return: True if the string has a valid uniform resource identifier scheme prefix.
    :rtype: bool
    """
    return "://" in value and value.split("://", 1)[0].isidentifier()


def _filename_from_url_path(path: str) -> str:
    """
    Derive a filename from a uniform resource locator path.

    :param path: Uniform resource locator path component.
    :type path: str
    :return: Filename or a fallback name.
    :rtype: str
    """
    filename = Path(unquote(path)).name
    return filename or "download"


def _sanitize_filename_component(name: str) -> str:
    allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._() ")
    sanitized_name = "".join(
        (character if character in allowed_characters else "_") for character in name
    ).strip()
    return sanitized_name or "file"


def _namespaced_filename(
    *, source_uri: Optional[str], fallback_name: Optional[str], media_type: str
) -> str:
    base_name = ""
    if source_uri:
        base_name = quote(source_uri, safe="")
    if not base_name and fallback_name:
        base_name = _sanitize_filename_component(fallback_name)
    if not base_name:
        base_name = "file"
    return _ensure_extension_for_media_type(base_name, media_type)


def _media_type_from_filename(name: str) -> str:
    """
    Guess media type from a filename.

    :param name: Filename to inspect.
    :type name: str
    :return: Guessed media type or application/octet-stream.
    :rtype: str
    """
    media_type, _ = mimetypes.guess_type(name)
    return media_type or "application/octet-stream"


def _sniff_media_type_from_bytes(data: bytes) -> Optional[str]:
    """
    Sniff a media type from leading bytes for a small set of common formats.

    :param data: Raw bytes to inspect.
    :type data: bytes
    :return: Detected media type or None.
    :rtype: str or None
    """
    prefix = data[:32]
    if prefix.startswith(b"%PDF-"):
        return "application/pdf"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
        return "audio/x-wav"
    if prefix.startswith(b"ID3") or (
        len(prefix) >= 2 and prefix[0] == 0xFF and (prefix[1] & 0xE0) == 0xE0
    ):
        return "audio/mpeg"
    if prefix.startswith(b"OggS"):
        return "audio/ogg"
    if prefix.lstrip().lower().startswith(b"<!doctype html") or prefix.lstrip().lower().startswith(
        b"<html"
    ):
        return "text/html"
    return None


def _normalize_media_type(*, filename: str, media_type: str) -> str:
    """
    Normalize media types that are commonly mislabelled by upstream sources.

    This function exists to keep the corpus usable for humans. When a source provides a filename
    extension that users recognize (for example, ``.ogg``), Biblicus prefers a matching media type
    so that downstream processing can make reasonable decisions.

    :param filename: Filename associated with the payload.
    :type filename: str
    :param media_type: Media type reported or guessed for the payload.
    :type media_type: str
    :return: Normalized media type.
    :rtype: str
    """
    suffix = Path(filename).suffix.lower()
    if media_type in {"application/ogg", "application/x-ogg"} and suffix in {
        ".ogg",
        ".oga",
        ".ogx",
    }:
        return "audio/ogg"
    return media_type


def _ensure_extension_for_media_type(filename: str, media_type: str) -> str:
    """
    Ensure the filename has a usable extension for the media type.

    :param filename: Filename candidate.
    :type filename: str
    :param media_type: Media type to target.
    :type media_type: str
    :return: Filename with extension.
    :rtype: str
    """
    if Path(filename).suffix:
        return filename
    media_type_overrides = {
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "image/jpeg": ".jpg",
        "text/html": ".html",
    }
    if media_type in media_type_overrides:
        ext = media_type_overrides[media_type]
    else:
        ext = mimetypes.guess_extension(media_type) or ""
    return filename + ext if ext else filename


@dataclass(frozen=True)
class SourcePayload:
    """
    Loaded source payload for ingestion.

    :ivar data: Raw bytes from the source.
    :vartype data: bytes
    :ivar filename: Suggested filename for the payload.
    :vartype filename: str
    :ivar media_type: Internet Assigned Numbers Authority media type for the payload.
    :vartype media_type: str
    :ivar source_uri: Source uniform resource identifier used to load the payload.
    :vartype source_uri: str
    """

    data: bytes
    filename: str
    media_type: str
    source_uri: str


def load_source(source: str | Path, *, source_uri: Optional[str] = None) -> SourcePayload:
    """
    Load bytes from a source reference.

    :param source: File path or uniform resource locator to load.
    :type source: str or Path
    :param source_uri: Optional override for the source uniform resource identifier.
    :type source_uri: str or None
    :return: Source payload with bytes and metadata.
    :rtype: SourcePayload
    :raises ValueError: If a file:// uniform resource identifier has a non-local host.
    :raises NotImplementedError: If the uniform resource identifier scheme is unsupported.
    """
    if isinstance(source, Path):
        path = source.resolve()
        payload_bytes = path.read_bytes()
        media_type = _media_type_from_filename(path.name)
        if media_type == "application/octet-stream":
            sniffed = _sniff_media_type_from_bytes(payload_bytes)
            if sniffed:
                media_type = sniffed
                filename = _ensure_extension_for_media_type(path.name, media_type)
            else:
                filename = path.name
        else:
            filename = path.name
        if path.suffix.lower() in {".md", ".markdown"}:
            media_type = "text/markdown"
        resolved_source_uri = source_uri or path.as_uri()
        return SourcePayload(
            data=payload_bytes,
            filename=filename,
            media_type=media_type,
            source_uri=resolved_source_uri,
        )

    if _looks_like_uri(source):
        parsed = urlparse(source)
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError(
                    f"Unsupported file uniform resource identifier host: {parsed.netloc!r}"
                )
            path = Path(unquote(parsed.path)).resolve()
            if path.is_dir():
                index_path = path / "index.html"
                if index_path.exists():
                    return load_source(index_path, source_uri=source_uri or source)
                raise IsADirectoryError(f"Directory source lacks index: {path}")
            return load_source(path, source_uri=source_uri or source)

        if parsed.scheme in {"http", "https"}:
            request = Request(source, headers={"User-Agent": "biblicus/0"})
            with urlopen(request, timeout=30) as response:
                response_bytes = response.read()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
                fallback_filename = _filename_from_url_path(parsed.path)
                media_type = content_type or _media_type_from_filename(fallback_filename)
                if media_type == "application/octet-stream":
                    sniffed = _sniff_media_type_from_bytes(response_bytes)
                    if sniffed:
                        media_type = sniffed
                        fallback_filename = _ensure_extension_for_media_type(
                            fallback_filename, media_type
                        )
                media_type = _normalize_media_type(
                    filename=fallback_filename, media_type=media_type
                )
                if Path(fallback_filename).suffix.lower() in {".md", ".markdown"}:
                    media_type = "text/markdown"
                resolved_source_uri = source_uri or source
                return SourcePayload(
                    data=response_bytes,
                    filename=fallback_filename,
                    media_type=media_type,
                    source_uri=resolved_source_uri,
                )

        raise NotImplementedError(
            f"Unsupported source uniform resource identifier scheme: {parsed.scheme}://"
        )

    path = Path(source).resolve()
    return load_source(path, source_uri=source_uri)
