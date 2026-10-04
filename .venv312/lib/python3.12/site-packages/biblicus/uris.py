"""
Uniform resource identifier and path helpers for Biblicus corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
from urllib.parse import unquote, urlparse


def _looks_like_uri(value: str) -> bool:
    """
    Check whether a string resembles a uniform resource identifier.

    :param value: Candidate string.
    :type value: str
    :return: True if the string has a valid uniform resource identifier scheme prefix.
    :rtype: bool
    """
    return "://" in value and value.split("://", 1)[0].isidentifier()


def corpus_ref_to_path(ref: Union[str, Path]) -> Path:
    """
    Convert a corpus reference to a filesystem path.

    :param ref: Filesystem path or file:// uniform resource identifier.
    :type ref: str or Path
    :return: Resolved filesystem path.
    :rtype: Path
    :raises NotImplementedError: If a non-file uniform resource identifier scheme is used.
    :raises ValueError: If a file:// uniform resource identifier has a non-local host.
    """
    if isinstance(ref, Path):
        return ref.resolve()

    if _looks_like_uri(ref):
        parsed = urlparse(ref)
        if parsed.scheme != "file":
            raise NotImplementedError(
                "Only file:// corpus uniform resource identifiers are supported in version zero "
                f"(got {parsed.scheme}://)"
            )
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"Unsupported file uniform resource identifier host: {parsed.netloc!r}"
            )
        return Path(unquote(parsed.path)).resolve()

    return Path(ref).resolve()


def normalize_corpus_uri(ref: Union[str, Path]) -> str:
    """
    Normalize a corpus reference into a file:// uniform resource identifier.

    :param ref: Filesystem path or file:// uniform resource identifier.
    :type ref: str or Path
    :return: Canonical file:// uniform resource identifier.
    :rtype: str
    """
    return corpus_ref_to_path(ref).as_uri()
