"""
Error types for Biblicus.
"""

from __future__ import annotations


class ExtractionSnapshotFatalError(RuntimeError):
    """
    Fatal extraction snapshot error that should abort the entire snapshot.

    This exception is used for conditions that indicate a configuration or environment problem
    rather than a per-item extraction failure. For example, a selection extractor that depends
    on referenced extraction snapshot manifests treats missing manifests as fatal.
    """


class IngestCollisionError(RuntimeError):
    """
    Ingest collision for an already ingested source.

    :param source_uri: Source uniform resource identifier that caused the collision.
    :type source_uri: str
    :param existing_item_id: Identifier of the existing catalog item.
    :type existing_item_id: str
    :param existing_relpath: Raw storage relpath of the existing item.
    :type existing_relpath: str
    """

    def __init__(self, *, source_uri: str, existing_item_id: str, existing_relpath: str) -> None:
        self.source_uri = source_uri
        self.existing_item_id = existing_item_id
        self.existing_relpath = existing_relpath
        message = (
            "Source already ingested"
            f": source_uri={source_uri} existing_item_id={existing_item_id}"
            f" existing_relpath={existing_relpath}"
        )
        super().__init__(message)


class RemoteSourceDependencyError(RuntimeError):
    """
    Raised when a remote source dependency is unavailable.

    :param message: Error message describing the missing dependency.
    :type message: str
    """
