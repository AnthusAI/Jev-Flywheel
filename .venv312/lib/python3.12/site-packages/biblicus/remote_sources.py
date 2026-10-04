"""
Remote source adapters for Biblicus corpora.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

from .errors import RemoteSourceDependencyError
from .models import RemoteCorpusSourceConfig, RemoteSourceItem
from .user_config import SourceProfileConfig


def _isoformat_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_etag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    return stripped or None


class S3RemoteSource:
    """
    Remote source adapter for Amazon S3.
    """

    def __init__(self, config: RemoteCorpusSourceConfig, aws: SourceProfileConfig) -> None:
        self._config = config
        self._aws = aws
        self._client = self._build_client()

    def _build_client(self):
        try:
            import boto3
        except ImportError as import_error:
            raise RemoteSourceDependencyError(
                'Remote S3 sources require boto3. Install it with pip install "biblicus[aws]".'
            ) from import_error
        return boto3.client(
            "s3",
            region_name=self._aws.region,
            aws_access_key_id=self._aws.access_key_id,
            aws_secret_access_key=self._aws.secret_access_key,
            aws_session_token=self._aws.session_token,
            endpoint_url=self._aws.endpoint_url,
        )

    def list_items(self) -> list[RemoteSourceItem]:
        """
        List items available in the configured S3 bucket and prefix.

        :return: Remote source items describing each object.
        :rtype: list[RemoteSourceItem]
        :raises ValueError: If required configuration fields are missing.
        """
        bucket = self._config.bucket
        prefix = self._config.prefix or ""
        response = self._client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents") or []
        items = []
        for entry in contents:
            key = entry.get("Key")
            if not key or key.endswith("/"):
                continue
            etag = _normalize_etag(entry.get("ETag"))
            last_modified = _isoformat_timestamp(entry.get("LastModified"))
            size = int(entry.get("Size") or 0)
            content_type = entry.get("ContentType")
            items.append(
                RemoteSourceItem(
                    key=key,
                    source_uri=f"s3://{bucket}/{key}",
                    etag=etag,
                    last_modified=last_modified,
                    size=size,
                    content_type=content_type,
                )
            )
        return items

    def fetch_bytes(self, item: RemoteSourceItem) -> Tuple[bytes, Optional[str]]:
        """
        Fetch the raw bytes for a remote S3 object.

        :param item: Remote source item to download.
        :type item: RemoteSourceItem
        :return: Tuple containing the object bytes and content type.
        :rtype: tuple[bytes, Optional[str]]
        """
        response = self._client.get_object(Bucket=self._config.bucket, Key=item.key)
        body = response["Body"].read()
        content_type = response.get("ContentType") or item.content_type
        return body, content_type


class AzureBlobRemoteSource:
    """
    Remote source adapter for Azure Blob Storage.
    """

    def __init__(self, config: RemoteCorpusSourceConfig, azure: SourceProfileConfig) -> None:
        self._config = config
        self._azure = azure
        self._client = self._build_client()

    def _build_client(self):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as import_error:
            raise RemoteSourceDependencyError(
                'Remote Azure Blob sources require azure-storage-blob. '
                'Install it with pip install "biblicus[azure]".'
            ) from import_error
        if self._azure.connection_string:
            return ContainerClient.from_connection_string(
                self._azure.connection_string, self._config.container
            )
        account_name = self._azure.account_name
        account_url = self._azure.account_url or f"https://{account_name}.blob.core.windows.net"
        return ContainerClient.from_account_url(
            account_url, self._config.container, credential=self._azure.account_key
        )

    def list_items(self) -> list[RemoteSourceItem]:
        """
        List items available in the configured Azure blob container and prefix.

        :return: Remote source items describing each blob.
        :rtype: list[RemoteSourceItem]
        :raises ValueError: If required configuration fields are missing.
        """
        prefix = self._config.prefix or ""
        items = []
        for entry in self._client.list_blobs(name_starts_with=prefix):
            key = entry.name
            if not key or key.endswith("/"):
                continue
            etag = _normalize_etag(getattr(entry, "etag", None))
            last_modified = _isoformat_timestamp(getattr(entry, "last_modified", None))
            size = int(getattr(entry, "size", 0) or 0)
            content_type = None
            content_settings = getattr(entry, "content_settings", None)
            if content_settings is not None:
                content_type = getattr(content_settings, "content_type", None)
            account = self._azure.account_name or "account"
            items.append(
                RemoteSourceItem(
                    key=key,
                    source_uri=f"azure-blob://{account}/{self._config.container}/{key}",
                    etag=etag,
                    last_modified=last_modified,
                    size=size,
                    content_type=content_type,
                )
            )
        return items

    def fetch_bytes(self, item: RemoteSourceItem) -> Tuple[bytes, Optional[str]]:
        """
        Fetch the raw bytes for a remote Azure blob.

        :param item: Remote source item to download.
        :type item: RemoteSourceItem
        :return: Tuple containing the blob bytes and content type.
        :rtype: tuple[bytes, Optional[str]]
        """
        downloader = self._client.download_blob(item.key)
        content = downloader.readall()
        return content, item.content_type


def iter_items(source: object) -> Iterable[RemoteSourceItem]:
    """
    Return the iterable of items for a supported remote source adapter.

    :param source: Remote source adapter instance.
    :type source: object
    :return: Iterable of remote source items.
    :rtype: Iterable[RemoteSourceItem]
    :raises ValueError: If the source adapter type is unsupported.
    """
    if isinstance(source, S3RemoteSource):
        return source.list_items()
    if isinstance(source, AzureBlobRemoteSource):
        return source.list_items()
    raise ValueError("Unsupported remote source")
