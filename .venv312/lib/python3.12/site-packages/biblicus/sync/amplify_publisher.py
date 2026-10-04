"""
AWS Amplify Publisher for syncing Biblicus corpus data to cloud.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import boto3
import requests


@dataclass
class SyncResult:
    """Result of a catalog sync operation."""

    skipped: bool = False
    reason: str = ""
    created: int = 0
    updated: int = 0
    deleted: int = 0
    errors: List[str] = field(default_factory=list)
    hash: str = ""


class AmplifyPublisher:
    """Publishes corpus events to AWS Amplify backend."""

    def __init__(self, corpus_name: str):
        self.corpus_name = corpus_name

        # Load config from environment or config file
        self._load_config()

        if not all([self.appsync_endpoint, self.api_key, self.s3_bucket]):
            raise ValueError(
                "Missing required Amplify configuration. "
                "Run 'biblicus dashboard configure' to set up credentials."
            )

        self.s3_client = boto3.client('s3', region_name=self.region)

    def _load_config(self):
        """Load configuration from environment or ~/.biblicus/amplify.env."""
        # Try environment variables first
        self.appsync_endpoint = os.getenv('AMPLIFY_APPSYNC_ENDPOINT')
        self.api_key = os.getenv('AMPLIFY_API_KEY')
        self.region = os.getenv('AWS_REGION', 'us-west-2')
        self.s3_bucket = os.getenv('AMPLIFY_S3_BUCKET')

        # If not in environment, try config file
        if not all([self.appsync_endpoint, self.api_key, self.s3_bucket]):
            config_path = Path.home() / '.biblicus' / 'amplify.env'
            if config_path.exists():
                for line in config_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key == 'AMPLIFY_APPSYNC_ENDPOINT' and not self.appsync_endpoint:
                            self.appsync_endpoint = value
                        elif key == 'AMPLIFY_API_KEY' and not self.api_key:
                            self.api_key = value
                        elif key == 'AWS_REGION':
                            self.region = value
                        elif key == 'AMPLIFY_S3_BUCKET':
                            if not self.s3_bucket:
                                self.s3_bucket = value

    def create_corpus(self) -> Dict[str, Any]:
        """Create or update corpus record."""
        mutation = """
        mutation CreateCorpus($input: CreateCorpusInput!) {
          createCorpus(input: $input) {
            name
            s3Prefix
            status
          }
        }
        """
        variables = {
            'input': {
                'name': self.corpus_name,
                's3Prefix': f'{self.corpus_name}',
                'status': 'IDLE',
            }
        }
        return self._execute_graphql(mutation, variables)

    def start_snapshot(
        self,
        snapshot_id: str,
        snapshot_type: str,
        total_items: int
    ) -> Dict[str, Any]:
        """Publish intent event when snapshot starts (before S3)."""
        mutation = """
        mutation CreateSnapshot($input: CreateSnapshotInput!) {
          createSnapshot(input: $input) {
            corpusId
            snapshotId
            status
            totalItems
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'snapshotId': snapshot_id,
                'type': snapshot_type,
                'status': 'RUNNING',
                'totalItems': total_items,
                'completedItems': 0,
                'startTime': datetime.now(timezone.utc).isoformat(),
            }
        }
        return self._execute_graphql(mutation, variables)

    def update_snapshot_progress(
        self,
        snapshot_id: str,
        completed_items: int
    ) -> Dict[str, Any]:
        """Update snapshot progress during processing."""
        mutation = """
        mutation UpdateSnapshot($input: UpdateSnapshotInput!) {
          updateSnapshot(input: $input) {
            corpusId
            snapshotId
            completedItems
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'snapshotId': snapshot_id,
                'completedItems': completed_items,
            }
        }
        return self._execute_graphql(mutation, variables)

    def complete_snapshot(
        self,
        snapshot_id: str,
        total_items: int,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark snapshot as completed or failed."""
        mutation = """
        mutation UpdateSnapshot($input: UpdateSnapshotInput!) {
          updateSnapshot(input: $input) {
            corpusId
            snapshotId
            status
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'snapshotId': snapshot_id,
                'status': 'FAILED' if error_message else 'COMPLETED',
                'completedItems': total_items,
                'endTime': datetime.now(timezone.utc).isoformat(),
            }
        }
        if error_message:
            variables['input']['errorMessage'] = error_message

        return self._execute_graphql(mutation, variables)

    def register_file(
        self,
        file_path: str,
        relative_to: Path
    ) -> Dict[str, Any]:
        """Register file as LOCAL_ONLY before upload."""
        relative_path = str(Path(file_path).relative_to(relative_to))

        mutation = """
        mutation CreateFileMetadata($input: CreateFileMetadataInput!) {
          createFileMetadata(input: $input) {
            corpusId
            filePath
            status
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'filePath': relative_path,
                'status': 'LOCAL_ONLY',
            }
        }
        return self._execute_graphql(mutation, variables)

    def upload_file(
        self,
        local_path: Path,
        relative_to: Path
    ) -> Dict[str, Any]:
        """Upload file to S3 (direct mirror of local structure)."""
        relative_path = str(local_path.relative_to(relative_to))
        s3_key = f'{self.corpus_name}/{relative_path}'

        # Update status to UPLOADING
        mutation = """
        mutation UpdateFileMetadata($input: UpdateFileMetadataInput!) {
          updateFileMetadata(input: $input) {
            corpusId
            filePath
            status
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'filePath': relative_path,
                'status': 'UPLOADING',
            }
        }
        self._execute_graphql(mutation, variables)

        # Upload to S3 (this will trigger Lambda → AVAILABLE)
        with open(local_path, 'rb') as f:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=f
            )

        return {'s3Key': s3_key}

    def sync_catalog(self, catalog_path: Path, force: bool = False) -> SyncResult:
        """Sync catalog.json to Amplify (idempotent)."""
        from biblicus.models import CorpusCatalog

        # 1. Load and hash catalog
        catalog = CorpusCatalog.model_validate_json(catalog_path.read_text())
        catalog_hash = self._compute_catalog_hash(catalog)

        # 2. Check if changed (idempotency)
        try:
            metadata = self._get_catalog_metadata()
            if not force and metadata and metadata.get('catalogHash') == catalog_hash:
                return SyncResult(skipped=True, reason="Catalog unchanged", hash=catalog_hash)
        except Exception:
            metadata = None

        # 3. Choose sync strategy
        if len(catalog.items) < 100 or force:
            result = self._full_replacement_sync(catalog, catalog_hash)
        else:
            result = self._incremental_sync(catalog, metadata, catalog_hash)

        # 4. Update metadata
        self._sync_catalog_metadata(catalog, catalog_hash)

        return result

    def _compute_catalog_hash(self, catalog) -> str:
        """Hash catalog excluding timestamps for idempotency."""
        items_data = sorted([
            (item.id, item.sha256, item.relpath)
            for item in catalog.items.values()
        ])
        return hashlib.sha256(json.dumps(items_data).encode()).hexdigest()

    def _get_catalog_metadata(self) -> Optional[Dict[str, Any]]:
        """Get current catalog metadata from Amplify."""
        query = """
        query GetCatalogMetadata($corpusId: ID!) {
          getCatalogMetadata(corpusId: $corpusId) {
            corpusId
            catalogHash
            itemCount
            lastSyncedAt
          }
        }
        """
        variables = {'corpusId': self.corpus_name}
        try:
            result = self._execute_graphql(query, variables)
            return result.get('getCatalogMetadata')
        except Exception:
            return None

    def _sync_catalog_metadata(self, catalog, catalog_hash: str):
        """Update catalog metadata in Amplify."""
        mutation = """
        mutation UpdateCatalogMetadata($input: UpdateCatalogMetadataInput!) {
          updateCatalogMetadata(input: $input) {
            corpusId
            catalogHash
          }
        }
        """
        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'catalogHash': catalog_hash,
                'itemCount': len(catalog.items),
                'lastSyncedAt': datetime.now(timezone.utc).isoformat(),
                'schemaVersion': catalog.schema_version,
                'configurationId': catalog.configuration_id if hasattr(catalog, 'configuration_id') else None,
                'corpusUri': catalog.corpus_uri if hasattr(catalog, 'corpus_uri') else None,
            }
        }
        try:
            self._execute_graphql(mutation, variables)
        except Exception:
            # Try create if update failed
            mutation = mutation.replace('updateCatalogMetadata', 'createCatalogMetadata')
            mutation = mutation.replace('UpdateCatalogMetadataInput', 'CreateCatalogMetadataInput')
            self._execute_graphql(mutation, variables)

    def _full_replacement_sync(self, catalog, catalog_hash: str) -> SyncResult:
        """Replace all catalog items (for small catalogs)."""
        errors = []
        created = 0

        # Create all items
        for item_id, item in catalog.items.items():
            try:
                self._create_catalog_item(item)
                created += 1
            except Exception as e:
                errors.append(f"Failed to sync item {item.id}: {e}")

        return SyncResult(
            skipped=False,
            created=created,
            updated=0,
            deleted=0,
            errors=errors,
            hash=catalog_hash
        )

    def _incremental_sync(self, catalog, metadata: Optional[Dict], catalog_hash: str) -> SyncResult:
        """Sync only changed items (for large catalogs)."""
        # For MVP, just do full replacement
        # TODO: Implement proper diff logic
        return self._full_replacement_sync(catalog, catalog_hash)

    def _retry_attempts(self) -> Iterable[int]:
        return range(3)

    def _create_catalog_item(self, item):
        """Create or update a catalog item with retry."""
        mutation = """
        mutation CreateCatalogItem($input: CreateCatalogItemInput!) {
          createCatalogItem(input: $input) {
            corpusId
            itemId
          }
        }
        """

        variables = {
            'input': {
                'corpusId': self.corpus_name,
                'itemId': item.id,
                'relpath': item.relpath,
                'sha256': item.sha256,
                'bytes': item.bytes,
                'mediaType': item.media_type,
                'title': item.title if hasattr(item, 'title') else None,
                'tags': item.tags if hasattr(item, 'tags') else [],
                'metadataJson': json.dumps(item.metadata) if hasattr(item, 'metadata') and item.metadata else None,
                'createdAt': datetime.now(timezone.utc).isoformat(),
                'sourceUri': item.source_uri if hasattr(item, 'source_uri') else None,
                'hasExtraction': False,
            }
        }

        published = False
        last_error: Exception | None = None
        for attempt in self._retry_attempts():
            try:
                self._execute_graphql(mutation, variables)
                published = True
                break
            except Exception as e:
                last_error = e
                if attempt < 2 and 'Network' in str(e):
                    time.sleep(2 ** attempt)
                    continue
                break
        if published:
            return
        if last_error is not None:
            raise last_error

    def _execute_graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GraphQL mutation against AppSync."""
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
        }

        payload = {
            'query': query,
            'variables': variables,
        }

        response = requests.post(
            self.appsync_endpoint,
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

        result = response.json()
        if 'errors' in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        return result.get('data', {})
