"""
AWS Transcribe speech to text extractor plugin.

Uses Amazon Transcribe for audio transcription with support for multiple
languages, custom vocabularies, and speaker identification.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class AwsTranscribeSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for AWS Transcribe speech to text extraction.

    :ivar language_code: Language code (en-US, es-ES, etc.).
    :vartype language_code: str
    :ivar region_name: AWS region (us-east-1, eu-west-1, etc.).
    :vartype region_name: str
    :ivar s3_bucket: S3 bucket name for temporary audio uploads.
    :vartype s3_bucket: str
    :ivar identify_speakers: Enable speaker identification.
    :vartype identify_speakers: bool
    :ivar max_speakers: Maximum number of speakers (2-10).
    :vartype max_speakers: int or None
    :ivar vocabulary_name: Custom vocabulary name.
    :vartype vocabulary_name: str or None
    :ivar show_alternatives: Include alternative transcriptions.
    :vartype show_alternatives: bool
    :ivar max_alternatives: Number of alternative transcriptions (2-10).
    :vartype max_alternatives: int or None
    :ivar max_wait_seconds: Maximum time to wait for job completion.
    :vartype max_wait_seconds: float
    :ivar poll_interval_seconds: Interval between job status polls.
    :vartype poll_interval_seconds: float
    """

    model_config = ConfigDict(extra="forbid")

    language_code: str = Field(default="en-US", min_length=1)
    region_name: str = Field(default="us-east-1", min_length=1)
    s3_bucket: str = Field(min_length=1)
    identify_speakers: bool = Field(default=False)
    max_speakers: Optional[int] = Field(default=None, ge=2, le=10)
    vocabulary_name: Optional[str] = Field(default=None, min_length=1)
    show_alternatives: bool = Field(default=False)
    max_alternatives: Optional[int] = Field(default=None, ge=2, le=10)
    max_wait_seconds: float = Field(default=600.0, gt=0)
    poll_interval_seconds: float = Field(default=5.0, gt=0)


class AwsTranscribeSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio using AWS Transcribe.

    Uses Amazon Transcribe API for high-quality speech recognition with
    support for custom vocabularies and speaker diarization.

    Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    and boto3 package.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-aws-transcribe"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: AwsTranscribeSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        try:
            import boto3  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "AWS Transcribe speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[aws]" or pip install boto3.'
            ) from import_error

        return AwsTranscribeSpeechToTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item using AWS Transcribe.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: AwsTranscribeSpeechToTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        _ = previous_extractions
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, AwsTranscribeSpeechToTextExtractorConfig)
            else AwsTranscribeSpeechToTextExtractorConfig.model_validate(config)
        )

        try:
            import boto3
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "AWS Transcribe speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[aws]" or pip install boto3.'
            ) from import_error

        # Initialize AWS clients
        s3_client = boto3.client("s3", region_name=parsed_config.region_name)
        transcribe_client = boto3.client("transcribe", region_name=parsed_config.region_name)

        source_path = corpus.root / item.relpath

        # Generate unique job name
        import uuid

        job_name = f"biblicus-{item.id}-{uuid.uuid4().hex[:8]}"

        # Upload audio to S3 (Transcribe requires S3 URI)
        bucket_name = parsed_config.s3_bucket
        s3_key = f"biblicus-audio/{item.id}/{source_path.name}"

        try:
            # Upload audio to S3
            with source_path.open("rb") as audio_file:
                s3_client.upload_fileobj(audio_file, bucket_name, s3_key)

            # Detect media format from media type
            media_format = self._detect_media_format(item.media_type)

            # Start transcription job
            job_args = {
                "TranscriptionJobName": job_name,
                "Media": {"MediaFileUri": f"s3://{bucket_name}/{s3_key}"},
                "MediaFormat": media_format,
                "LanguageCode": parsed_config.language_code,
            }

            # Add optional settings
            settings = {}
            if parsed_config.identify_speakers:
                settings["ShowSpeakerLabels"] = True
                if parsed_config.max_speakers is not None:
                    settings["MaxSpeakerLabels"] = parsed_config.max_speakers

            if parsed_config.vocabulary_name:
                settings["VocabularyName"] = parsed_config.vocabulary_name

            if parsed_config.show_alternatives:
                settings["ShowAlternatives"] = True
                if parsed_config.max_alternatives is not None:
                    settings["MaxAlternatives"] = parsed_config.max_alternatives

            if settings:
                job_args["Settings"] = settings

            transcribe_client.start_transcription_job(**job_args)

            # Poll for completion
            max_wait_seconds = parsed_config.max_wait_seconds
            poll_interval = parsed_config.poll_interval_seconds
            elapsed = 0

            while elapsed < max_wait_seconds:
                response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]

                if status == "COMPLETED":
                    # Get transcript
                    transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

                    import json as json_module
                    import urllib.request

                    with urllib.request.urlopen(transcript_uri) as response_data:
                        transcript_data = json_module.loads(response_data.read().decode("utf-8"))

                    # Extract text
                    transcript_text = transcript_data["results"]["transcripts"][0]["transcript"]

                    # Extract metadata
                    metadata = {
                        "job_name": job_name,
                        "language_code": parsed_config.language_code,
                        "speaker_labels": parsed_config.identify_speakers,
                    }

                    # Add speaker information if available
                    if "speaker_labels" in transcript_data["results"]:
                        metadata["speakers"] = transcript_data["results"]["speaker_labels"]

                    return ExtractedText(
                        text=transcript_text.strip(),
                        producer_extractor_id=self.extractor_id,
                        metadata=metadata,
                    )

                elif status == "FAILED":
                    failure_reason = response["TranscriptionJob"].get("FailureReason", "Unknown")
                    raise ExtractionSnapshotFatalError(
                        f"AWS Transcribe job failed: {failure_reason}"
                    )

                # Wait before polling again
                time.sleep(poll_interval)
                elapsed += poll_interval

            raise ExtractionSnapshotFatalError(
                f"AWS Transcribe job timed out after {max_wait_seconds} seconds"
            )

        finally:
            # Cleanup: Delete transcription job and S3 object
            try:
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception as cleanup_error:
                print(
                    f"Warning: failed to delete AWS Transcribe job {job_name}: {cleanup_error}",
                    file=sys.stderr,
                )

            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            except Exception as cleanup_error:
                print(
                    f"Warning: failed to delete temporary S3 object {s3_key}: {cleanup_error}",
                    file=sys.stderr,
                )

    def _detect_media_format(self, media_type: str) -> str:
        """
        Detect AWS Transcribe media format from MIME type.

        :param media_type: MIME media type.
        :type media_type: str
        :return: AWS Transcribe media format.
        :rtype: str
        """
        media_type_lower = media_type.lower()

        if "flac" in media_type_lower:
            return "flac"
        elif "wav" in media_type_lower or "wave" in media_type_lower:
            return "wav"
        elif "mp3" in media_type_lower or "mpeg" in media_type_lower:
            return "mp3"
        elif "mp4" in media_type_lower or "m4a" in media_type_lower:
            return "mp4"
        elif "ogg" in media_type_lower:
            return "ogg"
        elif "webm" in media_type_lower:
            return "webm"
        else:
            return "wav"  # default
