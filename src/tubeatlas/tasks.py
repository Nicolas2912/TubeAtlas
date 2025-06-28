"""Celery tasks for TubeAtlas."""

import asyncio
import logging
from typing import Any, Dict, Optional

from .config.celery_app import celery_app
from .config.database import AsyncSessionLocal
from .repositories.transcript_repository import TranscriptRepository
from .repositories.video_repository import VideoRepository
from .services.transcript_service import TranscriptService
from .services.youtube_service import YouTubeService
from .utils.exceptions import QuotaExceededError, TransientAPIError
from .utils.validators import extract_video_id

logger = logging.getLogger(__name__)


async def get_async_session():
    """Get async database session context manager."""
    return AsyncSessionLocal()


@celery_app.task(bind=True, max_retries=3, acks_late=True)
def download_channel(
    self,
    channel_url: str,
    include_shorts: bool = False,
    max_videos: Optional[int] = None,
    update_existing: bool = False,
) -> Dict[str, Any]:
    """
    Celery task that fetches all (or limited) videos for a channel, pulls transcripts,
    and persists both video and transcript records via upsert semantics.

    Args:
        channel_url: YouTube channel URL
        include_shorts: Whether to include YouTube Shorts (default: False)
        max_videos: Maximum number of videos to fetch (default: unlimited)
        update_existing: Whether to update existing video records (default: False)

    Returns:
        dict with counts {processed, succeeded, failed, skipped}
    """
    logger.info(f"Starting channel download task for: {channel_url}")
    logger.info(
        f"Parameters: include_shorts={include_shorts}, max_videos={max_videos}, update_existing={update_existing}"
    )

    try:
        # Run the async implementation
        result = asyncio.run(
            _download_channel_async(
                channel_url, include_shorts, max_videos, update_existing
            )
        )

        logger.info(f"Channel download completed: {result}")
        return result

    except (QuotaExceededError, TransientAPIError) as exc:
        logger.error(f"YouTube API error in channel download: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2**self.request.retries, max_retries=3)

    except Exception as exc:
        logger.error(f"Unexpected error in channel download: {exc}")
        # Don't retry on unexpected errors, but log them
        raise exc


@celery_app.task(bind=True, max_retries=3, acks_late=True)
def download_video(self, video_url: str) -> Dict[str, Any]:
    """
    Celery task that fetches a single video (by URL or ID), pulls its transcript,
    and persists the data.

    Args:
        video_url: YouTube video URL or video ID

    Returns:
        dict with status and video_id
    """
    logger.info(f"Starting video download task for: {video_url}")

    try:
        # Run the async implementation
        result = asyncio.run(_download_video_async(video_url))

        logger.info(f"Video download completed: {result}")
        return result

    except (QuotaExceededError, TransientAPIError) as exc:
        logger.error(f"YouTube API error in video download: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2**self.request.retries, max_retries=3)

    except Exception as exc:
        logger.error(f"Unexpected error in video download: {exc}")
        # Don't retry on unexpected errors, but log them
        raise exc


async def _download_channel_async(
    channel_url: str,
    include_shorts: bool,
    max_videos: Optional[int],
    update_existing: bool,
) -> Dict[str, Any]:
    """Async implementation of channel download logic."""
    processed = 0
    succeeded = 0
    failed = 0
    skipped = 0

    async with AsyncSessionLocal() as session:
        async with session.begin():
            try:
                # Initialize services
                youtube_service = YouTubeService()
                transcript_service = TranscriptService(session, youtube_service)
                video_repo = VideoRepository(session)
                transcript_repo = TranscriptRepository(session)

                logger.info("Services initialized, starting video processing")

                # Fetch videos from channel
                for video_meta in youtube_service.fetch_channel_videos(
                    channel_url=channel_url,
                    include_shorts=include_shorts,
                    max_videos=max_videos,
                ):
                    processed += 1
                    video_id = video_meta.get("video_id")

                    if not video_id:
                        logger.warning(f"Skipping video with missing ID: {video_meta}")
                        failed += 1
                        continue

                    try:
                        # Check if video already exists and skip if not updating
                        if not update_existing and await video_repo.exists(video_id):
                            logger.debug(f"Skipping existing video: {video_id}")
                            skipped += 1
                            continue

                        # Persist video metadata
                        await video_repo.upsert(video_meta)
                        logger.debug(f"Persisted video: {video_id}")

                        # Get transcript
                        transcript_data = await transcript_service.get_transcript(
                            video_id
                        )

                        # Prepare transcript record for persistence
                        transcript_record_data = {
                            "video_id": video_id,
                            "status": transcript_data["status"],
                            "language_code": transcript_data.get("language_code"),
                            "is_generated": transcript_data.get("is_generated"),
                            "total_token_count": transcript_data.get(
                                "total_token_count"
                            ),
                            "channel_name": video_meta.get("channel_title", ""),
                        }

                        # Add transcript text if available
                        segments = transcript_data.get("segments")
                        if segments:
                            transcript_text = "\n".join(
                                [segment["text"] for segment in segments]
                            )
                            transcript_record_data["transcript_text"] = transcript_text

                        # Persist transcript
                        await transcript_repo.upsert(transcript_record_data)
                        logger.debug(
                            f"Persisted transcript for video: {video_id} (status: {transcript_data['status']})"
                        )

                        succeeded += 1

                    except Exception as e:
                        logger.error(f"Failed to process video {video_id}: {e}")
                        failed += 1
                        continue

                # Commit all changes
                await session.commit()
                logger.info(
                    f"Channel processing completed. Processed: {processed}, Succeeded: {succeeded}, Failed: {failed}, Skipped: {skipped}"
                )

            except Exception as e:
                logger.error(f"Error in channel download: {e}")
                await session.rollback()
                raise

    return {
        "status": "completed",
        "channel_url": channel_url,
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "include_shorts": include_shorts,
        "max_videos": max_videos,
        "update_existing": update_existing,
    }


async def _download_video_async(video_url: str) -> Dict[str, Any]:
    """Async implementation of single video download logic."""
    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    if not video_id:
        # Maybe it's already a video ID
        if len(video_url) == 11 and video_url.isalnum():
            video_id = video_url
        else:
            raise ValueError(f"Could not extract video ID from: {video_url}")

    logger.info(f"Processing video ID: {video_id}")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            try:
                # Initialize services
                youtube_service = YouTubeService()
                transcript_service = TranscriptService(session, youtube_service)
                video_repo = VideoRepository(session)
                transcript_repo = TranscriptRepository(session)

                # Get video metadata - we need to create a minimal metadata dict
                # Since YouTubeService doesn't have a single video fetch method,
                # we'll use the batch method with one video
                video_metadata_list = youtube_service._batch_get_video_metadata(
                    [video_id]
                )

                if not video_metadata_list:
                    raise ValueError(f"Video not found or not accessible: {video_id}")

                video_data = video_metadata_list[0]
                video_meta = youtube_service._normalize_video_metadata(video_data)

                # Persist video metadata
                await video_repo.upsert(video_meta)
                logger.info(f"Persisted video: {video_id}")

                # Get transcript
                transcript_data = await transcript_service.get_transcript(video_id)

                # Prepare transcript record for persistence
                transcript_record_data = {
                    "video_id": video_id,
                    "status": transcript_data["status"],
                    "language_code": transcript_data.get("language_code"),
                    "is_generated": transcript_data.get("is_generated"),
                    "total_token_count": transcript_data.get("total_token_count"),
                    "channel_name": video_meta.get("channel_title", ""),
                }

                # Add transcript text if available
                segments = transcript_data.get("segments")
                if segments:
                    transcript_text = "\n".join(
                        [segment["text"] for segment in segments]
                    )
                    transcript_record_data["transcript_text"] = transcript_text

                # Persist transcript
                await transcript_repo.upsert(transcript_record_data)
                logger.info(
                    f"Persisted transcript for video: {video_id} (status: {transcript_data['status']})"
                )

                # Commit changes
                await session.commit()

            except Exception as e:
                logger.error(f"Error in video download: {e}")
                await session.rollback()
                raise

    return {
        "status": "completed",
        "video_id": video_id,
        "video_url": video_url,
        "transcript_status": transcript_data["status"],
        "title": video_meta.get("title", ""),
        "channel_title": video_meta.get("channel_title", ""),
    }


@celery_app.task(bind=True, name="process_transcript")
def process_transcript(self, video_id: str) -> Dict[str, Any]:
    """Process video transcript extraction and analysis."""
    logger.info(f"Processing transcript for video: {video_id}")

    try:
        # TODO: Implement actual transcript processing
        # This would typically:
        # 1. Extract transcript from YouTube
        # 2. Clean and normalize text
        # 3. Store in database
        # 4. Update processing status

        result = {
            "video_id": video_id,
            "status": "completed",
            "message": "Transcript processed successfully",
            "task_id": self.request.id,
        }

        logger.info(f"Transcript processing completed for video: {video_id}")
        return result

    except Exception as exc:
        logger.error(f"Transcript processing failed for video {video_id}: {exc}")
        raise self.retry(countdown=60, max_retries=3, exc=exc)


@celery_app.task(bind=True, name="generate_knowledge_graph")
def generate_knowledge_graph(self, video_id: str) -> Dict[str, Any]:
    """Generate knowledge graph from video transcript."""
    logger.info(f"Generating knowledge graph for video: {video_id}")

    try:
        # TODO: Implement actual knowledge graph generation
        # This would typically:
        # 1. Retrieve processed transcript
        # 2. Extract entities and relationships using LLM
        # 3. Build knowledge graph structure
        # 4. Store graph data
        # 5. Update processing status

        result = {
            "video_id": video_id,
            "status": "completed",
            "message": "Knowledge graph generated successfully",
            "entities_count": 0,  # TODO: Actual count
            "relationships_count": 0,  # TODO: Actual count
            "task_id": self.request.id,
        }

        logger.info(f"Knowledge graph generation completed for video: {video_id}")
        return result

    except Exception as exc:
        logger.error(f"Knowledge graph generation failed for video {video_id}: {exc}")
        raise self.retry(countdown=120, max_retries=3, exc=exc)


@celery_app.task(bind=True, name="process_video")
def process_video(self, video_id: str) -> Dict[str, Any]:
    """Complete video processing pipeline."""
    logger.info(f"Starting complete video processing for: {video_id}")

    try:
        # TODO: Implement complete video processing pipeline
        # This would typically:
        # 1. Extract video metadata from YouTube
        # 2. Process transcript
        # 3. Generate knowledge graph
        # 4. Index for search
        # 5. Update all statuses

        result = {
            "video_id": video_id,
            "status": "completed",
            "message": "Video processing pipeline completed successfully",
            "steps_completed": [
                "metadata",
                "transcript",
                "knowledge_graph",
                "indexing",
            ],
            "task_id": self.request.id,
        }

        logger.info(f"Complete video processing finished for: {video_id}")
        return result

    except Exception as exc:
        logger.error(f"Video processing failed for video {video_id}: {exc}")
        raise self.retry(countdown=180, max_retries=3, exc=exc)


@celery_app.task(bind=True, name="health_check")
def health_check(self) -> Dict[str, Any]:
    """Health check task for monitoring Celery worker status."""
    logger.info("Executing Celery health check")

    return {
        "status": "healthy",
        "message": "Celery worker is operational",
        "task_id": self.request.id,
        "worker": self.request.hostname,
    }
