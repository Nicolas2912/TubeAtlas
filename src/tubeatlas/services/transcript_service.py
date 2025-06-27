"""Transcript processing service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from src.tubeatlas.utils.token_counter import count_tokens as count_tokens_util

if TYPE_CHECKING:  # pragma: no cover
    from .youtube_service import YouTubeService

logger = logging.getLogger(__name__)


class TranscriptSegment(TypedDict):
    """A single segment of a transcript."""

    text: str
    start: float
    duration: float
    token_count: int


class Transcript(TypedDict):
    """Structured transcript data."""

    status: str  # 'success', 'not_found', 'disabled'
    video_id: str
    language_code: Optional[str]
    is_generated: Optional[bool]
    segments: Optional[List[TranscriptSegment]]
    total_token_count: Optional[int]


class TranscriptService:
    """Service for transcript processing and persistence."""

    def __init__(
        self,
        session=None,
        youtube_service: "YouTubeService" | None = None,  # forward reference
    ):
        """Initialize transcript service with DB session and YouTubeService.

        Args:
            session: SQLAlchemy async session
            youtube_service: Instance of `YouTubeService` for fetching video data
        """
        self.session = session
        self.youtube_service = youtube_service

        if session is not None:
            from ..repositories.transcript_repository import TranscriptRepository
            from ..repositories.video_repository import VideoRepository  # local import

            self.video_repo = VideoRepository(session)
            self.transcript_repo = TranscriptRepository(session)
        else:
            self.video_repo = None  # type: ignore[assignment]
            self.transcript_repo = None  # type: ignore[assignment]

    def _find_transcript(self, transcript_list_iterator, language_codes: List[str]):
        """Find the best transcript from a list based on language codes."""
        # Convert iterator to a list to allow multiple iterations
        transcript_list = list(transcript_list_iterator)

        # Try to find a transcript in the preferred languages
        for lang_code in language_codes:
            try:
                # The find_transcript method from the original list object is needed.
                # This is a bit of a workaround for the mock. In reality,
                # transcript_list_iterator would be the list object itself.
                original_list_obj = transcript_list_iterator
                transcript = original_list_obj.find_transcript([lang_code])
                return transcript
            except NoTranscriptFound:
                continue

        # If no preferred transcript is found, try to find any manually created one
        for t in transcript_list:
            if not t.is_generated:
                return t

        # If still no transcript, take the first available one
        try:
            return transcript_list[0]
        except IndexError:
            return None

    async def get_transcript(
        self, video_id: str, language_codes: Optional[List[str]] = None
    ) -> Transcript:
        """
        Fetches the transcript for a given video ID with fallback logic.

        Args:
            video_id: The ID of the YouTube video.
            language_codes: A list of preferred language codes (e.g., ['en', 'en-US']).
                            If None, defaults to ['en'].

        Returns:
            A structured transcript dictionary.
        """
        if language_codes is None:
            language_codes = ["en"]

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video: {video_id}")
            return {
                "status": "disabled",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
                "total_token_count": None,
            }
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video: {video_id}")
            return {
                "status": "not_found",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
                "total_token_count": None,
            }
        except Exception as exc:
            logger.error("Failed to fetch transcript for video %s: %s", video_id, exc)
            return {
                "status": "fetch_error",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
                "total_token_count": None,
            }

        transcript = self._find_transcript(transcript_list, language_codes)

        if not transcript:
            logger.warning(f"No suitable transcript found for video: {video_id}")
            return {
                "status": "not_found",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
                "total_token_count": None,
            }

        try:
            fetched_segments = transcript.fetch()

            segments_with_tokens: List[TranscriptSegment] = []
            total_tokens = 0

            for s in fetched_segments:
                token_count = count_tokens_util(s["text"])
                segment: TranscriptSegment = {
                    "text": s["text"],
                    "start": s["start"],
                    "duration": s["duration"],
                    "token_count": token_count,
                }
                segments_with_tokens.append(segment)
                total_tokens += token_count

            return {
                "status": "success",
                "video_id": video_id,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "segments": segments_with_tokens,
                "total_token_count": total_tokens,
            }
        except Exception as exc:
            logger.error("Failed to fetch transcript for video %s: %s", video_id, exc)
            return {
                "status": "fetch_error",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
                "total_token_count": None,
            }

    async def extract_transcript(
        self, video_id: str, language: str = "en"
    ) -> Optional[str]:
        """Extract transcript for a video."""
        # TODO: This method might be deprecated in favor of get_transcript
        logger.info(f"Extracting transcript for video: {video_id}")
        transcript_data = await self.get_transcript(video_id, [language])
        if (
            transcript_data
            and transcript_data["status"] == "success"
            and transcript_data["segments"] is not None
        ):
            return " ".join(segment["text"] for segment in transcript_data["segments"])
        return None

    async def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for specified model."""
        logger.info(f"Counting tokens for model: {model}")
        return count_tokens_util(text, model)

    async def process_channel_transcripts(  # noqa: C901
        self,
        channel_url: str,
        update_existing: bool = False,
        include_shorts: bool = False,
        max_videos: int | None = None,
    ) -> Dict[str, Any]:
        """Download and persist video metadata & transcripts for a YouTube channel.

        Implements *incremental mode*: if ``update_existing`` is ``False`` the
        consumer stops as soon as it encounters the **first** video that already
        exists in the database under the assumption that all subsequent videos
        have been processed in a previous run (YouTube API returns videos in
        reverse chronological order).

        Args:
            channel_url: The YouTube channel URL or handle
            update_existing: If ``True`` the method re-processes videos even if
                they already exist ("full refresh"). If ``False`` processing
                stops at the first existing video ("incremental").
            include_shorts: Pass-through flag for the underlying fetcher
            max_videos: Optional hard cap on number of videos to process

        Returns:
            A summary dictionary with counters for created/updated/skipped
            videos and transcripts.
        """
        if self.youtube_service is None:
            raise ValueError("YouTubeService instance required for processing")

        created_videos = updated_videos = skipped_videos = 0
        created_trans = updated_trans = skipped_trans = 0

        video_iter = self.youtube_service.fetch_channel_videos(
            channel_url,
            include_shorts=include_shorts,
            max_videos=max_videos,
        )

        async def _process_video(video_meta):
            nonlocal created_videos, updated_videos, skipped_videos, created_trans, updated_trans, skipped_trans

            video_id: str = video_meta["id"]

            if not update_existing and await self.video_repo.exists(video_id):
                logger.info(
                    "Encountered existing video %s – incremental stop.", video_id
                )
                return True  # signal to stop processing further videos

            # Upsert video
            existing_video = await self.video_repo.get_by_id(video_id)
            await self.video_repo.upsert(video_meta)
            if existing_video:
                updated_videos += 1
            else:
                created_videos += 1

            # Fetch transcript
            transcript_data = await self.get_transcript(video_id)

            if transcript_data["status"] != "success":
                skipped_videos += 1
                skipped_trans += 1
                return False

            content_text = " ".join(
                seg["text"] for seg in transcript_data["segments"] or []
            )

            transcript_row = {
                "video_id": video_id,
                "video_title": video_meta["title"],
                "video_url": f"https://youtu.be/{video_id}",
                "channel_name": video_meta.get("channelTitle", ""),
                "channel_url": channel_url,
                "content": content_text,
                "language": transcript_data["language_code"] or "en",
                "openai_tokens": transcript_data["total_token_count"],
                "processing_status": "completed",
            }

            existing_trans_row = await self.transcript_repo.get_by_id(video_id)
            await self.transcript_repo.upsert(transcript_row)
            if existing_trans_row:
                updated_trans += 1
            else:
                created_trans += 1

            return False  # continue processing

        # Handle async vs sync iterables
        if hasattr(video_iter, "__aiter__"):
            async for video_meta in video_iter:  # type: ignore[async-iter-compatible]
                should_stop = await _process_video(video_meta)
                if should_stop:
                    break
        else:
            for video_meta in video_iter:  # type: ignore[misc]
                should_stop = await _process_video(video_meta)
                if should_stop:
                    break

        return {
            "videos": {
                "created": created_videos,
                "updated": updated_videos,
                "skipped": skipped_videos,
            },
            "transcripts": {
                "created": created_trans,
                "updated": updated_trans,
                "skipped": skipped_trans,
            },
        }
