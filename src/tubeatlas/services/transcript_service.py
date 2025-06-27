"""Transcript processing service."""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

logger = logging.getLogger(__name__)


class TranscriptSegment(TypedDict):
    """A single segment of a transcript."""

    text: str
    start: float
    duration: float


class Transcript(TypedDict):
    """Structured transcript data."""

    status: str  # 'success', 'not_found', 'disabled'
    video_id: str
    language_code: Optional[str]
    is_generated: Optional[bool]
    segments: Optional[List[TranscriptSegment]]


class TranscriptService:
    """Service for transcript processing."""

    def __init__(self):
        """Initialize transcript service."""
        # TODO: Initialize transcript API clients
        pass

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
            }
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video: {video_id}")
            return {
                "status": "not_found",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
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
            }

        try:
            fetched_segments = transcript.fetch()
            segments: List[TranscriptSegment] = [
                {"text": s["text"], "start": s["start"], "duration": s["duration"]}
                for s in fetched_segments
            ]
            return {
                "status": "success",
                "video_id": video_id,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "segments": segments,
            }
        except Exception as e:
            logger.error(f"Failed to fetch transcript for video {video_id}: {e}")
            return {
                "status": "fetch_error",
                "video_id": video_id,
                "language_code": None,
                "is_generated": None,
                "segments": None,
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

    async def count_tokens(self, text: str, model: str = "openai") -> int:
        """Count tokens in text for specified model."""
        # TODO: Implement token counting
        logger.info(f"Counting tokens for model: {model}")
        raise NotImplementedError("Token counting not yet implemented")

    async def process_channel_transcripts(self, channel_id: str) -> Dict[str, Any]:
        """Process transcripts for all videos in a channel."""
        # TODO: Implement channel transcript processing
        logger.info(f"Processing transcripts for channel: {channel_id}")
        raise NotImplementedError("Channel transcript processing not yet implemented")
