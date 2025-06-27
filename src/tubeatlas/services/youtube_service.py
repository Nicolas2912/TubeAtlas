"""YouTube API integration service."""

import logging
import random
import re
import time
from typing import Any, Dict, Generator, List, Optional

import isodate
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config.settings import settings
from ..utils.exceptions import QuotaExceededError, TransientAPIError

logger = logging.getLogger(__name__)


class YouTubeService:
    """Service for YouTube API integration with pagination and retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize YouTube service with API key."""
        self.api_key = api_key or settings.youtube_api_key or settings.google_api_key
        if not self.api_key:
            raise ValueError("YouTube API key is required")

        self._youtube_client = None

    @property
    def youtube_client(self):
        """Get or create YouTube API client."""
        if self._youtube_client is None:
            try:
                self._youtube_client = build("youtube", "v3", developerKey=self.api_key)
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API client: {e}")
                raise TransientAPIError(f"Failed to initialize YouTube API client: {e}")
        return self._youtube_client

    def _paged_request(
        self,
        request_method,
        initial_params: Dict[str, Any],
        max_retries: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 64.0,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute paginated YouTube API requests with exponential back-off.

        Args:
            request_method: The API method to call
            initial_params: Initial parameters for the request
            max_retries: Maximum number of retry attempts
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds

        Yields:
            Raw JSON response items from each page
        """
        params = initial_params.copy()
        page_token = None

        while True:
            if page_token:
                params["pageToken"] = page_token
            elif "pageToken" in params:
                # Remove pageToken for first request
                params.pop("pageToken", None)

            # Execute request with retry logic
            response = self._execute_with_retry(
                request_method, params, max_retries, initial_wait, max_wait
            )

            # Yield items from this page
            if "items" in response:
                for item in response["items"]:
                    yield item

            # Check for next page
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def _handle_http_error(
        self, error: HttpError, attempt: int, max_retries: int
    ) -> None:
        """
        Handle HTTP errors from YouTube API with appropriate exception raising.

        Args:
            error: The HttpError that occurred
            attempt: Current attempt number (0-based)
            max_retries: Maximum number of retries allowed

        Raises:
            QuotaExceededError: When quota is exceeded after all retries
            TransientAPIError: When other errors occur after all retries
        """
        error_code = error.resp.status
        error_reason = error.resp.reason

        logger.warning(
            f"YouTube API error on attempt {attempt + 1}: "
            f"Status {error_code}, Reason: {error_reason}"
        )

        # Check if this is a quota exceeded error
        if error_code == 403 and "quota" in str(error).lower():
            if attempt >= max_retries:
                raise QuotaExceededError(
                    f"YouTube API quota exceeded after {max_retries + 1} attempts"
                )

        # Check if this is a transient error we should retry
        elif error_code in [429, 500, 502, 503, 504]:
            if attempt >= max_retries:
                raise TransientAPIError(
                    f"YouTube API transient error after "
                    f"{max_retries + 1} attempts: Status {error_code}"
                )
        else:
            # Non-retryable error
            raise TransientAPIError(
                f"YouTube API error: Status {error_code}, {error_reason}"
            )

    def _calculate_wait_time(
        self, attempt: int, initial_wait: float, max_wait: float
    ) -> float:
        """
        Calculate wait time with exponential back-off and jitter.

        Args:
            attempt: Current attempt number (0-based)
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds

        Returns:
            Wait time in seconds with jitter applied
        """
        wait_time = min(initial_wait * (2**attempt), max_wait)
        # Add jitter (±25% of wait time)
        jitter = wait_time * 0.25 * (2 * random.random() - 1)
        return max(0, wait_time + jitter)

    def _execute_with_retry(
        self,
        request_method,
        params: Dict[str, Any],
        max_retries: int,
        initial_wait: float,
        max_wait: float,
    ) -> Dict[str, Any]:
        """
        Execute a single API request with exponential back-off retry logic.

        Args:
            request_method: The API method to call
            params: Request parameters
            max_retries: Maximum number of retry attempts
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds

        Returns:
            Raw JSON response

        Raises:
            QuotaExceededError: When quota is exceeded after all retries
            TransientAPIError: When other errors occur after all retries
        """
        last_exception = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Build and execute request
                request = request_method(**params)
                response = request.execute()

                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"Request succeeded on attempt {attempt + 1}")

                return response

            except HttpError as e:
                last_exception = e
                self._handle_http_error(e, attempt, max_retries)

                # Calculate wait time for next attempt
                if attempt < max_retries:
                    actual_wait = self._calculate_wait_time(
                        attempt, initial_wait, max_wait
                    )
                    logger.info(f"Retrying in {actual_wait:.2f} seconds...")
                    time.sleep(actual_wait)

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

                if attempt >= max_retries:
                    raise TransientAPIError(
                        f"Unexpected error after {max_retries + 1} attempts: {e}"
                    )

                # Wait before retry for unexpected errors too
                if attempt < max_retries:
                    wait_time = min(initial_wait * (2**attempt), max_wait)
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        # This should never be reached, but just in case
        raise TransientAPIError(f"Request failed after all retries: {last_exception}")

    def _extract_channel_id(self, channel_url: str) -> str:
        """
        Extract channel ID from various YouTube channel URL formats.

        Args:
            channel_url: YouTube channel URL (various formats supported)

        Returns:
            The YouTube channel ID

        Raises:
            TransientAPIError: If channel cannot be found or resolved
        """
        # Clean up the URL
        url = channel_url.strip()

        # Direct channel ID format: /channel/UC...
        if "/channel/" in url:
            channel_id = url.split("/channel/")[1].split("/")[0]
            logger.debug(f"Extracted channel ID from URL: {channel_id}")
            return channel_id

        # Handle format: @username
        handle_match = re.search(r"@([a-zA-Z0-9_-]+)", url)
        if handle_match:
            return self._resolve_handle(handle_match.group(1))

        # Handle legacy user format: /user/username
        user_match = re.search(r"/user/([a-zA-Z0-9_-]+)", url)
        if user_match:
            return self._resolve_username(user_match.group(1))

        # Handle custom URL format: /c/customname
        custom_match = re.search(r"/c/([a-zA-Z0-9_-]+)", url)
        if custom_match:
            return self._resolve_custom_name(custom_match.group(1))

        raise TransientAPIError(f"Unsupported channel URL format: {url}")

    def _resolve_handle(self, handle: str) -> str:
        """Resolve @handle to channel ID."""
        try:
            response = self._execute_with_retry(
                self.youtube_client.channels().list,
                {"part": "id", "forHandle": handle},
                max_retries=3,
                initial_wait=1.0,
                max_wait=64.0,
            )
            if response.get("items"):
                channel_id = response["items"][0]["id"]
                logger.debug(f"Resolved handle @{handle} to channel ID: {channel_id}")
                return channel_id
            else:
                raise TransientAPIError(f"Handle @{handle} not found")
        except Exception as e:
            logger.error(f"Failed to resolve handle @{handle}: {e}")
            raise TransientAPIError(f"Failed to resolve handle @{handle}: {e}")

    def _resolve_username(self, username: str) -> str:
        """Resolve /user/username to channel ID."""
        try:
            response = self._execute_with_retry(
                self.youtube_client.channels().list,
                {"part": "id", "forUsername": username},
                max_retries=3,
                initial_wait=1.0,
                max_wait=64.0,
            )
            if response.get("items"):
                channel_id = response["items"][0]["id"]
                logger.debug(
                    f"Resolved username {username} to channel ID: {channel_id}"
                )
                return channel_id
            else:
                raise TransientAPIError(f"Username {username} not found")
        except Exception as e:
            logger.error(f"Failed to resolve username {username}: {e}")
            raise TransientAPIError(f"Failed to resolve username {username}: {e}")

    def _resolve_custom_name(self, custom_name: str) -> str:
        """Resolve /c/customname to channel ID."""
        try:
            response = self._execute_with_retry(
                self.youtube_client.channels().list,
                {"part": "id", "forUsername": custom_name},
                max_retries=3,
                initial_wait=1.0,
                max_wait=64.0,
            )
            if response.get("items"):
                channel_id = response["items"][0]["id"]
                logger.debug(
                    f"Resolved custom name {custom_name} to channel ID: {channel_id}"
                )
                return channel_id
            else:
                raise TransientAPIError(f"Custom name {custom_name} not found")
        except Exception as e:
            logger.error(f"Failed to resolve custom name {custom_name}: {e}")
            raise TransientAPIError(f"Failed to resolve custom name {custom_name}: {e}")

    def _get_uploads_playlist_id(self, channel_id: str) -> str:
        """
        Get the uploads playlist ID for a given channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            The uploads playlist ID

        Raises:
            TransientAPIError: If channel not found or uploads playlist unavailable
        """
        try:
            response = self._execute_with_retry(
                self.youtube_client.channels().list,
                {"part": "contentDetails", "id": channel_id},
                max_retries=3,
                initial_wait=1.0,
                max_wait=64.0,
            )

            if not response.get("items"):
                raise TransientAPIError(f"Channel not found: {channel_id}")

            uploads_playlist_id = response["items"][0]["contentDetails"][
                "relatedPlaylists"
            ]["uploads"]
            logger.debug(f"Found uploads playlist: {uploads_playlist_id}")
            return uploads_playlist_id

        except Exception as e:
            logger.error(
                f"Failed to get uploads playlist for channel {channel_id}: {e}"
            )
            raise TransientAPIError(f"Failed to get uploads playlist: {e}")

    def _is_short(self, video_metadata: Dict[str, Any]) -> bool:
        """
        Detect if a video is a YouTube Short using multiple heuristics.

        Args:
            video_metadata: Video metadata from videos().list API call

        Returns:
            True if the video is likely a Short, False otherwise
        """
        try:
            # Check category ID (42 = Shorts)
            if video_metadata.get("snippet", {}).get("categoryId") == "42":
                logger.debug(
                    f"Video {video_metadata.get('id')} detected as Short via categoryId=42"
                )
                return True

            # Check duration (≤ 60 seconds)
            duration_str = video_metadata.get("contentDetails", {}).get("duration", "")
            if duration_str:
                try:
                    duration = isodate.parse_duration(duration_str).total_seconds()
                    if duration <= 60:
                        logger.debug(
                            f"Video {video_metadata.get('id')} detected as Short via duration={duration}s"
                        )
                        return True
                except Exception as e:
                    logger.warning(f"Failed to parse duration {duration_str}: {e}")

            # Check for #shorts in title or tags
            title = video_metadata.get("snippet", {}).get("title", "").lower()
            tags = [
                tag.lower() for tag in video_metadata.get("snippet", {}).get("tags", [])
            ]

            shorts_indicators = {"#shorts", "shorts", "#short"}

            if any(indicator in title for indicator in shorts_indicators):
                logger.debug(
                    f"Video {video_metadata.get('id')} detected as Short via title"
                )
                return True

            if any(
                any(indicator in tag for indicator in shorts_indicators) for tag in tags
            ):
                logger.debug(
                    f"Video {video_metadata.get('id')} detected as Short via tags"
                )
                return True

            return False

        except Exception as e:
            logger.warning(
                f"Error detecting Short status for video {video_metadata.get('id')}: {e}"
            )
            return False

    def _batch_get_video_metadata(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get metadata for multiple videos in a single API call.

        Args:
            video_ids: List of video IDs (max 50 per call)

        Returns:
            List of video metadata dictionaries
        """
        if not video_ids:
            return []

        # Limit to 50 IDs per API call
        if len(video_ids) > 50:
            video_ids = video_ids[:50]
            logger.warning("Truncating video ID batch to 50 items")

        try:
            response = self._execute_with_retry(
                self.youtube_client.videos().list,
                {
                    "part": "snippet,contentDetails,statistics,status",
                    "id": ",".join(video_ids),
                    "maxResults": 50,
                },
                max_retries=3,
                initial_wait=1.0,
                max_wait=64.0,
            )

            videos = response.get("items", [])
            logger.debug(
                f"Retrieved metadata for {len(videos)} videos out of {len(video_ids)} requested"
            )
            return videos

        except Exception as e:
            logger.error(f"Failed to get video metadata for batch: {e}")
            raise TransientAPIError(f"Failed to get video metadata: {e}")

    def _normalize_video_metadata(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize video metadata into a consistent format.

        Args:
            video_data: Raw video data from YouTube API

        Returns:
            Normalized metadata dictionary
        """
        snippet = video_data.get("snippet", {})
        content_details = video_data.get("contentDetails", {})
        statistics = video_data.get("statistics", {})
        status = video_data.get("status", {})

        # Parse duration
        duration_seconds = None
        duration_str = content_details.get("duration", "")
        if duration_str:
            try:
                duration_seconds = int(
                    isodate.parse_duration(duration_str).total_seconds()
                )
            except Exception as e:
                logger.warning(f"Failed to parse duration {duration_str}: {e}")

        normalized = {
            "video_id": video_data.get("id"),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration_seconds": duration_seconds,
            "duration_iso": duration_str,
            "category_id": snippet.get("categoryId", ""),
            "tags": snippet.get("tags", []),
            "view_count": (
                int(statistics.get("viewCount", 0))
                if statistics.get("viewCount")
                else 0
            ),
            "like_count": (
                int(statistics.get("likeCount", 0))
                if statistics.get("likeCount")
                else 0
            ),
            "comment_count": (
                int(statistics.get("commentCount", 0))
                if statistics.get("commentCount")
                else 0
            ),
            "privacy_status": status.get("privacyStatus", ""),
            "embeddable": status.get("embeddable", True),
            "made_for_kids": status.get("madeForKids", False),
            "is_short": self._is_short(video_data),
            "raw_json": video_data,  # Store raw data for debugging
        }

        return normalized

    def fetch_channel_videos(
        self,
        channel_url: str,
        include_shorts: bool = False,
        max_videos: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch videos from a YouTube channel with pagination and filtering.

        Args:
            channel_url: YouTube channel URL in any supported format
            include_shorts: Whether to include YouTube Shorts (default: False)
            max_videos: Maximum number of videos to fetch (default: unlimited)

        Yields:
            Normalized video metadata dictionaries

        Raises:
            TransientAPIError: On API failures
            QuotaExceededError: When quota is exceeded
        """
        logger.info(f"Fetching videos from channel: {channel_url}")
        logger.info(f"Include shorts: {include_shorts}, Max videos: {max_videos}")

        # Step 1: Resolve channel URL to channel ID
        channel_id = self._extract_channel_id(channel_url)
        logger.info(f"Resolved channel ID: {channel_id}")

        # Step 2: Get uploads playlist ID
        uploads_playlist_id = self._get_uploads_playlist_id(channel_id)
        logger.info(f"Uploads playlist ID: {uploads_playlist_id}")

        # Step 3: Iterate through playlist items
        video_count = 0
        video_ids_batch = []

        for playlist_item in self._paged_request(
            self.youtube_client.playlistItems().list,
            {
                "part": "contentDetails,snippet",
                "playlistId": uploads_playlist_id,
                "maxResults": 50,
            },
        ):
            video_id = playlist_item["contentDetails"]["videoId"]
            video_ids_batch.append(video_id)

            # Process in batches of 50 (API limit)
            if len(video_ids_batch) >= 50:
                for video in self._process_video_batch(
                    video_ids_batch, include_shorts, max_videos, video_count
                ):
                    yield video
                    video_count += 1

                    # Check if we've reached max_videos
                    if max_videos and video_count >= max_videos:
                        logger.info(f"Reached maximum video limit: {max_videos}")
                        return

                video_ids_batch = []

        # Process remaining videos in the last batch
        if video_ids_batch:
            for video in self._process_video_batch(
                video_ids_batch, include_shorts, max_videos, video_count
            ):
                yield video
                video_count += 1

                # Check if we've reached max_videos
                if max_videos and video_count >= max_videos:
                    logger.info(f"Reached maximum video limit: {max_videos}")
                    return

    def _process_video_batch(
        self,
        video_ids: List[str],
        include_shorts: bool,
        max_videos: Optional[int],
        current_count: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a batch of video IDs and yield normalized metadata.

        Args:
            video_ids: List of video IDs to process
            include_shorts: Whether to include YouTube Shorts
            max_videos: Maximum total videos to process
            current_count: Current number of videos processed

        Yields:
            Normalized video metadata dictionaries
        """
        videos_metadata = self._batch_get_video_metadata(video_ids)

        for video_data in videos_metadata:
            # Skip private/deleted videos
            if (
                not video_data
                or video_data.get("status", {}).get("privacyStatus") != "public"
            ):
                logger.debug(
                    f"Skipping non-public video: {video_data.get('id') if video_data else 'unknown'}"
                )
                continue

            normalized = self._normalize_video_metadata(video_data)

            # Filter shorts if not requested
            if not include_shorts and normalized["is_short"]:
                logger.debug(f"Skipping Short: {normalized['video_id']}")
                continue

            logger.debug(
                f"Yielding video: {normalized['video_id']} - {normalized['title'][:50]}..."
            )
            yield normalized
