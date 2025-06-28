"""Unit tests for YouTube service."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from googleapiclient.errors import HttpError

from src.tubeatlas.services.youtube_service import YouTubeService
from src.tubeatlas.utils.exceptions import QuotaExceededError, TransientAPIError


class TestYouTubeService:
    """Test cases for YouTubeService class."""

    @pytest.fixture
    def api_key(self):
        """Mock API key for testing."""
        return "test_api_key_123"

    @pytest.fixture
    def youtube_service(self, api_key):
        """Create YouTubeService instance for testing."""
        return YouTubeService(api_key=api_key)

    def test_init_with_api_key(self, api_key):
        """Test service initialization with API key."""
        service = YouTubeService(api_key=api_key)
        assert service.api_key == api_key
        assert service._youtube_client is None

    def test_init_without_api_key_raises_error(self):
        """Test service initialization without API key raises ValueError."""
        with patch("src.tubeatlas.services.youtube_service.settings") as mock_settings:
            mock_settings.youtube_api_key = None
            mock_settings.google_api_key = None

            with pytest.raises(ValueError, match="YouTube API key is required"):
                YouTubeService()

    @patch("src.tubeatlas.services.youtube_service.build")
    def test_youtube_client_property_creates_client(self, mock_build, youtube_service):
        """Test that youtube_client property creates and caches the client."""
        mock_client = Mock()
        mock_build.return_value = mock_client

        # First access should create client
        client1 = youtube_service.youtube_client
        assert client1 == mock_client
        mock_build.assert_called_once_with(
            "youtube", "v3", developerKey=youtube_service.api_key
        )

        # Second access should return cached client
        client2 = youtube_service.youtube_client
        assert client2 == mock_client
        assert mock_build.call_count == 1  # Should not be called again

    @patch("src.tubeatlas.services.youtube_service.build")
    def test_youtube_client_property_handles_build_error(
        self, mock_build, youtube_service
    ):
        """Test that youtube_client property handles build errors gracefully."""
        mock_build.side_effect = Exception("Build failed")

        with pytest.raises(
            TransientAPIError, match="Failed to initialize YouTube API client"
        ):
            _ = youtube_service.youtube_client

    def test_execute_with_retry_success_first_attempt(self, youtube_service):
        """Test successful request on first attempt."""
        mock_request_method = Mock()
        mock_request = Mock()
        mock_request.execute.return_value = {"items": [{"id": "test"}]}
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        result = youtube_service._execute_with_retry(
            mock_request_method, params, max_retries=3, initial_wait=1.0, max_wait=64.0
        )

        assert result == {"items": [{"id": "test"}]}
        mock_request_method.assert_called_once_with(**params)
        mock_request.execute.assert_called_once()

    def test_execute_with_retry_success_after_retries(self, youtube_service):
        """Test successful request after some retries."""
        mock_request_method = Mock()
        mock_request = Mock()

        # Fail twice, then succeed
        mock_request.execute.side_effect = [
            HttpError(Mock(status=503, reason="Service Unavailable"), b""),
            HttpError(Mock(status=503, reason="Service Unavailable"), b""),
            {"items": [{"id": "test"}]},
        ]
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = youtube_service._execute_with_retry(
                mock_request_method,
                params,
                max_retries=3,
                initial_wait=0.1,
                max_wait=1.0,
            )

        assert result == {"items": [{"id": "test"}]}
        assert mock_request.execute.call_count == 3

    def test_execute_with_retry_quota_exceeded_error(self, youtube_service):
        """Test that quota exceeded errors are properly handled."""
        mock_request_method = Mock()
        mock_request = Mock()

        # Simulate quota exceeded error
        mock_response = Mock(status=403, reason="Forbidden")
        error_content = b'{"error": {"code": 403, "message": "quota exceeded"}}'
        mock_request.execute.side_effect = HttpError(mock_response, error_content)
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(QuotaExceededError, match="YouTube API quota exceeded"):
                youtube_service._execute_with_retry(
                    mock_request_method,
                    params,
                    max_retries=2,
                    initial_wait=0.1,
                    max_wait=1.0,
                )

        assert mock_request.execute.call_count == 3  # Initial + 2 retries

    def test_execute_with_retry_transient_error_exhausted(self, youtube_service):
        """Test that transient errors raise TransientAPIError after max retries."""
        mock_request_method = Mock()
        mock_request = Mock()

        # Simulate persistent server error
        mock_response = Mock(status=503, reason="Service Unavailable")
        mock_request.execute.side_effect = HttpError(mock_response, b"")
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(TransientAPIError, match="YouTube API transient error"):
                youtube_service._execute_with_retry(
                    mock_request_method,
                    params,
                    max_retries=2,
                    initial_wait=0.1,
                    max_wait=1.0,
                )

        assert mock_request.execute.call_count == 3  # Initial + 2 retries

    def test_execute_with_retry_non_retryable_error(self, youtube_service):
        """Test that non-retryable errors are raised immediately."""
        mock_request_method = Mock()
        mock_request = Mock()

        # Simulate non-retryable error (404)
        mock_response = Mock(status=404, reason="Not Found")
        mock_request.execute.side_effect = HttpError(mock_response, b"")
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        with pytest.raises(TransientAPIError, match="YouTube API error: Status 404"):
            youtube_service._execute_with_retry(
                mock_request_method,
                params,
                max_retries=2,
                initial_wait=0.1,
                max_wait=1.0,
            )

        assert mock_request.execute.call_count == 1  # Should not retry

    def test_execute_with_retry_unexpected_error(self, youtube_service):
        """Test handling of unexpected errors."""
        mock_request_method = Mock()
        mock_request = Mock()

        # Simulate unexpected error
        mock_request.execute.side_effect = [
            Exception("Unexpected error"),
            Exception("Unexpected error"),
            {"items": [{"id": "test"}]},
        ]
        mock_request_method.return_value = mock_request

        params = {"part": "snippet", "maxResults": 50}

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = youtube_service._execute_with_retry(
                mock_request_method,
                params,
                max_retries=3,
                initial_wait=0.1,
                max_wait=1.0,
            )

        assert result == {"items": [{"id": "test"}]}
        assert mock_request.execute.call_count == 3

    def test_paged_request_single_page(self, youtube_service):
        """Test paginated request with single page of results."""
        mock_request_method = Mock()

        with patch.object(youtube_service, "_execute_with_retry") as mock_execute:
            mock_execute.return_value = {
                "items": [
                    {"id": "video1", "snippet": {"title": "Test Video 1"}},
                    {"id": "video2", "snippet": {"title": "Test Video 2"}},
                ]
                # No nextPageToken = single page
            }

            params = {"part": "snippet", "maxResults": 50}
            items = list(youtube_service._paged_request(mock_request_method, params))

            assert len(items) == 2
            assert items[0]["id"] == "video1"
            assert items[1]["id"] == "video2"

            # Should call execute once
            mock_execute.assert_called_once()

    def test_paged_request_multiple_pages(self, youtube_service):
        """Test paginated request with multiple pages of results."""
        mock_request_method = Mock()

        with patch.object(youtube_service, "_execute_with_retry") as mock_execute:
            # Simulate two pages of results
            mock_execute.side_effect = [
                {
                    "items": [
                        {"id": "video1", "snippet": {"title": "Test Video 1"}},
                        {"id": "video2", "snippet": {"title": "Test Video 2"}},
                    ],
                    "nextPageToken": "page2_token",
                },
                {
                    "items": [{"id": "video3", "snippet": {"title": "Test Video 3"}}]
                    # No nextPageToken = last page
                },
            ]

            params = {"part": "snippet", "maxResults": 50}
            items = list(youtube_service._paged_request(mock_request_method, params))

            assert len(items) == 3
            assert items[0]["id"] == "video1"
            assert items[1]["id"] == "video2"
            assert items[2]["id"] == "video3"

            # Should call execute twice (once per page)
            assert mock_execute.call_count == 2

            # Check that pageToken was passed in second call
            second_call_params = mock_execute.call_args_list[1][0][1]
            assert second_call_params["pageToken"] == "page2_token"

    def test_paged_request_empty_response(self, youtube_service):
        """Test paginated request with empty response."""
        mock_request_method = Mock()

        with patch.object(youtube_service, "_execute_with_retry") as mock_execute:
            mock_execute.return_value = {}  # No items key

            params = {"part": "snippet", "maxResults": 50}
            items = list(youtube_service._paged_request(mock_request_method, params))

            assert len(items) == 0
            mock_execute.assert_called_once()

    def test_paged_request_with_retry_propagation(self, youtube_service):
        """Test that pagination properly propagates retry parameters."""
        mock_request_method = Mock()

        with patch.object(youtube_service, "_execute_with_retry") as mock_execute:
            mock_execute.return_value = {"items": [{"id": "test"}]}

            params = {"part": "snippet", "maxResults": 50}

            # Call with custom retry parameters
            list(
                youtube_service._paged_request(
                    mock_request_method,
                    params,
                    max_retries=5,
                    initial_wait=2.0,
                    max_wait=128.0,
                )
            )

            # Verify retry parameters were passed through
            mock_execute.assert_called_once_with(
                mock_request_method,
                params,
                5,  # max_retries
                2.0,  # initial_wait
                128.0,  # max_wait
            )

    def test_extract_channel_id_from_channel_url(self, youtube_service):
        """Test channel ID extraction from /channel/ URL format."""
        url = "https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw"
        result = youtube_service._extract_channel_id(url)
        assert result == "UC_x5XG1OV2P6uZZ5FSM9Ttw"

    def test_extract_channel_id_from_handle(self, youtube_service):
        """Test channel ID extraction from @handle format."""
        url = "https://www.youtube.com/@testchannel"

        # Mock the API response
        mock_response = {"items": [{"id": "UC_test_channel_id"}]}
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        result = youtube_service._extract_channel_id(url)
        assert result == "UC_test_channel_id"

    def test_extract_channel_id_from_user(self, youtube_service):
        """Test channel ID extraction from /user/ format."""
        url = "https://www.youtube.com/user/testuser"

        # Mock the API response
        mock_response = {"items": [{"id": "UC_test_user_id"}]}
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        result = youtube_service._extract_channel_id(url)
        assert result == "UC_test_user_id"

    def test_extract_channel_id_from_custom(self, youtube_service):
        """Test channel ID extraction from /c/ format."""
        url = "https://www.youtube.com/c/customname"

        # Mock the API response
        mock_response = {"items": [{"id": "UC_custom_id"}]}
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        result = youtube_service._extract_channel_id(url)
        assert result == "UC_custom_id"

    def test_extract_channel_id_unsupported_format(self, youtube_service):
        """Test channel ID extraction with unsupported URL format."""
        url = "https://example.com/invalid"

        with pytest.raises(TransientAPIError, match="Unsupported channel URL format"):
            youtube_service._extract_channel_id(url)

    def test_get_uploads_playlist_id_success(self, youtube_service):
        """Test successful uploads playlist ID retrieval."""
        channel_id = "UC_test_channel"
        mock_response = {
            "items": [
                {
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU_test_channel_uploads"}
                    }
                }
            ]
        }
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        result = youtube_service._get_uploads_playlist_id(channel_id)
        assert result == "UU_test_channel_uploads"

    def test_get_uploads_playlist_id_channel_not_found(self, youtube_service):
        """Test uploads playlist ID retrieval when channel not found."""
        channel_id = "UC_nonexistent"
        mock_response: dict = {"items": []}
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        with pytest.raises(TransientAPIError, match="Channel not found"):
            youtube_service._get_uploads_playlist_id(channel_id)

    def test_is_short_by_category_id(self, youtube_service):
        """Test Short detection by category ID."""
        video_data = {
            "id": "test_video",
            "snippet": {"categoryId": "42"},
            "contentDetails": {"duration": "PT2M30S"},
        }

        result = youtube_service._is_short(video_data)
        assert result is True

    def test_is_short_by_duration(self, youtube_service):
        """Test Short detection by duration."""
        video_data = {
            "id": "test_video",
            "snippet": {"categoryId": "24"},
            "contentDetails": {"duration": "PT45S"},
        }

        result = youtube_service._is_short(video_data)
        assert result is True

    def test_is_short_by_title_hashtag(self, youtube_service):
        """Test Short detection by #shorts in title."""
        video_data = {
            "id": "test_video",
            "snippet": {"categoryId": "24", "title": "Amazing trick #shorts"},
            "contentDetails": {"duration": "PT2M30S"},
        }

        result = youtube_service._is_short(video_data)
        assert result is True

    def test_is_short_by_tags(self, youtube_service):
        """Test Short detection by tags."""
        video_data = {
            "id": "test_video",
            "snippet": {
                "categoryId": "24",
                "title": "Normal video",
                "tags": ["tutorial", "#shorts", "howto"],
            },
            "contentDetails": {"duration": "PT2M30S"},
        }

        result = youtube_service._is_short(video_data)
        assert result is True

    def test_is_not_short(self, youtube_service):
        """Test normal video (not a Short) detection."""
        video_data = {
            "id": "test_video",
            "snippet": {
                "categoryId": "24",
                "title": "Normal video tutorial",
                "tags": ["tutorial", "howto"],
            },
            "contentDetails": {"duration": "PT5M30S"},
        }

        result = youtube_service._is_short(video_data)
        assert result is False

    def test_batch_get_video_metadata_success(self, youtube_service):
        """Test successful batch video metadata retrieval."""
        video_ids = ["video1", "video2"]
        mock_response = {
            "items": [
                {"id": "video1", "snippet": {"title": "Video 1"}},
                {"id": "video2", "snippet": {"title": "Video 2"}},
            ]
        }
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        result = youtube_service._batch_get_video_metadata(video_ids)
        assert len(result) == 2
        assert result[0]["id"] == "video1"
        assert result[1]["id"] == "video2"

    def test_batch_get_video_metadata_empty_list(self, youtube_service):
        """Test batch metadata retrieval with empty video list."""
        result = youtube_service._batch_get_video_metadata([])
        assert result == []

    def test_batch_get_video_metadata_truncation(self, youtube_service):
        """Test batch metadata retrieval with >50 videos (should truncate)."""
        video_ids = [f"video{i}" for i in range(60)]  # 60 videos
        mock_response: dict = {"items": []}
        youtube_service._execute_with_retry = Mock(return_value=mock_response)

        youtube_service._batch_get_video_metadata(video_ids)

        # Should only call with first 50 videos
        args, kwargs = youtube_service._execute_with_retry.call_args
        params = args[1]  # Second argument contains the parameters
        assert "," in params["id"]
        assert len(params["id"].split(",")) == 50

    def test_normalize_video_metadata(self, youtube_service):
        """Test video metadata normalization."""
        video_data = {
            "id": "test_video",
            "snippet": {
                "title": "Test Video",
                "description": "Test description",
                "channelId": "UC_test",
                "channelTitle": "Test Channel",
                "publishedAt": "2024-01-01T00:00:00Z",
                "categoryId": "24",
                "tags": ["test", "video"],
            },
            "contentDetails": {"duration": "PT5M30S"},
            "statistics": {
                "viewCount": "1000",
                "likeCount": "100",
                "commentCount": "10",
            },
            "status": {
                "privacyStatus": "public",
                "embeddable": True,
                "madeForKids": False,
            },
        }

        result = youtube_service._normalize_video_metadata(video_data)

        assert result["video_id"] == "test_video"
        assert result["title"] == "Test Video"
        assert result["description"] == "Test description"
        assert result["channel_id"] == "UC_test"
        assert result["channel_title"] == "Test Channel"
        assert result["published_at"] == "2024-01-01T00:00:00Z"
        assert result["duration_seconds"] == 330  # 5m30s
        assert result["duration_iso"] == "PT5M30S"
        assert result["category_id"] == "24"
        assert result["tags"] == ["test", "video"]
        assert result["view_count"] == 1000
        assert result["like_count"] == 100
        assert result["comment_count"] == 10
        assert result["privacy_status"] == "public"
        assert result["embeddable"] is True
        assert result["made_for_kids"] is False
        assert result["is_short"] is False
        assert result["raw_json"] == video_data

    def test_normalize_video_metadata_missing_fields(self, youtube_service):
        """Test metadata normalization with missing fields."""
        video_data = {"id": "test_video", "snippet": {"title": "Minimal Video"}}

        result = youtube_service._normalize_video_metadata(video_data)

        assert result["video_id"] == "test_video"
        assert result["title"] == "Minimal Video"
        assert result["description"] == ""
        assert result["duration_seconds"] is None
        assert result["view_count"] == 0
        assert result["tags"] == []

    def test_persist_raw_response_success(self, youtube_service):
        """Test successful persistence of raw JSON response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override the raw_data_dir for testing
            youtube_service.raw_data_dir = temp_dir

            video_id = "test_video_123"
            raw_data = {
                "id": video_id,
                "snippet": {"title": "Test Video"},
                "statistics": {"viewCount": "1000"},
            }

            # Call the persistence method
            youtube_service._persist_raw_response(video_id, raw_data)

            # Verify file was created
            expected_filepath = os.path.join(temp_dir, f"{video_id}.json")
            assert os.path.exists(expected_filepath)

            # Verify file contents
            with open(expected_filepath, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data == raw_data

    def test_persist_raw_response_io_error(self, youtube_service):
        """Test that IOError during persistence doesn't abort processing."""
        # Use an invalid directory path to trigger IOError
        youtube_service.raw_data_dir = "/invalid/path/that/does/not/exist"

        video_id = "test_video_123"
        raw_data = {"id": video_id, "snippet": {"title": "Test Video"}}

        # Should not raise exception, just log warning
        with patch("src.tubeatlas.services.youtube_service.logger") as mock_logger:
            youtube_service._persist_raw_response(video_id, raw_data)
            mock_logger.warning.assert_called_once()
            assert "Failed to persist raw JSON" in str(mock_logger.warning.call_args)

    def test_normalize_video_metadata_persists_raw_data(self, youtube_service):
        """Test that normalize_video_metadata calls _persist_raw_response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override the raw_data_dir for testing
            youtube_service.raw_data_dir = temp_dir

            video_data = {
                "id": "test_video_456",
                "snippet": {
                    "title": "Test Video",
                    "description": "Test Description",
                    "channelId": "test_channel",
                    "channelTitle": "Test Channel",
                    "publishedAt": "2023-01-01T00:00:00Z",
                },
                "contentDetails": {"duration": "PT5M30S"},
                "statistics": {"viewCount": "1000"},
                "status": {"privacyStatus": "public"},
            }

            # Call normalize_video_metadata
            normalized = youtube_service._normalize_video_metadata(video_data)

            # Verify the raw data was persisted
            expected_filepath = os.path.join(temp_dir, "test_video_456.json")
            assert os.path.exists(expected_filepath)

            # Verify file contents match the input
            with open(expected_filepath, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data == video_data
            assert normalized["video_id"] == "test_video_456"

    def test_process_video_batch_filters_shorts(self, youtube_service):
        """Test video batch processing filters out Shorts when not requested."""
        video_ids = ["video1", "video2"]

        # Mock batch metadata retrieval
        mock_videos = [
            {
                "id": "video1",
                "snippet": {"title": "Normal Video", "categoryId": "24"},
                "contentDetails": {"duration": "PT5M00S"},
                "status": {"privacyStatus": "public"},
            },
            {
                "id": "video2",
                "snippet": {"title": "Short Video", "categoryId": "42"},
                "contentDetails": {"duration": "PT30S"},
                "status": {"privacyStatus": "public"},
            },
        ]
        youtube_service._batch_get_video_metadata = Mock(return_value=mock_videos)

        # Process batch without including shorts
        results = list(
            youtube_service._process_video_batch(
                video_ids, include_shorts=False, max_videos=None, current_count=0
            )
        )

        # Should only return the normal video
        assert len(results) == 1
        assert results[0]["video_id"] == "video1"

    def test_process_video_batch_includes_shorts(self, youtube_service):
        """Test video batch processing includes Shorts when requested."""
        video_ids = ["video1", "video2"]

        # Mock batch metadata retrieval
        mock_videos = [
            {
                "id": "video1",
                "snippet": {"title": "Normal Video", "categoryId": "24"},
                "contentDetails": {"duration": "PT5M00S"},
                "status": {"privacyStatus": "public"},
            },
            {
                "id": "video2",
                "snippet": {"title": "Short Video", "categoryId": "42"},
                "contentDetails": {"duration": "PT30S"},
                "status": {"privacyStatus": "public"},
            },
        ]
        youtube_service._batch_get_video_metadata = Mock(return_value=mock_videos)

        # Process batch with shorts included
        results = list(
            youtube_service._process_video_batch(
                video_ids, include_shorts=True, max_videos=None, current_count=0
            )
        )

        # Should return both videos
        assert len(results) == 2
        assert results[0]["video_id"] == "video1"
        assert results[1]["video_id"] == "video2"

    def test_process_video_batch_filters_private_videos(self, youtube_service):
        """Test video batch processing filters out private/deleted videos."""
        video_ids = ["video1", "video2"]

        # Mock batch metadata retrieval
        mock_videos = [
            {
                "id": "video1",
                "snippet": {"title": "Public Video", "categoryId": "24"},
                "contentDetails": {"duration": "PT5M00S"},
                "status": {"privacyStatus": "public"},
            },
            {
                "id": "video2",
                "snippet": {"title": "Private Video", "categoryId": "24"},
                "contentDetails": {"duration": "PT3M00S"},
                "status": {"privacyStatus": "private"},
            },
        ]
        youtube_service._batch_get_video_metadata = Mock(return_value=mock_videos)

        # Process batch
        results = list(
            youtube_service._process_video_batch(
                video_ids, include_shorts=True, max_videos=None, current_count=0
            )
        )

        # Should only return the public video
        assert len(results) == 1
        assert results[0]["video_id"] == "video1"

    @patch("src.tubeatlas.services.youtube_service.YouTubeService._paged_request")
    def test_fetch_channel_videos_integration(
        self, mock_paged_request, youtube_service
    ):
        """Test full fetch_channel_videos workflow."""
        # Mock channel resolution
        youtube_service._extract_channel_id = Mock(return_value="UC_test_channel")
        youtube_service._get_uploads_playlist_id = Mock(return_value="UU_test_uploads")

        # Mock paginated playlist items
        mock_playlist_items = [
            {"contentDetails": {"videoId": "video1"}},
            {"contentDetails": {"videoId": "video2"}},
        ]
        mock_paged_request.return_value = iter(mock_playlist_items)

        # Mock video metadata
        mock_videos = [
            {
                "id": "video1",
                "snippet": {"title": "Video 1", "categoryId": "24"},
                "contentDetails": {"duration": "PT5M00S"},
                "status": {"privacyStatus": "public"},
            },
            {
                "id": "video2",
                "snippet": {"title": "Video 2", "categoryId": "24"},
                "contentDetails": {"duration": "PT3M00S"},
                "status": {"privacyStatus": "public"},
            },
        ]
        youtube_service._batch_get_video_metadata = Mock(return_value=mock_videos)

        # Test the generator
        videos = list(
            youtube_service.fetch_channel_videos(
                "https://www.youtube.com/channel/UC_test_channel",
                include_shorts=False,
                max_videos=10,
            )
        )

        assert len(videos) == 2
        assert videos[0]["video_id"] == "video1"
        assert videos[1]["video_id"] == "video2"

    @patch("src.tubeatlas.services.youtube_service.YouTubeService._paged_request")
    def test_fetch_channel_videos_max_videos_limit(
        self, mock_paged_request, youtube_service
    ):
        """Test fetch_channel_videos respects max_videos limit."""
        # Mock channel resolution
        youtube_service._extract_channel_id = Mock(return_value="UC_test_channel")
        youtube_service._get_uploads_playlist_id = Mock(return_value="UU_test_uploads")

        # Mock many playlist items
        mock_playlist_items = [
            {"contentDetails": {"videoId": f"video{i}"}} for i in range(10)
        ]
        mock_paged_request.return_value = iter(mock_playlist_items)

        # Mock video metadata
        mock_videos = [
            {
                "id": f"video{i}",
                "snippet": {"title": f"Video {i}", "categoryId": "24"},
                "contentDetails": {"duration": "PT5M00S"},
                "status": {"privacyStatus": "public"},
            }
            for i in range(10)
        ]
        youtube_service._batch_get_video_metadata = Mock(return_value=mock_videos)

        # Test with max_videos=3
        videos = list(
            youtube_service.fetch_channel_videos(
                "https://www.youtube.com/channel/UC_test_channel",
                include_shorts=False,
                max_videos=3,
            )
        )

        # Should only return 3 videos
        assert len(videos) == 3
