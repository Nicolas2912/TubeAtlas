"""Tests for Celery tasks."""

from unittest.mock import AsyncMock, patch

import pytest

from tubeatlas.config.celery_app import celery_app
from tubeatlas.tasks import download_channel, download_video


@pytest.fixture
def celery_eager_mode():
    """Configure Celery to run tasks eagerly (synchronously) for testing."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield celery_app
    # Reset after test
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


class TestCeleryTasks:
    """Test suite for Celery tasks."""

    def test_download_channel_task_registration(self):
        """Test that download_channel task is properly registered."""
        assert "tubeatlas.tasks.download_channel" in celery_app.tasks
        task = celery_app.tasks["tubeatlas.tasks.download_channel"]
        assert task.max_retries == 3
        assert task.acks_late is True

    def test_download_video_task_registration(self):
        """Test that download_video task is properly registered."""
        assert "tubeatlas.tasks.download_video" in celery_app.tasks
        task = celery_app.tasks["tubeatlas.tasks.download_video"]
        assert task.max_retries == 3
        assert task.acks_late is True

    @patch("tubeatlas.tasks.asyncio.run")
    @patch("tubeatlas.tasks._download_channel_async")
    def test_download_channel_success(
        self, mock_async_func, mock_asyncio_run, celery_eager_mode
    ):
        """Test successful channel download task execution."""
        # Mock the async function return value
        expected_result = {
            "status": "completed",
            "channel_url": "https://www.youtube.com/channel/UC_test",
            "processed": 5,
            "succeeded": 4,
            "failed": 1,
            "skipped": 0,
            "include_shorts": False,
            "max_videos": None,
            "update_existing": False,
        }
        mock_async_func.return_value = expected_result
        mock_asyncio_run.return_value = expected_result

        # Execute the task
        result = download_channel.apply(
            args=["https://www.youtube.com/channel/UC_test"],
            kwargs={
                "include_shorts": False,
                "max_videos": None,
                "update_existing": False,
            },
        )

        # Verify results
        assert result.successful()
        assert result.result == expected_result
        mock_asyncio_run.assert_called_once()

    @patch("tubeatlas.tasks.asyncio.run")
    @patch("tubeatlas.tasks._download_video_async")
    def test_download_video_success(
        self, mock_async_func, mock_asyncio_run, celery_eager_mode
    ):
        """Test successful video download task execution."""
        # Mock the async function return value
        expected_result = {
            "status": "completed",
            "video_id": "test_video_123",
            "video_url": "https://www.youtube.com/watch?v=test_video_123",
            "transcript_status": "success",
            "title": "Test Video",
            "channel_title": "Test Channel",
        }
        mock_async_func.return_value = expected_result
        mock_asyncio_run.return_value = expected_result

        # Execute the task
        result = download_video.apply(
            args=["https://www.youtube.com/watch?v=test_video_123"]
        )

        # Verify results
        assert result.successful()
        assert result.result == expected_result
        mock_asyncio_run.assert_called_once()

    @patch("tubeatlas.tasks._download_channel_async", new_callable=AsyncMock)
    @patch("tubeatlas.tasks.asyncio.run")
    def test_download_channel_retry_on_api_error(
        self, mock_asyncio_run, mock_async_func, celery_eager_mode
    ):
        """Test that channel download retries on API errors."""
        from celery.exceptions import Retry

        from tubeatlas.utils.exceptions import QuotaExceededError

        # Configure the async mock to raise an API error
        mock_async_func.side_effect = QuotaExceededError("Quota exceeded")
        mock_asyncio_run.side_effect = QuotaExceededError("Quota exceeded")

        # Execute the task - should raise Retry exception since we're in eager mode
        with pytest.raises(Retry):
            download_channel.apply(args=["https://www.youtube.com/channel/UC_test"])

    @patch("tubeatlas.tasks._download_video_async", new_callable=AsyncMock)
    @patch("tubeatlas.tasks.asyncio.run")
    def test_download_video_retry_on_api_error(
        self, mock_asyncio_run, mock_async_func, celery_eager_mode
    ):
        """Test that video download retries on API errors."""
        from celery.exceptions import Retry

        from tubeatlas.utils.exceptions import TransientAPIError

        # Configure the async mock to raise an API error
        mock_async_func.side_effect = TransientAPIError("API temporarily unavailable")
        mock_asyncio_run.side_effect = TransientAPIError("API temporarily unavailable")

        # Execute the task - should raise Retry exception since we're in eager mode
        with pytest.raises(Retry):
            download_video.apply(
                args=["https://www.youtube.com/watch?v=test_video_123"]
            )

    def test_download_channel_task_routing(self):
        """Test that download_channel is routed to the correct queue."""
        routing = celery_app.conf.task_routes
        assert "tubeatlas.tasks.download_channel" in routing
        assert routing["tubeatlas.tasks.download_channel"]["queue"] == "youtube"

    def test_download_video_task_routing(self):
        """Test that download_video is routed to the correct queue."""
        routing = celery_app.conf.task_routes
        assert "tubeatlas.tasks.download_video" in routing
        assert routing["tubeatlas.tasks.download_video"]["queue"] == "youtube"
