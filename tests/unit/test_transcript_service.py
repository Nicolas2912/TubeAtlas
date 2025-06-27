from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from src.tubeatlas.services.transcript_service import TranscriptService


@pytest.fixture
def transcript_service():
    """Fixture for TranscriptService."""
    return TranscriptService()


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_success_preferred_language(
    mock_api, transcript_service: TranscriptService
):
    """Test successful transcript retrieval with a preferred language."""
    # Arrange
    video_id = "test_video_id"
    mock_transcript = MagicMock()
    mock_transcript.language_code = "en"
    mock_transcript.is_generated = False
    mock_transcript.fetch.return_value = [
        {"text": "Hello", "start": 0.0, "duration": 1.0}
    ]

    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.return_value = mock_transcript
    mock_api.list_transcripts.return_value = mock_transcript_list

    # Act
    result = await transcript_service.get_transcript(video_id, language_codes=["en"])

    # Assert
    assert result["status"] == "success"
    assert result["video_id"] == video_id
    assert result["language_code"] == "en"
    assert not result["is_generated"]
    assert result["segments"] == [{"text": "Hello", "start": 0.0, "duration": 1.0}]
    mock_api.list_transcripts.assert_called_once_with(video_id)
    mock_transcript_list.find_transcript.assert_called_once_with(["en"])
    mock_transcript.fetch.assert_called_once()


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_fallback_to_manual(
    mock_api, transcript_service: TranscriptService
):
    """Test fallback to a manually created transcript when preferred is not found."""
    # Arrange
    video_id = "test_video_id"
    mock_manual_transcript = MagicMock()
    mock_manual_transcript.language_code = "de"
    mock_manual_transcript.is_generated = False
    mock_manual_transcript.fetch.return_value = [
        {"text": "Hallo", "start": 0.0, "duration": 1.0}
    ]

    # Simulate iterator for transcript_list
    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.side_effect = NoTranscriptFound(
        video_id, ["en"], {"de": "German"}
    )
    mock_transcript_list.__iter__.return_value = iter([mock_manual_transcript])
    mock_api.list_transcripts.return_value = mock_transcript_list

    # Act
    result = await transcript_service.get_transcript(video_id, language_codes=["en"])

    # Assert
    assert result["status"] == "success"
    assert result["language_code"] == "de"
    assert not result["is_generated"]
    assert result["segments"] is not None
    assert result["segments"][0]["text"] == "Hallo"
    assert mock_transcript_list.find_transcript.call_count == 1


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_fallback_to_generated(
    mock_api, transcript_service: TranscriptService
):
    """Test fallback to the first available transcript (generated)."""
    # Arrange
    video_id = "test_video_id"
    mock_generated_transcript_es = MagicMock()
    mock_generated_transcript_es.language_code = "es"
    mock_generated_transcript_es.is_generated = True
    mock_generated_transcript_es.fetch.return_value = [
        {"text": "Hola", "start": 0.0, "duration": 1.0}
    ]

    mock_generated_transcript_fr = MagicMock()
    mock_generated_transcript_fr.language_code = "fr"
    mock_generated_transcript_fr.is_generated = True

    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.side_effect = NoTranscriptFound(
        video_id, ["en"], {"es": "Spanish", "fr": "French"}
    )
    # The list of available transcripts only contains generated ones.
    # The logic should not find a manual one and fallback to the first in the list.
    mock_transcript_list.__iter__.return_value = iter(
        [mock_generated_transcript_es, mock_generated_transcript_fr]
    )
    mock_api.list_transcripts.return_value = mock_transcript_list

    # Act
    result = await transcript_service.get_transcript(video_id, language_codes=["en"])

    # Assert
    assert result["status"] == "success"
    # It should pick the first one from the list, which is 'es'.
    assert result["language_code"] == "es"
    assert result["is_generated"] is True
    assert result["segments"] is not None
    assert result["segments"][0]["text"] == "Hola"


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_transcripts_disabled(
    mock_api, transcript_service: TranscriptService
):
    """Test handling of TranscriptsDisabled exception."""
    # Arrange
    video_id = "test_video_id"
    mock_api.list_transcripts.side_effect = TranscriptsDisabled(video_id)

    # Act
    result = await transcript_service.get_transcript(video_id)

    # Assert
    assert result["status"] == "disabled"
    assert result["video_id"] == video_id
    assert result["segments"] is None


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_no_transcript_found_at_all(
    mock_api, transcript_service: TranscriptService
):
    """Test handling of NoTranscriptFound exception."""
    # Arrange
    video_id = "test_video_id"
    mock_api.list_transcripts.side_effect = NoTranscriptFound(video_id, [], {})

    # Act
    result = await transcript_service.get_transcript(video_id)

    # Assert
    assert result["status"] == "not_found"
    assert result["video_id"] == video_id
    assert result["segments"] is None


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_no_suitable_transcript_found_after_iteration(
    mock_api, transcript_service: TranscriptService
):
    """Test case where list is returned but is empty."""
    # Arrange
    video_id = "test_video_id"

    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.side_effect = NoTranscriptFound(
        video_id, ["en"], {}
    )
    # Make it an empty iterator
    mock_transcript_list.__iter__.return_value = iter([])
    mock_api.list_transcripts.return_value = mock_transcript_list

    # Act
    result = await transcript_service.get_transcript(video_id, language_codes=["en"])

    # Assert
    assert result["status"] == "not_found"
    assert result["video_id"] == video_id


@pytest.mark.asyncio
@patch("src.tubeatlas.services.transcript_service.YouTubeTranscriptApi")
async def test_get_transcript_fetch_error(
    mock_api, transcript_service: TranscriptService
):
    """Test handling of an exception during the final fetch."""
    # Arrange
    video_id = "test_video_id"
    mock_transcript = MagicMock()
    mock_transcript.fetch.side_effect = Exception("Fetch failed!")

    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.return_value = mock_transcript
    mock_api.list_transcripts.return_value = mock_transcript_list

    # Act
    result = await transcript_service.get_transcript(video_id)

    # Assert
    assert result["status"] == "fetch_error"
    assert result["video_id"] == video_id
    assert result["segments"] is None


@pytest.mark.asyncio
async def test_extract_transcript_success(transcript_service: TranscriptService):
    """Test extract_transcript successfully concatenates text."""
    # Arrange
    video_id = "test_video_id"
    mock_transcript_data = {
        "status": "success",
        "video_id": video_id,
        "language_code": "en",
        "is_generated": False,
        "segments": [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
        ],
    }

    with patch.object(
        transcript_service, "get_transcript", return_value=mock_transcript_data
    ) as mock_get:
        # Act
        result = await transcript_service.extract_transcript(video_id)
        # Assert
        assert result == "Hello world"
        mock_get.assert_called_once_with(video_id, ["en"])


@pytest.mark.asyncio
async def test_extract_transcript_failure(transcript_service: TranscriptService):
    """Test extract_transcript returns None when get_transcript fails."""
    # Arrange
    video_id = "test_video_id"
    mock_transcript_data = {
        "status": "disabled",
        "video_id": video_id,
        "language_code": None,
        "is_generated": None,
        "segments": None,
    }

    with patch.object(
        transcript_service, "get_transcript", return_value=mock_transcript_data
    ):
        # Act
        result = await transcript_service.extract_transcript(video_id)
        # Assert
        assert result is None
