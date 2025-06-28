from typing import AsyncGenerator, Dict, List

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tubeatlas.config.database import Base
from tubeatlas.repositories.video_repository import VideoRepository
from tubeatlas.services.transcript_service import TranscriptService


class _FakeYouTubeService:
    """A minimal fake of YouTubeService for testing."""

    def __init__(self, videos: List[Dict[str, str]]):
        self._videos = videos

    async def fetch_channel_videos(self, *args, **kwargs):  # type: ignore[override]
        for v in self._videos:
            yield v


@pytest_asyncio.fixture()
async def async_session() -> AsyncGenerator[AsyncSession, None]:  # type: ignore[misc]
    """Provide an in-memory SQLite AsyncSession for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async_session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)

    # Create all tables in the in-memory database.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


# ---------------------------------------------------------------------------
# VideoRepository.upsert -----------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_repository_upsert(async_session: AsyncSession):
    repo = VideoRepository(async_session)

    data = {
        "id": "vid1",
        "channel_id": "chan1",
        "title": "original title",
    }

    created = await repo.upsert(data)
    assert created.id == "vid1"
    assert created.title == "original title"

    # Upsert with updated title
    data_updated = {
        "id": "vid1",
        "channel_id": "chan1",
        "title": "updated title",
    }

    updated = await repo.upsert(data_updated)
    assert updated.id == "vid1"
    assert updated.title == "updated title"

    # Ensure only one row exists
    rows = await repo.list_all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# TranscriptService.process_channel_transcripts incremental mode ------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_channel_transcripts_incremental(
    async_session: AsyncSession, monkeypatch
):
    # Prepare fake YouTube videos: vid1 already exists, vid2 new
    videos = [
        {
            "id": "vid1",
            "channel_id": "chan1",
            "title": "Video One",
            "channelTitle": "Demo Channel",
        },
        {
            "id": "vid2",
            "channel_id": "chan1",
            "title": "Video Two",
            "channelTitle": "Demo Channel",
        },
    ]

    fake_yt = _FakeYouTubeService(videos)

    # Pre-insert vid1 to simulate existing data
    video_repo = VideoRepository(async_session)
    await video_repo.upsert(videos[0])

    # Patch get_transcript to avoid external call
    async def _mock_get_transcript(self, video_id, *_, **__):  # type: ignore[unused-argument]
        return {
            "status": "success",
            "video_id": video_id,
            "language_code": "en",
            "is_generated": False,
            "segments": [
                {"text": "hello", "start": 0.0, "duration": 1.0, "token_count": 1}
            ],
            "total_token_count": 1,
        }

    monkeypatch.setattr(TranscriptService, "get_transcript", _mock_get_transcript)

    service = TranscriptService(async_session, fake_yt)

    summary = await service.process_channel_transcripts(
        channel_url="https://youtube.com/@demo", update_existing=False
    )

    # Since vid1 exists, service should stop before processing vid2
    assert summary["videos"]["created"] == 0  # no new videos
    assert summary["videos"]["updated"] == 0  # existing not updated because stop

    # Now run again with update_existing=True
    summary2 = await service.process_channel_transcripts(
        channel_url="https://youtube.com/@demo", update_existing=True
    )

    assert summary2["videos"]["updated"] >= 1  # vid1 updated
