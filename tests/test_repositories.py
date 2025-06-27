"""Tests for repository layer CRUD operations."""

import asyncio
from datetime import UTC, datetime
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tubeatlas.config.database import Base
from tubeatlas.models import ProcessingTask, Transcript, Video
from tubeatlas.repositories import ProcessingTaskRepository, TranscriptRepository


@pytest.fixture(scope="module")
def event_loop():  # type: ignore[override]
    """Create an instance of the default event loop for the test module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture()
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an in-memory SQLite AsyncSession for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async_session_maker = async_sessionmaker(bind=engine, expire_on_commit=False)

    # Create all tables in the in-memory database.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


# -----------------------------------------------------------------------------
# TranscriptRepository CRUD ----------------------------------------------------
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_repository_crud(async_session: AsyncSession):
    """Full CRUD cycle for TranscriptRepository."""
    # First insert a video for the FK constraint
    video = Video(
        id="vid1",
        channel_id="chan1",
        title="Demo video",
    )
    async_session.add(video)
    await async_session.commit()

    repo = TranscriptRepository(async_session)

    # CREATE
    transcript = Transcript(
        video_id="vid1",
        video_title="Demo video",
        video_url="https://youtube.com/watch?v=vid1",
        channel_name="Demo Channel",
        channel_url="https://youtube.com/@demo",
        content="hello world",
        language="en",
    )
    created = await repo.create(transcript)
    assert created.video_id == "vid1"
    assert created.processing_status == "pending"

    # READ
    fetched = await repo.get_by_id("vid1")
    assert fetched is not None and fetched.video_url.endswith("vid1")

    # UPDATE
    fetched.content = "updated content"
    updated = await repo.update(fetched)
    assert updated.content == "updated content"

    # LIST ALL
    all_items = await repo.list_all()
    assert len(all_items) == 1

    # LIST BY CHANNEL
    by_channel = await repo.list_by_channel("Demo Channel")
    assert len(by_channel) == 1

    # DELETE
    deleted = await repo.delete("vid1")
    assert deleted is True
    assert await repo.get_by_id("vid1") is None


# -----------------------------------------------------------------------------
# ProcessingTaskRepository CRUD ------------------------------------------------
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_processing_task_repository_crud(async_session: AsyncSession):
    """Full CRUD cycle for ProcessingTaskRepository."""
    repo = ProcessingTaskRepository(async_session)

    # CREATE
    task = ProcessingTask(
        id="task1",
        task_type="transcript",
        target_id="vid999",
        status="pending",
        progress_percent=0,
        created_at=datetime.now(UTC),
    )
    created = await repo.create(task)
    assert created.id == "task1"

    # READ
    fetched = await repo.get_by_id("task1")
    assert fetched is not None and fetched.status == "pending"

    # UPDATE
    fetched.status = "done"
    fetched.progress_percent = 100
    updated = await repo.update(fetched)
    assert updated.status == "done"
    assert updated.progress_percent == 100

    # LIST ALL
    tasks = await repo.list_all()
    assert len(tasks) == 1

    # LIST BY STATUS
    done_tasks = await repo.list_by_status("done")
    assert len(done_tasks) == 1 and done_tasks[0].id == "task1"

    pending_tasks = await repo.list_pending()
    assert pending_tasks == []

    # DELETE
    deleted = await repo.delete("task1")
    assert deleted is True
    assert await repo.get_by_id("task1") is None
