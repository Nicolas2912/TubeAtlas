"""Tests for ORM models."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from tubeatlas.config.database import Base
from tubeatlas.models import KnowledgeGraph, ProcessingTask, Transcript, Video


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_create_all_tables(engine):
    """Test that all tables are created successfully."""
    # Create all tables
    Base.metadata.create_all(engine)

    # Verify all expected tables exist
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]

        expected_tables = [
            "videos",
            "transcripts",
            "knowledge_graphs",
            "processing_tasks",
        ]
        for table in expected_tables:
            assert table in tables, f"Table {table} was not created"


def test_video_model_schema(engine):
    """Test Video model schema and constraints."""
    Base.metadata.create_all(engine)

    # Check table structure
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(videos)"))
        columns = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in result
        }

        # Verify primary key
        assert columns["id"]["pk"] == 1

        # Verify required fields
        assert columns["channel_id"]["notnull"] == 1
        assert columns["title"]["notnull"] == 1

        # Verify optional fields
        assert columns["description"]["notnull"] == 0
        assert columns["view_count"]["notnull"] == 0


def test_transcript_model_schema(engine):
    """Test Transcript model schema and foreign key."""
    Base.metadata.create_all(engine)

    # Check table structure
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(transcripts)"))
        columns = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in result
        }

        # Verify primary key
        assert columns["video_id"]["pk"] == 1

        # Verify required fields
        assert columns["video_title"]["notnull"] == 1
        assert columns["video_url"]["notnull"] == 1
        assert columns["channel_name"]["notnull"] == 1
        assert columns["channel_url"]["notnull"] == 1

        # Check foreign key constraint
        result = conn.execute(text("PRAGMA foreign_key_list(transcripts)"))
        fks = list(result)
        assert len(fks) == 1
        assert fks[0][2] == "videos"  # references videos table


def test_knowledge_graph_model_schema(engine):
    """Test KnowledgeGraph model schema and indexes."""
    Base.metadata.create_all(engine)

    # Check table structure
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(knowledge_graphs)"))
        columns = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in result
        }

        # Verify primary key
        assert columns["id"]["pk"] == 1

        # Check indexes exist
        result = conn.execute(text("PRAGMA index_list(knowledge_graphs)"))
        indexes = [row[1] for row in result]
        assert "ix_knowledge_graphs_video_id" in indexes
        assert "ix_knowledge_graphs_channel_id" in indexes


def test_processing_task_model_schema(engine):
    """Test ProcessingTask model schema."""
    Base.metadata.create_all(engine)

    # Check table structure
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(processing_tasks)"))
        columns = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]} for row in result
        }

        # Verify primary key
        assert columns["id"]["pk"] == 1

        # Verify required fields
        assert columns["task_type"]["notnull"] == 1
        assert columns["target_id"]["notnull"] == 1


def test_video_model_crud(session):
    """Test Video model CRUD operations."""
    # Create
    video = Video(
        id="test_video_123",
        channel_id="test_channel_456",
        title="Test Video Title",
        description="Test video description",
        duration_seconds=300,
        view_count=1000,
    )
    session.add(video)
    session.commit()

    # Read
    retrieved_video = session.query(Video).filter_by(id="test_video_123").first()
    assert retrieved_video is not None
    assert retrieved_video.title == "Test Video Title"
    assert retrieved_video.channel_id == "test_channel_456"
    assert retrieved_video.duration_seconds == 300

    # Update
    retrieved_video.view_count = 2000
    session.commit()

    updated_video = session.query(Video).filter_by(id="test_video_123").first()
    assert updated_video.view_count == 2000

    # Delete
    session.delete(updated_video)
    session.commit()

    deleted_video = session.query(Video).filter_by(id="test_video_123").first()
    assert deleted_video is None


def test_transcript_model_crud(session):
    """Test Transcript model CRUD operations."""
    # First create a video for the foreign key
    video = Video(
        id="test_video_456",
        channel_id="test_channel_789",
        title="Test Video for Transcript",
    )
    session.add(video)
    session.commit()

    # Create transcript
    transcript = Transcript(
        video_id="test_video_456",
        video_title="Test Video for Transcript",
        video_url="https://youtube.com/watch?v=test_video_456",
        channel_name="Test Channel",
        channel_url="https://youtube.com/@testchannel",
        content="This is the transcript content",
        language="en",
        openai_tokens=100,
        gemini_tokens=95,
    )
    session.add(transcript)
    session.commit()

    # Read
    retrieved_transcript = (
        session.query(Transcript).filter_by(video_id="test_video_456").first()
    )
    assert retrieved_transcript is not None
    assert retrieved_transcript.content == "This is the transcript content"
    assert retrieved_transcript.language == "en"
    assert retrieved_transcript.processing_status == "pending"  # default value


def test_knowledge_graph_model_crud(session):
    """Test KnowledgeGraph model CRUD operations."""
    # Create knowledge graph
    kg = KnowledgeGraph(
        id="kg_test_123",
        video_id="test_video_789",
        video_title="Test Video for KG",
        channel_name="Test Channel",
        channel_id="test_channel_123",
        graph_type="video",
        entities_count=50,
        relationships_count=75,
        model_used="gpt-4",
        processing_time_seconds=120,
        storage_path="/path/to/kg.json",
    )
    session.add(kg)
    session.commit()

    # Read
    retrieved_kg = session.query(KnowledgeGraph).filter_by(id="kg_test_123").first()
    assert retrieved_kg is not None
    assert retrieved_kg.graph_type == "video"
    assert retrieved_kg.entities_count == 50
    assert retrieved_kg.relationships_count == 75


def test_processing_task_model_crud(session):
    """Test ProcessingTask model CRUD operations."""
    # Create processing task
    task = ProcessingTask(
        id="task_test_123",
        task_type="transcript",
        target_id="test_video_999",
        status="in_progress",
        progress_percent=50,
        error_message="No errors so far",
    )
    session.add(task)
    session.commit()

    # Read
    retrieved_task = session.query(ProcessingTask).filter_by(id="task_test_123").first()
    assert retrieved_task is not None
    assert retrieved_task.task_type == "transcript"
    assert retrieved_task.status == "in_progress"
    assert retrieved_task.progress_percent == 50


def test_model_repr_methods():
    """Test __repr__ methods for all models."""
    video = Video(id="test_123", title="Test Video Title")
    assert "test_123" in repr(video)
    assert "Test Video" in repr(video)

    transcript = Transcript(video_id="test_456", processing_status="done")
    assert "test_456" in repr(transcript)
    assert "done" in repr(transcript)

    kg = KnowledgeGraph(id="kg_789", graph_type="channel")
    assert "kg_789" in repr(kg)
    assert "channel" in repr(kg)

    task = ProcessingTask(id="task_012", task_type="kg_video", status="pending")
    assert "task_012" in repr(task)
    assert "kg_video" in repr(task)
    assert "pending" in repr(task)


def test_datetime_defaults(session):
    """Test that datetime defaults work correctly."""
    # Create a video and check created_at/updated_at
    video = Video(
        id="datetime_test_123", channel_id="test_channel", title="DateTime Test Video"
    )
    session.add(video)
    session.commit()

    # Verify datetime fields are set
    assert video.created_at is not None
    assert video.updated_at is not None
    assert isinstance(video.created_at, datetime)
    assert isinstance(video.updated_at, datetime)


def test_foreign_key_constraint(session):
    """Test foreign key constraint between transcripts and videos."""
    # Create transcript without corresponding video should work (nullable FK)
    transcript = Transcript(
        video_id="nonexistent_video",
        video_title="Test Title",
        video_url="https://test.com",
        channel_name="Test Channel",
        channel_url="https://test-channel.com",
    )
    session.add(transcript)
    # This should not raise an error since FKs are not enforced by default in SQLite
    session.commit()

    assert transcript.video_id == "nonexistent_video"
