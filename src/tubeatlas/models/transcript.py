"""Transcript models."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from tubeatlas.config.database import Base


class Transcript(Base):
    """Transcript model."""

    __tablename__ = "transcripts"

    video_id = Column(String, ForeignKey("videos.id"), primary_key=True)
    video_title = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    channel_name = Column(String, nullable=False)
    channel_url = Column(String, nullable=False)
    content = Column(Text)
    language = Column(String, default="en")
    openai_tokens = Column(Integer)
    gemini_tokens = Column(Integer)
    processing_status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    def __repr__(self) -> str:
        """String representation of Transcript model."""
        return f"<Transcript(video_id='{self.video_id}', status='{self.processing_status}')>"
