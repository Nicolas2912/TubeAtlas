"""Transcript models."""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative.api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()


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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        """String representation of Transcript model."""
        return f"<Transcript(video_id='{self.video_id}', status='{self.processing_status}')>"
