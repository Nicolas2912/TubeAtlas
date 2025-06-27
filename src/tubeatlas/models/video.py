"""Video metadata models."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative.api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()


class Video(Base):
    """Video metadata model."""

    __tablename__ = "videos"

    id = Column(String, primary_key=True)
    channel_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    publish_date = Column(DateTime)
    duration_seconds = Column(Integer)
    view_count = Column(Integer)
    like_count = Column(Integer)
    dislike_count = Column(Integer)
    comment_count = Column(Integer)
    category_id = Column(String)
    category_name = Column(String)
    tags = Column(JSON)  # JSON array
    thumbnail_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        """String representation of Video model."""
        return f"<Video(id='{self.id}', title='{self.title[:50]}...')>"
