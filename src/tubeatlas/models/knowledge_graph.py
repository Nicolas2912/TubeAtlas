"""Knowledge graph data models."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative.api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()


class KnowledgeGraph(Base):
    """Knowledge graph model."""

    __tablename__ = "knowledge_graphs"

    id = Column(String, primary_key=True)
    video_id = Column(String)
    video_title = Column(String)
    channel_name = Column(String)
    channel_id = Column(String)
    graph_type = Column(String)  # 'video' or 'channel'
    entities_count = Column(Integer)
    relationships_count = Column(Integer)
    model_used = Column(String)
    processing_time_seconds = Column(Integer)
    storage_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_video_id", "video_id"),
        Index("idx_channel_id", "channel_id"),
    )

    def __repr__(self) -> str:
        """String representation of KnowledgeGraph model."""
        return f"<KnowledgeGraph(id='{self.id}', type='{self.graph_type}')>"


class ProcessingTask(Base):
    """Processing task model."""

    __tablename__ = "processing_tasks"

    id = Column(String, primary_key=True)
    task_type = Column(String, nullable=False)  # 'transcript', 'kg_video', 'kg_channel'
    target_id = Column(String, nullable=False)  # video_id or channel_id
    status = Column(String, default="pending")
    progress_percent = Column(Integer, default=0)
    error_message = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    def __repr__(self) -> str:
        """String representation of ProcessingTask model."""
        return f"<ProcessingTask(id='{self.id}', type='{self.task_type}', status='{self.status}')>"
