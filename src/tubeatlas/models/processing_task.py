"""Processing task models."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from tubeatlas.config.database import Base


class ProcessingTask(Base):
    """Processing task model."""

    __tablename__ = "processing_tasks"

    id = Column(String, primary_key=True)
    task_type = Column(String, nullable=False)  # 'transcript', 'kg_video', 'kg_channel'
    target_id = Column(String, nullable=False)  # video_id or channel_id
    status = Column(String, default="pending")
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)  # Text type for longer error messages
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    def __repr__(self) -> str:
        """String representation of ProcessingTask model."""
        return f"<ProcessingTask(id='{self.id}', type='{self.task_type}', status='{self.status}')>"
