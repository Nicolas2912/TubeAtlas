"""Knowledge graph data models."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Index, Integer, String

from tubeatlas.config.database import Base


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
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Indexes as specified in PRD
    __table_args__ = (
        Index("ix_knowledge_graphs_video_id", "video_id"),
        Index("ix_knowledge_graphs_channel_id", "channel_id"),
    )

    def __repr__(self) -> str:
        """String representation of KnowledgeGraph model."""
        return f"<KnowledgeGraph(id='{self.id}', type='{self.graph_type}')>"
