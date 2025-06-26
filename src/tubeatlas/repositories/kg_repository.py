"""Knowledge graph storage repository."""

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.knowledge_graph import KnowledgeGraph
from .base_repository import BaseRepository


class KnowledgeGraphRepository(BaseRepository[KnowledgeGraph]):
    """Repository for knowledge graph storage."""
    
    async def create(self, entity: KnowledgeGraph) -> KnowledgeGraph:
        """Create a new knowledge graph."""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[KnowledgeGraph]:
        """Get knowledge graph by ID."""
        result = await self.session.execute(
            select(KnowledgeGraph).where(KnowledgeGraph.id == entity_id)
        )
        return result.scalar_one_or_none()
    
    async def update(self, entity: KnowledgeGraph) -> KnowledgeGraph:
        """Update an existing knowledge graph."""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete knowledge graph by ID."""
        kg = await self.get_by_id(entity_id)
        if kg:
            await self.session.delete(kg)
            await self.session.commit()
            return True
        return False
    
    async def list_all(self, limit: int = 100, offset: int = 0) -> List[KnowledgeGraph]:
        """List all knowledge graphs with pagination."""
        result = await self.session.execute(
            select(KnowledgeGraph).offset(offset).limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_video_id(self, video_id: str) -> List[KnowledgeGraph]:
        """Get knowledge graphs for a specific video."""
        result = await self.session.execute(
            select(KnowledgeGraph).where(KnowledgeGraph.video_id == video_id)
        )
        return list(result.scalars().all())
    
    async def get_by_channel_id(self, channel_id: str) -> List[KnowledgeGraph]:
        """Get knowledge graphs for a specific channel."""
        result = await self.session.execute(
            select(KnowledgeGraph).where(KnowledgeGraph.channel_id == channel_id)
        )
        return list(result.scalars().all()) 