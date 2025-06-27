"""Video data access repository."""

from typing import List, Optional

from sqlalchemy import select

from ..models.video import Video
from .base_repository import BaseRepository


class VideoRepository(BaseRepository[Video]):
    """Repository for video data access."""

    async def create(self, entity: Video) -> Video:
        """Create a new video."""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: str) -> Optional[Video]:
        """Get video by ID."""
        result = await self.session.execute(select(Video).where(Video.id == entity_id))
        return result.scalar_one_or_none()

    async def update(self, entity: Video) -> Video:
        """Update an existing video."""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete video by ID."""
        video = await self.get_by_id(entity_id)
        if video:
            await self.session.delete(video)
            await self.session.commit()
            return True
        return False

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[Video]:
        """List all videos with pagination."""
        result = await self.session.execute(select(Video).offset(offset).limit(limit))
        return list(result.scalars().all())

    async def get_by_channel_id(self, channel_id: str) -> List[Video]:
        """Get all videos for a specific channel."""
        result = await self.session.execute(
            select(Video).where(Video.channel_id == channel_id)
        )
        return list(result.scalars().all())
