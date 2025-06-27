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

    async def exists(self, entity_id: str) -> bool:
        """Return True if a video with the given ID exists."""
        result = await self.session.execute(
            select(Video.id).where(Video.id == entity_id)
        )
        return result.scalar_one_or_none() is not None

    async def upsert(self, data: dict) -> Video:
        """Insert or update a video record based on the primary key (id)."""
        video_id = data.get("id")
        if video_id is None:
            raise ValueError("Video data must include 'id' field for upsert")

        # Only keep keys that correspond to model columns
        valid_keys = {c.key for c in Video.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        existing = await self.get_by_id(video_id)
        if existing:
            # Update existing fields
            for key, value in filtered_data.items():
                setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        # Create new video entity
        new_video = Video(**filtered_data)
        self.session.add(new_video)
        await self.session.commit()
        await self.session.refresh(new_video)
        return new_video
