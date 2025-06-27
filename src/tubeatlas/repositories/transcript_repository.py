"""Transcript data access repository."""

from typing import List, Optional

from sqlalchemy import select

from ..models.transcript import Transcript
from .base_repository import BaseRepository


class TranscriptRepository(BaseRepository[Transcript]):
    """Repository for transcript data access."""

    async def create(self, entity: Transcript) -> Transcript:  # type: ignore[override]
        """Create a new transcript record.

        Using `expire_on_commit=False` (configured in the session factory) allows
        us to safely refresh the instance after commit to retrieve any database
        defaults (e.g. timestamps).
        """
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: str) -> Optional[Transcript]:  # type: ignore[override]
        """Return a transcript by the associated video ID (primary key)."""
        result = await self.session.execute(
            select(Transcript).where(Transcript.video_id == entity_id)
        )
        return result.scalar_one_or_none()

    async def update(self, entity: Transcript) -> Transcript:  # type: ignore[override]
        """Persist updates made to the Transcript entity instance."""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: str) -> bool:  # type: ignore[override]
        """Delete a transcript by video ID. Returns True if deleted."""
        transcript = await self.get_by_id(entity_id)
        if transcript:
            await self.session.delete(transcript)
            await self.session.commit()
            return True
        return False

    async def list_all(  # type: ignore[override]
        self, limit: int = 100, offset: int = 0
    ) -> List[Transcript]:
        """Return a paginated list of transcripts."""
        result = await self.session.execute(
            select(Transcript).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    # Additional convenience look-ups -------------------------------------------------

    async def list_by_channel(self, channel_name: str) -> List[Transcript]:
        """Return all transcripts for a given channel name."""
        result = await self.session.execute(
            select(Transcript).where(Transcript.channel_name == channel_name)
        )
        return list(result.scalars().all())

    async def exists(self, entity_id: str) -> bool:  # type: ignore[override]
        """Return True if a transcript for the given video ID exists."""
        result = await self.session.execute(
            select(Transcript.video_id).where(Transcript.video_id == entity_id)
        )
        return result.scalar_one_or_none() is not None

    async def upsert(self, data: dict) -> Transcript:  # type: ignore[override]
        """Insert or update a transcript record based on video_id."""
        video_id = data.get("video_id")
        if video_id is None:
            raise ValueError("Transcript data must include 'video_id' field for upsert")

        valid_keys = {c.key for c in Transcript.__table__.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        existing = await self.get_by_id(video_id)
        if existing:
            for key, value in filtered_data.items():
                setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        new_transcript = Transcript(**filtered_data)  # type: ignore[arg-type]
        self.session.add(new_transcript)
        await self.session.commit()
        await self.session.refresh(new_transcript)
        return new_transcript
