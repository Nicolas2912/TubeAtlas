"""Processing task repository layer."""

from typing import List, Optional

from sqlalchemy import select

from ..models.processing_task import ProcessingTask
from .base_repository import BaseRepository


class ProcessingTaskRepository(BaseRepository[ProcessingTask]):
    """Repository providing CRUD operations for `ProcessingTask` entities."""

    async def create(self, entity: ProcessingTask) -> ProcessingTask:  # type: ignore[override]
        """Insert a new `ProcessingTask` into the database."""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def get_by_id(self, entity_id: str) -> Optional[ProcessingTask]:  # type: ignore[override]
        """Retrieve a `ProcessingTask` by primary key."""
        result = await self.session.execute(
            select(ProcessingTask).where(ProcessingTask.id == entity_id)
        )
        return result.scalar_one_or_none()

    async def update(self, entity: ProcessingTask) -> ProcessingTask:  # type: ignore[override]
        """Persist changes made to a `ProcessingTask` instance."""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity_id: str) -> bool:  # type: ignore[override]
        """Delete a `ProcessingTask` by ID. Returns True if deleted."""
        task = await self.get_by_id(entity_id)
        if task:
            await self.session.delete(task)
            await self.session.commit()
            return True
        return False

    async def list_all(  # type: ignore[override]
        self, limit: int = 100, offset: int = 0
    ) -> List[ProcessingTask]:
        """Return a paginated list of processing tasks."""
        result = await self.session.execute(
            select(ProcessingTask).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    # Convenience helpers -------------------------------------------------------------

    async def list_by_status(self, status: str) -> List[ProcessingTask]:
        """Return all tasks matching a specific status."""
        result = await self.session.execute(
            select(ProcessingTask).where(ProcessingTask.status == status)
        )
        return list(result.scalars().all())

    async def list_pending(self) -> List[ProcessingTask]:
        """Return tasks with status `pending`."""
        return await self.list_by_status("pending")
