"""Celery application configuration."""

from celery import Celery

from .settings import settings

# Create Celery app
celery_app = Celery(
    "tubeatlas",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["tubeatlas.tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Task routing (optional - for when we have multiple queues)
celery_app.conf.task_routes = {
    "tubeatlas.tasks.process_transcript": {"queue": "transcripts"},
    "tubeatlas.tasks.generate_knowledge_graph": {"queue": "kg_generation"},
    "tubeatlas.tasks.process_video": {"queue": "video_processing"},
}

# Auto-discover tasks
celery_app.autodiscover_tasks()


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery functionality."""
    print(f"Request: {self.request!r}")
    return "Debug task executed successfully"
