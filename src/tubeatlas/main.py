"""Main FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .api.middleware import DatabaseHealthMiddleware
from .api.routes import chat, health, knowledge_graphs, transcripts
from .config.database import async_engine, create_all
from .config.settings import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles database initialization and connection testing on startup,
    and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting TubeAtlas application...")

    try:
        # Create all database tables
        logger.info("Initializing database tables...")
        await create_all()
        logger.info("Database tables initialized successfully")

        # Test database connectivity
        logger.info("Testing database connectivity...")
        async with async_engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            if result.fetchone()[0] == 1:
                logger.info("Database connectivity test passed")
            else:
                logger.error("Database connectivity test failed")
                raise RuntimeError("Database connectivity test failed")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down TubeAtlas application...")
    await async_engine.dispose()
    logger.info("Database connections closed")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced YouTube Knowledge Graph & RAG Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add database health middleware (before CORS to catch all requests)
app.add_middleware(DatabaseHealthMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router)  # Health checks first
app.include_router(transcripts.router)
app.include_router(knowledge_graphs.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name} v{settings.app_version}",
        "status": "running",
        "docs": "/docs",
        "health_checks": {
            "basic": "/health",
            "database": "/health/db",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "tubeatlas.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
    )
