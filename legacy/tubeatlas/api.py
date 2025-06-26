"""
TubeAtlas API

This module provides a FastAPI-based REST API for TubeAtlas functionality,
including YouTube transcript downloading and knowledge graph building.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv

from transcripts import YouTubeTranscriptManager, validate_channel_url
from kg_builder_langchain import KnowledgeGraphBuilder, GraphBuilderConfig

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TubeAtlas API",
    description="API for downloading YouTube transcripts and building knowledge graphs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ChannelDownloadRequest(BaseModel):
    channel_url: HttpUrl = Field(..., description="YouTube channel URL")
    storage_type: str = Field("sqlite", description="Storage type ('file' or 'sqlite')")
    region_code: str = Field("US", description="Region code for category mapping")

class DownloadResponse(BaseModel):
    task_id: str
    channel_url: str
    status: str
    message: str
    timestamp: datetime

class KnowledgeGraphRequest(BaseModel):
    channel_name: str = Field(..., description="Channel name to build KG for")
    model_name: str = Field("gpt-4.1-mini", description="LLM model to use")
    temperature: float = Field(0.0, description="Model temperature")
    strict_mode: bool = Field(True, description="Strict mode for graph building")
    additional_instructions: Optional[str] = Field(None, description="Additional instructions for graph building")

class KnowledgeGraphResponse(BaseModel):
    task_id: str
    channel_name: str
    status: str
    message: str
    timestamp: datetime
    graph_path: Optional[str] = None

# In-memory task storage (replace with proper database in production)
download_tasks: Dict[str, Dict[str, Any]] = {}
kg_tasks: Dict[str, Dict[str, Any]] = {}

def generate_task_id() -> str:
    """Generate a unique task ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

@app.post("/download/channel", response_model=DownloadResponse)
async def download_channel_transcripts(
    request: ChannelDownloadRequest,
    background_tasks: BackgroundTasks
) -> DownloadResponse:
    """
    Start downloading transcripts from a YouTube channel.
    
    Args:
        request: Channel download request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        DownloadResponse with task information
    """
    try:
        # Validate channel URL
        if not validate_channel_url(str(request.channel_url)):
            raise HTTPException(status_code=400, detail="Invalid YouTube channel URL")
        
        # Generate task ID
        task_id = generate_task_id()
        
        # Create task entry
        task = {
            "task_id": task_id,
            "channel_url": str(request.channel_url),
            "status": "pending",
            "message": "Task created",
            "timestamp": datetime.now(),
            "storage_type": request.storage_type,
            "region_code": request.region_code
        }
        download_tasks[task_id] = task
        
        # Add background task
        background_tasks.add_task(
            process_channel_download,
            task_id,
            str(request.channel_url),
            request.storage_type,
            request.region_code
        )
        
        return DownloadResponse(**task)
        
    except Exception as e:
        logger.error(f"Error creating download task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kg/build", response_model=KnowledgeGraphResponse)
async def build_knowledge_graph(
    request: KnowledgeGraphRequest,
    background_tasks: BackgroundTasks
) -> KnowledgeGraphResponse:
    """
    Start building a knowledge graph from channel transcripts.
    
    Args:
        request: Knowledge graph building parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        KnowledgeGraphResponse with task information
    """
    try:
        # Generate task ID
        task_id = generate_task_id()
        
        # Create task entry
        task = {
            "task_id": task_id,
            "channel_name": request.channel_name,
            "status": "pending",
            "message": "Task created",
            "timestamp": datetime.now(),
            "model_name": request.model_name,
            "temperature": request.temperature,
            "strict_mode": request.strict_mode,
            "additional_instructions": request.additional_instructions
        }
        kg_tasks[task_id] = task
        
        # Add background task
        background_tasks.add_task(
            process_knowledge_graph,
            task_id,
            request.channel_name,
            request.model_name,
            request.temperature,
            request.strict_mode,
            request.additional_instructions
        )
        
        return KnowledgeGraphResponse(**task)
        
    except Exception as e:
        logger.error(f"Error creating KG task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/download/{task_id}", response_model=DownloadResponse)
async def get_download_status(task_id: str = FastAPIPath(..., description="Task ID")) -> DownloadResponse:
    """Get the status of a download task."""
    if task_id not in download_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return DownloadResponse(**download_tasks[task_id])

@app.get("/tasks/kg/{task_id}", response_model=KnowledgeGraphResponse)
async def get_kg_status(task_id: str = FastAPIPath(..., description="Task ID")) -> KnowledgeGraphResponse:
    """Get the status of a knowledge graph building task."""
    if task_id not in kg_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return KnowledgeGraphResponse(**kg_tasks[task_id])

async def process_channel_download(
    task_id: str,
    channel_url: str,
    storage_type: str,
    region_code: str
) -> None:
    """Background task for processing channel downloads."""
    try:
        # Update task status
        download_tasks[task_id]["status"] = "processing"
        download_tasks[task_id]["message"] = "Downloading transcripts..."
        
        # Initialize YouTube service
        from googleapiclient.discovery import build
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            raise ValueError("YouTube API key not found")
            
        youtube_service = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Set up storage paths
        if storage_type == 'sqlite':
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True, parents=True)
            channel_name = channel_url.split('/')[-1]
            db_path = data_dir / f"{channel_name}.db"
        else:
            db_path = None
            
        # Initialize transcript manager
        transcript_manager = YouTubeTranscriptManager(
            youtube_service=youtube_service,
            storage_type=storage_type,
            db_path=str(db_path) if db_path else None,
            region_code=region_code
        )
        
        # Download transcripts
        successful = transcript_manager.download_channel_data(channel_url)
        
        # Update task status
        download_tasks[task_id]["status"] = "completed"
        download_tasks[task_id]["message"] = f"Successfully processed {successful} videos"
        
    except Exception as e:
        logger.error(f"Error processing download task {task_id}: {e}")
        download_tasks[task_id]["status"] = "failed"
        download_tasks[task_id]["message"] = f"Error: {str(e)}"

async def process_knowledge_graph(
    task_id: str,
    channel_name: str,
    model_name: str,
    temperature: float,
    strict_mode: bool,
    additional_instructions: Optional[str]
) -> None:
    """Background task for building knowledge graphs."""
    try:
        # Update task status
        kg_tasks[task_id]["status"] = "processing"
        kg_tasks[task_id]["message"] = "Building knowledge graph..."
        
        # Configure and initialize graph builder
        config = GraphBuilderConfig(
            model_name=model_name,
            temperature=temperature,
            strict_mode=strict_mode,
            db_path=f"data/{channel_name}.db",
            output_path=f"data/{channel_name}_kg.json",
            additional_instructions=additional_instructions or ""
        )
        
        builder = KnowledgeGraphBuilder(config)
        
        # Load text and build graph
        text = builder.load_text_from_db()
        graph_data = builder.extract_knowledge_graph(text)
        builder.save_knowledge_graph(graph_data)
        
        # Update task status
        kg_tasks[task_id]["status"] = "completed"
        kg_tasks[task_id]["message"] = "Knowledge graph built successfully"
        kg_tasks[task_id]["graph_path"] = config.output_path
        
    except Exception as e:
        logger.error(f"Error processing KG task {task_id}: {e}")
        kg_tasks[task_id]["status"] = "failed"
        kg_tasks[task_id]["message"] = f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 