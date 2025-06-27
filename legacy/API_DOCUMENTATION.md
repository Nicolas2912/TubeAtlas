# TubeAtlas API Documentation

## Overview

TubeAtlas contains two distinct API servers that serve different purposes in the YouTube transcript analysis and knowledge graph pipeline:

1. **`api_server.py`** - Knowledge Graph Serving API (Read-only)
2. **`tubeatlas/api.py`** - Full-featured Processing API (Read/Write operations)

## Architecture Relationship

```
┌─────────────────────┐    ┌─────────────────────┐
│   tubeatlas/api.py  │    │   api_server.py     │
│  (Processing API)   │    │  (Serving API)      │
│                     │    │                     │
│ • Download videos   │    │ • Serve KG data     │
│ • Build KG          │────▶│ • Read-only access  │
│ • Background tasks  │    │ • Statistics        │
└─────────────────────┘    └─────────────────────┘
           │                          │
           ▼                          ▼
    ┌─────────────────────────────────────┐
    │           data/ directory           │
    │  • {channel}.db (SQLite files)      │
    │  • complete_kg_*.json (KG files)    │
    └─────────────────────────────────────┘
```

---

## 1. api_server.py - Knowledge Graph Serving API

### Purpose
A lightweight FastAPI server specifically designed to **serve pre-generated knowledge graph data** to frontend applications. This is a read-only API focused on data consumption.

### Key Features

#### Core Endpoints
- **`GET /`** - API information and available endpoints
- **`GET /health`** - Health check endpoint
- **`GET /api/datasets`** - List all available knowledge graph datasets
- **`GET /api/kg/{dataset}`** - Retrieve complete knowledge graph for a dataset
- **`GET /api/kg/{dataset}/stats`** - Get statistics about a knowledge graph

#### Dataset Management
```python
DATASET_FILES = {
    "bryanjohnson": "complete_kg_langchain_bryanjohnson.json",
    "andreykarpathy": "complete_kg_langchain_AndrejKarpathy.json",
    "example": "example_kg_langchain.json"
}
```

#### Data Format
Returns knowledge graphs with enhanced metadata:
```json
{
  "triples": [...],
  "metadata": {
    "dataset": "bryanjohnson",
    "filename": "complete_kg_langchain_bryanjohnson.json",
    "file_size_bytes": 1234567,
    "last_modified": 1699123456.789,
    "triple_count": 450
  }
}
```

#### Statistics Features
- **Entity Analysis**: Unique subjects, objects, and total entities
- **Predicate Analysis**: Most common relationships and their frequencies
- **Graph Metrics**: Total triples, unique predicates, distribution analysis

### CORS Configuration
Configured for frontend development environments:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)
- `http://localhost:4173` (Vite preview)

---

## 2. tubeatlas/api.py - Full Processing API

### Purpose
A comprehensive FastAPI server that handles the **entire TubeAtlas workflow** from YouTube transcript downloading to knowledge graph generation. This is a full-featured API with background task processing.

### Key Features

#### Core Workflow Endpoints

##### 1. Transcript Download
- **`POST /download/channel`** - Start downloading transcripts from a YouTube channel
- **`GET /tasks/download/{task_id}`** - Check download task status

**Request Format:**
```json
{
  "channel_url": "https://youtube.com/@channel",
  "storage_type": "sqlite",
  "region_code": "US"
}
```

##### 2. Knowledge Graph Building
- **`POST /kg/build`** - Start building a knowledge graph from transcripts
- **`GET /tasks/kg/{task_id}`** - Check knowledge graph building status

**Request Format:**
```json
{
  "channel_name": "bryanjohnson",
  "model_name": "gpt-4.1-mini",
  "temperature": 0.0,
  "strict_mode": true,
  "additional_instructions": "Focus on health and longevity topics"
}
```

#### Background Task Processing
Uses FastAPI's background tasks for long-running operations:
- **Asynchronous Processing**: Tasks run in background without blocking API
- **Task Tracking**: In-memory storage for task status (suitable for development)
- **Status Updates**: Real-time status tracking (pending → processing → completed/failed)

#### Integration Components

##### YouTube Integration
```python
# Uses Google YouTube API v3
youtube_service = build('youtube', 'v3', developerKey=youtube_api_key)
transcript_manager = YouTubeTranscriptManager(
    youtube_service=youtube_service,
    storage_type=storage_type,
    db_path=db_path,
    region_code=region_code
)
```

##### Knowledge Graph Integration
```python
# Uses LangChain-based KG builder
config = GraphBuilderConfig(
    model_name=model_name,
    temperature=temperature,
    strict_mode=strict_mode,
    db_path=f"data/{channel_name}.db",
    output_path=f"data/{channel_name}_kg.json"
)
```

---

## Data Flow Between APIs

### Sequential Workflow
1. **Data Collection** (`tubeatlas/api.py`)
   ```
   POST /download/channel → Downloads transcripts → Saves to {channel}.db
   ```

2. **Knowledge Graph Generation** (`tubeatlas/api.py`)
   ```
   POST /kg/build → Reads from {channel}.db → Generates complete_kg_{channel}.json
   ```

3. **Data Serving** (`api_server.py`)
   ```
   GET /api/kg/{dataset} → Reads complete_kg_{channel}.json → Serves to frontend
   ```

### File Dependencies
```
tubeatlas/api.py creates:
├── data/{channel}.db (SQLite transcripts)
└── data/complete_kg_{channel}.json (Knowledge graph)

api_server.py reads:
└── data/complete_kg_{channel}.json (Serves to frontend)
```

---

## Use Cases and Deployment Scenarios

### Development Workflow
1. **Research Phase**: Use `tubeatlas/api.py` to download and analyze new channels
2. **Analysis Phase**: Generate knowledge graphs for specific topics
3. **Presentation Phase**: Use `api_server.py` to serve data to visualization frontend

### Production Scenarios

#### Option 1: Unified Deployment
- Deploy both APIs together
- `tubeatlas/api.py` for admin/research operations
- `api_server.py` for public data access

#### Option 2: Separated Deployment
- **Processing Server**: `tubeatlas/api.py` for internal operations
- **Public API**: `api_server.py` for frontend and external access
- Better security and resource isolation

### Port Configuration
- **`api_server.py`**: Port 8000 (default)
- **`tubeatlas/api.py`**: Port 8000 (default, but can be configured differently)

---

## Technical Considerations

### Performance
- **`api_server.py`**: Optimized for fast read operations, minimal dependencies
- **`tubeatlas/api.py`**: Handles heavy processing with background tasks

### Scalability
- **In-memory task storage**: Suitable for development, should be replaced with Redis/Database for production
- **File-based KG storage**: Simple but could be moved to database for complex queries

### Error Handling
Both APIs include comprehensive error handling:
- **Validation errors**: Invalid URLs, missing files
- **Processing errors**: API failures, parsing issues
- **HTTP status codes**: Proper REST API error responses

### Security
- **CORS configuration**: Different policies for different use cases
- **API key management**: Environment variable based configuration
- **Input validation**: Pydantic models for request validation

---

## Frontend Integration

### Knowledge Graph Visualization
The frontend (`frontend/src/`) integrates primarily with `api_server.py`:

```javascript
// Example frontend integration
const kgData = await fetch('http://localhost:8000/api/kg/bryanjohnson');
const stats = await fetch('http://localhost:8000/api/kg/bryanjohnson/stats');
```

### Admin Interface
Could integrate with `tubeatlas/api.py` for:
- Triggering new downloads
- Monitoring processing status
- Managing datasets

---

## Summary

| Aspect | api_server.py | tubeatlas/api.py |
|--------|---------------|------------------|
| **Purpose** | Serve KG data | Full workflow processing |
| **Operations** | Read-only | Read/Write + Processing |
| **Complexity** | Lightweight | Full-featured |
| **Frontend Integration** | Primary visualization API | Admin/management API |
| **Data Flow** | Consumer | Producer |
| **Deployment** | Production-ready serving | Research and processing |

Both APIs work together to provide a complete pipeline from YouTube content to interactive knowledge graph visualizations, with clear separation of concerns between data processing and data serving.
