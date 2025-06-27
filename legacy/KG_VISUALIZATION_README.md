# Knowledge Graph Visualization System

This system provides an interactive visualization of knowledge graphs extracted from YouTube transcripts using sigma.js and React.

## 🏗️ Architecture

The system consists of three main components:

1. **Backend Knowledge Graph Builder** (`tubeatlas/kg_builder_langchain.py`)
   - Extracts knowledge graphs from transcript data using LangChain and OpenAI
   - Saves results as JSON files in the `data/` directory

2. **FastAPI Server** (`api_server.py`)
   - Serves knowledge graph data via REST API
   - Provides endpoints for different datasets
   - Handles CORS for frontend requests

3. **React Frontend** (`frontend/`)
   - Interactive sigma.js visualization
   - Search, filtering, and layout controls
   - Real-time node/edge information display

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Node.js Environment**
   ```bash
   cd frontend
   npm install
   ```

3. **OpenAI API Key**
   ```bash
   # Create .env file in project root
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Step 1: Generate Knowledge Graph

Generate a knowledge graph from your Bryan Johnson data:

```bash
# Run the generation script
python run_kg_generation.py
```

This will:
- Check for the Bryan Johnson database (`data/bryanjohnson.db`)
- Calculate estimated API costs
- Generate knowledge graph using OpenAI GPT-4
- Save results to `data/complete_kg_langchain_bryanjohnson.json`

### Step 2: Start the API Server

```bash
# Start the FastAPI server
python api_server.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Step 3: Start the Frontend

```bash
# In a new terminal
cd frontend
npm run dev
```

The frontend will be available at: http://localhost:5173

### Step 4: View the Knowledge Graph

1. Navigate to http://localhost:5173/knowledge-graph
2. Select "Bryan Johnson" from the dataset dropdown
3. If no backend data is available, toggle "Use mock data" for testing

## 📊 Features

### Interactive Visualization
- **Pan & Zoom**: Navigate the graph with mouse
- **Node Selection**: Click nodes to see details
- **Search**: Find specific entities by name
- **Force Layout**: Run physics simulation to organize nodes
- **Color Coding**: Different colors for entity types (people, research, health, etc.)

### Controls
- **Search Bar**: Find and highlight specific nodes
- **Zoom Controls**: Zoom in/out, reset view
- **Layout Button**: Run force-directed layout algorithm
- **Dataset Selector**: Switch between different knowledge graphs

### Information Panels
- **Node Stats**: See distribution of entity types
- **Node Details**: View information about selected nodes
- **Edge Details**: See relationship information

## 🔧 API Endpoints

### Main Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/datasets` - List available datasets
- `GET /api/kg/{dataset}` - Get knowledge graph data
- `GET /api/kg/{dataset}/stats` - Get graph statistics

### Example Usage
```bash
# Get Bryan Johnson knowledge graph
curl http://localhost:8000/api/kg/bryanjohnson

# Get graph statistics
curl http://localhost:8000/api/kg/bryanjohnson/stats

# List available datasets
curl http://localhost:8000/api/datasets
```

## 📁 Data Format

Knowledge graphs are stored as JSON files with this structure:

```json
{
  "triples": [
    {
      "subject": "Bryan Johnson",
      "predicate": "founded",
      "object": "Kernel"
    },
    {
      "subject": "Bryan Johnson",
      "predicate": "practices",
      "object": "Biohacking"
    }
  ]
}
```

The frontend transforms this into sigma.js format with nodes and edges.

## 🎨 Customization

### Adding New Datasets

1. **Generate Knowledge Graph**
   ```python
   # Modify run_kg_generation.py or use kg_builder_langchain.py directly
   config = GraphBuilderConfig(
       db_path="data/your_dataset.db",
       output_path="data/complete_kg_langchain_your_dataset.json"
   )
   ```

2. **Update API Server**
   ```python
   # Add to DATASET_FILES in api_server.py
   DATASET_FILES = {
       "bryanjohnson": "complete_kg_langchain_bryanjohnson.json",
       "your_dataset": "complete_kg_langchain_your_dataset.json"
   }
   ```

3. **Update Frontend**
   ```javascript
   // Add to datasets array in KnowledgeGraphPage.jsx
   const datasets = [
     { id: 'bryanjohnson', name: 'Bryan Johnson', description: 'Biohacking content' },
     { id: 'your_dataset', name: 'Your Dataset', description: 'Your description' }
   ];
   ```

### Styling and Colors

Node colors are defined in `KGService.getNodeColor()`:

```javascript
// Customize colors based on entity types
if (label.includes('research')) {
  return '#3b82f6'; // Blue for research
} else if (label.includes('person')) {
  return '#ef4444'; // Red for people
}
// Add more categories...
```

## 🐛 Troubleshooting

### Common Issues

1. **"Backend not available" error**
   - Make sure the API server is running on port 8000
   - Check CORS settings in `api_server.py`
   - Try using mock data for testing

2. **"No knowledge graph data found"**
   - Ensure the JSON file exists in the `data/` directory
   - Run `python run_kg_generation.py` to generate data
   - Check file permissions

3. **Sigma.js rendering issues**
   - Clear browser cache
   - Check browser console for JavaScript errors
   - Ensure graph data has valid nodes and edges

4. **OpenAI API errors**
   - Verify your API key in `.env` file
   - Check your OpenAI account balance
   - Ensure rate limits aren't exceeded

### Debug Mode

Enable debug logging by setting environment variables:

```bash
# For API server
export LOG_LEVEL=debug
python api_server.py

# For knowledge graph generation
export PYTHONPATH=.
python -m tubeatlas.kg_builder_langchain
```

## 📈 Performance Tips

1. **Batch Size**: Adjust batch size in knowledge graph generation
   ```python
   # Smaller batches for memory efficiency
   graph_data = builder.build_complete_knowledge_graph(batch_size=5)
   ```

2. **Async Processing**: Use async version for better performance
   ```python
   # Async version with concurrency control
   graph_data = await builder.build_complete_knowledge_graph_async(
       batch_size=10,
       max_concurrent=3
   )
   ```

3. **Token Limits**: Monitor token usage to avoid API limits
   ```python
   # Check costs before processing
   total_tokens, estimated_cost = builder._calc_api_costs()
   ```

## 🤝 Contributing

To contribute to the knowledge graph visualization system:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test with both mock and real data
5. Submit a pull request

## 📄 License

This project is part of TubeAtlas and follows the same license terms.
