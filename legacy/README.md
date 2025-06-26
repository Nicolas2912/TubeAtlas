# TubeAtlas
Gather deep insights from YouTube channels

## Project Structure

- data/                  # SQLite database files per channel
- tubeatlas/             # Core library modules
  - transcripts.py       # Download transcripts & metadata manager
  - query.py             # SQLite query utility
  - kg_builder.py        # Knowledge graph builder classes
  - legacy/              # Legacy code versions
    - transcripts_old.py # Old transcript downloader & cleaner
- tubeatlas/__init__.py   # Package initializer
- tubeatlas/legacy/__init__.py
- knowledge_graph.html   # Example KG visualization output
- requirements.txt       # Python dependencies (to be created)
- README.md              # Project overview and usage

## Usage

Run the core workflows as modules:

1. Download transcripts & metadata:
   ```bash
   python -m tubeatlas.transcripts
   ```

2. Query the SQLite database:
   ```bash
   python -m tubeatlas.query
   ```

3. Build and visualize the knowledge graph (as a library):
   ```python
   from tubeatlas.kg_builder import KnowledgeGraphBuilderGoogle
   builder = KnowledgeGraphBuilderGoogle(
       model_name="models/gemini-1.5-flash-001",
       yt_channel="AndrejKarpathy"
   )
   builder.build_kg()
   ```

For more details on configuration and environment variables, refer to the inline documentation in each module.
