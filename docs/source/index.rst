.. TubeAtlas documentation master file

TubeAtlas Documentation
=======================

Advanced YouTube Knowledge Graph & RAG Platform for transcript analysis and insight generation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   rag/index
   services/index
   models/index

Overview
--------

TubeAtlas is a comprehensive platform for analyzing YouTube content through advanced RAG (Retrieval-Augmented Generation) techniques and knowledge graph construction.

Key Features
------------

* **Multi-Strategy Chunking**: Fixed-length and semantic chunking strategies
* **Vector Storage**: FAISS-based vector store for efficient similarity search
* **Embedding Support**: OpenAI embeddings with extensible interface
* **Modular Pipeline**: Configurable ingestion and retrieval pipelines
* **Knowledge Graphs**: Advanced knowledge graph construction from transcripts
* **RESTful API**: FastAPI-based REST API for all operations

Quick Start
-----------

.. code-block:: python

   from tubeatlas.rag.pipeline import IngestPipeline, RetrievalPipeline

   # Create ingestion pipeline
   pipeline = IngestPipeline(
       chunker_name="semantic",
       embedder_name="openai",
       vector_store_name="faiss"
   )

   # Ingest documents
   documents = [{"id": "doc1", "text": "Your content here"}]
   result = await pipeline.ingest_documents(documents)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
