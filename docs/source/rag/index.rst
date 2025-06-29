RAG Module
==========

The RAG (Retrieval-Augmented Generation) module provides a comprehensive framework for document processing, embedding, and retrieval.

.. toctree::
   :maxdepth: 2

   chunking
   embedding
   vector_store
   pipeline
   registry

Overview
--------

The RAG module consists of several key components:

* **Chunking**: Text splitting strategies (fixed-length and semantic)
* **Embedding**: Text-to-vector conversion using various models
* **Vector Store**: Efficient storage and retrieval of embeddings
* **Pipeline**: End-to-end processing workflows
* **Registry**: Component registration and factory system

Architecture
------------

The RAG system follows a modular, interface-based design that allows for easy extension and customization:

.. code-block:: python

   # Example usage
   from tubeatlas.rag.registry import get_registry

   registry = get_registry()
   chunker = registry.create_chunker("semantic")
   embedder = registry.create_embedder("openai")
   vector_store = registry.create_vector_store("faiss")

Components
----------

Chunking Strategies
^^^^^^^^^^^^^^^^^^^

.. automodule:: tubeatlas.rag.chunking
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Models
^^^^^^^^^^^^^^^^

.. automodule:: tubeatlas.rag.embedding
   :members:
   :undoc-members:
   :show-inheritance:

Vector Stores
^^^^^^^^^^^^^

.. automodule:: tubeatlas.rag.vector_store
   :members:
   :undoc-members:
   :show-inheritance:

Processing Pipelines
^^^^^^^^^^^^^^^^^^^^

.. automodule:: tubeatlas.rag.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Component Registry
^^^^^^^^^^^^^^^^^^

.. automodule:: tubeatlas.rag.registry
   :members:
   :undoc-members:
   :show-inheritance:
