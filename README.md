# TubeAtlas

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: flake8](https://img.shields.io/badge/linter-flake8-lightgrey.svg)](https://flake8.pycqa.org/en/latest/)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](http://mypy-lang.org/)
[![Framework: FastAPI](https://img.shields.io/badge/framework-FastAPI-05998b.svg)](https://fastapi.tiangolo.com/)
[![CI/CD: GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-blue.svg)](.github/workflows/main.yml)

**An advanced platform for large-scale YouTube content analysis, transforming unstructured video transcripts into structured knowledge graphs and enabling powerful, retrieval-augmented querying.**

---

## 📖 Overview

TubeAtlas is engineered to unlock the vast knowledge repository within YouTube. It automates the entire pipeline from content ingestion to insight generation, allowing researchers, creators, and analysts to deeply understand and interact with video content at scale.

The core of TubeAtlas is a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline that intelligently processes transcripts, builds comprehensive knowledge graphs, and facilitates natural language conversations with the content of entire YouTube channels, even those containing millions of tokens.

## 🏗️ Architecture

The system is designed with a clean, scalable architecture that separates concerns into distinct layers. It leverages asynchronous processing to handle long-running, resource-intensive tasks efficiently.

```mermaid
graph TD
    subgraph "Input Layer"
        A[YouTube Channels/Videos] --> B(YouTube Service)
    end

    subgraph "Processing Layer (Async)"
        B --> C{Celery + Redis}
        C --> D[Transcript Service]
        D --> E[RAG Pipeline]
    end

    subgraph "RAG Pipeline"
        direction LR
        E_Chunk[Chunking<br/>(Semantic, Fixed)]
        E_Embed[Embedding<br/>(OpenAI)]
        E_KG[Graph Extraction<br/>(LLM)]
        E_Store[Vector Store<br/>(FAISS)]
        E_Chunk --> E_Embed --> E_Store
        E_Chunk --> E_KG
    end

    subgraph "Data Persistence"
        E --> F[SQLite Database<br/>(Metadata, KGs)]
        E --> G[FAISS Vector Store<br/>(Embeddings)]
    end

    subgraph "API & Query Layer"
        H[User] --> I(FastAPI)
        I --> J[Chat & KG Services]
        J --> E
        J --> F
        J --> G
        J --> H
    end

    linkStyle 8 stroke-width:2px,stroke:green,stroke-dasharray: 5 5;
```

---

## ✨ Core Features

-   **Automated Content Ingestion**: Seamlessly fetches video metadata and transcripts for individual videos or entire YouTube channels.
-   **Advanced RAG Pipeline**: Implements a multi-faceted retrieval strategy combining semantic search (via embeddings) and knowledge graph traversal to provide accurate, context-aware answers.
-   **Intelligent Chunking**: Employs sophisticated text chunking strategies, including fixed-size and semantic chunking, to prepare transcripts for efficient processing by Large Language Models (LLMs).
-   **High-Performance Vector Storage**: Utilizes **FAISS** (Facebook AI Similarity Search) for efficient storage and retrieval of text embeddings, forming the backbone of the semantic search capability.
-   **Knowledge Graph Generation**: Leverages LLMs to extract structured entities and their relationships from unstructured text, building a comprehensive knowledge graph of the content.
-   **Asynchronous Task Processing**: Uses a powerful combination of **Celery** and **Redis** to manage long-running tasks like transcript downloading and knowledge graph creation in the background, ensuring the API remains responsive.
-   **Robust & Modern Backend**: Built with **FastAPI** for high-performance, asynchronous API endpoints with automatic OpenAPI and Swagger documentation.
-   **Clean Architecture**: Follows a repository pattern to separate business logic from data access, enhancing maintainability and testability. The data layer is powered by **SQLAlchemy ORM**.
-   **Comprehensive Tooling**: Fully containerized with **Docker** and **Docker Compose** for easy setup and deployment. Includes a suite of code quality tools (`black`, `flake8`, `mypy`) and a CI/CD pipeline orchestrated with **GitHub Actions**.

---

## 🛠️ Technology Stack

| Category              | Technology                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/)                                                                    |
| **Database**          | [SQLite](https://www.sqlite.org/index.html) with [SQLAlchemy](https://www.sqlalchemy.org/) (Async)          |
| **Async Tasks**       | [Celery](https://docs.celeryq.dev/en/stable/) with [Redis](https://redis.io/) Broker                          |
| **LLM Integration**   | [LangChain](https://www.langchain.com/), [OpenAI API](https://openai.com/blog/openai-api)                    |
| **Vector Store**      | [FAISS](https://faiss.ai/) (Facebook AI Similarity Search)                                                    |
| **Dependency Mgmt**   | [Poetry](https://python-poetry.org/)                                                                        |
| **Containerization**  | [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)                       |
| **Testing**           | [pytest](https://docs.pytest.org/en/7.4.x/)                                                                 |
| **Code Quality**      | [black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/en/latest/), [mypy](http://mypy-lang.org/) |
| **CI/CD**             | [GitHub Actions](https://github.com/features/actions)                                                       |

---

## 🚀 Getting Started

### Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
-   [Poetry](https://python-poetry.org/docs/#installation)
-   An OpenAI API key

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/TubeAtlas.git
    cd TubeAtlas
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the project root by copying the example:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and add your `OPENAI_API_KEY`.

3.  **Install dependencies:**
    Use Poetry to install the project dependencies.
    ```bash
    poetry install
    ```

4.  **Launch the application:**
    Use Docker Compose to build and run all the services (API, Celery workers, Redis).
    ```bash
    docker-compose up --build
    ```

The API will be available at `http://localhost:8000`.

---

## 🔌 API Usage

Once the application is running, you can access the interactive API documentation:

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Example: Process a video

You can submit a YouTube video for transcript processing using a `curl` command:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/transcripts/video' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=your_video_id"
  }'
```

This will trigger an asynchronous background task to download and process the transcript.

---

## 🔬 Development & Testing

This project is equipped with a full suite of development tools to ensure code quality and correctness.

-   **Run tests:**
    Execute the test suite using `pytest`.
    ```bash
    poetry run pytest
    ```

-   **Check code formatting and linting:**
    ```bash
    poetry run black . --check
    poetry run flake8 src tests
    ```

-   **Run static type checking:**
    ```bash
    poetry run mypy src
    ```

---

## 🗺️ Roadmap

This project is under active development. The future roadmap includes:

-   [ ] **Advanced RAG Strategies**: Implementing hierarchical summarization and hybrid retrieval methods as outlined in the PRD.
-   [ ] **Knowledge Graph Enhancements**: Full implementation of KG-based retrieval, graph merging, and interactive visualizations.
-   [ ] **Chat Interface**: Building out the conversational chat endpoints for querying channels and knowledge graphs.
-   [ ] **Scalability Improvements**: Migrating to PostgreSQL for enhanced database performance and implementing more robust caching layers.
-   [ ] **Frontend**: Building a modern and responsive frontend for the application.


## 📄 License

TBD
