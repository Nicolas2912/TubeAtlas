# =============================================================================
# Builder Stage - Install dependencies
# =============================================================================
FROM python:3.12-slim AS builder

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies needed for building (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry with specific version for reproducibility
RUN pip install poetry==1.8.3

# Set work directory
WORKDIR /app

# Copy Poetry configuration files first for better layer caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry: Don't create virtual env since we're in a container
RUN poetry config virtualenvs.create false

# Install dependencies directly with Poetry (production only)
RUN poetry install --only=main --no-interaction --no-ansi --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src/ ./src/

# =============================================================================
# Runtime Stage - Minimal runtime environment
# =============================================================================
FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:$PATH" \
    PYTHONPATH="/app/src"

# Install runtime dependencies only (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser -d /home/appuser -s /bin/bash -c "App User" appuser && \
    mkdir -p /home/appuser/.local && \
    chown -R appuser:appuser /home/appuser

# Copy installed packages from builder stage
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin

# Set work directory and copy source code
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/src ./src

# Copy additional necessary files
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser .env.template ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with shorter intervals for faster startup detection
HEALTHCHECK --interval=15s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "tubeatlas.main:app", "--host", "0.0.0.0", "--port", "8000"]
