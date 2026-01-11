# CCBA RAG System - Production Dockerfile
# Multi-stage build for smaller image size

# Stage 1: Base with Python
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install ".[dev]"

# Stage 3: Production
FROM dependencies as production

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY entrypoint/ ./entrypoint/
COPY server.py app.py ./

# Create data directory
RUN mkdir -p data/documents data/uploads

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/status || exit 1

# Default command: start API server
CMD ["python", "server.py"]
