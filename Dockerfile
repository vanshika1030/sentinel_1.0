FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install .

# Copy source code
COPY . .

# Install the package itself
RUN pip install -e .

# Default command: run API
CMD ["uvicorn", "sentinel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
