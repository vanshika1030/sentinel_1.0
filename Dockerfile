FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY sentinel/ ./sentinel/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy remaining files (competitors.yaml, etc.)
COPY . .

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "sentinel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
