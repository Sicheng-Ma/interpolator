# Backend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.21.0 \
    torch>=2.0.0 \
    scikit-learn>=1.0.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.22.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.0.0

# Copy application code
COPY fivedreg/ ./fivedreg/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "fivedreg.main:app", "--host", "0.0.0.0", "--port", "8000"]