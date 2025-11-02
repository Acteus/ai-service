# Flask AI Service Dockerfile for Fly.io
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl needed for healthcheck)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --timeout=900 --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs

# Set environment variables for Fly.io
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_PORT=8080
ENV FLASK_DEBUG=false

# Expose port for Fly.io (8080 is default)
EXPOSE 8080

# Health check for Fly.io
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
