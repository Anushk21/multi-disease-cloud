# Cloud Run container for Multi‑Disease ML app
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create models dir if not exists
RUN mkdir -p models

# Default: run via gunicorn on port 8080 for Cloud Run
ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-b", ":8080", "app:app"]
