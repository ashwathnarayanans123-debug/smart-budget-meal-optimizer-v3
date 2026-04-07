# Use Python 3.11 Bullseye image
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port (HF preference)
EXPOSE 7860

# Start the environment server
# Using uvicorn to serve the FastAPI app from server/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
