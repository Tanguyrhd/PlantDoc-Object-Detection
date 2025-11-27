# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only the necessary files
COPY api/ ./api/
COPY results/ ./results/

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
CMD ["sh", "-c", "uvicorn api.fast:app --host 0.0.0.0 --port ${PORT:-8000}"]
