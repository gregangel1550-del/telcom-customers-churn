# Use an official Python base image (slim = smaller size, no unnecessary tools)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If only your code changes (not requirements), Docker skips reinstalling packages
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model artifacts
COPY models/ ./models/

# Copy the API source code
COPY api/ ./api/

# Set working directory to the api folder so imports work correctly
WORKDIR /app/api

# Expose the port the API runs on
EXPOSE 8000

# Start the server
# host=0.0.0.0 means accept connections from outside the container (required for cloud hosting)
# workers=2 means two parallel processes handling requests simultaneously
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]