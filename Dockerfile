# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., git for some pip packages if needed, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose any necessary ports (not needed for this client-side app, but good practice)
# EXPOSE 8080

# Command to run the application
CMD ["python", "LiveExecution/main.py"]
