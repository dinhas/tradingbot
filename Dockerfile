# Use a lightweight official Python 3.11 image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (Crucial for real-time trade logs)
ENV PYTHONUNBUFFERED=1

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set workspace directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code into the container
COPY . .

# Launch the trading bot
CMD ["python", "main.py"]