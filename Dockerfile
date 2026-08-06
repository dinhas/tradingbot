FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FRED_API_KEY=b30c77a87d01838e84f6760b17d5070b

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose a dummy port to satisfy Back4App requirements
EXPOSE 8080

CMD ["python", "main.py"]