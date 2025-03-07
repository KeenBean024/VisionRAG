# Dockerfile
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    gcc \
    g++

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Backend service
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Frontend service
EXPOSE 8501
CMD ["streamlit", "run", "frontend.py"]
