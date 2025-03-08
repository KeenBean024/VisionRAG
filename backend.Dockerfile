# Backend Dockerfile
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    gcc \
    g++ \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
