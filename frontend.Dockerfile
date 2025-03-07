# Frontend Dockerfile
FROM python:3.10-slim
WORKDIR /app

# COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install streamlit

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
