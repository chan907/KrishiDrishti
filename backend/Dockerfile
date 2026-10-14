# Build from the REPO ROOT:  docker build -f backend/Dockerfile -t krishidrishti-api .
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_DIR=/models STORAGE_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
# inference deps only - no training libraries (matplotlib, pandas, sklearn, mlflow ...)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY core core
RUN pip install ./core
COPY backend/app backend/app
WORKDIR /srv/backend
# Model is MOUNTED at /models (see docker-compose.yml) so the image stays small and models are swappable.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
