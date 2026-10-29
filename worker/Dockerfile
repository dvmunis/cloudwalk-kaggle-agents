FROM python:3.11-slim

# LightGBM and XGBoost need the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV DATA_DIR=/workspace/data RUNS_DIR=/workspace/runs PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
