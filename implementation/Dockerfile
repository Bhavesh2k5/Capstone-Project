FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/results /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

ENV CAPSTONE_OUTPUT_ROOT=/app/results

EXPOSE 8501

CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
