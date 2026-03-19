FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY processing_pipeline/ /app/processing_pipeline/
COPY web_service/ /app/web_service/

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/pipeline_uploads /tmp/pipeline_outputs \
    && chown appuser:appuser /tmp/pipeline_uploads /tmp/pipeline_outputs

ENV PYTHONPATH=/app

EXPOSE 5000

USER appuser

CMD ["python", "web_service/app.py"]
