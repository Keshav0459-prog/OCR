# SecureChain DMS - AI Service (OCR + Classification)
FROM python:3.10-slim

LABEL maintainer="SecureChain DMS - SIH26190"
LABEL description="FastAPI + PaddleOCR document classification microservice"

# ---------------------------------------------------------------------------
# CRITICAL SYSTEM DEPENDENCIES
# libgl1 / libglib2.0-0 -> required by opencv/paddleocr's C bindings
# poppler-utils                  -> required by pdf2image (pdftoppm/pdfinfo)
# libgomp1                       -> required by paddlepaddle's OpenMP runtime
# Without these the container will crash at import-time with
# "ImportError: libGL.so.1: cannot open shared object file"
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        libgomp1 \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . .

# Non-root user for defense-in-depth.
RUN useradd --create-home --shell /bin/bash aiuser \
    && chown -R aiuser:aiuser /app
USER aiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]