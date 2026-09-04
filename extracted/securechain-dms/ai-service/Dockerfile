# SecureChain DMS — AI Service (SIH26190)
FROM python:3.10-slim

LABEL maintainer="SecureChain DMS - SIH26190"
LABEL description="FastAPI + OCR + spaCy document classification microservice"

# ---------------------------------------------------------------------------
# System dependencies
#   libgl1 / libglib2.0-0  -> opencv / paddleocr C bindings
#   poppler-utils          -> pdf2image (pdftoppm / pdfinfo)
#   libgomp1               -> paddlepaddle OpenMP runtime
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

# Install Python dependencies first (leverages Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Download spaCy English model.
# Done as a separate RUN step so it is cached independently of requirements.
# en_core_web_sm is small (~12 MB) and required for named-entity extraction.
# ---------------------------------------------------------------------------
RUN python -m spacy download en_core_web_sm --no-cache-dir

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