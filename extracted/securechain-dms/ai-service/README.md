# SecureChain DMS — AI Service
### Member 6: AI + DevOps + Testing | SIH26190

FastAPI microservice that powers the AI layer of SecureChain DMS:
- **OCR extraction** — RapidOCR (primary) → EasyOCR → PaddleOCR fallback chain
- **Bilingual support** — English and Hindi/Devanagari with auto-detection
- **Legal document classification** — FIR, Chargesheet, Forensic Report, Witness Statement, Case Diary, Internal Progress Note
- **AI sensitivity suggestion** — LOW / MEDIUM / HIGH tier with automatic escalation
- **Named entity extraction** — Persons, organisations, dates, locations via spaCy NER
- **Blockchain quorum recommendation** — 1-of-1 / 2-of-3 / 3-of-5 per sensitivity tier

---

## Quick Start (Local)

```bash
# Clone and enter the ai-service directory
cd ai-service

# One-command bootstrap (creates venv, installs deps + spaCy model)
bash setup.sh

# Activate the virtualenv (Linux/macOS)
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# Start the server
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 for the browser upload UI, or http://localhost:8000/docs for the Swagger UI.

---

## Run with Docker (Recommended)

```bash
# From the repo root (securechain-dms/)
cp .env.example .env          # edit with real secrets if needed

# Start only the AI service + PostgreSQL
docker compose up -d --build

# Start the full stack (AI + backend + frontend)
docker compose --profile full up -d --build
```

The AI service will be available at http://localhost:8000.

> **Note:** The Dockerfile bakes `en_core_web_sm` into the image during build so spaCy NER is available from the first request with no cold-start download.

---

## API Reference

### `POST /api/v1/ai/analyze-document`

Upload a PDF, JPEG, or PNG document for OCR + classification.

**Form fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | PDF / JPEG / PNG, max 25 MB |
| `language` | string | `auto` (default), `en`, or `hi` |

**Response (200 OK):**

```jsonc
{
  "document_type": "Chargesheet",
  "detected_sections": ["BNS Section 103", "CRPC Section 173"],
  "sensitivity_tier": "HIGH",
  "recommended_quorum": { "required": 3, "pool_size": 5 },
  "confidence_score": 0.87,
  "extracted_summary": "Chargesheet filed under BNS Section 103...",
  "extracted_text": "...",
  "sensitivity_keywords_found": ["ballistics", "forensic"],
  "raw_text_char_count": 4821,
  "ocr_language_used": "English",
  "extracted_entities": {
    "persons": ["Rajesh Kumar"],
    "organizations": ["Delhi Police"],
    "dates": ["10 April 2024"],
    "locations": ["Delhi"]
  }
}
```

### `GET /health`
Returns engine readiness status.

### `GET /api/v1/status`
Returns whether models are still loading in the background.

---

## Running Tests

```bash
# Full test suite with coverage (mirrors GitLab CI exactly)
pytest tests/ -v \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=xml:coverage.xml \
    --junitxml=report.xml

# Quick run
pytest tests/ -v
```

### Test Coverage

| Test Class | What it covers |
|------------|---------------|
| `TestAIClassification` | Classification, sensitivity tiers, quorum, error handling |
| `TestLanguageMismatchDetection` | Devanagari OCR retry, gibberish score heuristic |
| `TestMakerCheckerRule` | Self-approval block (HTTP 403 contract) |
| `TestInstantFlaggingRule` | Edit → PENDING QUORUM without mutating hash |
| `TestQuorumThreshold` | 3-of-5 quorum state machine |
| `TestSpacyEntityExtraction` | NER entities, deduplication, graceful degradation, end-to-end |

### spaCy Model

The `conftest.py` auto-downloads `en_core_web_sm` if it is missing before the test session starts — no manual step required.

---

## Sensitivity & Quorum Rules

| Document Type | Base Tier | Quorum |
|---------------|-----------|--------|
| FIR | HIGH | 3-of-5 |
| Chargesheet | HIGH | 3-of-5 |
| Forensic Report | HIGH | 3-of-5 |
| Witness Statement | MEDIUM | 2-of-3 |
| Case Diary | MEDIUM | 2-of-3 |
| Internal Progress Note | LOW | 1-of-1 |

> **Escalation rule:** If 2+ high-risk keywords (`ballistics`, `dna`, `terrorism`, etc.) are detected and the base tier is not already HIGH, the tier is escalated one level.

---

## CI/CD Pipeline (GitLab)

The `.gitlab-ci.yml` at repo root defines four stages:

```
lint → test → security_scan → build_and_push
```

| Stage | Jobs |
|-------|------|
| `lint` | `flake8` (hard fail), `black` (advisory) |
| `test` | `pytest` with Cobertura + JUnit report upload |
| `security_scan` | GitLab SAST + Secret Detection |
| `build_and_push` | Docker build + push to GitLab registry (main branch / tags only) |

---

## Environment Variables

See [`.env.example`](../.env.example) at repo root for all variables.

---

## Member 6 Deliverables Checklist

- [x] OCR extraction — RapidOCR → EasyOCR → PaddleOCR chain
- [x] Bilingual OCR — English + Hindi/Devanagari with auto-detection & retry
- [x] Document classification — keyword-weighted scoring for all 6 document types
- [x] AI sensitivity suggestion — base tier + escalation logic
- [x] spaCy NER — persons, organisations, dates, locations (runnable ✅)
- [x] Docker setup — `Dockerfile` with spaCy model baked in
- [x] `docker-compose.yml` — full stack (postgres + ai-service + backend + frontend)
- [x] Integration tests — 5 test classes, 20+ test cases
- [x] `conftest.py` — auto model download + OCR mock fixtures
- [x] GitLab CI/CD — lint → test → security scan → build & push
- [x] `setup.sh` — one-command local dev bootstrap
- [x] `.env.example` — environment variable documentation
- [x] `README.md` — this document
