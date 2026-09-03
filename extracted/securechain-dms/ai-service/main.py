"""
SecureChain DMS - AI Service (SIH26190)
-----------------------------------------
Standalone FastAPI microservice that performs OCR extraction (PaddleOCR)
and automatic legal-document classification for the Ministry of Home
Affairs blockchain-anchored Document Management System.

Endpoint:
    POST /api/v1/ai/analyze-document

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("securechain.ai-service")

# --------------------------------------------------------------------------
# Global OCR engines (initialised once on startup to avoid cold starts)
# --------------------------------------------------------------------------
# Indian legal/administrative documents (FIRs, chargesheets, case diaries)
# are very commonly bilingual Hindi (Devanagari) + English, sometimes with
# regional-language variants. Running everything through a single English
# model silently mis-recognises Devanagari glyphs as Latin look-alikes,
# producing readable-looking but meaningless text (e.g. "3Tcc19T a 3y").
# We therefore keep one PaddleOCR instance per supported script and pick
# the right one per request (with an automatic-detection fallback).
SUPPORTED_OCR_LANGUAGES = {
    "en": "en",  # English / Latin script
    "hi": "hi",  # Hindi / Devanagari script (also covers Marathi, Nepali text using Devanagari)
}
OCR_ENGINES: Dict[str, Optional[object]] = {lang: None for lang in SUPPORTED_OCR_LANGUAGES}

# Optional spaCy layer for entity extraction (names, dates, orgs) on the
# ENGLISH portion of correctly-OCR'd text. This is additive — it never
# runs on Hindi text and never substitutes for correct OCR language
# selection, which is the actual fix for garbled output.
SPACY_NLP = None
SPACY_AVAILABLE = False

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB safety cap
MAX_PDF_PAGES = 15  # cap OCR pass for very large scanned dossiers

# Heuristic threshold: fraction of "mixed" tokens (letters + digits jammed
# together, e.g. "3Tcc19T") above which we suspect the wrong OCR language
# model was used and retry with the alternate script.
GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD = 0.20
_MIXED_TOKEN_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{3,}$")
_DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise PaddleOCR (en + hi) and, if available, spaCy exactly once."""
    global OCR_ENGINES, SPACY_NLP, SPACY_AVAILABLE

    logger.info("Initialising PaddleOCR engines (en, hi)... this happens once.")
    try:
        from paddleocr import PaddleOCR

        for lang_key, paddle_lang in SUPPORTED_OCR_LANGUAGES.items():
            logger.info("Loading PaddleOCR model for language='%s'", paddle_lang)
            OCR_ENGINES[lang_key] = PaddleOCR(
                use_angle_cls=True,
                lang=paddle_lang,
                show_log=False,
            )
        logger.info("PaddleOCR engines initialised successfully.")
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.exception("Failed to initialise PaddleOCR engines: %s", exc)

    logger.info("Attempting to load optional spaCy English model for entity extraction...")
    try:
        import spacy

        SPACY_NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("spaCy 'en_core_web_sm' loaded — entity extraction enabled.")
    except Exception as exc:
        # Not fatal: entity extraction is an optional enhancement, OCR/
        # classification work fine without it.
        logger.warning(
            "spaCy model unavailable (%s). Continuing without entity extraction. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm",
            exc,
        )
        SPACY_NLP = None
        SPACY_AVAILABLE = False

    yield
    logger.info("Shutting down AI service.")


app = FastAPI(
    title="SecureChain DMS - AI Service",
    description="OCR extraction & legal document classification microservice",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------
# Domain constants: classification, sensitivity & quorum rules
# --------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "FIR",
    "Chargesheet",
    "Forensic Report",
    "Witness Statement",
    "Case Diary",
    "Internal Progress Note",
]

# Keyword banks used for scoring each candidate document type against the
# extracted OCR text. Weighted so that strong / unambiguous phrases score
# higher than generic ones.
DOCUMENT_TYPE_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
    "FIR": [
        ("first information report", 3.0),
        ("fir no", 2.5),
        ("f.i.r", 2.5),
        ("fir number", 2.0),
        ("police station", 1.0),
        ("under section", 0.5),
        ("complainant", 0.5),
    ],
    "Chargesheet": [
        ("chargesheet", 3.0),
        ("charge sheet", 3.0),
        ("final report under section 173", 3.0),
        ("section 173", 2.0),
        ("charge-sheet", 3.0),
        ("investigating officer", 0.5),
    ],
    "Forensic Report": [
        ("forensic science laboratory", 3.0),
        ("forensic report", 3.0),
        ("fsl report", 2.5),
        ("ballistic", 2.0),
        ("ballistics", 2.0),
        ("post-mortem", 2.0),
        ("postmortem", 2.0),
        ("dna profiling", 2.5),
        ("dna analysis", 2.0),
        ("viscera report", 2.0),
        ("forensic", 1.0),
    ],
    "Witness Statement": [
        ("statement of witness", 3.0),
        ("witness statement", 3.0),
        ("section 161", 2.0),
        ("deposition", 1.5),
        ("recorded statement", 1.0),
        ("i state as follows", 1.0),
    ],
    "Case Diary": [
        ("case diary", 3.0),
        ("daily diary", 2.5),
        ("cd entry", 2.0),
        ("station diary", 1.5),
        ("general diary", 1.5),
    ],
    "Internal Progress Note": [
        ("progress note", 2.5),
        ("internal note", 2.5),
        ("office note", 2.0),
        ("dispatch slip", 2.0),
        ("dispatch", 1.0),
        ("inter-office memo", 1.5),
        ("for internal circulation", 1.5),
    ],
}

# Base sensitivity tier assigned strictly by document type.
DOCUMENT_TYPE_SENSITIVITY: Dict[str, str] = {
    "FIR": "HIGH",
    "Chargesheet": "HIGH",
    "Forensic Report": "HIGH",
    "Witness Statement": "MEDIUM",
    "Case Diary": "MEDIUM",
    "Internal Progress Note": "LOW",
}

# Sensitive-content keyword bank used purely for detection/escalation signal.
SENSITIVITY_KEYWORDS = [
    "post-mortem",
    "postmortem",
    "ballistics",
    "ballistic",
    "confidential",
    "dna",
    "forensic",
    "top secret",
    "classified",
]

# Quorum matrix — strict, per the SecureChain DMS governance policy.
QUORUM_MATRIX: Dict[str, Dict[str, int]] = {
    "LOW": {"required": 1, "pool_size": 1},
    "MEDIUM": {"required": 2, "pool_size": 3},
    "HIGH": {"required": 3, "pool_size": 5},
}

TIER_ORDER = ["LOW", "MEDIUM", "HIGH"]

# Legal section reference patterns: "IPC Section 302", "Section 302 IPC",
# "BNS Section 103", "Sec. 302 IPC", "U/S 302 IPC", etc.
LEGAL_ACTS = r"(IPC|BNS|BNSS|CrPC|CPC|Evidence Act|POCSO|NDPS)"
SECTION_PATTERNS = [
    re.compile(
        rf"\b{LEGAL_ACTS}\s+(?:Section|Sec\.?|S\.)\s+(\d+[A-Za-z]*(?:\(\w+\))?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:Section|Sec\.?|S\.|U/S|u/s)\s*\.?\s*(\d+[A-Za-z]*(?:\(\w+\))?)\s+(?:of\s+)?{LEGAL_ACTS}\b",
        re.IGNORECASE,
    ),
]


# --------------------------------------------------------------------------
# Pydantic response models
# --------------------------------------------------------------------------

class QuorumInfo(BaseModel):
    required: int = Field(..., description="Minimum number of approvals needed")
    pool_size: int = Field(..., description="Total size of the approver pool")


class DocumentAnalysisResponse(BaseModel):
    document_type: str = Field(..., description="Detected legal document category")
    detected_sections: List[str] = Field(
        default_factory=list, description="Legal sections referenced in the document"
    )
    sensitivity_tier: str = Field(..., description="LOW | MEDIUM | HIGH")
    recommended_quorum: QuorumInfo
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the classification, 0-1"
    )
    extracted_summary: str = Field(..., description="Short extractive summary of OCR text")
    extracted_text: str = Field(..., description="Full text extracted by OCR")
    sensitivity_keywords_found: List[str] = Field(
        default_factory=list, description="Sensitive terms detected in the document"
    )
    raw_text_char_count: int = Field(..., description="Length of full OCR text extracted")
    ocr_language_used: str = Field(
        ..., description="OCR language model(s) actually used, e.g. 'en', 'hi', or 'en+hi'"
    )
    extracted_entities: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Named entities (persons, organizations, dates, locations) from English "
        "text, extracted via spaCy if available. Null when spaCy is not installed.",
    )


class HealthResponse(BaseModel):
    status: str
    ocr_engines_ready: Dict[str, bool]
    spacy_entity_extraction_ready: bool


# --------------------------------------------------------------------------
# OCR helpers
# --------------------------------------------------------------------------

def _pil_to_ndarray(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB numpy array as expected by PaddleOCR."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def _run_ocr_on_image(image_array: np.ndarray, lang: str) -> str:
    """Run PaddleOCR (for the given language) on a single image array."""
    engine = OCR_ENGINES.get(lang)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OCR engine for language '{lang}' is not initialised. Please retry shortly.",
        )
    result = engine.ocr(image_array, cls=True)
    lines: List[str] = []
    if not result:
        return ""
    # PaddleOCR returns a list per input image: [[ [box, (text, conf)], ... ]]
    for page_result in result:
        if not page_result:
            continue
        for line in page_result:
            try:
                text = line[1][0]
                if text:
                    lines.append(text)
            except (IndexError, TypeError):
                continue
    return "\n".join(lines)


def _gibberish_score(text: str) -> float:
    """
    Heuristic used to detect a wrong-language OCR pass. When PaddleOCR runs
    a Devanagari (or other non-Latin) page through the English model, it
    forces glyphs onto the nearest Latin letters/digits it knows, producing
    tokens that jam letters and digits together mid-word in a way normal
    English text almost never does (e.g. "3Tcc19T", "aTdT-HTdT"). A high
    ratio of such tokens is a strong signal the wrong model was used.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    mixed = sum(1 for t in tokens if _MIXED_TOKEN_PATTERN.match(t))
    return mixed / len(tokens)


def _devanagari_ratio(text: str) -> float:
    """Return the fraction of non-whitespace characters in Devanagari."""
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return 0.0
    return sum(bool(_DEVANAGARI_PATTERN.match(character)) for character in characters) / len(characters)


def _ocr_page(image_array: np.ndarray, requested_lang: str) -> Tuple[str, str]:
    """
    Run OCR on a single page, honouring the requested language. When
    requested_lang == "auto", try English first; if the result looks like
    garbled cross-script noise, automatically retry with the Hindi
    (Devanagari) model and keep whichever result is cleaner.

    Returns (text, language_used).
    """
    if requested_lang in ("en", "hi"):
        return _run_ocr_on_image(image_array, requested_lang), requested_lang

    # auto mode
    en_text = _run_ocr_on_image(image_array, "en")
    en_score = _gibberish_score(en_text)

    if en_score < GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD or OCR_ENGINES.get("hi") is None:
        if en_text or OCR_ENGINES.get("hi") is None:
            return en_text, "en"
        hi_text = _run_ocr_on_image(image_array, "hi")
        return hi_text, "hi"

    logger.info(
        "English OCR pass looked like garbled cross-script text (mixed-token ratio=%.2f). "
        "Retrying page with Hindi (Devanagari) model.",
        en_score,
    )
    hi_text = _run_ocr_on_image(image_array, "hi")
    hi_score = _gibberish_score(hi_text)

    if _devanagari_ratio(hi_text) > _devanagari_ratio(en_text):
        return hi_text, "hi"

    # Prefer whichever pass produced cleaner output; Hindi output is scored
    # against the same Latin-mixed-token heuristic, which is still a useful
    # (if imperfect) signal that recognisable Latin-script content (headers,
    # section numbers, English boilerplate) came through intact rather than
    # being mangled.
    if hi_score < en_score:
        return hi_text, "hi"
    return en_text, "en"


def extract_text_from_upload(
    file_bytes: bytes, kind: str, language: str = "auto"
) -> Tuple[str, str]:
    """
    Dispatch OCR extraction depending on whether the upload is a PDF or an
    image, and which OCR language was requested ('en', 'hi', or 'auto').

    Returns (full_text, language_used_summary).
    """
    all_text: List[str] = []
    languages_used: List[str] = []

    if kind == "pdf":
        try:
            pages = convert_from_bytes(file_bytes, dpi=200)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to render PDF pages: {exc}",
            )
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF contained no renderable pages.",
            )
        for page in pages[:MAX_PDF_PAGES]:
            arr = _pil_to_ndarray(page)
            text, lang_used = _ocr_page(arr, language)
            if text:
                all_text.append(text)
            languages_used.append(lang_used)
    else:  # image
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.load()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to read image file: {exc}",
            )
        arr = _pil_to_ndarray(image)
        text, lang_used = _ocr_page(arr, language)
        if text:
            all_text.append(text)
        languages_used.append(lang_used)

    # Summarise which language(s) actually got used across pages, e.g. "en"
    # or "hi" or "en+hi" for a mixed-language multi-page dossier.
    unique_langs = sorted(set(languages_used)) or ["en"]
    language_summary = "+".join(unique_langs)

    return "\n".join(all_text).strip(), language_summary


# --------------------------------------------------------------------------
# Optional spaCy entity extraction (English text only)
# --------------------------------------------------------------------------

def extract_entities_with_spacy(text: str) -> Dict[str, List[str]]:
    """
    Pull structured entities (person names, organisations, dates, locations)
    out of the English portions of correctly-OCR'd text using spaCy's NER.

    This is a pure enhancement layer: if spaCy isn't installed/loaded, it
    degrades gracefully to empty results rather than failing the request.
    It intentionally does NOT run on Hindi/Devanagari text — spaCy's small
    English model has no meaningful signal there, and running it would
    produce noise, not entities.
    """
    empty: Dict[str, List[str]] = {"persons": [], "organizations": [], "dates": [], "locations": []}
    if not SPACY_AVAILABLE or SPACY_NLP is None or not text:
        return empty

    # Cap input length for latency safety on very long OCR dumps.
    doc = SPACY_NLP(text[:20000])

    def _unique(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in seq:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    persons = _unique([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    orgs = _unique([ent.text for ent in doc.ents if ent.label_ == "ORG"])
    dates = _unique([ent.text for ent in doc.ents if ent.label_ == "DATE"])
    locations = _unique([ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")])

    return {
        "persons": persons,
        "organizations": orgs,
        "dates": dates,
        "locations": locations,
    }


# --------------------------------------------------------------------------
# Classification / NLP helpers
# --------------------------------------------------------------------------

def classify_document(text: str) -> Tuple[str, float]:
    """
    Score the OCR text against each document-type keyword bank and return
    the best-matching type along with a normalised confidence score.
    """
    lowered = text.lower()

    scores: Dict[str, float] = {doc_type: 0.0 for doc_type in DOCUMENT_TYPES}
    max_possible: Dict[str, float] = {
        doc_type: sum(weight for _, weight in kws)
        for doc_type, kws in DOCUMENT_TYPE_KEYWORDS.items()
    }

    for doc_type, keyword_weights in DOCUMENT_TYPE_KEYWORDS.items():
        for phrase, weight in keyword_weights:
            if phrase in lowered:
                scores[doc_type] += weight

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score <= 0:
        # No keyword signal at all — fall back to the lowest-risk default
        # and flag low confidence rather than guessing HIGH sensitivity.
        return "Internal Progress Note", 0.15

    normalised_confidence = best_score / max_possible[best_type] if max_possible[best_type] else 0.0
    # Clamp and keep a realistic floor/ceiling.
    confidence = max(0.35, min(0.99, round(normalised_confidence, 2)))
    return best_type, confidence


def detect_legal_sections(text: str) -> List[str]:
    """Extract unique legal section references, e.g. 'IPC Section 302'."""
    found: List[str] = []
    seen = set()

    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            # Normalise to "<ACT> Section <number>" regardless of pattern order
            if groups[0].upper() in {"IPC", "BNS", "BNSS", "CRPC", "CPC", "POCSO", "NDPS"} or groups[0].lower() == "evidence act":
                act, number = groups[0], groups[1]
            else:
                number, act = groups[0], groups[1]
            normalised = f"{act.upper()} Section {number}"
            key = normalised.lower()
            if key not in seen:
                seen.add(key)
                found.append(normalised)

    return found


def detect_sensitivity_keywords(text: str) -> List[str]:
    lowered = text.lower()
    found = [kw for kw in SENSITIVITY_KEYWORDS if kw in lowered]
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for kw in found:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def resolve_sensitivity_tier(document_type: str, sensitive_hits: List[str]) -> str:
    """
    Strict base rule: tier is determined by document_type per the governance
    policy. If two or more high-risk sensitivity keywords are present and the
    base tier is not already HIGH, escalate by one tier as a safety measure
    (defense-in-depth against mis-classification), never de-escalate.
    """
    base_tier = DOCUMENT_TYPE_SENSITIVITY.get(document_type, "MEDIUM")

    if len(sensitive_hits) >= 2 and base_tier != "HIGH":
        idx = TIER_ORDER.index(base_tier)
        base_tier = TIER_ORDER[min(idx + 1, len(TIER_ORDER) - 1)]

    return base_tier


def build_summary(text: str, max_sentences: int = 4, max_chars: int = 600) -> str:
    """Very lightweight extractive summary: first N sentences, capped by length."""
    if not text:
        return "No text could be extracted from the document."

    # Split on sentence-ish boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    summary = " ".join(sentences[:max_sentences])
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    if not summary:
        summary = text[:max_chars]
    return summary


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def upload_page() -> HTMLResponse:
    """Serve the browser upload page for manually testing OCR."""
    with open("index.html", encoding="utf-8") as page:
        return HTMLResponse(page.read())


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ocr_engines_ready={lang: (engine is not None) for lang, engine in OCR_ENGINES.items()},
        spacy_entity_extraction_ready=SPACY_AVAILABLE,
    )


@app.post(
    "/api/v1/ai/analyze-document",
    response_model=DocumentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["ai"],
)
async def analyze_document(
    file: UploadFile = File(...),
    language: str = Form(
        "auto",
        description="OCR language: 'auto' (default, detects & retries across scripts), 'en', or 'hi'.",
    ),
) -> DocumentAnalysisResponse:
    """
    Accepts a PDF/JPEG/PNG document, runs OCR, classifies the document type,
    detects legal sections & sensitivity keywords, and returns the
    recommended blockchain-approval quorum.

    `language` lets the caller pin the OCR script when known (e.g. "hi" for
    a Devanagari FIR) to skip the auto-detection retry pass and save time;
    left as "auto" it will detect and correct a wrong-language OCR pass
    automatically (see `_ocr_page`).
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES.keys())}"
            ),
        )

    if language not in ("auto", "en", "hi"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="language must be one of: 'auto', 'en', 'hi'.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    kind = ALLOWED_CONTENT_TYPES[file.content_type]

    logger.info(
        "Received document '%s' (%s, %d bytes, language=%s) for analysis",
        file.filename,
        kind,
        len(file_bytes),
        language,
    )

    extracted_text, ocr_language_used = extract_text_from_upload(file_bytes, kind, language)

    if not extracted_text:
        logger.warning("OCR produced no text for file '%s'", file.filename)

    document_type, confidence_score = classify_document(extracted_text)
    detected_sections = detect_legal_sections(extracted_text)
    sensitivity_hits = detect_sensitivity_keywords(extracted_text)
    sensitivity_tier = resolve_sensitivity_tier(document_type, sensitivity_hits)
    quorum = QUORUM_MATRIX[sensitivity_tier]
    summary = build_summary(extracted_text)
    entities = extract_entities_with_spacy(extracted_text) if SPACY_AVAILABLE else None

    response = DocumentAnalysisResponse(
        document_type=document_type,
        detected_sections=detected_sections,
        sensitivity_tier=sensitivity_tier,
        recommended_quorum=QuorumInfo(required=quorum["required"], pool_size=quorum["pool_size"]),
        confidence_score=confidence_score,
        extracted_summary=summary,
        extracted_text=extracted_text,
        sensitivity_keywords_found=sensitivity_hits,
        raw_text_char_count=len(extracted_text),
        ocr_language_used=ocr_language_used,
        extracted_entities=entities,
    )

    logger.info(
        "Analysis complete: type=%s tier=%s quorum=%s confidence=%.2f ocr_lang=%s",
        document_type,
        sensitivity_tier,
        quorum,
        confidence_score,
        ocr_language_used,
    )

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):  # pragma: no cover
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)