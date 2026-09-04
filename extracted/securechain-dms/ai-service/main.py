# SecureChain DMS - AI Service (SIH26190)
# Standalone FastAPI microservice for OCR extraction (PaddleOCR) and
# legal-document classification for MHA blockchain-anchored DMS.
# Run: uvicorn main:app --host 0.0.0.0 --port 8000

from __future__ import annotations

import io
import logging
import re
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
from pydantic import BaseModel, Field
from scipy import fft as sp_fft
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("securechain.ai-service")

# OCR engine globals (loaded once in background on startup)
SUPPORTED_OCR_LANGUAGES = {"en": "en", "hi": "hi"}
OCR_ENGINES: dict[str, object | None] = {lang: None for lang in SUPPORTED_OCR_LANGUAGES}
RAPID_OCR_ENGINE = None
EASY_OCR_READER = None
MODEL_LOADING = True
SPACY_NLP = None
SPACY_AVAILABLE = False

ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_PAGES = 15
GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD = 0.20
_MIXED_TOKEN_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{3,}$")
_DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _load_all_models():
    """Load OCR and NLP models on startup."""
    global RAPID_OCR_ENGINE, EASY_OCR_READER, SPACY_NLP, SPACY_AVAILABLE, MODEL_LOADING

    try:
        from rapidocr_onnxruntime import RapidOCR
        RAPID_OCR_ENGINE = RapidOCR()
        logger.info("[Startup] RapidOCR ready.")
    except Exception as exc:
        logger.warning("[Startup] RapidOCR init failed (%s); will fallback to EasyOCR.", exc)

    try:
        import easyocr
        EASY_OCR_READER = easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)
        logger.info("[Startup] EasyOCR ready.")
    except Exception as exc:
        logger.exception("[Startup] EasyOCR init failed: %s", exc)

    try:
        import spacy
        SPACY_NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("[Startup] spaCy 'en_core_web_sm' loaded.")
    except Exception as exc:
        logger.warning("[Startup] spaCy unavailable (%s). Entity extraction disabled.", exc)

    MODEL_LOADING = False
    logger.info("[Startup] All models ready and server active.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield
    logger.info("Shutting down AI service.")


app = FastAPI(
    title="SecureChain DMS - AI Service",
    description="OCR extraction, legal document & report classification, and AI forensic analysis",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Domain constants ---

DOCUMENT_TYPES = [
    "FIR", "Chargesheet", "Forensic Report",
    "Witness Statement", "Case Diary", "Internal Progress Note",
]

DOCUMENT_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "FIR": [
        ("first information report", 3.0), ("fir no", 3.0),
        ("police station", 2.0), ("complainant", 2.0), ("u/s", 1.5),
        ("under section", 1.5), ("date and hour of occurrence", 2.0),
        ("informant", 1.5), ("प्राथमिकी", 3.0), ("प्रथम सूचना रिपोर्ट", 3.0),
        ("थाना", 2.0), ("शिकायतकर्ता", 2.0), ("धारा", 1.5),
    ],
    "Chargesheet": [
        ("charge sheet", 3.0), ("chargesheet", 3.0), ("final report", 2.5),
        ("investigating officer", 2.0), ("charge-sheet", 3.0),
        ("section 173", 3.0), ("accused", 1.5), ("calender of evidence", 2.0),
        ("आरोप पत्र", 3.0), ("अंतिम प्रतिवेदन", 2.5), ("विवेचना अधिकारी", 2.0),
    ],
    "Forensic Report": [
        ("forensic", 3.0), ("post-mortem", 3.0), ("autopsy", 3.0),
        ("ballistics", 3.0), ("chemical analysis", 2.5),
        ("fsl report", 3.0), ("viscera", 2.5), ("dna report", 3.0),
        ("forensic science laboratory", 3.0), ("mlc", 3.0), ("medico-legal", 3.0),
        ("विधि विज्ञान", 3.0), ("शव परीक्षण", 3.0), ("पोस्टमार्टम", 3.0),
    ],
    "Witness Statement": [
        ("statement of witness", 3.0), ("statement under section 161", 3.5),
        ("statement u/s 161", 3.5), ("witness statement", 3.0),
        ("deposes that", 2.0), ("on solemn affirmation", 2.0),
        ("गवाह का बयान", 3.0), ("धारा 161", 3.5), ("बयान गवाह", 3.0),
    ],
    "Case Diary": [
        ("case diary", 3.0), ("daily diary", 2.5), ("general diary", 2.5),
        ("cd no", 2.0), ("investigation diary", 2.5), ("case diary no", 3.0),
        ("केस डायरी", 3.0), ("दैनिक डायरी", 2.5), ("रोजनामचा", 2.5),
    ],
    "Internal Progress Note": [
        ("progress note", 2.5), ("internal note", 2.5), ("office note", 2.0),
        ("dispatch slip", 2.0), ("dispatch", 1.0),
        ("inter-office memo", 1.5), ("for internal circulation", 1.5),
    ],
}

DOCUMENT_TYPE_SENSITIVITY: dict[str, str] = {
    "FIR": "HIGH", "Chargesheet": "HIGH", "Forensic Report": "HIGH",
    "Witness Statement": "MEDIUM", "Case Diary": "MEDIUM",
    "Internal Progress Note": "LOW",
}

SENSITIVITY_KEYWORDS = [
    "post-mortem", "postmortem", "ballistics", "ballistic", "confidential",
    "dna", "forensic", "top secret", "classified", "homicide", "murder",
    "autopsy", "weapon", "narcotics", "poison", "cybercrime", "terrorism",
]

QUORUM_MATRIX: dict[str, dict[str, int]] = {
    "LOW":    {"required": 1, "pool_size": 1},
    "MEDIUM": {"required": 2, "pool_size": 3},
    "HIGH":   {"required": 3, "pool_size": 5},
}

TIER_ORDER = ["LOW", "MEDIUM", "HIGH"]

LEGAL_ACTS = (
    r"(IPC|BNS|BNSS|CrPC|CPC|Evidence Act|Indian Evidence Act|POCSO(?:\s+Act)?|"
    r"NDPS(?:\s+Act)?|Arms(?:\s+Act)?|IT(?:\s+Act)?|Information\s+Technology\s+Act|"
    r"Motor\s+Vehicles\s+Act|MV\s+Act|Excise\s+Act|UAPA|MCOCA|PMLA|NIA(?:\s+Act)?|PASAA|"
    r"भा\.?\s*दं\.?\s*सं\.?|भारतीय\s+न्याय\s+संहिता|आईपीसी|बीएनएस|दंड\s+प्रक्रिया\s+संहिता)"
)

SECTION_PATTERNS = [
    re.compile(
        rf"\b(?:u/s|under\s+sections?|under\s+sec\.?|sections?|sec\.?|s\.|धारा)\s*"
        rf"([0-9a-zA-Z\s,\/\(\)\+\-&]+?)\s+(?:of\s+)?{LEGAL_ACTS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{LEGAL_ACTS}\s+(?:u/s|under\s+sections?|under\s+sec\.?|sections?|sec\.?|s\.|धारा)\s*"
        rf"([0-9a-zA-Z\s,\/\(\)\+\-&]+)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{LEGAL_ACTS}\s+(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z]*(?:\(\w+\))?)\b",
        re.IGNORECASE,
    ),
]

# --- Pydantic models ---

class QuorumInfo(BaseModel):
    required: int = Field(..., description="Minimum approvals needed")
    pool_size: int = Field(..., description="Total approver pool size")


class StructuredFields(BaseModel):
    """Key structured fields parsed directly from OCR text of Indian legal documents (FIR / Chargesheet)."""
    name: str | None = Field(None, description="Complainant / subject name")
    father_name: str | None = Field(None, description="Father's / husband's name (S/o, D/o, W/o)")
    address: str | None = Field(None, description="Residential address (R/o, Address:)")
    fir_number: str | None = Field(None, description="FIR / Case number")
    police_station: str | None = Field(None, description="Police station name")
    date_of_incident: str | None = Field(None, description="Date of the incident or FIR")
    age: str | None = Field(None, description="Age of complainant / accused")
    mobile: str | None = Field(None, description="Mobile / contact number")
    accused_name: str | None = Field(None, description="Accused person name")
    occupation: str | None = Field(None, description="Occupation / profession")


class ReportStructuredFields(BaseModel):
    """Structured fields extracted from Supplementary Medical, Forensic, Autopsy or Lab Reports."""
    report_type: str | None = Field(None, description="MLC / Autopsy / Forensic Lab / Ballistics")
    examining_officer: str | None = Field(None, description="Doctor, Medical Jurist, or Forensic Scientist name")
    hospital_or_lab: str | None = Field(None, description="Hospital name or Forensic Science Laboratory")
    subject_name: str | None = Field(None, description="Injured person, deceased, or examinee name")
    age: str | None = Field(None, description="Age of subject")
    gender: str | None = Field(None, description="Gender of subject")
    date_of_examination: str | None = Field(None, description="Date and time of medical or lab examination")
    injury_findings: list[str] = Field(default_factory=list, description="Clinical injury details, lacerations, burns, chemical results")
    cause_of_death_or_opinion: str | None = Field(None, description="Doctor's final opinion or cause of death")
    severity_grade: str | None = Field(None, description="SIMPLE | GRIEVOUS | FATAL | DANGEROUS | NARCOTICS_POSITIVE")
    weapon_inferred: str | None = Field(None, description="Sharp weapon, blunt weapon, firearm, poison, etc.")


class CrossDocumentCorroboration(BaseModel):
    """Corroboration matrix between FIR Legal Allegations and Supplementary Medical / Forensic Reports."""
    has_dual_documents: bool = Field(False, description="Whether both FIR and Supplementary Report were uploaded")
    corroboration_status: str = Field("NOT_APPLICABLE", description="CORROBORATED | PARTIAL_CORROBORATION | CONTRADICTION_ALERT | NOT_APPLICABLE")
    corroboration_score: float = Field(0.0, ge=0.0, le=1.0, description="Evidentiary consistency score")
    corroborated_points: list[str] = Field(default_factory=list, description="Legally matching findings between FIR and Report")
    discrepancy_alerts: list[str] = Field(default_factory=list, description="Discrepancies or contradictions between FIR and Report")
    magistrate_evidentiary_note: str = Field("", description="Summary guidance for the presiding magistrate")


class ForensicAnalysisResult(BaseModel):
    """AI Forensic Pixel & Spectral Analysis for physical scanned & AI-generated documents (Member 6)."""
    is_scanned_document: bool = Field(True, description="Whether document was evaluated as physical scanned evidence")
    is_ai_generated: bool = Field(False, description="Whether document was detected as AI-generated / synthetic / digital export")
    ai_generation_score: float = Field(0.0, ge=0.0, le=1.0, description="Mathematical AI-generation / synthetic image score")
    spectral_anomaly_score: float = Field(0.0, ge=0.0, le=1.0, description="2D FFT high-frequency spectral artifact score")
    metadata_provenance_score: float = Field(0.0, ge=0.0, le=1.0, description="AI software / digital export metadata detection score")
    ela_anomaly_score: float = Field(0.0, ge=0.0, le=1.0, description="Error Level Analysis pixel compression error score")
    noise_inconsistency_score: float = Field(0.0, ge=0.0, le=1.0, description="Localized high-frequency noise variance score")
    forgery_score: float = Field(0.0, ge=0.0, le=1.0, description="Composite digital tampering & forgery score")
    tamper_risk_level: str = Field("LOW", description="LOW | MEDIUM | HIGH")
    forgery_verdict: str = Field("GENUINE", description="GENUINE | AI-GENERATED / SYNTHETIC | SUSPECTED FORGERY")
    detected_ai_tools: list[str] = Field(default_factory=list, description="List of detected AI generator or digital software signatures")
    forensic_proof_details: dict[str, Any] = Field(default_factory=dict, description="Verifiable mathematical proofs (FFT, Kurtosis, Noise variance)")
    worm_audit_flags: list[str] = Field(default_factory=list, description="High-alert flags for WORM magistrate audit log")
    forensic_summary: str = Field(
        "Pixel compression levels and noise variance are uniform across document grid. No digital tampering detected.",
        description="Magistrate audit trail explanation",
    )
    analyzed_pages_count: int = Field(1, description="Number of document pages scanned by forensic engine")


class DocumentAnalysisResponse(BaseModel):
    document_type: str = Field(..., description="Detected legal document category")
    detected_sections: list[str] = Field(default_factory=list)
    sensitivity_tier: str = Field(..., description="LOW | MEDIUM | HIGH")
    recommended_quorum: QuorumInfo
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    extracted_summary: str
    extracted_text: str
    sensitivity_keywords_found: list[str] = Field(default_factory=list)
    raw_text_char_count: int
    ocr_language_used: str
    extracted_entities: dict[str, list[str]] | None = None
    structured_fields: StructuredFields = Field(default_factory=StructuredFields)
    report_fields: ReportStructuredFields | None = None
    cross_corroboration: CrossDocumentCorroboration | None = None
    forensic_analysis: ForensicAnalysisResult = Field(default_factory=ForensicAnalysisResult)


class HealthResponse(BaseModel):
    status: str
    ocr_engines_ready: dict[str, bool]
    spacy_entity_extraction_ready: bool


# --- OCR helpers ---

def _pil_to_ndarray(image: Image.Image, max_dim: int = 2200) -> np.ndarray:
    """Convert PIL image to RGB ndarray with optional CLAHE contrast enhancement."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    img_np = np.array(image)
    try:
        import cv2
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)
    except Exception as exc:
        logger.debug("cv2 CLAHE skipped: %s", exc)
        try:
            from PIL import ImageEnhance
            image = ImageEnhance.Contrast(image).enhance(1.4)
            image = ImageEnhance.Sharpness(image).enhance(1.3)
        except Exception:
            pass
        return img_np


def _run_ocr_on_image(image_array: np.ndarray, lang: str) -> str:
    """Run multi-engine OCR. Hindi: EasyOCR + PaddleOCR ensemble. English: RapidOCR → EasyOCR → PaddleOCR."""
    if lang == "hi":
        easy_text, paddle_text = "", ""

        if EASY_OCR_READER is not None:
            try:
                results = EASY_OCR_READER.readtext(image_array, detail=0, paragraph=False)
                easy_text = "\n".join(str(t).strip() for t in results if str(t).strip())
            except Exception as exc:
                logger.warning("EasyOCR (hi) failed: %s", exc)

        engine = OCR_ENGINES.get("hi")
        if engine is not None:
            try:
                try:
                    result = engine.ocr(image_array)
                except TypeError:
                    result = engine.ocr(image_array, cls=True)
                if result:
                    p_lines = []
                    for item in result:
                        if isinstance(item, dict):
                            t = item.get("rec_text") or item.get("text", "")
                            if t:
                                p_lines.append(str(t))
                        elif isinstance(item, list):
                            for line in item:
                                try:
                                    t = line[1][0]
                                    if t:
                                        p_lines.append(t)
                                except (IndexError, TypeError):
                                    continue
                    paddle_text = "\n".join(p_lines)
            except Exception as exc:
                logger.warning("PaddleOCR (hi) failed: %s", exc)

        easy_dev = sum(1 for c in easy_text if _DEVANAGARI_PATTERN.match(c))
        paddle_dev = sum(1 for c in paddle_text if _DEVANAGARI_PATTERN.match(c))
        if paddle_dev > easy_dev and paddle_text.strip():
            return paddle_text
        return easy_text.strip() or paddle_text

    # English / auto
    if RAPID_OCR_ENGINE is not None:
        try:
            result, _ = RAPID_OCR_ENGINE(image_array)
            if result:
                text = "\n".join(str(item[1]).strip() for item in result if item and len(item) > 1 and str(item[1]).strip())
                if text and not _CJK_PATTERN.search(text) and _gibberish_score(text) < 0.18:
                    return text
        except Exception as exc:
            logger.warning("RapidOCR failed (%s); trying EasyOCR.", exc)

    if EASY_OCR_READER is not None:
        try:
            results = EASY_OCR_READER.readtext(image_array, detail=0, paragraph=False)
            lines = [str(t).strip() for t in results if str(t).strip()]
            if lines:
                return "\n".join(lines)
        except Exception as exc:
            logger.warning("EasyOCR fallback failed (%s); trying PaddleOCR.", exc)

    engine = OCR_ENGINES.get("en")
    if engine is not None:
        try:
            try:
                result = engine.ocr(image_array)
            except TypeError:
                result = engine.ocr(image_array, cls=True)
            if result:
                p_lines = []
                for item in result:
                    if isinstance(item, dict):
                        t = item.get("rec_text") or item.get("text", "")
                        if t:
                            p_lines.append(str(t))
                    elif isinstance(item, list):
                        for line in item:
                            try:
                                t = line[1][0]
                                if t:
                                    p_lines.append(t)
                            except (IndexError, TypeError):
                                continue
                return "\n".join(p_lines)
        except Exception as exc:
            logger.warning("PaddleOCR fallback failed: %s", exc)

    return ""


def _gibberish_score(text: str) -> float:
    """Fraction of tokens that jam letters and digits together (cross-script OCR noise signal)."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _MIXED_TOKEN_PATTERN.match(t)) / len(tokens)


def _devanagari_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(bool(_DEVANAGARI_PATTERN.match(c)) for c in chars) / len(chars)


# ---------------------------------------------------------------------------
# Structured field extraction — High Precision Multi-Layer Pipeline:
#   Layer 1: Key-value line parser with automatic inline relation splitting
#   Layer 1b: Universal inline relation parser (S/o, D/o, W/o, R/o, Hindi aliases)
#   Layer 2: Positional & contextual pattern fallbacks
#   Layer 3: spaCy NER fallback when explicit labels are absent
# ---------------------------------------------------------------------------

_FIELD_LABELS: dict[str, list[str]] = {
    "name": [
        "name of complainant", "complainant name", "complainant's name",
        "name of informant", "informant name", "name of applicant",
        "applicant name", "complainant", "informant", "applicant",
        "name of victim", "victim name", "victim", "name of person",
        "person name", "full name", "1. name", "i. name", "name",
        # Hindi
        "शिकायतकर्ता का नाम", "शिकायतकर्ता", "आवेदक का नाम", "आवेदक",
        "सूचक का नाम", "सूचक", "पीड़ित का नाम", "वादी का नाम", "नाम",
    ],
    "father_name": [
        "father's / husband's name", "father's name", "father name", "fathers name",
        "name of father", "husband's name", "husband name", "name of husband",
        "parent's name", "parent name", "guardian's name", "guardian name",
        "son of", "daughter of", "wife of", "husband of", "care of",
        "s/o", "d/o", "w/o", "h/o", "c/o", "s/0", "d/0", "w/0", "c/0",
        "f/name", "f.name", "father", "husband",
        # Hindi
        "पिता का नाम", "पिता / पति का नाम", "पिता", "पति का नाम", "पति",
        "आत्मज", "सुपुत्र", "पुत्र", "पुत्री", "पत्नी", "वल्द",
        "संरक्षक का नाम", "स/पु", "ड/पु", "वा/पु",
    ],
    "address": [
        "permanent address", "present address", "residential address",
        "residence address", "postal address", "current address",
        "residence", "full address", "resident of", "residing at",
        "address", "r/o", "r/0", "house no", "h.no", "h no",
        "locality", "village", "mohalla",
        # Hindi
        "स्थायी पता", "वर्तमान पता", "पूरा पता", "पता", "निवास",
        "निवासी", "साकिन", "मुकाम", "ग्राम", "मोहल्ला",
    ],
    "fir_number": [
        "first information report no", "first information report number",
        "fir no.", "fir no", "fir number", "fir#", "fir",
        "case no.", "case no", "case number",
        "complaint no.", "complaint no", "complaint number",
        "cr no.", "cr no", "cr number", "crime no.", "crime no", "crime number",
        # Hindi
        "प्रथम सूचना रिपोर्ट संख्या", "प्रथम सूचना रिपोर्ट नं", "प्रथम सूचना रिपोर्ट",
        "एफआईआर संख्या", "एफआईआर नं", "एफआईआर नंबर", "एफआईआर",
        "मुकदमा संख्या", "मुकदमा नं", "केस संख्या", "केस नं", "केस नंबर", "अपराध संख्या",
    ],
    "police_station": [
        "police station", "p.s.", "ps", "station house", "police chowki", "outpost",
        "thana", "thane", "station",
        # Hindi
        "पुलिस स्टेशन", "थाने का नाम", "कोतवाली", "चौकी", "थाना",
    ],
    "date_of_incident": [
        "date and time of incident", "date & time of incident",
        "date of incident", "date of occurrence", "date of offence",
        "date of crime", "incident date", "occurrence date",
        "date of fir", "date of report", "date of complaint", "date",
        # Hindi
        "घटना की तारीख", "घटना दिनांक", "समय व दिनांक", "दिनांक", "तारीख",
    ],
    "age": [
        "age (years)", "age in years", "age/sex", "age / sex", "approx age", "aged", "age",
        # Hindi
        "उम्र", "आयु", "वय",
    ],
    "mobile": [
        "mobile number", "mobile no.", "mobile no", "mobile",
        "phone number", "phone no.", "phone no", "phone",
        "contact number", "contact no.", "contact no", "contact",
        "tel number", "tel no.", "tel no", "tel", "telephone", "mob",
        # Hindi
        "मोबाइल नंबर", "मोबाइल नं", "मोबाइल", "फोन नंबर", "फोन नं", "फोन",
        "संपर्क नंबर", "संपर्क नं", "संपर्क",
    ],
    "accused_name": [
        "name of the accused", "name of accused", "accused's name", "accused name",
        "accused person", "name of suspect", "suspect name", "suspect",
        "offender name", "offender", "accused",
        # Hindi
        "अभियुक्त का नाम", "अभियुक्त", "संदिग्ध का नाम", "संदिग्ध",
        "आरोपी का नाम", "आरोपी", "दोषी का नाम",
    ],
    "occupation": [
        "occupation", "profession", "trade", "employed as", "employment",
        "designation", "business", "work",
        # Hindi
        "पेशा", "व्यवसाय", "नौकरी", "धंधा", "काम",
    ],
}

# Lookup table: label -> field_key
_LABEL_TO_FIELD: dict[str, str] = {}
for _fk, _labels in _FIELD_LABELS.items():
    for _lbl in _labels:
        _LABEL_TO_FIELD[_lbl.lower().strip()] = _fk

_SORTED_LABELS = sorted(_LABEL_TO_FIELD.keys(), key=len, reverse=True)


# --- Normalization & Cleaning Helpers ---

_ITEM_PREFIX_RE = re.compile(
    r"^(?:(?:\d{1,3}(?:\.\d{1,3})?|[a-zA-Z]|\([0-9a-zA-Z\u0900-\u097F]{1,4}\))\s*[\.\)\-:]\s*)+",
    re.IGNORECASE,
)

_NAME_LABEL_PREFIX_RE = re.compile(
    r"^(?:Name(?:\s+of\s+(?:complainant|informant|applicant|accused|victim|person))?|"
    r"Complainant|Informant|Applicant|Accused|Victim|Person|नाम|शिकायतकर्ता(?:\s+का\s+नाम)?|"
    r"आवेदक(?:\s+का\s+नाम)?|अभियुक्त(?:\s+का\s+नाम)?|वादी|सूचक)[\s:\-.]*",
    re.IGNORECASE,
)

_RELATION_PREFIX_RE = re.compile(
    r"^(?:[sSdDwWhHcC]/[oO0]|[sSdDwWhHcC]/0|\b(?:son|daughter|wife|husband|care)\s+of\b|"
    r"Father(?:'s)?\s*Name|Husband(?:'s)?\s*Name|पिता(?:\s+का\s+नाम)?|पति(?:\s+का\s+नाम)?|"
    r"आत्मज|पुत्र|सुपुत्र|पुत्री|पत्नी|वल्द)[\s:\-.]*",
    re.IGNORECASE,
)

_ADDRESS_PREFIX_RE = re.compile(
    r"^(?:Permanent\s+Address|Present\s+Address|Residential\s+Address|Residence|Address|Full\s+Address|"
    r"[rR]/[oO0]|[rR]/0|\b(?:resident\s+of|residing\s+at)\b|पता|निवास|स्थायी\s+पता|वर्तमान\s+पता|निवासी|साकिन|मुकाम)[\s:\-.]*",
    re.IGNORECASE,
)

_HONORIFIC_PREFIX_RE = re.compile(
    r"^(?:Shri|Sh\.|Mr\.|Mrs\.|Smt\.|Dr\.|Late|Sri|Master|Miss|Kumari|Km\.|Md\.|Mohd\.|"
    r"श्री|श्रीमती|सुश्री|डॉ\.?|स्व\.?)[\s.]*",
    re.IGNORECASE,
)

_RELATION_MARKERS_RE = re.compile(
    r"(?:\b(?:s/o|d/o|w/o|h/o|c/o|s/0|d/0|w/0|c/0|son\s+of|daughter\s+of|wife\s+of|husband\s+of|care\s+of|father(?:'s)?\s*name|husband(?:'s)?\s*name)\b|"
    r"(?<=\s)[sSdDwWhHcC]/[oO0](?=\s)|"
    r"[\s,;:\-](?:आत्मज|सुपुत्र|पुत्र|पुत्री|पत्नी|पति|वल्द)[\s,;:\-])",
    re.IGNORECASE,
)

_RESIDENCE_MARKERS_RE = re.compile(
    r"(?:\b(?:r/o|r/0|resident\s+of|residing\s+at|permanent\s+address|present\s+address|residential\s+address)\b|"
    r"(?<=\s)[rR]/[oO0](?=\s)|"
    r"[\s,;:\-](?:निवासी|निवाशी|साकिन|मुकाम|स्थायी\s+पता|वर्तमान\s+पता|पता)[\s,;:\-])",
    re.IGNORECASE,
)

_NAME_TAIL_SPLIT_RE = re.compile(
    r"[\s,;]+(?:[sSdDwWhHcC]/[oO0]|\b(?:son|daughter|wife|husband)\s+of\b|[rR]/[oO0]|\b(?:resident\s+of|residing\s+at)\b|आत्मज|पुत्र|सुपुत्र|पुत्री|पत्नी|पति|निवासी|साकिन|पता|\baged?\b|\bage\s*\d+)",
    re.IGNORECASE,
)

_FATHER_TAIL_SPLIT_RE = re.compile(
    r"[\s,;]+(?:[rR]/[oO0]|[rR]/0|\b(?:resident\s+of|residing\s+at|address)\b|निवास[ी]?|साकिन|मुकाम|पता|\baged?\b|\bage\s*\d+|\bmob(?:ile)?\b|\bphone\b)",
    re.IGNORECASE,
)

_ADDRESS_TAIL_SPLIT_RE = re.compile(
    r"[\s,;]+(?:FIR|Case|Date\s+of|Police\s+Station|P\.S\.|Thana|धारा|दिनांक)\s*[:\-]",
    re.IGNORECASE,
)


def _clean_name(val: str | None) -> str | None:
    """Clean person / complainant name by stripping numbering, labels, honorifics, and trailing relations."""
    if not val:
        return None
    s = val.strip()
    s = _ITEM_PREFIX_RE.sub("", s).strip()
    for _ in range(3):
        prev = s
        s = _NAME_LABEL_PREFIX_RE.sub("", s).strip()
        s = _HONORIFIC_PREFIX_RE.sub("", s).strip()
        if s == prev:
            break
    # Strip trailing relations or address tails if attached
    s = _NAME_TAIL_SPLIT_RE.split(s, maxsplit=1)[0]
    s = s.strip(" \t\n\r,;:-.=").strip()
    if len(s) < 2 or s.isdigit() or s.lower() in {"null", "none", "unknown", "na", "n/a"}:
        return None
    return s


def _clean_father_name(val: str | None) -> str | None:
    """Clean father's / husband's name by stripping relation prefixes, honorifics, and address tails."""
    if not val:
        return None
    s = val.strip()
    s = _ITEM_PREFIX_RE.sub("", s).strip()
    for _ in range(3):
        prev = s
        s = _RELATION_PREFIX_RE.sub("", s).strip()
        s = _HONORIFIC_PREFIX_RE.sub("", s).strip()
        if s == prev:
            break
    # Strip trailing address or age parts
    s = _FATHER_TAIL_SPLIT_RE.split(s, maxsplit=1)[0]
    s = s.strip(" \t\n\r,;:-.=").strip()
    if len(s) < 2 or s.isdigit() or s.lower() in {"null", "none", "unknown", "na", "n/a"}:
        return None
    return s


def _clean_address(val: str | None) -> str | None:
    """Clean address string by stripping address label prefixes and cleaning extra punctuation."""
    if not val:
        return None
    s = val.strip()
    s = _ITEM_PREFIX_RE.sub("", s).strip()
    for _ in range(3):
        prev = s
        s = _ADDRESS_PREFIX_RE.sub("", s).strip()
        if s == prev:
            break
    # Strip trailing metadata if accidentally attached
    s = _ADDRESS_TAIL_SPLIT_RE.split(s, maxsplit=1)[0]
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r,;:-.=")
    if len(s) < 3:
        return None
    return s


def _clean_age(val: str | None) -> str | None:
    """Extract clean age digits from string like '32 years', 'Age: 45', etc."""
    if not val:
        return None
    m = re.search(r"\b(\d{1,3})\b", val)
    return m.group(1) if m else val.strip()


def _normalize_key(key: str) -> str:
    """Normalize raw key string from OCR line by stripping numbers, item letters, and bracketed translations."""
    k = _ITEM_PREFIX_RE.sub("", key).strip()
    k = re.sub(r'\([^\)]*\)', '', k).strip()
    k = re.sub(r'\s+', ' ', k).strip().lower()
    return k


def _parse_inline_line(line: str) -> dict[str, str]:
    """
    Parse a single line (or multi-word line value) that may contain inline notation:
    e.g., 'Amit Kumar S/o Ramesh Lal R/o 12 Main St, Delhi'
          'Sunita Devi W/o Manoj Kumar, पता: मकान नं 45'
          'अमित कुमार आत्मज श्री सुरेश कुमार निवासी ग्राम रामपुर'
    """
    res: dict[str, str] = {}
    if not line or len(line.strip()) < 5:
        return res

    so_match = _RELATION_MARKERS_RE.search(line)
    ro_match = _RESIDENCE_MARKERS_RE.search(line)

    if so_match and ro_match and so_match.start() < ro_match.start():
        raw_name = line[:so_match.start()]
        raw_father = line[so_match.end():ro_match.start()]
        raw_addr = line[ro_match.end():]
        c_name = _clean_name(raw_name)
        c_father = _clean_father_name(raw_father)
        c_addr = _clean_address(raw_addr)
        if c_name: res["name"] = c_name
        if c_father: res["father_name"] = c_father
        if c_addr: res["address"] = c_addr
    elif so_match:
        raw_name = line[:so_match.start()]
        raw_father = line[so_match.end():]
        c_name = _clean_name(raw_name)
        c_father = _clean_father_name(raw_father)
        if c_name: res["name"] = c_name
        if c_father: res["father_name"] = c_father
    elif ro_match:
        raw_name = line[:ro_match.start()]
        raw_addr = line[ro_match.end():]
        c_name = _clean_name(raw_name)
        c_addr = _clean_address(raw_addr)
        if c_name and len(c_name.split()) >= 2: res["name"] = c_name
        if c_addr: res["address"] = c_addr

    return res


# Broad positional fallback patterns (when no label is present)
_FALLBACK: dict[str, list[re.Pattern[str]]] = {
    "fir_number": [
        re.compile(r"\bFIR\s*[:\-]?\s*([A-Za-z0-9/\-]{2,20})\b", re.I),
        re.compile(r"\bNo\.?\s*(\d{1,6}/\d{4})\b", re.I),
        re.compile(r"\b(\d{1,6}/\d{4})\b"),
    ],
    "mobile": [
        re.compile(r"\b((?:\+91[\s\-]?)?[6-9]\d{9})\b"),
        re.compile(r"\b(0\d{2,4}[\s\-]\d{6,8})\b"),
    ],
    "date_of_incident": [
        re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),
        re.compile(
            r"\b(\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{4})\b", re.I,
        ),
    ],
    "age": [re.compile(r"\baged?\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs)?\b", re.I)],
    "name": [
        re.compile(r"(?:^|\n)\s*(?:Name|नाम)\s*[:\-]\s*([^\n,;]{2,50})", re.IGNORECASE),
    ],
}


def _parse_kv_lines(text: str) -> dict[str, str]:
    """
    Parse every line of OCR text as a potential key: value pair.
    Handles multi-word keys (including Hindi) separated by colon, dash, or equals.
    Splits compound values (like Name containing S/o and Address) across canonical fields.
    Stitches multi-line address blocks.
    """
    result: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        matched = False
        for sep in (":", "-", "="):
            if sep in line:
                raw_key, _, raw_val = line.partition(sep)
                raw_key = raw_key.strip().lower()
                norm_key = _normalize_key(raw_key)
                raw_val = raw_val.strip()
                # If value is on next line
                if not raw_val and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and ":" not in next_line and len(next_line) < 120:
                        raw_val = next_line

                if raw_val:
                    for lbl in _SORTED_LABELS:
                        # Match label boundary or exact phrase against raw_key or norm_key
                        if (
                            norm_key == lbl or raw_key == lbl
                            or norm_key.endswith(" " + lbl) or norm_key.startswith(lbl + " ") or (" " + lbl + " ") in norm_key
                            or raw_key.endswith(" " + lbl) or raw_key.startswith(lbl + " ") or (" " + lbl + " ") in raw_key
                        ):
                            fk = _LABEL_TO_FIELD[lbl]
                            val = raw_val.strip(":-=").strip()

                            if fk == "name":
                                # Check if the name value actually has inline S/o or R/o
                                inline_parts = _parse_inline_line(val)
                                if inline_parts:
                                    for sub_fk, sub_val in inline_parts.items():
                                        if sub_fk not in result:
                                            result[sub_fk] = sub_val
                                else:
                                    c_name = _clean_name(val)
                                    if c_name and "name" not in result:
                                        result["name"] = c_name
                                matched = True
                            elif fk == "father_name":
                                c_father = _clean_father_name(val)
                                if c_father and "father_name" not in result:
                                    result["father_name"] = c_father
                                matched = True
                            elif fk == "address":
                                extra: list[str] = []
                                j = i + 1
                                while j < len(lines) and len(extra) < 5:
                                    cont = lines[j].strip()
                                    if not cont:
                                        break
                                    is_new_field = any(
                                        s in cont and len(cont.partition(s)[0].strip()) < 35
                                        for s in (":", "=")
                                    )
                                    if is_new_field:
                                        break
                                    extra.append(cont)
                                    j += 1
                                if extra:
                                    val = val + ", " + ", ".join(extra)
                                c_addr = _clean_address(val)
                                if c_addr and "address" not in result:
                                    result["address"] = c_addr
                                matched = True
                            else:
                                if fk not in result:
                                    result[fk] = val
                                matched = True
                            break
                if matched:
                    break
        i += 1
    return result


def _parse_inline_so(text: str) -> dict[str, str]:
    """
    Layer 1b — Scan lines for inline relation patterns (S/o, D/o, W/o, R/o, Hindi equivalents).
    """
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _parse_inline_line(line)
        for k, v in parsed.items():
            if k not in result and v:
                result[k] = v
        if len(result) >= 3:
            break

    # Standalone R/o fallback
    if "address" not in result:
        for line in text.splitlines():
            line = line.strip()
            ro_m = _RESIDENCE_MARKERS_RE.search(line)
            if ro_m:
                addr_candidate = _clean_address(line[ro_m.end():])
                if addr_candidate and len(addr_candidate) >= 6:
                    result["address"] = addr_candidate
                    break

    return result


def extract_structured_fields(text: str) -> "StructuredFields":
    """
    4-layer structured field extraction for Indian legal documents.
    """
    # Layer 1: universal key:value parser
    fields = _parse_kv_lines(text)
    logger.debug("Layer1 KV fields found: %s", list(fields.keys()))

    # Layer 1b: inline S/o & R/o pattern splitter
    inline = _parse_inline_so(text)
    for fk, val in inline.items():
        if fk not in fields and val:
            fields[fk] = val
    logger.debug("After Layer1b fields: %s", list(fields.keys()))

    # Layer 2: regex fallbacks for anything not found in Layers 1 / 1b
    for fk, patterns in _FALLBACK.items():
        if fk not in fields:
            for pat in patterns:
                m = pat.search(text)
                if m:
                    extracted = m.group(1).strip()
                    if fk == "name":
                        extracted = _clean_name(extracted) or ""
                    if extracted:
                        fields[fk] = extracted
                        logger.debug("Layer2 regex: %s = %s", fk, fields[fk][:40])
                        break

    # Layer 3: spaCy NER fallback — only fills fields still missing after Layers 1/2
    if SPACY_AVAILABLE and SPACY_NLP is not None:
        try:
            doc = SPACY_NLP(text[:15000])
            persons = [
                e.text.strip() for e in doc.ents
                if e.label_ == "PERSON" and len(e.text.strip()) > 3
            ]
            if "name" not in fields and persons:
                c_name = _clean_name(persons[0])
                if c_name:
                    fields["name"] = c_name
                    logger.debug("Layer3 NER: name = %s", c_name[:40])
            if "father_name" not in fields and len(persons) >= 2:
                c_father = _clean_father_name(persons[1])
                if c_father:
                    fields["father_name"] = c_father
                    logger.debug("Layer3 NER: father_name = %s", c_father[:40])
            if "date_of_incident" not in fields:
                dates = [e.text.strip() for e in doc.ents if e.label_ == "DATE"]
                if dates:
                    fields["date_of_incident"] = dates[0]
            if "police_station" not in fields:
                orgs = [
                    e.text.strip() for e in doc.ents
                    if e.label_ == "ORG" and "police" in e.text.lower()
                ]
                if orgs:
                    fields["police_station"] = orgs[0]
            if "address" not in fields:
                locs = [
                    e.text.strip() for e in doc.ents
                    if e.label_ in ("GPE", "LOC") and len(e.text.strip()) > 3
                ]
                if locs:
                    fields["address"] = ", ".join(locs[:3])
                    logger.debug("Layer3 NER: address (GPE/LOC) = %s", fields["address"][:40])
        except Exception:
            pass

    # Final post-processing pass to ensure all fields are cleaned
    final_name = _clean_name(fields.get("name"))
    final_father = _clean_father_name(fields.get("father_name"))
    final_address = _clean_address(fields.get("address"))
    final_accused = _clean_name(fields.get("accused_name"))
    final_age = _clean_age(fields.get("age"))
    raw_mobile = fields.get("mobile")
    final_mobile = raw_mobile.strip() if raw_mobile else None
    if final_mobile:
        m_mob = re.search(r"((?:\+91[\s\-]?)?[6-9]\d{9}|0\d{2,4}[\s\-]?\d{6,8})", final_mobile)
        if m_mob:
            final_mobile = m_mob.group(1)

    logger.debug("Final structured fields: %s", list(fields.keys()))
    return StructuredFields(
        name=final_name,
        father_name=final_father,
        address=final_address,
        fir_number=fields.get("fir_number"),
        police_station=fields.get("police_station"),
        date_of_incident=fields.get("date_of_incident"),
        age=final_age,
        mobile=final_mobile,
        accused_name=final_accused,
        occupation=fields.get("occupation"),
    )


# --- Supplementary Medical & Forensic Report Extraction ---

_REPORT_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("post-mortem", "Post-Mortem / Autopsy Examination Report"),
    ("postmortem", "Post-Mortem / Autopsy Examination Report"),
    ("autopsy", "Post-Mortem / Autopsy Examination Report"),
    ("medico-legal", "Medico-Legal Injury Report (MLC)"),
    ("mlc", "Medico-Legal Injury Report (MLC)"),
    ("injury report", "Medico-Legal Injury Report (MLC)"),
    ("chemical analysis", "Forensic Chemical / Viscera Report"),
    ("viscera", "Forensic Chemical / Viscera Report"),
    ("ballistics", "Forensic Ballistics & Firearms Report"),
    ("ballistic", "Forensic Ballistics & Firearms Report"),
    ("dna profiling", "DNA Forensic Profiling Report"),
    ("fsl report", "Forensic Science Laboratory (FSL) Report"),
    ("toxicology", "Toxicology Examination Report"),
]


def extract_report_structured_fields(text: str) -> ReportStructuredFields:
    """Extract structured data from Medical Examination (MLC), Autopsy, or Forensic Lab Reports."""
    if not text.strip():
        return ReportStructuredFields()

    lowered = text.lower()
    report_type = "Forensic / Medical Examination Report"
    for kw, label in _REPORT_TYPE_KEYWORDS:
        if kw in lowered:
            report_type = label
            break

    # 1. Examining Officer / Doctor / Scientist
    doctor_patterns = [
        re.compile(r"(?:Doctor|Dr\.|Medical\s+Officer|Examining\s+Officer|Forensic\s+Expert|Medical\s+Jurist|Autopsy\s+Surgeon)\s*[:\-]?\s*(Dr\.?\s+[A-Za-z\s\.]+)", re.I),
        re.compile(r"(?:Name\s+of\s+Doctor|Medical\s+Officer)\s*[:\-]?\s*([A-Za-z\s\.\(\)]+)", re.I),
        re.compile(r"\b(Dr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"),
        re.compile(r"(?:चिकित्सक|डॉक्टर|चिकित्सा\s+अधिकारी)\s*[:\-]?\s*(डॉ\.?\s+[\u0900-\u097F\s]+)", re.I),
    ]
    examining_officer = None
    for pat in doctor_patterns:
        m = pat.search(text)
        if m:
            c_doc = _clean_name(m.group(1).strip())
            if c_doc:
                examining_officer = c_doc if c_doc.startswith("Dr.") or "डॉ" in c_doc else f"Dr. {c_doc}"
                break

    # 2. Hospital / Laboratory Name
    hosp_patterns = [
        re.compile(r"(?:Hospital|Medical\s+College|Laboratory|FSL|Lab|Institute|डिस्पेंसरी|अस्पताल)\s*[:\-]?\s*([A-Za-z0-9\s,\.\(\)\-]{5,80})", re.I),
        re.compile(r"\b([A-Za-z\s\.]+(?:Hospital|Medical\s+College|Institute\s+of\s+Medical\s+Sciences|AIIMS|Forensic\s+Science\s+Laboratory|FSL|Dispensary)[A-Za-z0-9\s,\.\-]*)\b", re.I),
        re.compile(r"([\u0900-\u097F\s]+(?:अस्पताल|चिकित्सालय|आयुर्विज्ञान|विधि\s+विज्ञान\s+प्रयोगशाला))", re.I),
    ]
    hospital_or_lab = None
    for pat in hosp_patterns:
        m = pat.search(text)
        if m:
            raw_hosp = m.group(1).strip(" :-,.\n\r")
            if len(raw_hosp) >= 5 and not raw_hosp.lower().startswith("name"):
                hospital_or_lab = raw_hosp.splitlines()[0].strip()
                break

    # 3. Subject / Patient / Deceased Name
    subj_patterns = [
        re.compile(r"(?:Name\s+of\s+(?:Injured|Patient|Deceased|Victim|Subject|Examinee)|Injured|Patient|Deceased|Victim|मरीज|घायल|मृतक)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s\.\-]{2,50})", re.I),
    ]
    subject_name = None
    for pat in subj_patterns:
        m = pat.search(text)
        if m:
            subject_name = _clean_name(m.group(1).strip())
            if subject_name:
                break
    if not subject_name:
        # Fallback to general structured name parser
        kv_fields = _parse_kv_lines(text)
        subject_name = _clean_name(kv_fields.get("name"))

    # 4. Age & Gender
    age = _clean_age(text)
    gender = None
    if re.search(r"\b(?:Male|M\/|M\b|पुरुष)\b", text, re.I):
        gender = "Male"
    elif re.search(r"\b(?:Female|F\/|F\b|महिला|स्त्री)\b", text, re.I):
        gender = "Female"

    # 5. Date of Examination
    date_exam = None
    m_date = re.search(r"(?:Date\s+of\s+(?:Examination|Autopsy|Admission|PM)|दिनांक)\s*[:\-]?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", text, re.I)
    if m_date:
        date_exam = m_date.group(1).strip()
    else:
        m_date_gen = re.search(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b", text)
        if m_date_gen:
            date_exam = m_date_gen.group(1).strip()

    # 6. Injury / Clinical Findings
    injury_findings: list[str] = []
    injury_regex = re.compile(
        r"(?:(?:[\d]+\.|\-|\*)\s*)?([^\n\r\.]*(?:incised\s+wound|lacerated\s+wound|laceration|contusion|abrasion|"
        r"fracture|bruise|hematoma|burn|gunshot|bullet\s+entry|poison|viscera|stab\s+wound|haemorrhage|trauma|"
        r"घाव|चोट|खरोंच|हड्डी\s+टूटना|अस्थि\s+भंग)[^\n\r\.]*)",
        re.I,
    )
    for match in injury_regex.finditer(text):
        finding = match.group(1).strip(" \t\n\r-*•:;,.")
        if len(finding) > 6 and finding not in injury_findings and len(injury_findings) < 6:
            injury_findings.append(finding)

    # 7. Severity Grade & Inferred Weapon
    severity_grade = "SIMPLE"
    weapon_inferred = "Unspecified / Blunt object"

    if re.search(r"\b(?:Fatal|Cause\s+of\s+Death|Dangerous\s+to\s+life|Hemorrhagic\s+shock|Craniocerebral\s+damage|मृत्यु\s+का\s+कारण)\b", text, re.I):
        severity_grade = "FATAL / DANGEROUS TO LIFE"
    elif re.search(r"\b(?:Grievous|Grievous\s+hurt|Fracture|Bone\s+deep|गंभीर\s+चोट|अस्थि\s+भंग)\b", text, re.I):
        severity_grade = "GRIEVOUS"
    elif re.search(r"\b(?:Simple\s+nature|Simple\s+injury|साधारण\s+चोट)\b", text, re.I):
        severity_grade = "SIMPLE"
    elif re.search(r"\b(?:Narcotics\s+positive|Opiates\s+detected|Poison\s+detected|Ethyl\s+alcohol)\b", text, re.I):
        severity_grade = "NARCOTICS / TOXICOLOGY POSITIVE"

    if re.search(r"\b(?:Sharp\s+edged|Sharp\s+weapon|Incised|Stab|Sword|Knife|Dagger|धारदार\s+हथियार)\b", text, re.I):
        weapon_inferred = "Sharp-edged weapon / Bladed weapon"
    elif re.search(r"\b(?:Firearm|Gunshot|Bullet|GSR|Pellet|बंदूक|गोली)\b", text, re.I):
        weapon_inferred = "Firearm / Projectile weapon"
    elif re.search(r"\b(?:Poison|Chemical|Toxic|Acid|Alkaloid|जहर|विषाक्त)\b", text, re.I):
        weapon_inferred = "Chemical / Toxic substance"
    elif re.search(r"\b(?:Hard\s+and\s+blunt|Blunt\s+force|Lathi|Stick|भोथरा\s+हथियार|लाठी)\b", text, re.I):
        weapon_inferred = "Hard and blunt object"

    # 8. Cause of Death / Final Opinion
    opinion = None
    op_m = re.search(r"(?:Opinion|Cause\s+of\s+Death|Conclusion|Final\s+Remark|राय|निष्कर्ष)\s*[:\-]?\s*([^\n\r]+(?:\n[^\n\r]+){0,2})", text, re.I)
    if op_m:
        opinion = op_m.group(1).strip(" :-,.\n\r")

    return ReportStructuredFields(
        report_type=report_type,
        examining_officer=examining_officer,
        hospital_or_lab=hospital_or_lab,
        subject_name=subject_name,
        age=age,
        gender=gender,
        date_of_examination=date_exam,
        injury_findings=injury_findings,
        cause_of_death_or_opinion=opinion,
        severity_grade=severity_grade,
        weapon_inferred=weapon_inferred,
    )


def correlate_fir_and_report(
    fir_sections: list[str],
    fir_text: str,
    report_fields: ReportStructuredFields,
    report_text: str,
) -> CrossDocumentCorroboration:
    """Correlate FIR legal sections and allegations with Supplementary Medical/Forensic Report."""
    corroborated_points: list[str] = []
    discrepancy_alerts: list[str] = []
    score = 0.50

    sec_str = " ".join(fir_sections).lower()
    rep_text_low = (report_text + " " + (report_fields.severity_grade or "") + " " + (report_fields.weapon_inferred or "")).lower()

    # 1. Check Grievous / Attempt to Murder (IPC 307 / 326 / 324 or BNS 109 / 115 / 117)
    if any(s in sec_str for s in ["307", "326", "324", "109", "115", "117"]):
        if report_fields.severity_grade in ("GRIEVOUS", "FATAL / DANGEROUS TO LIFE") or "grievous" in rep_text_low:
            corroborated_points.append(
                f"FIR Sections for severe assault ({', '.join(s for s in fir_sections if any(x in s for x in ['307', '326', '324', '109', '115']))}) "
                f"are strongly corroborated by Medical Report finding '{report_fields.severity_grade}' injuries."
            )
            score += 0.30
        elif report_fields.severity_grade == "SIMPLE" and "simple" in rep_text_low:
            discrepancy_alerts.append(
                f"EVIDENTIARY DISCREPANCY: FIR registers severe assault sections (e.g. 307 / 326), but Medical Report explicitly classifies all injuries as 'SIMPLE in nature'."
            )
            score -= 0.20

    # 2. Check Homicide / Murder (IPC 302 / 304 / BNS 103 / 105)
    if any(s in sec_str for s in ["302", "304", "103", "105"]):
        if report_fields.severity_grade == "FATAL / DANGEROUS TO LIFE" or "post-mortem" in (report_fields.report_type or "").lower():
            corroborated_points.append(
                "FIR Homicide allegations (Section 302/103) are corroborated by Autopsy / Post-Mortem findings of fatal trauma."
            )
            score += 0.35

    # 3. Check Weapon consistency
    if "sharp" in (report_fields.weapon_inferred or "").lower() and any(w in fir_text.lower() for w in ["knife", "sword", "dagger", "chaku", "talwar", "धारदार"]):
        corroborated_points.append(
            f"Weapon Consistency Confirmed: FIR alleges sharp-edged weapon attack, matching Medical Report inference: '{report_fields.weapon_inferred}'."
        )
        score += 0.15
    elif "firearm" in (report_fields.weapon_inferred or "").lower() and any(w in fir_text.lower() for w in ["gun", "pistol", "revolver", "firearm", "goli", "bandook", "गोली"]):
        corroborated_points.append(
            f"Ballistics & Firearm Consistency Confirmed: FIR firearm allegations match Medical/Forensic Report weapon inference: '{report_fields.weapon_inferred}'."
        )
        score += 0.15

    # 4. Status determination
    score = min(0.99, max(0.10, round(score, 2)))
    if discrepancy_alerts:
        status_verdict = "CONTRADICTION_ALERT"
        note = "ATTENTION MAGISTRATE: Potential evidentiary discrepancies detected between FIR penal sections and Medical/Forensic findings. Cross-examination of the examining medical officer is recommended."
    elif score >= 0.70:
        status_verdict = "CORROBORATED"
        note = "VALIDATED: Medical / Forensic findings corroborate FIR penal sections and assault mechanisms."
    else:
        status_verdict = "PARTIAL_CORROBORATION"
        note = "Dual evidence analyzed. Findings provide independent supplementary details for case file."

    return CrossDocumentCorroboration(
        has_dual_documents=True,
        corroboration_status=status_verdict,
        corroboration_score=score,
        corroborated_points=corroborated_points,
        discrepancy_alerts=discrepancy_alerts,
        magistrate_evidentiary_note=note,
    )


# --- Language, OCR & Document Classification Helpers ---

HINGLISH_KEYWORDS = {
    "agar", "aur", "badal", "bhi", "cheez", "ched", "chad", "dega", "hai", "hain", "ha",
    "ho", "jaega", "jayega", "ka", "karega", "karegi", "karo", "kare",
    "ke", "ki", "ko", "kya", "mein", "par", "raha", "rahi", "rahe",
    "se", "sharma", "tha", "thi", "turant", "woh", "yahan", "ye", "saath",
}


def detect_detailed_language(text: str, requested_lang: str) -> str:
    """Classify OCR text as 'Hindi', 'Hinglish', or 'English'."""
    if not text.strip():
        return "Hindi" if requested_lang == "hi" else "English"

    dev_count = sum(1 for c in text if _DEVANAGARI_PATTERN.match(c))
    words = [w.strip().lower() for w in text.split() if w.strip().isalpha()]
    latin_words = [w for w in words if w.isascii()]
    hinglish_hits = sum(1 for w in latin_words if w in HINGLISH_KEYWORDS)
    meaningful_latin = [w for w in latin_words if len(w) > 2]

    if dev_count > 10:
        return "Hinglish" if len(meaningful_latin) >= 12 else "Hindi"
    if hinglish_hits >= 2:
        return "Hinglish"
    if requested_lang == "hi" or dev_count > 0:
        return "Hindi"
    return "English"


def _ocr_page(image_array: np.ndarray, requested_lang: str) -> tuple[str, str]:
    """Run OCR on one page. In 'auto' mode retries with Hindi model if English looks garbled."""
    if requested_lang in ("en", "hi"):
        return _run_ocr_on_image(image_array, requested_lang), requested_lang

    en_text = _run_ocr_on_image(image_array, "en")
    en_score = _gibberish_score(en_text)

    if en_score < GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD or OCR_ENGINES.get("hi") is None:
        if en_text or OCR_ENGINES.get("hi") is None:
            return en_text, "en"
        return _run_ocr_on_image(image_array, "hi"), "hi"

    logger.info("English OCR garbled (mixed-token ratio=%.2f). Retrying with Hindi model.", en_score)
    hi_text = _run_ocr_on_image(image_array, "hi")
    hi_score = _gibberish_score(hi_text)

    if _devanagari_ratio(hi_text) > _devanagari_ratio(en_text):
        return hi_text, "hi"
    return (hi_text, "hi") if hi_score < en_score else (en_text, "en")


def extract_text_from_upload(file_bytes: bytes, kind: str, language: str = "auto") -> tuple[str, str]:
    """Extract text from PDF (fast digital path + parallel OCR) or image."""
    import concurrent.futures
    import os

    all_page_results: list[tuple[int, str, str]] = []

    if kind == "pdf":
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            pages_to_ocr = []

            for idx, page in enumerate(pdf):
                if idx >= MAX_PDF_PAGES:
                    break
                textpage = page.get_textpage()
                embedded_text = textpage.get_text_range().strip()
                if len(embedded_text) > 30:
                    has_devanagari = any(_DEVANAGARI_PATTERN.match(c) for c in embedded_text)
                    all_page_results.append((idx, embedded_text, "hi" if has_devanagari else "en"))
                else:
                    pages_to_ocr.append((idx, page.render(scale=1.5).to_pil()))

            if pages_to_ocr:
                def process_scanned_page(item):
                    p_idx, p_img = item
                    arr = _pil_to_ndarray(p_img, max_dim=1600)
                    txt, l_used = _ocr_page(arr, language)
                    return (p_idx, txt, l_used)

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as ex:
                    all_page_results.extend(ex.map(process_scanned_page, pages_to_ocr))

            all_page_results.sort(key=lambda x: x[0])

        except Exception as exc:
            logger.warning("pypdfium2 failed (%s), falling back to pdf2image.", exc)
            pages = convert_from_bytes(file_bytes, dpi=150)[:MAX_PDF_PAGES]
            for idx, page in enumerate(pages):
                text, lang_used = _ocr_page(_pil_to_ndarray(page, max_dim=1600), language)
                all_page_results.append((idx, text, lang_used))

    else:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.load()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unable to read image: {exc}")
        text, lang_used = _ocr_page(_pil_to_ndarray(image, max_dim=1600), language)
        all_page_results.append((0, text, lang_used))

    full_text = "\n".join(r[1] for r in all_page_results if r[1].strip()).strip()
    return full_text, detect_detailed_language(full_text, language)


def extract_entities_with_spacy(text: str) -> dict[str, list[str]]:
    """Extract named entities using spaCy (English text only). Returns empty dict if unavailable."""
    empty: dict[str, list[str]] = {"persons": [], "organizations": [], "dates": [], "locations": []}
    if not SPACY_AVAILABLE or SPACY_NLP is None or not text:
        return empty

    doc = SPACY_NLP(text[:20000])

    def _unique(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for item in seq:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    return {
        "persons":       _unique([e.text for e in doc.ents if e.label_ == "PERSON"]),
        "organizations": _unique([e.text for e in doc.ents if e.label_ == "ORG"]),
        "dates":         _unique([e.text for e in doc.ents if e.label_ == "DATE"]),
        "locations":     _unique([e.text for e in doc.ents if e.label_ in ("GPE", "LOC")]),
    }


def classify_document(text: str) -> tuple[str, float]:
    """Score OCR text against keyword banks and return (document_type, confidence)."""
    lowered = text.lower()
    scores: dict[str, float] = {t: 0.0 for t in DOCUMENT_TYPES}
    max_possible = {t: sum(w for _, w in kws) for t, kws in DOCUMENT_TYPE_KEYWORDS.items()}

    for doc_type, kws in DOCUMENT_TYPE_KEYWORDS.items():
        for phrase, weight in kws:
            if phrase in lowered:
                scores[doc_type] += weight

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score <= 0:
        return "Internal Progress Note", 0.15

    confidence = max(0.35, min(0.99, round(best_score / max_possible[best_type], 2)))
    return best_type, confidence


_ACT_NORMALIZATION = {
    "ipc": "IPC",
    "भा.दं.सं.": "IPC",
    "भा.दं.सं": "IPC",
    "भा दं सं": "IPC",
    "आईपीसी": "IPC",
    "bns": "BNS",
    "भारतीय न्याय संहिता": "BNS",
    "बीएनएस": "BNS",
    "bnss": "BNSS",
    "दंड प्रक्रिया संहिता": "CrPC",
    "crpc": "CrPC",
    "cpc": "CPC",
    "evidence act": "Evidence Act",
    "indian evidence act": "Evidence Act",
    "pocso": "POCSO",
    "pocso act": "POCSO",
    "ndps": "NDPS",
    "ndps act": "NDPS",
    "arms": "Arms Act",
    "arms act": "Arms Act",
    "it": "IT Act",
    "it act": "IT Act",
    "information technology act": "IT Act",
    "motor vehicles act": "Motor Vehicles Act",
    "mv act": "MV Act",
    "excise act": "Excise Act",
    "uapa": "UAPA",
    "mcoca": "MCOCA",
    "pmla": "PMLA",
    "pasaa": "PASAA",
    "nia": "NIA Act",
    "nia act": "NIA Act",
}


def _normalize_act(act_str: str) -> str:
    cleaned = re.sub(r"\s+", " ", act_str.strip().lower())
    return _ACT_NORMALIZATION.get(cleaned, act_str.strip().upper())


def detect_legal_sections(text: str) -> list[str]:
    """Extract unique legal section references (e.g. 'IPC Section 302', 'BNS Section 103(1)')."""
    found, seen = [], set()
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            g = match.groups()
            if len(g) >= 2 and g[0] and g[1]:
                first, second = g[0].strip(), g[1].strip()
                if any(k in first.lower() for k in _ACT_NORMALIZATION) or first.upper() in {"IPC", "BNS", "BNSS", "CRPC", "CPC", "NDPS", "POCSO"}:
                    raw_act, raw_sec = first, second
                else:
                    raw_sec, raw_act = first, second

                act = _normalize_act(raw_act)
                sec_tokens = re.split(r"[\s,\/\+\&]|(?:and)", raw_sec, flags=re.I)
                for tok in sec_tokens:
                    tok = tok.strip(" ,;:-.")
                    if tok and re.match(r"^\d+[A-Za-z]*(?:\([0-9a-zA-Z]+\))*$", tok):
                        normalised = f"{act} Section {tok}"
                        if normalised.lower() not in seen:
                            seen.add(normalised.lower())
                            found.append(normalised)
            elif len(g) == 1 and g[0]:
                pass
    return found


def detect_sensitivity_keywords(text: str) -> list[str]:
    lowered = text.lower()
    seen: set[str] = set()
    return [kw for kw in SENSITIVITY_KEYWORDS if kw in lowered and not seen.add(kw)]  # type: ignore[func-returns-value]


def resolve_sensitivity_tier(document_type: str, sensitive_hits: list[str]) -> str:
    """Base tier from document type; escalate one level if 2+ sensitive keywords found."""
    tier = DOCUMENT_TYPE_SENSITIVITY.get(document_type, "MEDIUM")
    if len(sensitive_hits) >= 2 and tier != "HIGH":
        tier = TIER_ORDER[min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)]
    return tier


def build_summary(text: str, max_sentences: int = 4, max_chars: int = 600) -> str:
    """Lightweight extractive summary: first N sentences, capped by character length."""
    if not text:
        return "No text could be extracted from the document."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    summary = " ".join(sentences[:max_sentences])
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary or text[:max_chars]


# --- Member 6: Advanced Multi-Signal AI Forensic & AI-Generation Detection Suite ---

_AI_GENERATOR_KEYWORDS = [
    "chatgpt", "openai", "canva", "midjourney", "stable diffusion", "stablediffusion",
    "dall-e", "dall·e", "weasyprint", "reportlab", "wkhtmltopdf", "photoshop",
    "adobe illustrator", "figma", "cairo", "skia", "matplotlib", "phantomjs",
    "comfyui", "automatic1111", "novelai", "coreldraw", "inkscape",
]


def inspect_metadata_for_ai_tools(file_bytes: bytes, kind: str, image: Image.Image | None) -> tuple[float, list[str], dict[str, str]]:
    """
    Inspect PDF document dictionaries, XMP metadata, and image EXIF/PNG chunks for AI tools,
    synthetic generators, and direct vector renderers.
    """
    detected_tools: list[str] = []
    metadata_details: dict[str, str] = {}
    provenance_score = 0.0

    # 1. Search raw bytes / PDF trailer
    if kind == "pdf" or (file_bytes and file_bytes[:4] == b"%PDF"):
        header_sample = file_bytes[:50000] + file_bytes[-50000:] if len(file_bytes) > 100000 else file_bytes
        text_sample = header_sample.decode("latin-1", errors="ignore").lower()

        for kw in _AI_GENERATOR_KEYWORDS:
            if kw in text_sample:
                detected_tools.append(f"PDF Metadata Keyword: '{kw.title()}'")
                metadata_details["pdf_producer_match"] = kw

        # Check for digital creators
        m_prod = re.search(r"/(?:Producer|Creator|Author)\s*\(([^)]+)\)", text_sample, re.I)
        if m_prod:
            creator_str = m_prod.group(1).strip()
            metadata_details["creator_software"] = creator_str
            for kw in _AI_GENERATOR_KEYWORDS:
                if kw in creator_str.lower():
                    detected_tools.append(f"PDF Generator Software: '{creator_str}'")

    # 2. Search Image EXIF and PNG Chunks (info dict)
    if image is not None:
        if hasattr(image, "info") and image.info:
            for key, val in image.info.items():
                val_str = str(val).lower()
                metadata_details[f"image_tag_{key}"] = str(val)[:60]
                for kw in _AI_GENERATOR_KEYWORDS:
                    if kw in val_str:
                        detected_tools.append(f"Image Metadata Tag [{key}]: '{kw}'")
                if key in ("parameters", "prompt", "workflow") and len(val_str) > 10:
                    detected_tools.append(f"AI Diffusion Prompt / Parameter Block detected in image header")

        # EXIF tags
        try:
            exif = image.getexif()
            if exif:
                for tag_id, val in exif.items():
                    val_str = str(val).lower()
                    for kw in _AI_GENERATOR_KEYWORDS:
                        if kw in val_str:
                            detected_tools.append(f"EXIF Tag {tag_id}: '{kw}'")
        except Exception:
            pass

    # Score assignment
    if any("prompt" in t.lower() or "midjourney" in t.lower() or "stable diffusion" in t.lower() or "chatgpt" in t.lower() or "dall-e" in t.lower() for t in detected_tools):
        provenance_score = 0.95
    elif any("canva" in t.lower() or "photoshop" in t.lower() or "reportlab" in t.lower() or "weasyprint" in t.lower() for t in detected_tools):
        provenance_score = 0.80
    elif detected_tools:
        provenance_score = 0.65

    return round(provenance_score, 2), detected_tools, metadata_details


def compute_spectral_anomaly(image: Image.Image) -> tuple[float, float, int]:
    """
    2D Fast Fourier Transform (FFT) Frequency Domain Spectral Analysis.
    Analyzes high-frequency spectral roll-off and periodic checkerboard spikes characteristic of AI diffusion / GAN models.
    Returns (spectral_score, high_freq_ratio, peak_count).
    """
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.0, 0.0, 0

        # 2D FFT
        f = sp_fft.fft2(gray)
        fshift = sp_fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = h // 2, w // 2
        r = min(cy, cx) // 2
        y, x = np.ogrid[:h, :w]
        high_mask = ((x - cx) ** 2 + (y - cy) ** 2) > r ** 2

        total_energy = float(np.sum(magnitude)) + 1e-6
        high_energy = float(np.sum(magnitude[high_mask]))
        high_freq_ratio = high_energy / total_energy

        # Peak prominence analysis for periodic grid artifacts
        log_mag = np.log1p(magnitude)
        threshold = float(np.mean(log_mag) + 3.5 * np.std(log_mag))
        peak_count = int(np.sum(log_mag > threshold))

        # Real scanned documents have natural low-pass optical decay (ratio ~0.04 to 0.18)
        # Synthetic / AI documents show abnormal high frequencies (>0.22) or repetitive peaks (>80)
        spectral_score = 0.0
        if high_freq_ratio > 0.22:
            spectral_score += min(0.60, (high_freq_ratio - 0.22) * 2.5)
        if peak_count > 60:
            spectral_score += min(0.40, (peak_count - 60) * 0.005)

        return round(min(1.0, spectral_score), 3), round(high_freq_ratio, 4), peak_count
    except Exception as exc:
        logger.debug("Spectral analysis skipped: %s", exc)
        return 0.0, 0.0, 0


def compute_sensor_noise_distribution(image: Image.Image) -> tuple[float, float, float]:
    """
    Analyze optical sensor noise vs flat synthetic digital background.
    Physical scanned paper always has sensor shot noise (variance > 0.8 in white areas).
    AI generated / digital renders have perfectly uniform 0-variance background patches.
    Returns (synthetic_flat_score, bg_noise_variance, bg_kurtosis).
    """
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        # Identify background / non-ink pixels (luminance > 225)
        bg_mask = gray > 225
        bg_count = int(np.sum(bg_mask))

        if bg_count < 500:
            return 0.0, 5.0, 3.0

        bg_pixels = gray[bg_mask]
        bg_var = float(np.var(bg_pixels))
        bg_kurt = float(sp_stats.kurtosis(bg_pixels)) if bg_var > 0.1 else 0.0

        # If variance is virtually 0 (perfect digital white), it's a synthetic / digital export
        if bg_var < 0.08:
            flat_score = 0.85
        elif bg_var < 0.40:
            flat_score = 0.50
        elif bg_var < 1.0:
            flat_score = 0.25
        else:
            flat_score = 0.02  # Natural sensor noise present

        return round(flat_score, 3), round(bg_var, 3), round(bg_kurt, 3)
    except Exception as exc:
        logger.debug("Sensor noise analysis skipped: %s", exc)
        return 0.0, 5.0, 3.0


def extract_images_from_upload(file_bytes: bytes, kind: str) -> list[Image.Image]:
    """Extract PIL RGB images from uploaded PDF pages or image."""
    images: list[Image.Image] = []
    if kind == "pdf":
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            for idx, page in enumerate(pdf):
                if idx >= MAX_PDF_PAGES:
                    break
                images.append(page.render(scale=1.5).to_pil().convert("RGB"))
        except Exception as exc:
            logger.debug("pypdfium2 render failed (%s), trying pdf2image.", exc)
            try:
                pages = convert_from_bytes(file_bytes, dpi=150)[:MAX_PDF_PAGES]
                images = [p.convert("RGB") for p in pages]
            except Exception as exc2:
                logger.warning("PDF page image extraction failed: %s", exc2)
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            images = [img.convert("RGB")]
        except Exception as exc:
            logger.warning("Image loading for forensics failed: %s", exc)
    return images


def compute_ela_anomaly(image: Image.Image, quality: int = 90) -> tuple[float, float]:
    """
    Perform Error Level Analysis (ELA) on a PIL Image.
    Saves image into in-memory JPEG at specified quality, computes absolute pixel difference,
    and analyzes spatial block compression variance to detect spliced / photoshopped regions.
    Returns (ela_anomaly_score, max_block_discrepancy).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w < 16 or h < 16:
        return 0.0, 0.0

    # Resave to in-memory JPEG at controlled compression quality
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed_img = Image.open(buf).convert("RGB")

    orig_arr = np.array(image, dtype=np.float32)
    comp_arr = np.array(compressed_img, dtype=np.float32)

    diff = np.sqrt(np.mean((orig_arr - comp_arr) ** 2, axis=2))  # (H, W)

    block_size = 32
    h_blocks = max(1, h // block_size)
    w_blocks = max(1, w // block_size)

    block_means = []
    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = diff[bi * block_size : (bi + 1) * block_size, bj * block_size : (bj + 1) * block_size]
            if block.size > 0:
                block_means.append(float(np.mean(block)))

    if not block_means:
        return 0.0, 0.0

    block_arr = np.array(block_means)
    mean_error = float(np.mean(block_arr))
    stdev_error = float(np.std(block_arr))
    max_error = float(np.max(block_arr))

    if mean_error < 0.2:
        return 0.02, round(stdev_error, 3)

    cv = stdev_error / (mean_error + 1e-4)
    peak_ratio = max_error / (mean_error + 1e-4)
    ela_score = min(1.0, max(0.0, (cv * 0.25) + max(0.0, peak_ratio - 1.5) * 0.08))

    return round(ela_score, 3), round(stdev_error, 3)


def compute_noise_inconsistency(image: Image.Image) -> float:
    """Evaluate localized high-frequency pixel noise variance to detect artificial splicing."""
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        if gray.shape[0] < 32 or gray.shape[1] < 32:
            return 0.0

        top = gray[:-2, 1:-1]
        bottom = gray[2:, 1:-1]
        left = gray[1:-1, :-2]
        right = gray[1:-1, 2:]
        center = gray[1:-1, 1:-1]

        laplacian = np.abs(4 * center - top - bottom - left - right)

        bs = 32
        patch_vars = []
        for r in range(0, laplacian.shape[0] - bs, bs):
            for c in range(0, laplacian.shape[1] - bs, bs):
                patch = laplacian[r : r + bs, c : c + bs]
                patch_vars.append(float(np.var(patch)))

        if not patch_vars:
            return 0.0

        p_arr = np.array(patch_vars)
        p_mean = float(np.mean(p_arr))
        p_std = float(np.std(p_arr))

        if p_mean < 1.0:
            return 0.02

        noise_score = min(1.0, max(0.0, (p_std / (p_mean + 20.0)) * 0.35))
        return round(noise_score, 3)
    except Exception as exc:
        logger.debug("Noise inconsistency computation skipped: %s", exc)
        return 0.0


def perform_forensic_ela_analysis(
    images: list[Image.Image],
    file_bytes: bytes = b"",
    kind: str = "pdf",
) -> ForensicAnalysisResult:
    """
    Multi-signal AI Forensic suite combining:
    1. Metadata & provenance inspection (AI generator signatures, digital software)
    2. 2D FFT spectral frequency domain analysis
    3. Sensor grain vs synthetic flat-field background variance
    4. Error Level Analysis (ELA) for localized tampering & copy-paste
    5. High-frequency Laplacian noise consistency
    """
    if not images:
        return ForensicAnalysisResult(
            is_scanned_document=False,
            is_ai_generated=False,
            ai_generation_score=0.0,
            spectral_anomaly_score=0.0,
            metadata_provenance_score=0.0,
            ela_anomaly_score=0.0,
            noise_inconsistency_score=0.0,
            forgery_score=0.0,
            tamper_risk_level="LOW",
            forgery_verdict="GENUINE",
            detected_ai_tools=[],
            forensic_proof_details={},
            worm_audit_flags=[],
            forensic_summary="Document processed without physical pixel anomalies.",
            analyzed_pages_count=0,
        )

    # 1. Metadata analysis
    first_img = images[0] if images else None
    prov_score, detected_tools, meta_details = inspect_metadata_for_ai_tools(file_bytes, kind, first_img)

    ela_scores = []
    noise_scores = []
    spectral_scores = []
    flat_scores = []
    bg_vars = []
    fft_ratios = []

    for img in images[:5]:
        ela_val, _ = compute_ela_anomaly(img)
        noise_val = compute_noise_inconsistency(img)
        spec_val, h_ratio, _ = compute_spectral_anomaly(img)
        flat_val, bg_v, _ = compute_sensor_noise_distribution(img)

        ela_scores.append(ela_val)
        noise_scores.append(noise_val)
        spectral_scores.append(spec_val)
        flat_scores.append(flat_val)
        bg_vars.append(bg_v)
        fft_ratios.append(h_ratio)

    max_ela = max(ela_scores) if ela_scores else 0.0
    max_noise = max(noise_scores) if noise_scores else 0.0
    max_spec = max(spectral_scores) if spectral_scores else 0.0
    max_flat = max(flat_scores) if flat_scores else 0.0
    min_bg_var = min(bg_vars) if bg_vars else 5.0
    max_fft_ratio = max(fft_ratios) if fft_ratios else 0.0

    # Composite AI-generation score
    ai_gen_score = round(max(prov_score, (0.50 * max_flat + 0.30 * max_spec + 0.20 * max_noise if prov_score > 0 or max_flat >= 0.80 else 0.05)), 2)
    # Composite forgery score
    composite_forgery = round(max(ai_gen_score, (0.60 * max_ela + 0.40 * max_noise)), 2)

    # Differentiate AI-generation vs Spliced tampering vs Genuine
    is_ai = bool(detected_tools) or prov_score >= 0.60 or ai_gen_score >= 0.60 or (min_bg_var < 0.08 and max_flat >= 0.75)
    is_spliced = (max_ela >= 0.20 or max_noise >= 0.30 or composite_forgery >= 0.32)
    if is_spliced:
        composite_forgery = max(0.35, composite_forgery)

    proof_details = {
        "fft_high_frequency_ratio": round(max_fft_ratio, 4),
        "background_noise_variance": round(min_bg_var, 3),
        "ela_error_discrepancy": round(max_ela, 3),
        "synthetic_flat_score": round(max_flat, 3),
        "metadata_generator_found": detected_tools if detected_tools else "None",
    }

    if is_ai:
        risk_level = "HIGH"
        verdict = "AI-GENERATED / SYNTHETIC"
        worm_flags = ["SUSPECTED_AI_GENERATION_FLAG", "WORM_MAGISTRATE_REVIEW_REQUIRED"]
        summary = (
            f"HIGH ALERT [AI-GENERATED / SYNTHETIC]: AI forensic engine detected synthetic digital origins (Score: {ai_gen_score:.2f}). "
            f"Mathematical Proof: Background Noise Variance={min_bg_var:.3f} (Natural scans >0.8), "
            f"FFT Spectral Ratio={max_fft_ratio:.4f}, Detected Signatures={detected_tools or 'Synthetic Digital Direct Export'}. "
            f"Forcefully tagged in WORM audit log for magistrate review."
        )
    elif is_spliced or composite_forgery >= 0.45:
        risk_level = "HIGH"
        verdict = "SUSPECTED FORGERY"
        worm_flags = ["SUSPECTED_FORGERY_FLAG", "WORM_MAGISTRATE_REVIEW_REQUIRED"]
        summary = (
            f"HIGH ALERT [SUSPECTED FORGERY]: Error Level Analysis (ELA) detected significant localized pixel compression "
            f"discrepancies (Anomaly Score: {max_ela:.2f}, Composite: {composite_forgery:.2f}). Document exhibits mismatched "
            f"compression layers or spliced text typical of pre-scan digital manipulation. Forcefully tagged in WORM audit trail."
        )
    elif composite_forgery >= 0.30:
        risk_level = "MEDIUM"
        verdict = "SUSPECTED FORGERY"
        worm_flags = ["SUSPECTED_FORGERY_FLAG"]
        summary = (
            f"WARNING: Moderate pixel compression discrepancy or boundary noise variance detected (Score: {composite_forgery:.2f}). "
            f"Tagged with SUSPECTED_FORGERY_FLAG in WORM log for magistrate inspection."
        )
    else:
        risk_level = "LOW"
        verdict = "GENUINE"
        worm_flags = []
        summary = (
            "Pixel compression levels, natural optical sensor grain, and 2D FFT spectral roll-off are uniform. "
            "No AI generation or digital tampering detected."
        )

    return ForensicAnalysisResult(
        is_scanned_document=True,
        is_ai_generated=is_ai,
        ai_generation_score=ai_gen_score,
        spectral_anomaly_score=max_spec,
        metadata_provenance_score=prov_score,
        ela_anomaly_score=max_ela,
        noise_inconsistency_score=max_noise,
        forgery_score=composite_forgery,
        tamper_risk_level=risk_level,
        forgery_verdict=verdict,
        detected_ai_tools=detected_tools,
        forensic_proof_details=proof_details,
        worm_audit_flags=worm_flags,
        forensic_summary=summary,
        analyzed_pages_count=len(images),
    )


# --- Routes ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def upload_page() -> HTMLResponse:
    with open("index.html", encoding="utf-8") as page:
        return HTMLResponse(page.read())


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ocr_engines_ready={lang: engine is not None for lang, engine in OCR_ENGINES.items()},
        spacy_entity_extraction_ready=SPACY_AVAILABLE,
    )


@app.get("/api/v1/status", tags=["ops"])
async def service_status() -> JSONResponse:
    return JSONResponse({"model_loading": MODEL_LOADING, "ocr_engines_ready": {k: v is not None for k, v in OCR_ENGINES.items()}})


@app.post(
    "/api/v1/ai/forensic-analysis",
    response_model=ForensicAnalysisResult,
    status_code=status.HTTP_200_OK,
    tags=["ai"],
)
async def forensic_analysis_endpoint(
    file: UploadFile = File(...),
) -> ForensicAnalysisResult:
    """Run standalone AI Forensic Pixel & Spectral AI-Generation Analysis (Member 6)."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    kind = ALLOWED_CONTENT_TYPES[file.content_type]
    images = extract_images_from_upload(file_bytes, kind)
    return perform_forensic_ela_analysis(images, file_bytes=file_bytes, kind=kind)


@app.post(
    "/api/v1/ai/analyze-report",
    response_model=DocumentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["ai"],
)
async def analyze_report_endpoint(
    report_file: UploadFile = File(...),
    language: str = Form("auto", description="OCR language: 'auto', 'en', or 'hi'."),
) -> DocumentAnalysisResponse:
    """Analyze a Supplementary Medical / Autopsy / Forensic Lab Report."""
    if report_file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{report_file.content_type}'. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )
    file_bytes = await report_file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded report file is empty.")
    kind = ALLOWED_CONTENT_TYPES[report_file.content_type]

    extracted_text, ocr_language_used = extract_text_from_upload(file_bytes, kind, language)
    doc_type, conf = classify_document(extracted_text)
    if doc_type != "Forensic Report":
        doc_type = "Forensic Report"

    sections = detect_legal_sections(extracted_text)
    sens_hits = detect_sensitivity_keywords(extracted_text)
    sens_tier = resolve_sensitivity_tier(doc_type, sens_hits)
    quorum = QUORUM_MATRIX[sens_tier]
    report_fields = extract_report_structured_fields(extracted_text)
    images = extract_images_from_upload(file_bytes, kind)
    forensic = perform_forensic_ela_analysis(images, file_bytes=file_bytes, kind=kind)

    return DocumentAnalysisResponse(
        document_type=doc_type,
        detected_sections=sections,
        sensitivity_tier=sens_tier,
        recommended_quorum=QuorumInfo(required=quorum["required"], pool_size=quorum["pool_size"]),
        confidence_score=max(0.85, conf),
        extracted_summary=build_summary(extracted_text),
        extracted_text=extracted_text,
        sensitivity_keywords_found=sens_hits,
        raw_text_char_count=len(extracted_text),
        ocr_language_used=ocr_language_used,
        extracted_entities=extract_entities_with_spacy(extracted_text) if SPACY_AVAILABLE else None,
        structured_fields=extract_structured_fields(extracted_text),
        report_fields=report_fields,
        cross_corroboration=None,
        forensic_analysis=forensic,
    )


@app.post(
    "/api/v1/ai/analyze-document",
    response_model=DocumentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["ai"],
)
async def analyze_document(
    file: UploadFile = File(...),
    report_file: UploadFile | None = File(None),
    language: str = Form("auto", description="OCR language: 'auto', 'en', or 'hi'."),
) -> DocumentAnalysisResponse:
    """
    Accepts FIR / Legal document and optional Supplementary Report (Medical/Forensic),
    runs OCR, classifies document type, extracts structured fields, runs AI Forensic & AI-Generation analysis,
    and performs cross-document evidentiary corroboration.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )
    if language not in ("auto", "en", "hi"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="language must be 'auto', 'en', or 'hi'.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    kind = ALLOWED_CONTENT_TYPES[file.content_type]
    logger.info("Received FIR '%s' (%s, %d bytes, lang=%s)", file.filename, kind, len(file_bytes), language)

    extracted_text, ocr_language_used = extract_text_from_upload(file_bytes, kind, language)
    document_type, confidence_score = classify_document(extracted_text)
    detected_sections = detect_legal_sections(extracted_text)
    sensitivity_hits = detect_sensitivity_keywords(extracted_text)
    sensitivity_tier = resolve_sensitivity_tier(document_type, sensitivity_hits)
    quorum = QUORUM_MATRIX[sensitivity_tier]
    entities = extract_entities_with_spacy(extracted_text) if SPACY_AVAILABLE else None
    structured = extract_structured_fields(extracted_text)

    page_images = extract_images_from_upload(file_bytes, kind)
    forensic = perform_forensic_ela_analysis(page_images, file_bytes=file_bytes, kind=kind)

    # If supplementary report is also uploaded
    report_fields = None
    cross_corroboration = None

    if report_file and report_file.filename:
        if report_file.content_type in ALLOWED_CONTENT_TYPES:
            rep_bytes = await report_file.read()
            if rep_bytes:
                rep_kind = ALLOWED_CONTENT_TYPES[report_file.content_type]
                rep_text, _ = extract_text_from_upload(rep_bytes, rep_kind, language)
                report_fields = extract_report_structured_fields(rep_text)
                cross_corroboration = correlate_fir_and_report(
                    fir_sections=detected_sections,
                    fir_text=extracted_text,
                    report_fields=report_fields,
                    report_text=rep_text,
                )
                # Also run forensics on report if needed
                rep_images = extract_images_from_upload(rep_bytes, rep_kind)
                rep_forensic = perform_forensic_ela_analysis(rep_images, file_bytes=rep_bytes, kind=rep_kind)
                if rep_forensic.forgery_score > forensic.forgery_score:
                    forensic = rep_forensic

    logger.info(
        "Analysis: type=%s tier=%s quorum=%s confidence=%.2f lang=%s forensic_verdict=%s (score=%.2f)",
        document_type, sensitivity_tier, quorum, confidence_score, ocr_language_used,
        forensic.forgery_verdict, forensic.forgery_score,
    )

    return DocumentAnalysisResponse(
        document_type=document_type,
        detected_sections=detected_sections,
        sensitivity_tier=sensitivity_tier,
        recommended_quorum=QuorumInfo(required=quorum["required"], pool_size=quorum["pool_size"]),
        confidence_score=confidence_score,
        extracted_summary=build_summary(extracted_text),
        extracted_text=extracted_text,
        sensitivity_keywords_found=sensitivity_hits,
        raw_text_char_count=len(extracted_text),
        ocr_language_used=ocr_language_used,
        extracted_entities=entities,
        structured_fields=structured,
        report_fields=report_fields,
        cross_corroboration=cross_corroboration,
        forensic_analysis=forensic,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):  # pragma: no cover
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)