# -*- coding: utf-8 -*-
# SecureChain DMS - Integration Test Suite
#
# HOW TO RUN
# ----------
# From the ai-service/ directory, with dependencies installed
# (pip install -r requirements.txt):
#
#     pytest tests/ -v \
#         --cov=. \
#         --cov-report=term-missing \
#         --cov-report=xml:coverage.xml \
#         --junitxml=report.xml
#
# This mirrors exactly what the GitLab CI test:pytest job runs.
# Quick run without coverage:
#     pytest tests/ -v
#
# SCOPE NOTE
# ----------
# Test 1  — AI classification (OCR layer is mocked for speed).
# Test 2  — Self-approval / maker-checker rule.
# Test 3  — Instant-flagging rule.
# Test 4  — 3-of-5 quorum state machine.
# Test 5  — spaCy entity extraction (Member 6 AI layer).

from __future__ import annotations

import io
import struct
import zlib
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main as ai_main
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures: dynamically generated dummy files (no external static assets)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_png_bytes() -> bytes:
    """
    Generate a minimal, valid 1x1 white PNG entirely in-memory so the suite
    has zero dependency on files checked into the repo.
    """
    width, height = 1, 1

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw_row = b"\x00" + b"\xff\xff\xff"  # filter byte + 1 white RGB pixel
    idat = zlib.compress(raw_row)
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return png


@pytest.fixture
def tiny_pdf_bytes() -> bytes:
    """
    Generate a minimal, syntactically valid single-page PDF entirely
    in-memory (no external static fixture files required).
    """
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000052 00000 n \n"
        b"0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n164\n"
        b"%%EOF"
    )
    return pdf


@pytest.fixture
def mock_ocr_text():
    """
    Factory fixture: patches the OCR extraction layer to return fixed text.

    `extract_text_from_upload` returns a (text, language_used) tuple in the
    real implementation, so the mock mirrors that contract.
    """

    def _apply(text: str, language_used: str = "en"):
        return patch.object(
            ai_main, "extract_text_from_upload", return_value=(text, language_used)
        )

    return _apply


# ---------------------------------------------------------------------------
# TEST 1 — AI Classification Validation
# ---------------------------------------------------------------------------

class TestAIClassification:
    def test_high_sensitivity_chargesheet_forensic_bns(self, mock_ocr_text, tiny_pdf_bytes):
        """
        A Chargesheet referencing a BNS section plus forensic ballistics
        content must be classified HIGH sensitivity with a 3-of-5 quorum.
        """
        ocr_text = (
            "CHARGESHEET\n"
            "Final Report under Section 173 CrPC\n"
            "The accused is charged under BNS Section 103 for the offence "
            "of murder. Attached herewith is the Forensic Science Laboratory "
            "Ballistics Report confirming the recovered weapon matches the "
            "cartridge cases recovered from the scene."
        )

        with mock_ocr_text(ocr_text):
            response = client.post(
                "/api/v1/ai/analyze-document",
                files={"file": ("chargesheet.pdf", tiny_pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 200
        body = response.json()

        assert body["sensitivity_tier"] == "HIGH"
        assert body["recommended_quorum"]["required"] == 3
        assert body["recommended_quorum"]["pool_size"] == 5
        assert body["document_type"] in {"Chargesheet", "Forensic Report"}
        assert "BNS Section 103" in body["detected_sections"]
        assert body["confidence_score"] > 0.0

    def test_low_sensitivity_internal_note(self, mock_ocr_text, tiny_png_bytes):
        """An internal dispatch/progress note must resolve to LOW / 1-of-1."""
        ocr_text = "INTERNAL PROGRESS NOTE\nFor internal circulation only. Dispatch slip attached."

        with mock_ocr_text(ocr_text):
            response = client.post(
                "/api/v1/ai/analyze-document",
                files={"file": ("note.png", tiny_png_bytes, "image/png")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["sensitivity_tier"] == "LOW"
        assert body["recommended_quorum"] == {"required": 1, "pool_size": 1}

    def test_medium_sensitivity_witness_statement(self, mock_ocr_text, tiny_png_bytes):
        """A witness statement must resolve to MEDIUM / 2-of-3."""
        ocr_text = "STATEMENT OF WITNESS recorded under Section 161 CrPC."

        with mock_ocr_text(ocr_text):
            response = client.post(
                "/api/v1/ai/analyze-document",
                files={"file": ("witness.png", tiny_png_bytes, "image/png")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["sensitivity_tier"] == "MEDIUM"
        assert body["recommended_quorum"] == {"required": 2, "pool_size": 3}

    def test_rejects_unsupported_content_type(self, tiny_png_bytes):
        response = client.post(
            "/api/v1/ai/analyze-document",
            files={"file": ("note.txt", b"plain text", "text/plain")},
        )
        assert response.status_code == 415

    def test_rejects_empty_file(self):
        response = client.post(
            "/api/v1/ai/analyze-document",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400

    def test_rejects_invalid_language_param(self, tiny_png_bytes):
        response = client.post(
            "/api/v1/ai/analyze-document",
            files={"file": ("note.png", tiny_png_bytes, "image/png")},
            data={"language": "fr"},
        )
        assert response.status_code == 400


class TestLanguageMismatchDetection:
    """
    Regression coverage for the Devanagari-OCR'd-as-English bug: PaddleOCR
    forces non-Latin glyphs onto Latin letters/digits when the wrong script
    model is used, producing tokens like '3Tcc19T'. `_gibberish_score` is
    the heuristic that flags this so `_ocr_page` can retry with the Hindi
    model in 'auto' mode.
    """

    def test_gibberish_score_flags_mixed_devanagari_like_tokens(self):
        garbled = "3Tcc19T a 3y yeyIT ach: 3fg 3R agc1ld aT 3h2 9ec1T HTdT 3Tchc1l91"
        score = ai_main._gibberish_score(garbled)
        assert score >= ai_main.GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD

    def test_gibberish_score_low_for_normal_english(self):
        clean = "The chargesheet was filed under Section 302 IPC by the investigating officer."
        score = ai_main._gibberish_score(clean)
        assert score < ai_main.GIBBERISH_MIXED_TOKEN_RATIO_THRESHOLD

    def test_auto_mode_tries_hindi_when_english_returns_no_text(self):
        clean_hi_reading = "अपराध की सूचना धारा 302 के तहत दर्ज की गई"

        with patch.object(ai_main, "_run_ocr_on_image") as mock_run_ocr:
            def _side_effect(image_array, lang):
                return "" if lang == "en" else clean_hi_reading

            mock_run_ocr.side_effect = _side_effect
            with patch.dict(ai_main.OCR_ENGINES, {"hi": object()}):
                text, lang_used = ai_main._ocr_page(
                    np.zeros((10, 10, 3), dtype="uint8"), "auto"
                )

        assert lang_used == "hi"
        assert text == clean_hi_reading

    def test_auto_mode_retries_with_hindi_model_on_garbled_english_pass(self):
        """
        When the English pass looks like garbled cross-script noise, auto
        mode must retry with the Hindi engine and prefer its output if it
        scores cleaner.
        """
        garbled_en = "3Tcc19T a 3y yeyIT ach 3fg 3R agc1ld aT 3h2 9ec1T HTdT"
        clean_hi_reading = "अपराध की सूचना धारा 302 के तहत दर्ज की गई"

        with patch.object(ai_main, "_run_ocr_on_image") as mock_run_ocr:
            def _side_effect(image_array, lang):
                return garbled_en if lang == "en" else clean_hi_reading

            mock_run_ocr.side_effect = _side_effect
            # Pretend the Hindi engine is loaded so auto-fallback is attempted.
            with patch.dict(ai_main.OCR_ENGINES, {"hi": object()}):
                text, lang_used = ai_main._ocr_page(np.zeros((10, 10, 3), dtype="uint8"), "auto")

        assert lang_used == "hi"
        assert text == clean_hi_reading

    def test_pinned_language_skips_auto_detection(self):
        """If the caller pins language='en', no fallback/retry should occur."""
        with patch.object(ai_main, "_run_ocr_on_image", return_value="some text") as mock_run_ocr:
            text, lang_used = ai_main._ocr_page(np.zeros((10, 10, 3), dtype="uint8"), "en")

        assert lang_used == "en"
        mock_run_ocr.assert_called_once_with(mock_run_ocr.call_args[0][0], "en")


# ---------------------------------------------------------------------------
# Reference implementation of the maker-checker / quorum state machine.
# Encodes the SIH26190 governance rules under test in cases 2-4.
# ---------------------------------------------------------------------------

class WorkflowError(Exception):
    pass


class ApprovalWorkflow:
    """
    Minimal reference state machine for the document edit-approval workflow.

    States: ACTIVE -> PENDING QUORUM -> APPROVED - v{n}
    """

    def __init__(self, document_id: str, base_version: str, quorum_required: int, requester_id: str):
        self.document_id = document_id
        self.original_hash = f"hash::{document_id}::{base_version}"
        self.current_version = base_version
        self.quorum_required = quorum_required
        self.requester_id = requester_id
        self.state = "ACTIVE"
        self.approvals: List[str] = []
        self._edit_initiated = False

    def initiate_edit(self) -> None:
        """Instant-flagging rule: initiating an edit immediately moves the
        document to PENDING QUORUM without mutating the original hash."""
        self._edit_initiated = True
        self.state = "PENDING QUORUM"
        # original_hash and current_version are intentionally untouched here;
        # they only change once quorum is met (see _finalize_approval).

    def submit_approval(self, approver_id: str) -> None:
        if not self._edit_initiated:
            raise WorkflowError("No edit is currently pending approval.")

        # Maker-Checker rule: the requester may never approve their own edit.
        if approver_id == self.requester_id:
            raise WorkflowError("SELF_APPROVAL_FORBIDDEN")

        if approver_id in self.approvals:
            raise WorkflowError("DUPLICATE_APPROVAL_FORBIDDEN")

        self.approvals.append(approver_id)

        if len(self.approvals) >= self.quorum_required:
            self._finalize_approval()

    def _finalize_approval(self) -> None:
        major, minor = self.current_version.lstrip("v").split(".")
        new_version = f"v{major}.{int(minor) + 1}"
        self.current_version = new_version
        self.state = f"APPROVED - {new_version}"


# ---------------------------------------------------------------------------
# TEST 2 — Self-Approval Block (Maker-Checker Rule)
# ---------------------------------------------------------------------------

class TestMakerCheckerRule:
    def test_requester_cannot_approve_own_edit(self):
        workflow = ApprovalWorkflow(
            document_id="DOC-001",
            base_version="v1.0",
            quorum_required=3,
            requester_id="officer_raj",
        )
        workflow.initiate_edit()

        with pytest.raises(WorkflowError) as exc_info:
            workflow.submit_approval("officer_raj")

        assert "SELF_APPROVAL_FORBIDDEN" in str(exc_info.value)

    def test_self_approval_block_via_api_contract(self):
        """
        Contract-level assertion for the HTTP layer: the backend must map a
        self-approval attempt to HTTP 403. Modelled here as the expected
        status code the Node.js `/api/v1/edits/{id}/approve` endpoint must
        return, verified against the reference state machine's error code.
        """
        workflow = ApprovalWorkflow("DOC-002", "v1.0", quorum_required=3, requester_id="officer_priya")
        workflow.initiate_edit()

        def approve_endpoint(approver_id: str) -> int:
            try:
                workflow.submit_approval(approver_id)
                return 200
            except WorkflowError as e:
                return 403 if "SELF_APPROVAL_FORBIDDEN" in str(e) else 400

        status_code = approve_endpoint("officer_priya")
        assert status_code == 403


# ---------------------------------------------------------------------------
# TEST 3 — Instant-Flagging Rule
# ---------------------------------------------------------------------------

class TestInstantFlaggingRule:
    def test_edit_initiation_flags_pending_without_altering_hash(self):
        workflow = ApprovalWorkflow(
            document_id="DOC-003",
            base_version="v1.0",
            quorum_required=3,
            requester_id="officer_raj",
        )
        original_hash_before = workflow.original_hash
        original_version_before = workflow.current_version

        assert workflow.state == "ACTIVE"

        workflow.initiate_edit()

        assert workflow.state == "PENDING QUORUM"
        # The original v1.0 hash must remain untouched until quorum is met.
        assert workflow.original_hash == original_hash_before
        assert workflow.current_version == original_version_before == "v1.0"


# ---------------------------------------------------------------------------
# TEST 4 — 3-of-5 Quorum Threshold State Machine
# ---------------------------------------------------------------------------

class TestQuorumThreshold:
    def test_two_approvals_remain_pending(self):
        workflow = ApprovalWorkflow("DOC-004", "v1.0", quorum_required=3, requester_id="officer_raj")
        workflow.initiate_edit()

        workflow.submit_approval("approver_1")
        workflow.submit_approval("approver_2")

        assert workflow.state == "PENDING QUORUM"
        assert workflow.current_version == "v1.0"

    def test_third_independent_approval_transitions_to_approved(self):
        workflow = ApprovalWorkflow("DOC-004", "v1.0", quorum_required=3, requester_id="officer_raj")
        workflow.initiate_edit()

        workflow.submit_approval("approver_1")
        workflow.submit_approval("approver_2")
        assert workflow.state == "PENDING QUORUM"

        workflow.submit_approval("approver_3")

        assert workflow.state == "APPROVED - v1.1"
        assert workflow.current_version == "v1.1"

    def test_duplicate_approver_is_rejected(self):
        workflow = ApprovalWorkflow("DOC-005", "v1.0", quorum_required=3, requester_id="officer_raj")
        workflow.initiate_edit()
        workflow.submit_approval("approver_1")

        with pytest.raises(WorkflowError) as exc_info:
            workflow.submit_approval("approver_1")

        assert "DUPLICATE_APPROVAL_FORBIDDEN" in str(exc_info.value)
        assert workflow.state == "PENDING QUORUM"

    def test_quorum_matches_high_sensitivity_matrix_from_ai_service(self):
        """
        Cross-check: the quorum_required used by the workflow for a
        HIGH-sensitivity document must equal the value the AI service
        recommends (3-of-5), keeping both layers of the system consistent.
        """
        assert ai_main.QUORUM_MATRIX["HIGH"] == {"required": 3, "pool_size": 5}
        assert ai_main.QUORUM_MATRIX["MEDIUM"] == {"required": 2, "pool_size": 3}
        assert ai_main.QUORUM_MATRIX["LOW"] == {"required": 1, "pool_size": 1}


# ---------------------------------------------------------------------------
# TEST 5 — spaCy Entity Extraction (Member 6 AI layer)
# ---------------------------------------------------------------------------

class TestSpacyEntityExtraction:
    """
    Validate that the spaCy NER layer correctly extracts named entities from
    English OCR text and degrades gracefully when spaCy is unavailable.
    These tests use the real spaCy model (en_core_web_sm) when available,
    making them genuine integration tests for the AI layer.
    """

    def test_extract_entities_returns_expected_structure(self):
        """extract_entities_with_spacy must always return the four canonical keys."""
        result = ai_main.extract_entities_with_spacy("John Smith filed the report on 1 January 2024.")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"persons", "organizations", "dates", "locations"}
        for v in result.values():
            assert isinstance(v, list)

    def test_extracts_person_names_from_english_text(self):
        """Real spaCy NER must find a person name in a simple English sentence."""
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed — skipping real-NER test.")

        with (
            patch.object(ai_main, "SPACY_AVAILABLE", True),
            patch.object(ai_main, "SPACY_NLP", spacy.load("en_core_web_sm")),
        ):
            result = ai_main.extract_entities_with_spacy(
                "Inspector Rajesh Kumar filed an FIR at Delhi Police Station."
            )
        # At least one person should be detected
        assert len(result["persons"]) >= 1

    def test_extracts_dates_from_legal_document_text(self):
        """spaCy should detect date expressions in legal boilerplate."""
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed — skipping real-NER test.")

        with (
            patch.object(ai_main, "SPACY_AVAILABLE", True),
            patch.object(ai_main, "SPACY_NLP", spacy.load("en_core_web_sm")),
        ):
            result = ai_main.extract_entities_with_spacy(
                "The incident occurred on 15 March 2023 at 11:30 PM."
            )
        assert len(result["dates"]) >= 1

    def test_returns_empty_when_spacy_unavailable(self):
        """When SPACY_AVAILABLE is False, function must return empty lists — never raise."""
        with patch.object(ai_main, "SPACY_AVAILABLE", False):
            result = ai_main.extract_entities_with_spacy(
                "John Smith filed an FIR at Delhi Police Station on 1 Jan 2024."
            )
        assert result == {"persons": [], "organizations": [], "dates": [], "locations": []}

    def test_returns_empty_for_blank_text(self):
        """Empty / whitespace-only input must return empty entity lists."""
        result = ai_main.extract_entities_with_spacy("   ")
        assert result == {"persons": [], "organizations": [], "dates": [], "locations": []}

    def test_no_duplicate_entities(self):
        """Repeated entity mentions must be de-duplicated."""
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed.")

        text = "John Smith and John Smith were both present. John Smith gave a statement."
        with (
            patch.object(ai_main, "SPACY_AVAILABLE", True),
            patch.object(ai_main, "SPACY_NLP", nlp),
        ):
            result = ai_main.extract_entities_with_spacy(text)
        persons = result["persons"]
        assert len(persons) == len(set(p.lower() for p in persons)), "Duplicate persons detected"

    def test_analyze_endpoint_includes_entities_field_when_spacy_ready(
        self, mock_ocr_text, tiny_png_bytes
    ):
        """
        End-to-end: POST /api/v1/ai/analyze-document must include
        'extracted_entities' with the correct structure when spaCy is available.
        """
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed.")

        ocr_text = "Inspector Rajesh Kumar filed a Witness Statement under Section 161 CrPC on 10 April 2024."

        with (
            mock_ocr_text(ocr_text, "en"),
            patch.object(ai_main, "SPACY_AVAILABLE", True),
            patch.object(ai_main, "SPACY_NLP", nlp),
        ):
            response = client.post(
                "/api/v1/ai/analyze-document",
                files={"file": ("statement.png", tiny_png_bytes, "image/png")},
            )

        assert response.status_code == 200
        body = response.json()
        assert "extracted_entities" in body
        entities = body["extracted_entities"]
        assert entities is not None
        assert set(entities.keys()) == {"persons", "organizations", "dates", "locations"}

    def test_analyze_endpoint_entities_null_when_spacy_unavailable(
        self, mock_ocr_text, tiny_png_bytes
    ):
        """
        When spaCy is not available, extracted_entities must be null (None)
        in the JSON response — the endpoint must not raise.
        """
        ocr_text = "Inspector Sharma filed an FIR at the local police station."

        with (
            mock_ocr_text(ocr_text, "en"),
            patch.object(ai_main, "SPACY_AVAILABLE", False),
        ):
            response = client.post(
                "/api/v1/ai/analyze-document",
                files={"file": ("fir.png", tiny_png_bytes, "image/png")},
            )

        assert response.status_code == 200
        assert response.json()["extracted_entities"] is None