# -*- coding: utf-8 -*-
"""
Tests for AI Forensic Pixel Analysis, Error Level Analysis (ELA), and WORM Tamper Tagging (Member 6).
"""
import io
import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

import main as ai_main
from main import (
    app,
    compute_ela_anomaly,
    compute_noise_inconsistency,
    perform_forensic_ela_analysis,
    extract_images_from_upload,
)

client = TestClient(app)


@pytest.fixture
def clean_scanned_image() -> Image.Image:
    """Generate a clean uniform optical scanned legal document page with natural sensor grain."""
    rng = np.random.default_rng(42)
    noise = rng.normal(245.0, 3.0, (300, 400, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(noise)
    draw = ImageDraw.Draw(img)
    for y in range(40, 360, 25):
        draw.line([(30, y), (270, y)], fill=(30, 30, 30), width=2)
    return img


@pytest.fixture
def spliced_tampered_image(clean_scanned_image) -> Image.Image:
    """
    Generate a tampered document where a patch (e.g. date/signature) was
    heavily compressed separately and spliced onto the page.
    """
    base = clean_scanned_image.copy()
    # Create a foreign patch with extreme high-frequency artificial compression noise
    np_patch = np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    foreign_patch = Image.fromarray(np_patch)
    # Save foreign patch at very low quality to create mismatched compression artifacts
    buf = io.BytesIO()
    foreign_patch.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    reloaded_patch = Image.open(buf).convert("RGB")
    # Paste the mismatched patch into the center of the clean document
    base.paste(reloaded_patch, (100, 150))
    return base


class TestForensicPixelAnalysis:
    def test_clean_document_verdict_genuine(self, clean_scanned_image):
        """A uniform document should yield LOW risk and GENUINE verdict without WORM flags."""
        result = perform_forensic_ela_analysis([clean_scanned_image])
        assert result.forgery_verdict in ("GENUINE", "LOW")
        assert result.tamper_risk_level in ("LOW", "MEDIUM")
        assert "SUSPECTED_FORGERY_FLAG" not in result.worm_audit_flags

    def test_spliced_tampered_document_flags_suspected_forgery(self, spliced_tampered_image):
        """
        A document with a spliced patch must be detected by ELA,
        triggering SUSPECTED FORGERY and the WORM magistrate flag.
        """
        result = perform_forensic_ela_analysis([spliced_tampered_image])
        assert result.forgery_verdict == "SUSPECTED FORGERY"
        assert "SUSPECTED_FORGERY_FLAG" in result.worm_audit_flags
        assert result.forgery_score > 0.20

    def test_empty_images_list_handled_gracefully(self):
        """Empty images input must return default non-error result."""
        result = perform_forensic_ela_analysis([])
        assert result.forgery_verdict == "GENUINE"
        assert result.tamper_risk_level == "LOW"
        assert result.analyzed_pages_count == 0

    def test_standalone_forensic_endpoint(self, clean_scanned_image):
        """Test POST /api/v1/ai/forensic-analysis standalone endpoint."""
        buf = io.BytesIO()
        clean_scanned_image.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/api/v1/ai/forensic-analysis",
            files={"file": ("doc.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "forgery_verdict" in data
        assert "tamper_risk_level" in data
        assert "ela_anomaly_score" in data
        assert "worm_audit_flags" in data

    def test_analyze_document_includes_forensic_analysis(self, clean_scanned_image):
        """Test that main analyze-document pipeline returns forensic_analysis object."""
        buf = io.BytesIO()
        clean_scanned_image.save(buf, format="PNG")
        buf.seek(0)

        response = client.post(
            "/api/v1/ai/analyze-document",
            files={"file": ("scanned_evidence.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "forensic_analysis" in body
        fa = body["forensic_analysis"]
        assert "forgery_verdict" in fa
        assert "worm_audit_flags" in fa
