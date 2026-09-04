# -*- coding: utf-8 -*-
"""
Tests for AI-Generated Document Detection, Medical & Forensic Report Extraction,
and Cross-Document Corroboration Matrix.
"""
import io
import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from main import (
    app,
    extract_report_structured_fields,
    correlate_fir_and_report,
    inspect_metadata_for_ai_tools,
    compute_spectral_anomaly,
    compute_sensor_noise_distribution,
    perform_forensic_ela_analysis,
    ReportStructuredFields,
)

client = TestClient(app)


@pytest.fixture
def synthetic_flat_ai_image() -> Image.Image:
    """Perfect 0-noise synthetic digital image created by digital tools / AI."""
    img = Image.new("RGB", (400, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Perfect digital vector rectangles
    draw.rectangle([(50, 50), (350, 80)], fill=(0, 0, 0))
    draw.rectangle([(50, 100), (350, 120)], fill=(0, 0, 0))
    return img


@pytest.fixture
def natural_scanned_image() -> Image.Image:
    """Simulate a physical document scanned via optical flatbed scanner with Poisson-Gaussian sensor noise."""
    rng = np.random.default_rng(123)
    noise = rng.normal(242.0, 3.5, (400, 500, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(noise)
    draw = ImageDraw.Draw(img)
    for y in range(60, 380, 40):
        draw.line([(40, y), (460, y)], fill=(40, 40, 40), width=2)
    return img


class TestAiGenerationDetection:
    def test_synthetic_flat_image_detected_as_ai_synthetic(self, synthetic_flat_ai_image):
        """A digitally generated document with 0 sensor noise must be flagged as AI-GENERATED / SYNTHETIC."""
        result = perform_forensic_ela_analysis([synthetic_flat_ai_image])
        assert result.is_ai_generated is True
        assert result.forgery_verdict in ("AI-GENERATED / SYNTHETIC", "SUSPECTED FORGERY")
        assert "SUSPECTED_AI_GENERATION_FLAG" in result.worm_audit_flags or "SUSPECTED_FORGERY_FLAG" in result.worm_audit_flags
        assert "background_noise_variance" in result.forensic_proof_details

    def test_natural_scan_passed_as_genuine(self, natural_scanned_image):
        """A natural scanned document with physical sensor grain must pass as GENUINE."""
        result = perform_forensic_ela_analysis([natural_scanned_image])
        assert result.is_ai_generated is False
        assert result.forgery_verdict == "GENUINE"
        assert result.tamper_risk_level == "LOW"

    def test_ai_generator_metadata_detection(self):
        """Test detection of explicit AI tool keywords in PDF bytes."""
        fake_pdf_bytes = b"%PDF-1.7 ... /Producer (ChatGPT PDF Exporter) /Creator (Canva AI Tool) ... %%EOF"
        prov_score, detected, meta = inspect_metadata_for_ai_tools(fake_pdf_bytes, "pdf", None)
        assert prov_score >= 0.80
        assert any("Chatgpt" in d or "Canva" in d for d in detected)

    def test_spectral_anomaly_fft(self, synthetic_flat_ai_image):
        """Test that 2D FFT spectral anomaly computation executes cleanly."""
        spec_score, h_ratio, peaks = compute_spectral_anomaly(synthetic_flat_ai_image)
        assert isinstance(spec_score, float)
        assert isinstance(h_ratio, float)
        assert isinstance(peaks, int)


class TestReportStructuredExtraction:
    def test_mlc_injury_report_extraction(self):
        sample_mlc = """
        MEDICO-LEGAL INJURY REPORT (MLC)
        Hospital: All India Institute of Medical Sciences (AIIMS), New Delhi
        Examining Medical Officer: Dr. Siddharth Verma
        Name of Injured: Ramesh Kumar
        Age: 34 Yrs   Sex: Male
        Date of Examination: 14/08/2026

        INJURY DETAILS:
        1. Deep incised wound of size 8cm x 2cm x bone deep over left parietal scalp.
        2. Multiple abrasions over right forearm.

        OPINION:
        Injury No. 1 is Grievous in nature caused by sharp cutting weapon.
        """
        report = extract_report_structured_fields(sample_mlc)
        assert "Medico-Legal" in report.report_type
        assert "Dr. Siddharth Verma" in (report.examining_officer or "")
        assert "AIIMS" in (report.hospital_or_lab or "")
        assert report.subject_name == "Ramesh Kumar"
        assert report.gender == "Male"
        assert report.severity_grade == "GRIEVOUS"
        assert "Sharp" in (report.weapon_inferred or "")
        assert len(report.injury_findings) >= 1

    def test_post_mortem_report_extraction(self):
        sample_pm = """
        POST-MORTEM EXAMINATION REPORT
        Hospital: SMS Medical College & Hospital, Jaipur
        Autopsy Surgeon: Dr. Rajesh Sharma, MD (Forensic Medicine)
        Name of Deceased: Late Mukesh Singh
        Date of PM: 02/09/2026

        FINDINGS & OPINION:
        Cause of Death: Hemorrhagic shock resulting from fatal blunt force trauma to cranial vault.
        """
        report = extract_report_structured_fields(sample_pm)
        assert "Post-Mortem" in report.report_type
        assert "Dr. Rajesh Sharma" in (report.examining_officer or "")
        assert report.severity_grade == "FATAL / DANGEROUS TO LIFE"
        assert report.cause_of_death_or_opinion is not None


class TestCrossDocumentCorroboration:
    def test_corroboration_matching_sections(self):
        fir_sections = ["IPC Section 326", "IPC Section 307"]
        fir_text = "Accused attacked complainant with sharp knife causing severe bleeding."
        report = ReportStructuredFields(
            report_type="Medico-Legal Injury Report",
            severity_grade="GRIEVOUS",
            weapon_inferred="Sharp-edged weapon / Bladed weapon",
            injury_findings=["Deep incised wound 8cm x 2cm"],
        )
        corrob = correlate_fir_and_report(
            fir_sections=fir_sections,
            fir_text=fir_text,
            report_fields=report,
            report_text="Grievous sharp weapon injury",
        )
        assert corrob.has_dual_documents is True
        assert corrob.corroboration_status == "CORROBORATED"
        assert corrob.corroboration_score >= 0.70
        assert len(corrob.corroborated_points) >= 1
        assert len(corrob.discrepancy_alerts) == 0

    def test_discrepancy_alert_on_contradiction(self):
        fir_sections = ["IPC Section 307", "IPC Section 326"]
        fir_text = "Accused brutally stabbed complainant with intent to kill."
        report = ReportStructuredFields(
            report_type="Medico-Legal Injury Report",
            severity_grade="SIMPLE",
            weapon_inferred="Hard and blunt object",
            injury_findings=["Simple abrasion on finger"],
        )
        corrob = correlate_fir_and_report(
            fir_sections=fir_sections,
            fir_text=fir_text,
            report_fields=report,
            report_text="Injuries are simple in nature",
        )
        assert corrob.corroboration_status == "CONTRADICTION_ALERT"
        assert len(corrob.discrepancy_alerts) >= 1


class TestDualUploadEndpoints:
    def test_analyze_report_standalone_endpoint(self):
        img = Image.new("RGB", (300, 300), color=(250, 250, 250))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = client.post(
            "/api/v1/ai/analyze-report",
            files={"report_file": ("report.png", buf.getvalue(), "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["document_type"] == "Forensic Report"
        assert "report_fields" in data

    def test_analyze_dual_documents(self):
        img1 = Image.new("RGB", (300, 300), color=(250, 250, 250))
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG")
        buf1.seek(0)

        img2 = Image.new("RGB", (300, 300), color=(250, 250, 250))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        buf2.seek(0)

        res = client.post(
            "/api/v1/ai/analyze-document",
            files={
                "file": ("fir.png", buf1.getvalue(), "image/png"),
                "report_file": ("report.png", buf2.getvalue(), "image/png"),
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "structured_fields" in data
        assert "report_fields" in data
        assert "cross_corroboration" in data
        assert data["cross_corroboration"]["has_dual_documents"] is True
