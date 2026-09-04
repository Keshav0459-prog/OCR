# -*- coding: utf-8 -*-
"""
Tests for structured field extraction (Name, Father's Name, Address, Legal Sections, etc.)
"""
import pytest
import main as ai_main
from main import extract_structured_fields, detect_legal_sections, app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestNameFatherAddressExtraction:
    def test_inline_so_standard_fir(self):
        text = "1. Complainant: Rajesh Sharma S/o Ramesh Sharma R/o Flat 402, Sector 12, Dwarka, New Delhi - 110078"
        fields = extract_structured_fields(text)
        assert fields.name == "Rajesh Sharma"
        assert fields.father_name == "Ramesh Sharma"
        assert "Dwarka" in fields.address
        assert "110078" in fields.address

    def test_inline_so_with_titles_and_ocr_noise(self):
        text = "Shri Vikram Singh S/0 Late Sh. Mohan Singh R/0 H.No 45, Gali No 2, Civil Lines, Meerut"
        fields = extract_structured_fields(text)
        assert fields.name == "Vikram Singh"
        assert fields.father_name == "Mohan Singh"
        assert "Civil Lines" in fields.address

    def test_inline_daughter_and_wife_notations(self):
        text1 = "Priya Verma D/o Suresh Verma R/o Village Rampur"
        f1 = extract_structured_fields(text1)
        assert f1.name == "Priya Verma"
        assert f1.father_name == "Suresh Verma"
        assert "Village Rampur" in f1.address

        text2 = "Smt. Sunita Devi W/o Manoj Kumar R/o 12 MG Road, Jaipur"
        f2 = extract_structured_fields(text2)
        assert f2.name == "Sunita Devi"
        assert f2.father_name == "Manoj Kumar"
        assert "12 MG Road" in f2.address

    def test_hindi_devanagari_inline_relations(self):
        text = "शिकायतकर्ता: अमित कुमार आत्मज श्री सुरेश कुमार निवासी ग्राम रामपुर, थाना सदर, जिला मेरठ"
        fields = extract_structured_fields(text)
        assert fields.name == "अमित कुमार"
        assert fields.father_name == "सुरेश कुमार"
        assert "ग्राम रामपुर" in fields.address

    def test_key_value_format_on_separate_lines(self):
        text = (
            "Name: Rahul Sharma\n"
            "Father's Name: Sh. Dinesh Sharma\n"
            "Age: 32 years\n"
            "Mobile: 9876543210\n"
            "Address: Flat 101, Sunshine Heights\n"
            "Sector 15, Rohini\n"
            "Delhi - 110085\n"
            "FIR No: 123/2024\n"
            "Police Station: Rohini North\n"
        )
        fields = extract_structured_fields(text)
        assert fields.name == "Rahul Sharma"
        assert fields.father_name == "Dinesh Sharma"
        assert fields.age == "32"
        assert fields.mobile == "9876543210"
        assert "Sunshine Heights" in fields.address
        assert "Rohini" in fields.address
        assert fields.fir_number == "123/2024"
        assert fields.police_station == "Rohini North"

    def test_key_value_name_with_embedded_so(self):
        """When key Name contains S/o and Address, it should split properly."""
        text = "Name of Complainant : Amit Kumar S/o Rajesh Kumar R/o 14 Park Street, Kolkata"
        fields = extract_structured_fields(text)
        assert fields.name == "Amit Kumar"
        assert fields.father_name == "Rajesh Kumar"
        assert "14 Park Street" in fields.address


class TestLegalSectionsDetection:
    def test_multi_section_ipc(self):
        text = "The accused is booked U/S 302, 307, 34 IPC and Section 120B IPC."
        sections = detect_legal_sections(text)
        assert "IPC Section 302" in sections
        assert "IPC Section 307" in sections
        assert "IPC Section 34" in sections
        assert "IPC Section 120B" in sections

    def test_slash_separated_sections(self):
        text = "Case registered under Section 376/511 IPC at Police Station."
        sections = detect_legal_sections(text)
        assert "IPC Section 376" in sections
        assert "IPC Section 511" in sections

    def test_bns_sections(self):
        text = "Offence under BNS Section 103(1) and BNS Section 115(2)."
        sections = detect_legal_sections(text)
        assert "BNS Section 103(1)" in sections
        assert "BNS Section 115(2)" in sections

    def test_special_acts(self):
        text = "Booked under Section 4 POCSO Act and Section 25 Arms Act and Section 66D IT Act."
        sections = detect_legal_sections(text)
        assert any("POCSO Section 4" in s or "POCSO Act Section 4" in s for s in sections)
        assert "Arms Act Section 25" in sections
        assert "IT Act Section 66D" in sections
