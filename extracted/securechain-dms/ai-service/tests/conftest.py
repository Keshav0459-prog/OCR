# conftest.py — shared pytest fixtures and session-level bootstrap
# for the SecureChain DMS AI service test suite.
#
# What this file does:
#   1. Auto-downloads the spaCy en_core_web_sm model if it is missing, so
#      every developer / CI runner gets a green test run without manual setup.
#   2. Provides a `mock_all_ocr_engines` session fixture that stubs out the
#      heavy OCR models (PaddleOCR, EasyOCR, RapidOCR) so the test suite
#      runs in seconds without downloading multi-GB model weights.

from __future__ import annotations

import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Session-level: ensure spaCy en_core_web_sm is present
# ---------------------------------------------------------------------------

def _ensure_spacy_model() -> None:
    """Download en_core_web_sm if not already installed."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("\n[conftest] en_core_web_sm not found — downloading now...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[conftest] en_core_web_sm download complete.", flush=True)
    except ImportError:
        # spaCy itself is not installed — skip silently; tests will handle it.
        pass


_ensure_spacy_model()


# ---------------------------------------------------------------------------
# Session fixture: patch heavy OCR engines so tests don't need model weights
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False, scope="session")
def mock_all_ocr_engines():
    """
    Stub out all three OCR engines at the module level for the duration of
    the test session. Tests that need real OCR behaviour should NOT use this
    fixture (opt out by not requesting it).

    Because the app initialises engines in a background thread that starts on
    lifespan startup, and TestClient triggers that lifespan, we patch the
    globals directly rather than via dependency injection.
    """
    import main as ai_main
    from unittest.mock import MagicMock, patch

    rapid_mock = MagicMock()
    rapid_mock.return_value = ([], None)  # (result, elapsed)

    easy_mock = MagicMock()
    easy_mock.readtext.return_value = []

    paddle_mock = MagicMock()
    paddle_mock.ocr.return_value = []

    with (
        patch.object(ai_main, "RAPID_OCR_ENGINE", rapid_mock),
        patch.object(ai_main, "EASY_OCR_READER", easy_mock),
        patch.dict(ai_main.OCR_ENGINES, {"en": paddle_mock, "hi": paddle_mock}),
    ):
        yield
