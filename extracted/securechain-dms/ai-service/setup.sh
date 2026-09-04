#!/usr/bin/env bash
# setup.sh — One-command local dev bootstrap for SecureChain DMS AI Service
# Usage: bash setup.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo ""
echo "================================================="
echo "  SecureChain DMS — AI Service Local Setup"
echo "  Member 6: AI + DevOps + Testing"
echo "================================================="
echo ""

# 1. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists — skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"

# 2. Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --quiet --upgrade pip

# 3. Install all dependencies (including spaCy model via direct URL in requirements.txt)
echo "[3/4] Installing Python dependencies + spaCy en_core_web_sm model..."
pip install --quiet -r requirements.txt

# 4. Verify spaCy model is loadable
echo "[4/4] Verifying spaCy model..."
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('    spaCy en_core_web_sm OK —', len(list(nlp.pipe_names)), 'pipeline components')"

echo ""
echo "✅  Setup complete!"
echo ""
echo "To run the AI service:"
echo "    source $VENV_DIR/bin/activate   # (or Scripts/activate on Windows)"
echo "    uvicorn main:app --reload --port 8000"
echo ""
echo "To run the full test suite with coverage:"
echo "    pytest tests/ -v --cov=. --cov-report=term-missing"
echo ""
echo "To build & run via Docker:"
echo "    docker compose up -d --build"
echo ""
