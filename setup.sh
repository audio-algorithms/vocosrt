#!/usr/bin/env bash
# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
#
# vocos_rt one-click setup (Linux / WSL / macOS bash). Idempotent.
# See setup.bat for the Windows equivalent and prompt sec.10 for the spec.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
MARKER="${REPO_ROOT}/.setup_complete"

echo "============================================================"
echo " vocos_rt setup (POSIX)"
echo " Repo: ${REPO_ROOT}"
echo "============================================================"

# ---- Idempotency check ----
if [[ -f "${MARKER}" ]]; then
    echo "[setup] Already set up. Nothing to do."
    echo "[setup] Delete .setup_complete to force re-run."
    exit 0
fi

# ---- Phase 1: prerequisites ----
echo
echo "[1/8] Verifying prerequisites..."

PY_BIN=""
for candidate in python3.12 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        ver=$("${candidate}" -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
        if [[ "${ver}" == "3.12" ]]; then
            PY_BIN="${candidate}"
            break
        fi
    fi
done
if [[ -z "${PY_BIN}" ]]; then
    echo "[setup] ERROR: Python 3.12 not found. Install via your package manager or python.org."
    exit 1
fi
echo "[setup] Using Python: $(${PY_BIN} --version) at $(command -v ${PY_BIN})"

if ! command -v git >/dev/null 2>&1; then
    echo "[setup] ERROR: git not found."
    exit 1
fi

# CUDA driver check via nvidia-smi (NOT nvcc -- see DECISIONS.md D6)
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    HAS_GPU=1
else
    echo "[setup] WARNING: nvidia-smi not found. CPU-only mode will be used."
fi

# ---- Phase 2: venv ----
echo
echo "[2/8] Ensuring venv at ${VENV_DIR} ..."
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PY_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PY="${VENV_DIR}/bin/python"
"${VENV_PY}" -m pip install --upgrade pip wheel setuptools

# ---- Phase 3: install deps ----
echo
echo "[3/8] Installing dependencies ..."
"${VENV_PY}" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${VENV_PY}" -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}"

# ---- Phase 4: checkpoints ----
echo
echo "[4/8] Downloading checkpoints ..."
"${VENV_PY}" "${REPO_ROOT}/scripts/download_checkpoints.py"

# ---- Phase 5: corpora ----
echo
echo "[5/8] Downloading corpora ..."
"${VENV_PY}" "${REPO_ROOT}/scripts/download_corpora.py" --subsets test-clean dev-clean train-clean-100

# ---- Phase 6: ONNX export (skipped if exporter not yet built) ----
echo
echo "[6/8] Exporting streaming ONNX ..."
if [[ -f "${REPO_ROOT}/scripts/export_streaming_onnx.py" ]]; then
    "${VENV_PY}" "${REPO_ROOT}/scripts/export_streaming_onnx.py"
else
    echo "[setup] export_streaming_onnx.py not yet present (Phase 3 deliverable). Skipping."
fi

# ---- Phase 7: smoke test ----
echo
echo "[7/8] Running smoke tests ..."
"${VENV_PY}" -m pytest "${REPO_ROOT}/tests" -v -k "test_00 or test_01_bit_exactness" --no-header || \
    echo "[setup] WARNING: smoke tests failed. Check above output."

# ---- Phase 8: summary ----
echo
echo "[8/8] Summary"
echo "  Python:  3.12 (${VENV_PY})"
echo "  GPU:     ${HAS_GPU}"
echo "  Repo:    ${REPO_ROOT}"
echo
echo "[setup] DONE. Activate the venv with: source ${VENV_DIR}/bin/activate"

echo "setup completed" > "${MARKER}"
