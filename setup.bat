@echo off
REM Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
REM
REM vocos_rt one-click setup (Windows). Idempotent.
REM
REM Phases (per prompt sec.10):
REM   1. Verify prerequisites (Python 3.12, Git, NVIDIA driver)
REM   2. Create venv if absent
REM   3. Install requirements + dev requirements
REM   4. Download checkpoints (HF Hub)
REM   5. Download corpora (LibriTTS subsets)
REM   6. Export streaming ONNX (skipped if not yet built)
REM   7. Smoke test (test_00_forensic_log + test_01_bit_exactness if available)
REM   8. Print summary

setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0"
set "VENV_DIR=%REPO_ROOT%.venv"
set "MARKER=%REPO_ROOT%.setup_complete"

echo ============================================================
echo  vocos_rt setup (Windows)
echo  Repo: %REPO_ROOT%
echo ============================================================

REM ---- Idempotency check ----
if exist "%MARKER%" (
    echo [setup] Already set up. Nothing to do.
    echo [setup] Delete .setup_complete to force re-run.
    exit /b 0
)

REM ---- Phase 1: prerequisites ----
echo.
echo [1/8] Verifying prerequisites...

where py >nul 2>&1
if errorlevel 1 (
    echo [setup] ERROR: Python launcher 'py' not found. Install Python 3.12 from python.org.
    exit /b 1
)
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [setup] ERROR: Python 3.12 not installed. Install from python.org.
    exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
    echo [setup] ERROR: git not found in PATH.
    exit /b 1
)

REM CUDA driver check via nvidia-smi (NOT nvcc -- see DECISIONS.md D6)
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [setup] WARNING: nvidia-smi not found. CPU-only mode will be used.
    set "HAS_GPU=0"
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    set "HAS_GPU=1"
)

REM ---- Phase 2: venv ----
echo.
echo [2/8] Ensuring venv at %VENV_DIR% ...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [setup] ERROR: venv creation failed.
        exit /b 1
    )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools

REM ---- Phase 3: install deps ----
echo.
echo [3/8] Installing dependencies ...
"%VENV_PY%" -m pip install -r "%REPO_ROOT%requirements.txt"
if errorlevel 1 (
    echo [setup] ERROR: requirements.txt install failed.
    exit /b 1
)
"%VENV_PY%" -m pip install -r "%REPO_ROOT%requirements-dev.txt"
if errorlevel 1 (
    echo [setup] ERROR: requirements-dev.txt install failed.
    exit /b 1
)
"%VENV_PY%" -m pip install -e "%REPO_ROOT%"

REM ---- Phase 4: checkpoints ----
echo.
echo [4/8] Downloading checkpoints ...
"%VENV_PY%" "%REPO_ROOT%scripts\download_checkpoints.py"
if errorlevel 1 (
    echo [setup] ERROR: checkpoint download failed.
    exit /b 1
)

REM ---- Phase 5: corpora ----
echo.
echo [5/8] Downloading corpora ...
"%VENV_PY%" "%REPO_ROOT%scripts\download_corpora.py" --subsets test-clean dev-clean train-clean-100
if errorlevel 1 (
    echo [setup] ERROR: corpus download failed.
    exit /b 1
)

REM ---- Phase 6: ONNX export (skipped if exporter not yet built) ----
echo.
echo [6/8] Exporting streaming ONNX ...
if exist "%REPO_ROOT%scripts\export_streaming_onnx.py" (
    "%VENV_PY%" "%REPO_ROOT%scripts\export_streaming_onnx.py"
) else (
    echo [setup] export_streaming_onnx.py not yet present (Phase 3 deliverable). Skipping.
)

REM ---- Phase 7: smoke test ----
echo.
echo [7/8] Running smoke tests ...
"%VENV_PY%" -m pytest "%REPO_ROOT%tests" -v -k "test_00 or test_01_bit_exactness" --no-header
if errorlevel 1 (
    echo [setup] WARNING: smoke tests failed. Check above output.
    REM Do not abort: bootstrap is still complete, tests are diagnostic.
)

REM ---- Phase 8: summary ----
echo.
echo [8/8] Summary
echo   Python:  3.12 (%VENV_PY%)
echo   GPU:     %HAS_GPU%
echo   Repo:    %REPO_ROOT%
echo.
echo [setup] DONE. Activate the venv with: %VENV_DIR%\Scripts\activate.bat

REM Mark complete
echo setup completed > "%MARKER%"

endlocal
exit /b 0
