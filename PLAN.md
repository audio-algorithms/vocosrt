# vocos_rt — Work Breakdown Structure

**Document ID:** AAL-ENG-VOCOS_RT-PLAN-001
**Version:** 0.1 (initial draft, 2026-05-01)
**Sources:** `prompt/CLAUDE_CODE_PROMPT_VOCOS_REALTIME.md` (AAL-PROMPT-VOCOSRT-001 v1.0.0)

---

## Phase legend

| Symbol | Meaning |
|---|---|
| ⏳ | Pending |
| ▶ | In progress |
| ✓ | Complete (gate passed) |
| ⚠ | Complete with documented deviation in DECISIONS.md |
| ✗ | Blocked / failing |

---

## Phase 0 — Repo bootstrap (target: 0.5 day)

| ID | Task | Status | Est. (h) | Actual (h) |
|---|---|---|---|---|
| T0.1 | git init; pyproject.toml; .gitignore; .python-version; LICENSE notice; README.md | ▶ | 0.5 | — |
| T0.2 | Create venv (Python 3.12); install minimal deps (torch, numpy, scipy, vocos, pytest) | ⏳ | 0.5 | — |
| T0.3 | `vocos_rt/forensic_log.py` — port AAL-ENG-FLOG-001 v4.0 to Python (file/console/UDP sinks, 4 levels, `YYYY.MM.DD HH:MM:SS:MSEC` timestamps, hot-path short-circuit) | ⏳ | 1.5 | — |
| T0.4 | `tests/test_00_forensic_log.py` — sink fan-out, level filter, thread safety, hot-path zero-cost when disabled | ⏳ | 1.0 | — |
| T0.5 | `setup.bat` and `setup.sh` skeleton — idempotent, prereq checks via `nvidia-smi` (not `nvcc`), venv create, deps install, smoke test | ⏳ | 1.5 | — |
| T0.6 | `scripts/download_checkpoints.py` — HF Hub `charactr/vocos-mel-24khz` with SHA-256 verify, idempotent | ⏳ | 1.0 | — |
| T0.7 | `scripts/download_corpora.py` — LibriTTS test-clean + dev-clean + train-clean-100; pre-flight free-disk check; SHA-256 verify; resumable | ⏳ | 1.5 | — |
| T0.8 | First commit (`feat: initial repo skeleton with bootstrap, forensic logger, downloads`) | ⏳ | 0.1 | — |

**Phase 0 exit gate:** `pytest tests/test_00_forensic_log.py -v` green; `python scripts/download_checkpoints.py` succeeds; `setup.sh` is callable and exits 0 on this machine.

---

## Phase 1 — Mechanical streaming wrapper (target: 3 days)

Implements the streaming wrapper around the **as-is pretrained** `charactr/vocos-mel-24khz` weights. Quality will be poor (audible smearing under causal masking with no fine-tune); that is expected — we are validating the *mechanism* here, not the audio quality.

| ID | Task | Status | Est. (h) | Actual (h) |
|---|---|---|---|---|
| T1.1 | `vocos_rt/streaming_stft.py` — Hann window, n_fft=1024, hop=256; sliding overlap-add ISTFT accumulator with division by Σwₙ² | ⏳ | 3 | — |
| T1.2 | `vocos_rt/mel_frontend.py` — streaming mel extractor; bit-equivalent to upstream `vocos.feature_extractor` for matching settings | ⏳ | 2 | — |
| T1.3 | `vocos_rt/causal_convnext.py` — causal-converted ConvNeXt blocks: depthwise conv K=7 → padding=0 + explicit left-pad of K-1 from per-block ring buffer; pointwise stages unchanged | ⏳ | 4 | — |
| T1.4 | `vocos_rt/state.py` — `StreamingState` dataclass: per-block ring buffers (FP32), STFT/ISTFT accumulators, frame counter | ⏳ | 1 | — |
| T1.5 | `vocos_rt/streaming_vocos.py` — public `StreamingVocos` API: `__init__`, `reset_state`, `step`, `stream`, `algorithmic_latency_samples`, `algorithmic_latency_ms` | ⏳ | 3 | — |
| T1.6 | `vocos_rt/offline_vocos.py` — same causal weights, full-sequence path (no state externalization). The reference oracle for tests 01–03. | ⏳ | 1.5 | — |
| T1.7 | `vocos_rt/backends/torch_backend.py` — eager PyTorch baseline backend | ⏳ | 1 | — |
| T1.8 | `vocos_rt/checkpoints.py` — checkpoint resolver, cache, SHA-256 verify | ⏳ | 1 | — |
| T1.9 | `tests/test_01_bit_exactness.py` — streaming vs. own-offline ≤ 1e-5 (FP32) after 49-frame warmup, 100 random mel sequences | ⏳ | 1.5 | — |
| T1.10 | `tests/test_02_chunk_invariance.py` — chunks {1,4,16,64,256} produce identical outputs after warmup | ⏳ | 1 | — |
| T1.11 | `tests/test_03_state_warmup.py` — divergence converges to bit-exact within ≤ 49 frames | ⏳ | 1 | — |
| T1.12 | `tests/test_04_istft_cola.py` — synthetic identity-mel passthrough, reconstruction error ≤ −80 dB | ⏳ | 1 | — |
| T1.13 | `tests/test_05_stability_long_run.py` — ≥ 1 h stream, no NaN/Inf, RMS error stationary | ⏳ | 1.5 | — |

**Phase 1 exit gate:** tests 01–05 all pass with as-is pretrained weights wrapped causally. Audio quality is *not* asserted here.

---

## Phase 2 — Causal fine-tune (target: 3–5 days incl. 30–48 h training)

| ID | Task | Status | Est. (h) | Actual (h) |
|---|---|---|---|---|
| T2.1 | `scripts/download_corpora.py` invocation: pull LibriTTS `train-clean-100` (~7 GB) + `dev-clean` (~350 MB) | ⏳ | 1 | — |
| T2.2 | `scripts/finetune_causal.py` — resumable, idempotent; AdamW lr=1e-4 cosine→1e-5; bf16/fp16; gradient accumulation to keep effective batch ≥ 8 on 4 GB VRAM | ⏳ | 4 | — |
| T2.3 | Discriminators (MPD + MRD) reused from upstream Vocos repo's training code; bidirectional (only generator is causal) | ⏳ | 2 | — |
| T2.4 | Loss stack: multi-resolution STFT loss + mel loss + GAN adversarial; identical to upstream Vocos training | ⏳ | 1 | — |
| T2.5 | Validation loop: mel loss on `dev-clean` every 5000 steps; keep best by val mel loss | ⏳ | 1 | — |
| T2.6 | Run fine-tune to 50,000 steps (hard cap: 60 h wall-clock; if projected to exceed, halve to 25,000 and log in DECISIONS.md) | ⏳ | 30–48 | — |
| T2.7 | `tests/test_06_objective_quality.py`, `test_07_spectral_band.py`, `test_10_pitch_prosody.py` — run on **100-utt dev subset** only at this gate | ⏳ | 2 | — |

**Phase 2 exit gate:** validation mel loss within 5% of upstream Vocos paper-reported value AND tests 06/07/10 pass on the 100-utt dev subset.

---

## Phase 3 — Full validation (target: 3 days)

| ID | Task | Status | Est. (h) | Actual (h) |
|---|---|---|---|---|
| T3.1 | `vocos_rt/_internal/graph_surgery.py` — externalize per-block state buffers as ONNX inputs/outputs | ⏳ | 3 | — |
| T3.2 | `scripts/export_streaming_onnx.py` — export, `onnx.checker.check_model`, ORT vs. eager parity ≤ 1e-3 | ⏳ | 2 | — |
| T3.3 | `vocos_rt/backends/onnx_backend.py` — ONNX Runtime CPU + CUDA EPs; IO binding with pre-allocated GPU state tensors | ⏳ | 3 | — |
| T3.4 | tests 06–13 on **full corpora** (LibriTTS test-clean + VCTK + RAVDESS + MUSAN + RIRs) | ⏳ | 4 | — |
| T3.5 | `tests/test_16_rtf_deadline.py` — ≥ 10⁶ frames; p99 < 9.67 ms; zero deadline misses | ⏳ | 2 | — |
| T3.6 | `tests/test_17a_virtual_loopback.py` — software loopback chirp + cross-correlation; latency within 5 ms of analytical budget | ⏳ | 2 | — |
| T3.7 | tests 18–24 (determinism, multistream, error handling, backend equivalence, precision equivalence, memory leak, API contract) | ⏳ | 4 | — |
| T3.8 | tests 25–27 (static analysis ruff+mypy --strict; reproducibility; coverage ≥ 90% line / ≥ 85% branch) | ⏳ | 2 | — |
| T3.9 | `demos/demo_realtime.py` (D5) — synthetic mel default; `--mic` optional; deadline-miss=0 verdict line | ⏳ | 2 | — |
| T3.10 | `demos/demo_parity.py` (D6) — 20 sentences × 3 modes (original-Vocos / vocos_rt offline / vocos_rt streaming); WAVs + spectrogram triptychs + diff plots + HTML index | ⏳ | 3 | — |
| T3.11 | `demos/demo_abx.py` (D7) — randomized side-by-side HTML for ad-hoc human spot-check | ⏳ | 2 | — |
| T3.12 | `scripts/run_full_validation.py` — orchestrate full pytest run, generate `reports/parity_report.html` with embedded plots, metric tables, audio links, and self-review section per prompt §14 | ⏳ | 3 | — |

**Phase 3 exit gate:** all 27 tests pass; all 3 demos exit 0; `parity_report.html` opens cleanly in a browser.

---

## Phase 5 — Documentation deliverables (target: 1 day; runs after P3 green)

| ID | Task | Status | Est. (h) | Actual (h) |
|---|---|---|---|---|
| T5.1 | Run discovery phase per `Generic Documentation Request.txt` §2; emit discovery report | ⏳ | 1 | — |
| T5.2 | Generate `docs/generated/AAL-ENG-VOCOS_RT-INT-001_Internal_Reference.docx` per §5.1 | ⏳ | 2 | — |
| T5.3 | Generate `docs/generated/AAL-ENG-VOCOS_RT-EXT-001_External_Reference.docx` per §5.2 + §6 firewall | ⏳ | 2 | — |
| T5.4 | Run discovery + execution phases per `Generic prompt for Test Report.md` §2–§3; collect raw artifacts under `docs/generated/test_run_<timestamp>/` | ⏳ | 1 | — |
| T5.5 | Generate `docs/generated/AAL-ENG-VOCOS_RT-TEST-001_Test_Report.docx` per §5 | ⏳ | 2 | — |
| T5.6 | Mirror release artifacts (repo minus venv/datasets/intermediates + docs + reports) to `C:\Users\jakob\Desktop\google drive\VOCOSRT\vocos_rt_release\` | ⏳ | 0.5 | — |

**Phase 5 exit gate:** three `.docx` files exist, validate, pass §7 self-checks of their respective specs; release mirror landed in workspace.

---

## Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | 4 GB VRAM forces effective batch=1; 60 h budget may not finish 50k steps | Medium | High | Halve to 25k; document; observed step rate after 1 h is the gate decision |
| R2 | train-clean-100 alone may not reach quality thresholds in tests 06/07/10 | Medium | Medium | If P2 gate fails, add train-clean-360 if disk allows; else relax thresholds with rationale |
| R3 | Drive sync at workspace path could corrupt files if datasets leak from `C:\dev\vocos_rt` | Low | High | All large I/O strictly under `C:\dev\vocos_rt`; only release subset mirrored back |
| R4 | onnxruntime-gpu CUDA Graphs may not work on RTX 3050 4 GB | Medium | Low | Fall back to plain IO binding; document in DECISIONS.md |
| R5 | `vocos` PyPI package may not support Python 3.12 cleanly | Low | High | If install fails, vendor only the model definition files (still under MIT) |
| R6 | ViSQOL install on Windows fails | High | Low | Use PESQ + STOI + custom MOS proxy; document |
| R7 | Tests 17b (acoustic loopback) and 14/15 (MUSHRA, ABX) are documented exclusions | N/A | N/A | Documented in DECISIONS.md; reflected in parity_report.html |

---

## Total estimated effort

- Phase 0: 0.5 day
- Phase 1: 3 days
- Phase 2: 3–5 days (30–48 h training is the long pole)
- Phase 3: 3 days
- Phase 5: 1 day
- **Total: ~11–13 days wall-clock** (fine-tune dominates)
