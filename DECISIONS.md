# vocos_rt — Decisions Log

**Document ID:** AAL-ENG-VOCOS_RT-DECISIONS-001
**Maintained by:** Claude Code (autonomous), reviewed by Jakob Ashtar

Every non-trivial deviation from `prompt/CLAUDE_CODE_PROMPT_VOCOS_REALTIME.md`, every dependency choice, every threshold tightening or loosening, every interpretation of an ambiguous spec line gets a numbered entry here. Format:

```
## D<N>  <one-line title>           [decided YYYY-MM-DD]
**Source:** prompt §X / discovery / Jakob / pragmatics
**Decision:** what we did
**Rationale:** why
**Alternatives considered:** what we rejected
**Reversibility:** trivial / moderate / hard
**Worth Jakob reviewing first?** yes / no
```

---

## D1  Python 3.12 instead of 3.11           [decided 2026-05-01]
**Source:** discovery (Python 3.11 not installed on target machine; have 3.10, 3.12, 3.13, 3.14) → Jakob
**Decision:** Pin `requires-python = ">=3.12,<3.13"`. `.python-version` is `3.12`.
**Rationale:** Closest available to prompt's 3.11 pin; 3.12 has full wheel coverage for `torch>=2.2`, `onnxruntime-gpu`, `torchcrepe`, `speechbrain`, `pesq`. 3.10 is older than the prompt's lower bound. 3.13/3.14 still have spotty wheel coverage in the audio-ML ecosystem.
**Alternatives considered:** Install 3.11 via winget (adds a manual step contrary to setup.bat idempotency goal); use 3.10 (regress on language version).
**Reversibility:** moderate — would require recreating the venv and re-running tests.
**Worth Jakob reviewing first?** No (already approved).

## D2  Build at `C:\dev\vocos_rt`, not in Google Drive workspace           [decided 2026-05-01]
**Source:** discovery (workspace at `C:\Users\jakob\Desktop\google drive\VOCOSRT` is inside Drive sync) → Jakob
**Decision:** Build the entire repo, venv, datasets, training intermediates at `C:\dev\vocos_rt`. At end of Phase 5, mirror the release subset (repo minus `.venv`, `datasets/`, `checkpoints/*.pt`, training intermediates; plus `docs/generated/*.docx` and `reports/parity_report.html`) into `C:\Users\jakob\Desktop\google drive\VOCOSRT\vocos_rt_release\`.
**Rationale:** Drive sync would constantly upload multi-GB intermediate artifacts and risk file corruption mid-sync during heavy training I/O.
**Reversibility:** trivial.
**Worth Jakob reviewing first?** No (already approved).

## D3  Training corpus: LibriTTS `train-clean-100` only           [decided 2026-05-01]
**Source:** discovery (98 GB free on C:; `train-clean-360` adds ~24 GB and would leave ~10 GB free during training) → Jakob
**Decision:** Fine-tune on `train-clean-100` (~7 GB) only, not the prompt's specified `train-clean-100 + train-clean-360`. Keep 50,000 steps with gradient accumulation.
**Rationale:** Disk headroom; modest expected quality cost (less data diversity, but the same speaker count and recording conditions).
**Alternatives considered:** Both subsets (disk-tight); skip fine-tune entirely (rejected — quality regression unacceptable).
**Quality risk:** Tests 06/07/10 thresholds may be tight. If P2 exit gate fails, the contingency is to add `train-clean-360` if disk allows by then, or to relax thresholds with documented rationale.
**Reversibility:** trivial — re-run training with both subsets if disk allows.
**Worth Jakob reviewing first?** No (already approved).

## D4  Checkpoint local-only; remote hosting deferred           [decided 2026-05-01]
**Source:** prompt D3 ("checkpoint downloadable from artifact storage") + Jakob
**Decision:** Causal-fine-tuned checkpoint lives in `checkpoints/` in the repo, gitignored except for a manifest. `test_26_reproducibility.py` will verify checkpoint hash against a manifest file rather than re-downloading.
**Rationale:** No artifact storage is named in the prompt; setting one up isn't required for this build. Manifest-based hash check satisfies the spirit of test 26.
**Follow-up:** When Jakob designates a host (HF Hub, S3, etc.), wire `scripts/download_checkpoints.py` to fetch the canonical artifact and update test 26 to download-then-hash.
**Reversibility:** trivial.
**Worth Jakob reviewing first?** No (already approved).

## D5  Git remote: local-only           [decided 2026-05-01]
**Source:** Jakob
**Decision:** `git init` only; no `git remote add`. Conventional Commits style preserved for future push.
**Reversibility:** trivial.
**Worth Jakob reviewing first?** No (already approved).

## D6  `setup.bat`/`setup.sh` CUDA check: `nvidia-smi` not `nvcc --version`           [decided 2026-05-01]
**Source:** discovery (no `nvcc` on target; only the NVIDIA driver) + prompt §10 step 1
**Decision:** Replace the `nvcc --version` check with `nvidia-smi` (which is shipped with the NVIDIA driver) and parse `Driver Version` and `CUDA Version` from its output.
**Rationale:** We use prebuilt PyTorch and `onnxruntime-gpu` wheels — neither requires the CUDA toolkit (`nvcc`); only the runtime driver. Requiring `nvcc` would add an unnecessary user installation step.
**Reversibility:** trivial.
**Worth Jakob reviewing first?** No.

## D7  Adopt upstream `vocos` PyPI package as a runtime dependency           [decided 2026-05-01]
**Source:** prior-art survey (pengzhendong/streaming-vocos uses this pattern); prompt §5 omits `vocos` from the dep list
**Decision:** Add `vocos>=0.1.0` to runtime dependencies. Reuse upstream's `Vocos.from_pretrained("charactr/vocos-mel-24khz")` for model architecture loading and the original feature extractor. Implement causal conversion, ring-buffer state, streaming ISTFT, and fine-tuning code in `vocos_rt/`.
**Rationale:** The upstream is MIT-licensed, well-tested, and slow-moving (last update Jan 2024). Vendoring would duplicate ~2 KLOC and create maintenance burden. Depending on it gives us the model definition and the pretrained loader for free.
**Alternatives considered:** Vendor upstream into `third_party/` (more code to maintain); reimplement the model from the paper (massively more work, no quality benefit).
**Reversibility:** moderate — would require vendoring the model definition files if the upstream package becomes unavailable.
**Worth Jakob reviewing first?** No (low risk; upstream is MIT and stable).

## D8  Prior-art survey: pengzhendong/streaming-vocos rejected as basis           [decided 2026-05-01]
**Source:** Jakob requested survey; clean-room concern
**Decision:** Surveyed `https://github.com/pengzhendong/streaming-vocos` (Apache-2.0, last push 2025-06-10, ~170 LOC across two files). Adopted only the **dependency pattern** (D7); rejected the **streaming approach**.
**Rationale for rejection:**
1. Their approach is *chunked-with-padding* (default 300 ms chunks + 1-frame padding), not true frame-streaming. Latency floor = chunk size = 300 ms — incompatible with our ≤ 9.67 ms per-frame deadline.
2. Their padding default of 1 frame is incorrect for the model's ~49-frame causal receptive field; their reconstructions have unbounded boundary discontinuity error from the bidirectional ConvNeXt.
3. They run the full ISTFT independently per chunk and crop, producing chunk-boundary artifacts; no overlap-add accumulator.
4. They do not causalize the ConvNeXt depthwise convs — the model still uses bidirectional context implicitly via the padded chunk.
5. They ship no causal-fine-tuned checkpoint and no quality validation beyond a single cosine-similarity print.
**What we adopted:** the `vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")` loader pattern; confirmation that `self.head.istft.hop_length = 256`.
**Attribution:** Will note prior art in README.md "Acknowledgments" section to honor good practice (Apache-2.0 doesn't require it because we copied no code).
**Reversibility:** N/A (research note, not a code commitment).
**Worth Jakob reviewing first?** No.

## D9  ViSQOL replacement: PESQ + STOI + custom MOS proxy           [decided 2026-05-01]
**Source:** prompt §5 ("`visqol` ... OR `pesq + pystoi + custom MOS proxy` if visqol is hard to install on Windows")
**Decision:** Use `pesq` + `pystoi` + a small linear-combination MOS proxy. ViSQOL is not attempted on Windows.
**Rationale:** ViSQOL's Bazel build on Windows is a known pain. The prompt explicitly authorizes this fallback.
**Worth Jakob reviewing first?** No.

## D10  License: proprietary, no LICENSE file shipped           [decided 2026-05-01]
**Source:** Jakob
**Decision:** Repo has no `LICENSE` file. Each source file gets the header `Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.`
**Rationale:** Jakob has not yet decided the licensing model; absence of a LICENSE file makes the work proprietary by default under US copyright law.
**Reversibility:** trivial.
**Worth Jakob reviewing first?** Recommended before any external distribution.

## D11  Demo `demo_realtime.py` default: synthetic mel + `sounddevice` if device present           [decided 2026-05-01]
**Source:** prompt §8 (demo D5)
**Decision:** Default mode is to read the four WAVs in `audio/`, extract their mel via the `vocos_rt` mel front-end, and stream them through `StreamingVocos.step()` into `sounddevice` if a device is available, else into a virtual loopback (named pipe / shared-memory ring). `--mic` mode is implemented but optional.
**Rationale:** Autonomous build cannot guarantee a microphone is present; synthetic-mel + virtual-loopback gives a deterministic, headless-runnable verdict. The four bundled WAVs (keyboard, music, noise, speech) are convenient diverse inputs.
**Worth Jakob reviewing first?** No.

## D12  Excluded tests 14, 15, 17b — documented per prompt §7.3           [decided 2026-05-01]
**Source:** prompt §7.3
**Decision:** Tests 14 (MUSHRA), 15 (ABX), 17b (acoustic loopback) are excluded. Documented in `parity_report.html`.
**Worth Jakob reviewing first?** No (per prompt).

---

## Hosting / future-work items (NOT decisions; tracked here for visibility)

- **F1** Host the causal-fine-tuned checkpoint on a public artifact store and switch test 26 to download-then-hash (see D4).
- **F2** Reattempt ViSQOL on Windows once the Bazel toolchain is reliable; reinstate as primary objective metric.
- **F3** Re-run fine-tune with `train-clean-360` added if/when disk allows; compare against this build's metrics.
- **F4** TensorRT backend on RTX 3050 (currently `OPTIONAL` in prompt §4); attempt only after baseline is green.
