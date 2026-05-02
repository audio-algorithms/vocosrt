# vocos_rt

Real-time streaming refactor of the [Vocos](https://github.com/gemelo-ai/vocos) neural vocoder.

- **Target hardware:** AMD Ryzen 5 + NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)
- **Target OS:** Windows 11; Linux secondary
- **Python:** 3.12
- **Audio contract:** 24 kHz, 100 mel bins, n_fft=1024, hop=256 (10.67 ms period), 4-frame decoder lookahead (~43 ms)

## Status

Pre-Phase-1 bootstrap. See [PLAN.md](PLAN.md).

## Documents

- [PLAN.md](PLAN.md) — work breakdown structure, phase gates, risk register
- [DECISIONS.md](DECISIONS.md) — every non-trivial choice and its rationale

## Acknowledgments

- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) — the upstream vocoder this work refactors (MIT)
- [pengzhendong/streaming-vocos](https://github.com/pengzhendong/streaming-vocos) — surveyed as prior art for the upstream-as-dependency pattern (Apache-2.0); see DECISIONS.md D8

## License

Copyright © 2026 AudioAlgorithms LLC. All rights reserved. Proprietary; not licensed for redistribution.
