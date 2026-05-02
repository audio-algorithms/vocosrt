# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Download and verify the upstream Vocos pretrained checkpoint.

Source: ``charactr/vocos-mel-24khz`` on Hugging Face Hub.
Destination: ``<repo>/checkpoints/charactr_vocos_mel_24khz/``.

Idempotent: if the destination matches the manifest's recorded SHA-256, exits cleanly.

Usage:
    python scripts/download_checkpoints.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"

UPSTREAM_REPO_ID = "charactr/vocos-mel-24khz"
LOCAL_DIRNAME = "charactr_vocos_mel_24khz"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_manifest(data: dict[str, dict[str, str]]) -> None:
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def manifest_matches(local_dir: Path, recorded: dict[str, str]) -> bool:
    """Return True iff every recorded file exists at the recorded SHA-256."""
    if not recorded:
        return False
    for relpath, expected_sha in recorded.items():
        target = local_dir / relpath
        if not target.exists():
            return False
        if sha256_of(target) != expected_sha:
            return False
    return True


def download(force: bool) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download_checkpoints] ERROR: huggingface_hub not installed; run setup first.", file=sys.stderr)
        return 1

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    local_dir = CHECKPOINT_DIR / LOCAL_DIRNAME

    manifest = load_manifest()
    recorded = manifest.get(UPSTREAM_REPO_ID, {})

    if not force and manifest_matches(local_dir, recorded):
        print(f"[download_checkpoints] Already present and verified: {local_dir}")
        return 0

    print(f"[download_checkpoints] Downloading {UPSTREAM_REPO_ID} ...")
    snapshot_download(
        repo_id=UPSTREAM_REPO_ID,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        # Skip the .git/cache shenanigans; we only want the model files
        allow_patterns=["*.bin", "*.safetensors", "*.pt", "*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    )

    # Re-hash everything and update manifest
    new_recorded: dict[str, str] = {}
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        new_recorded[rel] = sha256_of(path)

    manifest[UPSTREAM_REPO_ID] = new_recorded
    write_manifest(manifest)

    total_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    print(f"[download_checkpoints] Verified {len(new_recorded)} files, {total_bytes / 1e6:.1f} MB total")
    print(f"[download_checkpoints] Manifest: {MANIFEST_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if manifest matches.")
    args = parser.parse_args(argv)
    return download(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
