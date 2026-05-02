# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Download and verify test/training corpora.

Currently supported subsets (per DECISIONS.md D3 — train-clean-360 deliberately omitted):

    test-clean        LibriTTS test partition           ~350 MB
    dev-clean         LibriTTS dev partition            ~350 MB
    train-clean-100   LibriTTS training partition         ~7 GB

Destination: ``<repo>/datasets/libritts/``. Resumable via HTTP Range.

Usage:
    python scripts/download_corpora.py --subsets test-clean dev-clean
    python scripts/download_corpora.py --subsets train-clean-100   # ~7 GB, slow
    python scripts/download_corpora.py --list                      # show known subsets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
LIBRITTS_DIR = DATASETS_DIR / "libritts"
MANIFEST_PATH = DATASETS_DIR / "manifest.json"

# Reserve N GB of free disk after estimated download (generous for unpacking)
MIN_FREE_GB_AFTER = 5.0


@dataclass(frozen=True)
class Subset:
    name: str
    url: str
    archive_filename: str
    approx_compressed_gb: float
    approx_uncompressed_gb: float
    extracted_dir: str  # name of the top-level dir produced by tar -xf


# OpenSLR LibriTTS mirror (https://www.openslr.org/60/). SHA-256s are recorded
# into manifest.json on first successful download; on subsequent runs we verify.
_LIBRITTS_BASE = "https://www.openslr.org/resources/60"
SUBSETS: dict[str, Subset] = {
    "test-clean": Subset(
        name="test-clean",
        url=f"{_LIBRITTS_BASE}/test-clean.tar.gz",
        archive_filename="test-clean.tar.gz",
        approx_compressed_gb=0.35,
        approx_uncompressed_gb=0.45,
        extracted_dir="LibriTTS/test-clean",
    ),
    "dev-clean": Subset(
        name="dev-clean",
        url=f"{_LIBRITTS_BASE}/dev-clean.tar.gz",
        archive_filename="dev-clean.tar.gz",
        approx_compressed_gb=0.35,
        approx_uncompressed_gb=0.45,
        extracted_dir="LibriTTS/dev-clean",
    ),
    "train-clean-100": Subset(
        name="train-clean-100",
        url=f"{_LIBRITTS_BASE}/train-clean-100.tar.gz",
        archive_filename="train-clean-100.tar.gz",
        approx_compressed_gb=6.4,
        approx_uncompressed_gb=8.7,
        extracted_dir="LibriTTS/train-clean-100",
    ),
}


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


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


def http_download_resumable(url: str, dest: Path) -> None:
    """Download with HTTP Range support and a tqdm progress bar."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code in (200, 206):
            total_remaining = int(resp.headers.get("Content-Length", 0))
            total = existing + total_remaining
            mode = "ab" if existing else "wb"
            with dest.open(mode) as f, tqdm(
                total=total, initial=existing, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        elif resp.status_code == 416:
            # Already complete
            return
        else:
            raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")


def extract_tar_gz(archive: Path, dest_root: Path) -> None:
    print(f"[download_corpora] Extracting {archive.name} ...")
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        # Python 3.12+ supports filter='data' for safe extraction
        tf.extractall(dest_root, filter="data")


def already_extracted(subset: Subset) -> bool:
    return (LIBRITTS_DIR / subset.extracted_dir).is_dir()


def estimate_required_gb(subsets: list[Subset]) -> float:
    return sum(s.approx_compressed_gb + s.approx_uncompressed_gb for s in subsets)


def fetch_subset(subset: Subset, manifest: dict[str, dict[str, str]], keep_archive: bool) -> None:
    if already_extracted(subset):
        print(f"[download_corpora] {subset.name}: already extracted, skipping.")
        return

    archive_path = DATASETS_DIR / "_archives" / subset.archive_filename
    print(f"[download_corpora] {subset.name}: downloading from {subset.url}")
    http_download_resumable(subset.url, archive_path)

    digest = sha256_of(archive_path)
    recorded = manifest.get(subset.name, {})
    if "sha256" in recorded and recorded["sha256"] != digest:
        raise RuntimeError(
            f"SHA-256 mismatch for {subset.archive_filename}: "
            f"recorded {recorded['sha256']!r}, got {digest!r}"
        )
    manifest[subset.name] = {"sha256": digest, "url": subset.url}

    extract_tar_gz(archive_path, LIBRITTS_DIR)

    if not keep_archive:
        archive_path.unlink()
        archives_dir = archive_path.parent
        if archives_dir.exists() and not any(archives_dir.iterdir()):
            archives_dir.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=["test-clean", "dev-clean"],
                        help="One or more of: " + ", ".join(SUBSETS.keys()))
    parser.add_argument("--list", action="store_true", help="List known subsets and exit.")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep the .tar.gz files after extraction (default: delete).")
    args = parser.parse_args(argv)

    if args.list:
        for s in SUBSETS.values():
            print(f"  {s.name:<20} ~{s.approx_compressed_gb:>4.1f} GB compressed, "
                  f"~{s.approx_uncompressed_gb:>4.1f} GB extracted")
        return 0

    selected: list[Subset] = []
    for name in args.subsets:
        if name not in SUBSETS:
            print(f"[download_corpora] ERROR: unknown subset {name!r}", file=sys.stderr)
            print(f"[download_corpora] Known: {', '.join(SUBSETS.keys())}", file=sys.stderr)
            return 2
        selected.append(SUBSETS[name])

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-flight disk check
    required = estimate_required_gb(selected)
    available = free_gb(DATASETS_DIR)
    headroom = available - required - MIN_FREE_GB_AFTER
    print(f"[download_corpora] Required ~{required:.1f} GB; available {available:.1f} GB; "
          f"headroom after = {headroom:.1f} GB")
    if headroom < 0:
        print(f"[download_corpora] ERROR: insufficient disk. Need at least "
              f"{MIN_FREE_GB_AFTER + required:.1f} GB free.", file=sys.stderr)
        return 3

    manifest = load_manifest()
    for subset in selected:
        fetch_subset(subset, manifest, keep_archive=args.keep_archives)
    write_manifest(manifest)

    print("[download_corpora] All requested subsets are present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
