#!/usr/bin/env python3
# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""End-to-end Vast.ai automation for vocos_rt remote GPU training.

Manual prereqs (one-time, ~3 minutes):
  1. Create account at https://vast.ai
  2. Add payment method (Account -> Billing); minimum $5 deposit
  3. Generate API key (Account -> Keys -> "Generate API Key")

Then run:
  python scripts/rent_and_train.py --api-key <YOUR_VAST_API_KEY>
or set the VAST_API_KEY env var and run with no args.

The script:
  1. Generates an SSH keypair if none exists at ~/.ssh/id_vast
  2. Uploads the pubkey to your Vast.ai account
  3. Searches for the best matching GPU (24+ GB, datacenter, >= 99% reliability, cheap)
  4. Asks once for confirmation
  5. Rents, waits for boot
  6. SSH's in, clones https://github.com/audio-algorithms/vocosrt.git, runs setup_remote.sh
  7. Kicks off train_remote.sh in the background (nohup; survives SSH disconnect)
  8. Polls progress every 60s
  9. SCPs the final checkpoint back when training completes
 10. ALWAYS destroys the instance (stops billing) even on error/Ctrl+C
 11. Runs the local demo to produce the listening kit

Resumes if interrupted: instance ID is saved to .rent_state.json; on re-run,
it picks up monitoring/training of the existing instance instead of renting again.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".rent_state.json"
SSH_KEY_PATH = Path.home() / ".ssh" / "id_vast"
SSH_PUBKEY_PATH = Path(str(SSH_KEY_PATH) + ".pub")
GITHUB_REPO_URL = "https://github.com/audio-algorithms/vocosrt.git"
LOCAL_CHECKPOINT_DEST = REPO_ROOT / "checkpoints" / "finetune" / "final_remote.pt"
LOCAL_LOG_DEST = REPO_ROOT / "checkpoints" / "finetune" / "training_remote.log"

VAST_BIN = shutil.which("vastai") or shutil.which("vastai.exe")


# --------------------------------------------------------------- state


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------- helpers


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a command, log it, raise on nonzero unless check=False is passed."""
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"  $ {pretty}")
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def vast(args: list[str]) -> Any:
    """Call the vastai CLI; return parsed JSON for --raw commands, else stdout."""
    cmd = [VAST_BIN] + args
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if cp.returncode != 0:
        print(f"  vastai stderr: {cp.stderr}", file=sys.stderr)
        raise RuntimeError(f"vastai {args[0]} failed: {cp.stderr.strip()}")
    out = cp.stdout.strip()
    if "--raw" in args:
        return json.loads(out)
    return out


def ssh(host: str, port: int, command: str, *, capture: bool = True) -> str:
    cmd = [
        "ssh", "-i", str(SSH_KEY_PATH),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(REPO_ROOT / ".rent_known_hosts"),
        "-o", "ServerAliveInterval=30",
        "-o", "ConnectTimeout=15",
        f"root@{host}",
        command,
    ]
    cp = subprocess.run(cmd, text=True, capture_output=capture)
    if cp.returncode != 0 and capture:
        print(f"  ssh stderr: {cp.stderr}", file=sys.stderr)
    return cp.stdout if capture else ""


def scp_from_remote(host: str, port: int, remote_path: str, local_path: Path) -> None:
    cmd = [
        "scp", "-i", str(SSH_KEY_PATH),
        "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=" + str(REPO_ROOT / ".rent_known_hosts"),
        f"root@{host}:{remote_path}",
        str(local_path),
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------- vast.ai workflow


def ensure_vastai_installed() -> None:
    global VAST_BIN
    if VAST_BIN:
        return
    print("[setup] vastai CLI not found, installing via pip ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "vastai"])
    VAST_BIN = _find_vastai_bin()
    if not VAST_BIN:
        raise RuntimeError("Failed to install/find vastai CLI; install manually with `pip install vastai`")


def ensure_ssh_key() -> str:
    """Generate ~/.ssh/id_vast if absent. Return the public key string."""
    if not SSH_PUBKEY_PATH.exists():
        SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] generating SSH key at {SSH_KEY_PATH}")
        subprocess.check_call([
            "ssh-keygen", "-t", "ed25519",
            "-f", str(SSH_KEY_PATH),
            "-N", "",
            "-C", "vast.ai vocos_rt automation",
        ])
    return SSH_PUBKEY_PATH.read_text().strip()


def configure_vastai(api_key: str, pubkey: str) -> None:
    """Set the API key and SSH key on the vastai CLI side."""
    print("[setup] configuring vastai CLI with API key + SSH key")
    vast(["set", "api-key", api_key])
    # Vast wants the public key in a file or as an arg; via CLI:
    pubkey_file = REPO_ROOT / ".rent_pubkey.tmp"
    pubkey_file.write_text(pubkey + "\n")
    try:
        vast(["create", "ssh-key", "--ssh-key", pubkey])
    except RuntimeError:
        # If the key already exists on the account, that's fine
        pass
    finally:
        pubkey_file.unlink(missing_ok=True)


def search_best_offer() -> dict[str, Any]:
    """Find the cheapest matching offer.

    Criteria (in priority order):
      - rentable now
      - reliability >= 0.99
      - GPU memory >= 24 GB
      - 1 GPU (we don't multi-GPU)
      - sort by $/hr ascending
    """
    print("[search] querying Vast.ai for cheapest 24 GB+ datacenter GPU ...")
    query = (
        "rentable=true "
        "reliability2>=0.99 "
        "num_gpus=1 "
        "gpu_ram>=24 "
        "dph_total<=1.50 "
        "disk_space>=50"
    )
    offers = vast(["search", "offers", query, "--raw"])
    if not offers:
        # Loosen the disk requirement (we'll resize at create time)
        query = "rentable=true reliability2>=0.99 num_gpus=1 gpu_ram>=24 dph_total<=1.50"
        offers = vast(["search", "offers", query, "--raw"])
    if not offers:
        raise RuntimeError("No offers matched. Try loosening filters.")
    offers.sort(key=lambda o: o["dph_total"])
    return offers[0]


def confirm_and_rent(offer: dict[str, Any]) -> int:
    """Show the offer, ask for confirmation, then create the instance."""
    print()
    print("=" * 72)
    print(f"  GPU:         {offer['gpu_name']} x {offer['num_gpus']}")
    print(f"  VRAM:        {offer['gpu_ram'] / 1024:.0f} GB")
    print(f"  Location:    {offer.get('geolocation', '?')}")
    print(f"  Reliability: {offer['reliability2'] * 100:.2f}%")
    print(f"  Disk avail:  {offer['disk_space']:.0f} GB")
    print(f"  Price:       ${offer['dph_total']:.3f}/hr")
    est_hours = 1.5
    print(f"  Est. cost:   ${offer['dph_total'] * est_hours:.2f} for ~{est_hours:.1f} h run")
    print("=" * 72)
    resp = input("Rent this instance? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(0)
    print(f"[rent] creating instance from offer {offer['id']} ...")
    result = vast([
        "create", "instance", str(offer["id"]),
        "--image", "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
        "--disk", "60",
        "--ssh",
        "--direct",
        "--raw",
    ])
    instance_id = result.get("new_contract", result.get("instance_id"))
    if not instance_id:
        raise RuntimeError(f"Could not parse instance ID from create response: {result}")
    print(f"[rent] created instance {instance_id}; waiting for boot ...")
    return int(instance_id)


def wait_for_ready(instance_id: int, timeout_s: int = 600) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = vast(["show", "instance", str(instance_id), "--raw"])
        status = info.get("actual_status", "?")
        print(f"  [wait] status={status}")
        if status == "running" and info.get("ssh_idx") is not None:
            host = info.get("public_ipaddr") or info.get("ssh_host")
            port = info.get("ssh_port")
            return host, int(port)
        time.sleep(15)
    raise RuntimeError(f"Instance {instance_id} did not become ready in {timeout_s} s")


def setup_and_train(host: str, port: int) -> None:
    print(f"[remote] verifying SSH access to {host}:{port} ...")
    out = ssh(host, port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"  GPU on remote: {out.strip()}")

    print("[remote] cloning repo + running setup_remote.sh (~10-15 min) ...")
    cmd = (
        f"git clone {GITHUB_REPO_URL} && "
        "cd vocosrt && "
        "bash scripts/setup_remote.sh 2>&1 | tee setup.log"
    )
    ssh(host, port, cmd, capture=False)

    print("[remote] kicking off train_remote.sh in background (nohup) ...")
    ssh(host, port,
        "cd vocosrt && nohup bash scripts/train_remote.sh > /workspace/train.log 2>&1 & disown",
        capture=False)


def monitor_until_done(host: str, port: int, poll_s: int = 60, max_wall_h: float = 4.0) -> None:
    print(f"[monitor] polling every {poll_s}s; max {max_wall_h:.1f} h ...")
    t0 = time.time()
    while True:
        if time.time() - t0 > max_wall_h * 3600:
            raise RuntimeError(f"Wall-clock budget {max_wall_h} h exceeded")
        out = ssh(host, port, "tail -3 /workspace/train.log 2>/dev/null || true")
        # Echo last log line
        last_lines = [l for l in out.splitlines() if l.strip()]
        if last_lines:
            print(f"  [{time.strftime('%H:%M:%S')}] {last_lines[-1][:200]}")
        if any("DONE" in l or "wall-clock=" in l for l in last_lines):
            print("[monitor] training complete")
            return
        if any("Traceback" in l or "Error" in l or "out of memory" in l for l in last_lines):
            raise RuntimeError(f"Training appears to have crashed: {last_lines[-1]}")
        time.sleep(poll_s)


def fetch_results(host: str, port: int) -> None:
    print("[fetch] downloading final checkpoint + log ...")
    LOCAL_CHECKPOINT_DEST.parent.mkdir(parents=True, exist_ok=True)
    scp_from_remote(host, port, "/root/vocosrt/checkpoints/finetune/final.pt", LOCAL_CHECKPOINT_DEST)
    print(f"  saved {LOCAL_CHECKPOINT_DEST} ({LOCAL_CHECKPOINT_DEST.stat().st_size / 1e6:.1f} MB)")
    scp_from_remote(host, port, "/workspace/train.log", LOCAL_LOG_DEST)
    print(f"  saved {LOCAL_LOG_DEST}")


def destroy_instance(instance_id: int) -> None:
    print(f"[cleanup] destroying instance {instance_id} (stops billing) ...")
    try:
        vast(["destroy", "instance", str(instance_id)])
        print("  destroyed")
    except Exception as e:
        print(f"  WARN: destroy failed: {e} -- check Vast.ai dashboard manually!")


def run_local_listening_kit() -> None:
    print("[local] running demo_finetune_compare.py to build listening kit ...")
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    # Copy the remote checkpoint into the slot the demo expects
    target = REPO_ROOT / "checkpoints" / "finetune" / "step_050000.pt"
    shutil.copy(LOCAL_CHECKPOINT_DEST, target)
    subprocess.run([str(venv_py), str(REPO_ROOT / "demos" / "demo_finetune_compare.py")], check=False)


# --------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("VAST_API_KEY"),
                        help="Vast.ai API key (or set VAST_API_KEY env var)")
    parser.add_argument("--max-wall-h", type=float, default=4.0,
                        help="Hard cap on training wall-clock hours (default 4)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip interactive rent confirmation (use with care)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: provide --api-key or set VAST_API_KEY env var", file=sys.stderr)
        return 1

    ensure_vastai_installed()
    pubkey = ensure_ssh_key()
    configure_vastai(args.api_key, pubkey)

    state = load_state()
    instance_id: int | None = state.get("instance_id")

    if instance_id is None:
        offer = search_best_offer()
        if not args.no_confirm:
            instance_id = confirm_and_rent(offer)
        else:
            # Non-interactive: just rent
            print(f"[rent] auto-confirming offer {offer['id']} (--no-confirm)")
            result = vast([
                "create", "instance", str(offer["id"]),
                "--image", "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
                "--disk", "60", "--ssh", "--direct", "--raw",
            ])
            instance_id = int(result.get("new_contract", result.get("instance_id")))
        state["instance_id"] = instance_id
        save_state(state)
    else:
        print(f"[resume] using existing instance {instance_id} from {STATE_FILE.name}")

    # Always destroy on exit
    destroyed = {"done": False}
    def _cleanup() -> None:
        if not destroyed["done"]:
            destroy_instance(instance_id)
            destroyed["done"] = True
            STATE_FILE.unlink(missing_ok=True)
    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: (_cleanup(), sys.exit(130)))
        except (ValueError, AttributeError):
            pass  # not all platforms support all signals

    try:
        host, port = wait_for_ready(instance_id)
        state.update({"host": host, "port": port})
        save_state(state)

        # Skip setup if log already shows it ran (resume case)
        already_setup = bool(ssh(host, port, "test -d /root/vocosrt && echo yes || echo no").strip() == "yes")
        if not already_setup:
            setup_and_train(host, port)
        else:
            print("[remote] /root/vocosrt exists -- skipping clone+setup")
            # Confirm training is still running or kick it off
            running = ssh(host, port, "pgrep -f finetune_causal.py >/dev/null && echo yes || echo no").strip()
            if running != "yes":
                print("[remote] training not running; kicking off")
                ssh(host, port,
                    "cd /root/vocosrt && nohup bash scripts/train_remote.sh > /workspace/train.log 2>&1 & disown",
                    capture=False)

        monitor_until_done(host, port, max_wall_h=args.max_wall_h)
        fetch_results(host, port)
    finally:
        _cleanup()

    run_local_listening_kit()

    print()
    print("=" * 72)
    print("DONE. Listen at C:\\Users\\jakob\\Desktop\\google drive\\VOCOSRT\\audio_out_finetune\\")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
