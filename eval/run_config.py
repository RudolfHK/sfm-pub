#!/usr/bin/env python3
"""Run one `run_sfm.py` configuration under measurement.

Captures wall-clock time, peak resident memory of the pipeline process tree,
the full log, and the pipeline's own `--export-cameras` JSON. Writes one
`<name>.run.json` per invocation into the results directory, so the reporting
stage never has to re-run anything.

Usage
-----
    python eval/run_config.py --name base20 --image_dir ./buddha_20 \
        --results-dir eval_results -- --n_features 8000 --ratio 0.75
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _sample_peak_rss(proc: subprocess.Popen, out: dict, stop: threading.Event) -> None:
    """Poll the process tree's RSS until it exits; record the maximum in MB."""
    try:
        import psutil
    except ImportError:
        out["peak_rss_mb"] = None
        return
    try:
        p = psutil.Process(proc.pid)
    except psutil.Error:
        out["peak_rss_mb"] = None
        return
    peak = 0.0
    while not stop.wait(0.25):
        try:
            total = p.memory_info().rss
            for c in p.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, total / 1e6)
        except psutil.Error:
            break
    out["peak_rss_mb"] = round(peak, 1) if peak else None


def run(name: str, image_dir: str, results_dir: Path, extra: list[str],
        timeout: float | None) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    ply = results_dir / f"{name}.ply"
    cams = results_dir / f"{name}.cameras.json"
    log_path = results_dir / f"{name}.log"

    cmd = [
        sys.executable, str(_ROOT / "run_sfm.py"),
        "--image_dir", image_dir,
        "--output", str(ply),
        "--export-cameras", str(cams),
        *extra,
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    mem: dict = {}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_peak_rss, args=(proc, mem, stop), daemon=True)
    sampler.start()

    timed_out = False
    try:
        log, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        log, _ = proc.communicate()
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    wall = time.time() - t0
    log_path.write_text(log or "", encoding="utf-8")

    record = {
        "name": name,
        "cmd": cmd,
        "image_dir": image_dir,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "wall_s": round(wall, 2),
        "peak_rss_mb": mem.get("peak_rss_mb"),
        "log": str(log_path),
        "ply": str(ply) if ply.exists() else None,
        "ply_bytes": ply.stat().st_size if ply.exists() else None,
        "cameras_json": str(cams) if cams.exists() else None,
    }
    if cams.exists():
        with open(cams, encoding="utf-8") as fh:
            payload = json.load(fh)
        record["summary"] = {
            k: payload[k] for k in
            ("n_images", "n_cameras_registered", "n_points",
             "n_observations", "mean_track_length", "stage_times_s")
            if k in payload
        }
        record["summary"]["reprojection"] = payload.get("reprojection")

    (results_dir / f"{name}.run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--timeout", type=float, default=None,
                    help="Kill the run after this many seconds.")
    ap.add_argument("extra", nargs="*",
                    help="Extra flags passed through to run_sfm.py (after --).")
    args = ap.parse_args()

    rec = run(args.name, args.image_dir, Path(args.results_dir),
              list(args.extra), args.timeout)
    print(json.dumps({k: v for k, v in rec.items() if k != "cmd"}, indent=2))
    return 0 if rec["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
