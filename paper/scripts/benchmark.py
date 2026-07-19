#!/usr/bin/env python3
"""Benchmark the Python SfM pipeline against COLMAP on identical input.

Runs `run_sfm.py` once per backend on the same image directory, measures wall
time, and parses cameras / points / final BA-RMSE from the pipeline's own stdout
(and counts points authoritatively from the output PLY). Emits a Markdown table
for Tab. 1 of the paper.

Usage
-----
    # Both backends (needs a COLMAP binary on PATH):
    python paper/scripts/benchmark.py --image_dir ./images

    # Python backend only:
    python paper/scripts/benchmark.py --image_dir ./images --backends python

    # Pass extra flags through to every run (after --):
    python paper/scripts/benchmark.py --image_dir ./images -- --n_features 10000

Notes
-----
* Final RMSE is parsed from "BA final RMSE: X.XXX px" (logged unconditionally by
  the Python backend). COLMAP does not emit this line, so its RMSE cell is "n/v".
* Run both backends on the SAME machine for a fair time comparison.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_RE_CAMERAS = re.compile(r"Cameras\s*:\s*(\d+)\s*registered")
_RE_POINTS = re.compile(r"Points\s*:\s*([\d,]+)")
_RE_RMSE = re.compile(r"BA final RMSE:\s*([\d.]+)\s*px")
_ROOT = Path(__file__).resolve().parents[2]  # repo root


def count_ply_vertices(path: Path) -> int | None:
    """Read only the PLY header to get the authoritative vertex count."""
    try:
        with open(path, "rb") as fh:
            if fh.readline().strip() != b"ply":
                return None
            for _ in range(60):
                line = fh.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[2])
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def run_backend(image_dir: str, backend: str, out_ply: Path,
                extra: list[str]) -> dict:
    cmd = [sys.executable, str(_ROOT / "run_sfm.py"),
           "--image_dir", image_dir, "--output", str(out_ply),
           "--backend", backend, *extra]
    print(f"\n[benchmark] === {backend} ===\n[benchmark] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    log = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(f"[benchmark] {backend} FAILED (exit {proc.returncode}). "
              "Last lines:\n" + "\n".join(log.strip().splitlines()[-15:]))

    cam = _RE_CAMERAS.search(log)
    pts = _RE_POINTS.search(log)
    rmse = list(_RE_RMSE.finditer(log))  # take the last BA round
    ply_pts = count_ply_vertices(out_ply)

    return {
        "backend": backend,
        "ok": proc.returncode == 0,
        "elapsed": elapsed,
        "cameras": int(cam.group(1)) if cam else None,
        "points": ply_pts if ply_pts is not None
        else (int(pts.group(1).replace(",", "")) if pts else None),
        "rmse": float(rmse[-1].group(1)) if rmse else None,
    }


def fmt(v, suffix="", nd=0):
    if v is None:
        return "n/v"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v:,}{suffix}"


def to_markdown(rows: list[dict], n_images: int) -> str:
    label = {"python": "Python-Pipeline", "colmap": "COLMAP",
             "colmap-mvs": "COLMAP (+MVS)"}
    out = [
        f"| Metrik | {' | '.join(label.get(r['backend'], r['backend']) for r in rows)} |",
        f"|---|{'---|' * len(rows)}",
        f"| Laufzeit ({n_images} Bilder) | "
        + " | ".join(fmt(r['elapsed'], ' s', 1) for r in rows) + " |",
        "| Registrierte Kameras | "
        + " | ".join(f"{fmt(r['cameras'])}/{n_images}" if r['cameras'] is not None
                     else "n/v" for r in rows) + " |",
        "| 3D-Punkte (sparse) | "
        + " | ".join(fmt(r['points']) for r in rows) + " |",
        "| Reprojektions-RMSE | "
        + " | ".join(fmt(r['rmse'], ' px', 3) for r in rows) + " |",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--backends", nargs="+", default=["python", "colmap"],
                    choices=["python", "colmap", "colmap-mvs"])
    ap.add_argument("--out-dir", default="benchmark_out")
    ap.add_argument("extra", nargs="*",
                    help="extra args passed to every run (after --)")
    a = ap.parse_args(argv)

    img_dir = Path(a.image_dir)
    n_images = len([p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif",
                                            ".tiff", ".bmp"}]) if img_dir.is_dir() else 0
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_backend(a.image_dir, b, out_dir / f"{b}.ply", a.extra)
            for b in a.backends]

    table = to_markdown(rows, n_images)
    print("\n" + "=" * 62 + "\nTab. 1 — Kennzahlenvergleich\n" + "=" * 62)
    print(table)
    (out_dir / "table1.md").write_text(table + "\n", encoding="utf-8")
    print(f"\n[benchmark] table written to {out_dir / 'table1.md'}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
