#!/usr/bin/env python3
"""Are two identical invocations actually identical?

`paper/paper_l.md` §6.9 measured up to 57 % variation in the number of
registered cameras between byte-identical calls, because `cv2.setRNGSeed` was
never called and every RANSAC drew from OpenCV's process-global generator.

This script runs the same command N times into separate output paths and
compares the results exactly: the SHA-256 of the point cloud, the camera count,
the point count and the reprojection error. "Roughly the same" is not the
question; a seeded pipeline must return the same bytes.

    python eval/reproducibility_check.py --image-dir <dir> [--runs 3] [--extra ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-features", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(_ROOT / "eval_results" / "repro"))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra flags passed through to run_sfm.py")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in range(args.runs):
        ply = out_dir / f"run{k}.ply"
        cams = out_dir / f"run{k}.cameras.json"
        cmd = [
            sys.executable, str(_ROOT / "run_sfm.py"),
            "--image_dir", args.image_dir,
            "--output", str(ply),
            "--export-cameras", str(cams),
            "--n_features", str(args.n_features),
            "--seed", str(args.seed),
            *args.extra,
        ]
        print(f"run {k + 1}/{args.runs} …", flush=True)
        subprocess.run(cmd, cwd=str(_ROOT), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not ply.exists() or not cams.exists():
            print(f"  run {k} produced no output")
            return 1
        rec = json.loads(cams.read_text(encoding="utf-8"))
        rows.append({
            "run": k,
            "sha256": sha256(ply),
            "bytes": ply.stat().st_size,
            "cameras": rec["n_cameras_registered"],
            "points": rec["n_points"],
            "observations": rec["n_observations"],
            "rmse_px": rec["reprojection"]["rmse_px"],
            "focal_px": rec["shared_K"][0][0],
        })

    identical = all(r["sha256"] == rows[0]["sha256"] for r in rows)
    same_counts = all(
        (r["cameras"], r["points"], r["observations"]) ==
        (rows[0]["cameras"], rows[0]["points"], rows[0]["observations"])
        for r in rows
    )
    report = {"runs": rows, "identical_ply": identical, "identical_counts": same_counts,
              "seed": args.seed, "image_dir": args.image_dir}
    (out_dir / "reproducibility.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print()
    hdr = "%-4s %8s %8s %8s %10s %10s %s" % (
        "run", "cams", "points", "obs", "rmse px", "focal px", "sha256[:16]")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-4d %8d %8d %8d %10.4f %10.2f %s" % (
            r["run"], r["cameras"], r["points"], r["observations"],
            r["rmse_px"], r["focal_px"], r["sha256"][:16]))
    print()
    print(f"identical point clouds : {identical}")
    print(f"identical counts       : {same_counts}")
    print(f"artefacts: {out_dir}")
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
