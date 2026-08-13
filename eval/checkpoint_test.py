#!/usr/bin/env python3
"""Checkpoint reuse, cache invalidation, and determinism isolation.

Three questions, one experiment:

1. How much time does `--resume` actually save?
2. Is the reuse *correct* — does a resumed run reproduce the cold run, and do
   parameter changes that should invalidate a cache actually invalidate it?
3. Where does run-to-run variation come from? Two resumed runs consume byte-
   identical features and matches, so any difference between them is produced
   downstream, in geometric verification / incremental SfM / BA.

Usage
-----
    python eval/checkpoint_test.py --image-dir <dir> --results-dir eval_results \
        --ckpt-dir <scratch>/ckpt_ct [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_RE_CAMS = re.compile(r"Cameras\s*:\s*(\d+)\s*registered")
_RE_PTS = re.compile(r"Points\s*:\s*([\d,]+)")
_RE_RMSE = re.compile(r"BA final RMSE:\s*([\d.]+)\s*px")
_RE_SKIP_FEAT = re.compile(r"\[2/6\][\s\S]{0,400}?Skipped \(loaded from checkpoint\)")
_RE_SKIP_MATCH = re.compile(r"Skipped \(loaded from checkpoint — (\d+) pairs\)")
_RE_STALE = re.compile(r"Checkpoint (\S+) is stale")


def _run(image_dir: str, out: Path, ckpt: Path, extra: list[str]) -> dict:
    cmd = [sys.executable, "-X", "utf8", str(_ROOT / "run_sfm.py"),
           "--image_dir", image_dir, "--output", str(out),
           "--checkpoint-dir", str(ckpt), *extra]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log = (p.stdout or "") + (p.stderr or "")
    cams = _RE_CAMS.search(log)
    pts = _RE_PTS.search(log)
    rmse = _RE_RMSE.findall(log)
    return {
        "wall_s": round(time.time() - t0, 2),
        "rc": p.returncode,
        "cameras": int(cams.group(1)) if cams else None,
        "points": int(pts.group(1).replace(",", "")) if pts else None,
        "final_ba_rmse": float(rmse[-1]) if rmse else None,
        "features_from_cache": bool(_RE_SKIP_FEAT.search(log)),
        "matches_from_cache": bool(_RE_SKIP_MATCH.search(log)),
        "stale_reported": _RE_STALE.findall(log),
        "log": log,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--n-features", default="8000")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir)
    if ckpt.exists():
        shutil.rmtree(ckpt)
    base = ["--n_features", args.n_features]
    out = res_dir / "_ckpt_test.ply"

    steps: dict = {}

    # 1 — cold run, populates the cache
    steps["cold"] = _run(args.image_dir, out, ckpt, base)
    # 2 — resumed run, must reuse both caches
    steps["warm1"] = _run(args.image_dir, out, ckpt, [*base, "--resume"])
    # 3 — second resumed run: identical inputs to warm1 by construction
    steps["warm2"] = _run(args.image_dir, out, ckpt, [*base, "--resume"])
    # 4 — a changed matching parameter must invalidate the match cache only
    steps["changed_ratio"] = _run(args.image_dir, out, ckpt,
                                  [*base, "--resume", "--ratio", "0.65"])
    # 5 — a changed image set must invalidate the feature cache
    tmp_dir = res_dir / "_ckpt_test_imgs"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    srcs = sorted(p for p in Path(args.image_dir).iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for p in srcs[:-1]:                       # drop one image
        shutil.copy2(p, tmp_dir / p.name)
    steps["changed_image_set"] = _run(str(tmp_dir), out, ckpt,
                                      [*base, "--resume"])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    ckpt_bytes = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())

    cold, w1, w2 = steps["cold"], steps["warm1"], steps["warm2"]
    verdict = {
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_files": {f.name: f.stat().st_size
                             for f in sorted(ckpt.rglob("*")) if f.is_file()},
        "speedup_resume": (round(cold["wall_s"] / w1["wall_s"], 2)
                           if w1["wall_s"] else None),
        "cache_hit_warm": w1["features_from_cache"] and w1["matches_from_cache"],
        "invalidation_on_ratio_change": (
            steps["changed_ratio"]["features_from_cache"]
            and not steps["changed_ratio"]["matches_from_cache"]
        ),
        "invalidation_on_image_change": (
            not steps["changed_image_set"]["features_from_cache"]
        ),
        "resume_reproduces_cold": (
            cold["cameras"] == w1["cameras"] and cold["points"] == w1["points"]
        ),
        "two_resumes_agree": (
            w1["cameras"] == w2["cameras"] and w1["points"] == w2["points"]
        ),
        "determinism_note": (
            "warm1 and warm2 consume identical cached features AND matches. "
            "Any difference between them is generated downstream of matching "
            "(geometric verification / incremental SfM / BA)."
        ),
    }

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "log"}
               for k, v in steps.items()}
    payload = {"steps": summary, "verdict": verdict}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
