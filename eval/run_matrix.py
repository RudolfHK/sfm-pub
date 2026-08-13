#!/usr/bin/env python3
"""Run the whole evaluation matrix, sequentially, one config at a time.

Sequential by design: peak-memory and wall-clock numbers are only comparable if
runs do not compete for CPU. Already-completed configs are skipped unless
--force is given, so the matrix is restartable after an interruption.

Usage
-----
    python eval/run_matrix.py --data-root <dir-with-buddha_XX-subsets> \
        --results-dir eval_results [--only n20_base,n67_base] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_config import run  # same directory

# name -> (dataset key, extra flags, timeout seconds)
# dataset key is resolved against --data-root as <key>/
MATRIX: list[tuple[str, str, list[str], float]] = [
    # ── Baselines at three scales (fresh runs; comparable timings) ────────
    ("n06_base",  "buddha_06",  ["--n_features", "8000"], 1800),
    ("n06_mini",  "buddha_mini6", ["--n_features", "8000"], 1800),
    ("n20_base",  "buddha_20",  ["--n_features", "8000"], 5400),
    ("n67_base",  "buddha_67",  ["--n_features", "8000"], 21600),

    # ── Efficiency levers, all at N=20 so they are comparable ─────────────
    ("n20_seq",       "buddha_20", ["--n_features", "8000",
                                    "--match_strategy", "sequential",
                                    "--sequential_window", "5"], 5400),
    ("n20_localba",   "buddha_20", ["--n_features", "8000",
                                    "--local-ba-window", "10"], 5400),
    ("n20_trackmerge", "buddha_20", ["--n_features", "8000",
                                     "--track-merge"], 5400),
    ("n20_feat4k",    "buddha_20", ["--n_features", "4000"], 5400),
    ("n20_feat12k",   "buddha_20", ["--n_features", "12000"], 7200),

    # ── Reproducibility: byte-identical command, repeated runs ────────────
    ("n20_repeat", "buddha_20", ["--n_features", "8000"], 5400),
    ("n20_repeat2", "buddha_20", ["--n_features", "8000"], 5400),
    ("n20_repeat3", "buddha_20", ["--n_features", "8000"], 5400),

    # ── The configuration documented in run_sfm.py's own usage header ─────
    ("n67_user", "buddha_67", ["--n_features", "12000", "--ratio", "0.7",
                               "--min_inliers", "25", "--max_reproj_error", "3.0",
                               "--dense"], 28800),

    # ── Optional heavy stages ─────────────────────────────────────────────
    ("n20_dense", "buddha_20", ["--n_features", "8000", "--dense",
                                "--mesh", "--mesh-quality", "medium"], 10800),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--ckpt-root", default=None,
                    help="Parent dir for per-dataset checkpoint dirs "
                         "(default: <data-root>/_ckpt).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of config names to run.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run configs that already have a .run.json.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else data_root / "_ckpt"
    wanted = set(args.only.split(",")) if args.only else None

    for name, dataset, extra, timeout in MATRIX:
        if wanted is not None and name not in wanted:
            continue
        marker = results / f"{name}.run.json"
        if marker.exists() and not args.force:
            print(f"[skip] {name} (already done)", flush=True)
            continue
        image_dir = data_root / dataset
        if not image_dir.is_dir():
            print(f"[skip] {name}: missing dataset {image_dir}", flush=True)
            continue

        # One checkpoint dir per dataset: prevents different image sets from
        # repeatedly invalidating each other's feature cache.
        flags = [*extra, "--checkpoint-dir", str(ckpt_root / dataset)]
        print(f"[run ] {name}  ({dataset})  {' '.join(extra)}", flush=True)
        t0 = time.time()
        rec = run(name, str(image_dir), results, flags, timeout)
        s = rec.get("summary", {})
        print(
            f"[done] {name}  rc={rec['returncode']} "
            f"{time.time()-t0:.0f}s  "
            f"cams={s.get('n_cameras_registered')}/{s.get('n_images')} "
            f"pts={s.get('n_points')} "
            f"peak={rec.get('peak_rss_mb')}MB",
            flush=True,
        )

    print(json.dumps(sorted(p.name for p in results.glob("*.run.json")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
