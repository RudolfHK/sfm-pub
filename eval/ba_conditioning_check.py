#!/usr/bin/env python3
"""Verify the bundle-adjustment conditioning fix (work item W2).

`paper/paper_l.md` section 6.5 established, on synthetic data with poses and
structure pinned to ground truth and only the focal started wrong, that the
solver never moved the focal because it terminated on `xtol` while the residual
was still above 100 px.  Analytic parameter scaling plus tighter tolerances
recovered it.  This script re-runs that experiment against the *shipped*
`BundleAdjuster` so the fix stays verified as the code changes.

It reuses the scene builder from `paper/scripts/focal_experiment.py`, so the
numbers are directly comparable with Tab. 8 of the paper.

Variants compared, all starting from f = 2736.0 px (+47.0 %), truth 1860.9 px:

    pre-fix defaults      uniform scaling, tolerances 1e-4  (what shipped before)
    scaling only          analytic scaling, tolerances 1e-4
    tolerances only       uniform scaling, tolerances 1e-8
    shipped default       analytic scaling, tolerances 1e-8  <- current default

Usage
-----
    python eval/ba_conditioning_check.py --gt-dir <buddha> \\
        [--out eval_results/ba_conditioning.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper" / "scripts"))

from focal_experiment import (  # noqa: E402
    F_GUESS,
    F_TRUE,
    build_scene,
    make_observations,
    K_of,
    run_ba_only,
)
from sfm.bundle_adjustment import BundleAdjuster  # noqa: E402

log = logging.getLogger("ba_conditioning_check")

# Tolerance settings. max_nfev is multiplied by the parameter count inside
# BundleAdjuster; 1 gives roughly 4500 evaluations on this problem, matching
# the budget used for Tab. 8 so the rows stay comparable.
_LOOSE = dict(ftol=1e-4, gtol=1e-4, xtol=1e-4)
_TIGHT = dict(ftol=1e-8, gtol=1e-8, xtol=1e-8)

VARIANTS = [
    ("pre-fix defaults", dict(**_LOOSE, max_nfev=1, param_scaling=False)),
    ("scaling only",     dict(**_LOOSE, max_nfev=1, param_scaling=True)),
    ("tolerances only",  dict(**_TIGHT, max_nfev=1, param_scaling=False)),
    ("shipped default",  dict(**_TIGHT, max_nfev=1, param_scaling=True)),
]

# Acceptance criterion from IMPROVEMENT_PROMPT.md W2/W3: the focal must come
# back to within 10 % of truth on this bench.
ACCEPT_FOCAL_ERR_PCT = 10.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="Directory holding *_P.txt")
    ap.add_argument("--out", default=str(REPO / "eval_results" / "ba_conditioning.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    rng = np.random.default_rng(args.seed)
    keys, poses, X, normals, centre = build_scene(Path(args.gt_dir), rng)
    feats, matches, widx = make_observations(poses, X, normals, K_of(F_TRUE), rng)
    log.info("scene: %d cameras, %d points, %d matchable pairs",
             len(poses), len(X), len(matches))

    rows = []
    for label, opts in VARIANTS:
        r = run_ba_only(F_GUESS, poses, X, widx, feats, args.seed,
                        adjuster=BundleAdjuster(**opts))
        r["variant"] = label
        r["options"] = {k: v for k, v in opts.items()}
        rows.append(r)
        log.info("%-18s f %.1f -> %.1f px (%+.2f %%)  RMSE %.1f -> %.1f px  "
                 "rot %.3f deg  %s  %.1f s",
                 label, r["f_start"], r["f_end"], r["f_end_err_pct"],
                 r["rmse_init_px"], r["rmse_final_px"], r["rot_err_med_deg"],
                 r["termination"], r["wall_s"])

    shipped = rows[-1]
    passed = abs(shipped["f_end_err_pct"]) <= ACCEPT_FOCAL_ERR_PCT

    result = {
        "setup": {
            "gt_dir": str(args.gt_dir),
            "seed": args.seed,
            "n_cameras": len(poses),
            "f_true_px": F_TRUE,
            "f_start_px": F_GUESS,
            "f_start_err_pct": 100.0 * (F_GUESS - F_TRUE) / F_TRUE,
            "images": [str(k) for k in keys],
        },
        "variants": rows,
        "acceptance": {
            "criterion": f"|focal error| <= {ACCEPT_FOCAL_ERR_PCT} % for the shipped default",
            "measured_pct": shipped["f_end_err_pct"],
            "passed": bool(passed),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("written: %s", out)

    print()
    print(f"{'variant':<18} {'focal px':>10} {'err %':>8} {'RMSE px':>9} "
          f"{'rot deg':>8} {'s':>7}  termination")
    for r in rows:
        print(f"{r['variant']:<18} {r['f_end']:>10.1f} {r['f_end_err_pct']:>+8.2f} "
              f"{r['rmse_final_px']:>9.2f} {r['rot_err_med_deg']:>8.3f} "
              f"{r['wall_s']:>7.1f}  {r['termination']}")
    print()
    print(f"acceptance (|focal err| <= {ACCEPT_FOCAL_ERR_PCT} %): "
          f"{'PASS' if passed else 'FAIL'} at {shipped['f_end_err_pct']:+.2f} %")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
