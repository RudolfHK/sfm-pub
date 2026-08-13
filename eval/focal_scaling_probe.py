#!/usr/bin/env python3
"""Diagnose why bundle adjustment never moves the focal length.

Observation being explained: on every dataset the reported focal stays pinned at
its initial value (`f=2736.0 (delta+0.00)`) while ground truth is ~1861 px, and
BA always terminates on `xtol`.

Hypothesis: `least_squares(..., method="trf")` is called with the default
`x_scale=1.0`, so one trust-region step is the same absolute size for the focal
(magnitude ~2.7e3) as for a rotation component (magnitude ~1). The focal is
effectively frozen before `xtol` fires.

Test: run the pipeline twice in-process, once unmodified and once with
`x_scale="jac"` injected into every `least_squares` call — nothing else changes.
If the hypothesis holds, only the second run moves the focal.

This probe never edits the repository; it wraps `scipy.optimize.least_squares`
before `run_sfm` imports it.

Usage
-----
    python eval/focal_scaling_probe.py --image-dir <dir> --results-dir eval_results \
        --gt-dir <dir-with-P-files> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import gt_pose_eval

_ROOT = Path(__file__).resolve().parents[1]

# Child-process driver: patch first, import run_sfm second.
_CHILD = textwrap.dedent(
    '''
    import sys
    sys.path.insert(0, {root!r})

    x_scale_mode = {mode!r}
    if x_scale_mode is not None:
        import scipy.optimize as _so
        _orig = _so.least_squares

        def _patched(*a, **kw):
            kw["x_scale"] = x_scale_mode
            return _orig(*a, **kw)

        _so.least_squares = _patched

    import run_sfm
    sys.exit(run_sfm.main({argv!r}))
    '''
)


def _run(image_dir: str, out_ply: Path, cams: Path, ckpt: Path,
         mode: str | None) -> dict:
    argv = [
        "--image_dir", image_dir,
        "--output", str(out_ply),
        "--export-cameras", str(cams),
        "--n_features", "8000",
        "--checkpoint-dir", str(ckpt),
        "--resume",
    ]
    code = _CHILD.format(root=str(_ROOT), mode=mode, argv=argv)
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       cwd=str(_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log = (p.stdout or "") + (p.stderr or "")
    refined = [ln.strip() for ln in log.splitlines() if "BA refined" in ln]
    return {"rc": p.returncode, "last_ba_line": refined[-1] if refined else None,
            "n_ba_calls": len(refined), "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir)
    out: dict = {}

    for label, mode in [("baseline_x_scale_1", None), ("patched_x_scale_jac", "jac")]:
        ply = res_dir / f"_probe_{label}.ply"
        cams = res_dir / f"_probe_{label}.cameras.json"
        r = _run(args.image_dir, ply, cams, Path(args.ckpt_dir), mode)
        entry = {"x_scale": mode or "1.0 (scipy default)",
                 "rc": r["rc"], "n_ba_calls": r["n_ba_calls"],
                 "last_ba_line": r["last_ba_line"]}
        if cams.exists():
            g = gt_pose_eval.evaluate(cams, gt_dir)
            entry["focal_est_px"] = g["focal"]["est_px"]
            entry["focal_gt_px"] = g["focal"]["gt_px"]
            entry["focal_error_pct"] = g["focal"]["error_pct"]
            entry["n_registered"] = g["n_registered"]
            entry["reproj_rmse_px"] = (g.get("reprojection") or {}).get("rmse_px")
            a = (g.get("alignment") or {}).get("ransac")
            if a:
                entry["pos_err_median_pct"] = a["position_err_pct_of_extent"]["median"]
                entry["rot_err_median_deg"] = a["rotation_err_deg"]["median"]
        out[label] = entry

    b, p = out["baseline_x_scale_1"], out["patched_x_scale_jac"]
    out["conclusion"] = (
        "CONFIRMED — focal only moves once the parameter vector is scaled"
        if (b.get("focal_error_pct") is not None
            and p.get("focal_error_pct") is not None
            and abs(p["focal_error_pct"]) < abs(b["focal_error_pct"]) - 1.0)
        else "NOT CONFIRMED — x_scale is not the limiting factor"
    )

    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
