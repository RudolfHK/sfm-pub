#!/usr/bin/env python3
"""Measure the impact of the accuracy fixes, before against after.

Runs the pipeline twice on the same images — once with every fix disabled
(reproducing the behaviour `EVALUATION_RESULTS.md` measured) and once with the
shipped defaults — and scores both against ground-truth poses.

The "before" configuration is not a different code path: it is the current code
with `--ba-no-param-scaling`, the old solver tolerances, no focal search and no
track completion.  Anything that moves between the two rows is therefore
attributable to the fixes rather than to unrelated drift.

Judged on ground-truth pose error, never on reprojection error: section 6.6 of
`paper/paper_l.md` established that the pipeline's self-reported RMSE is blind
to this failure mode, with the single most accurate configuration of the
original series posting a middling RMSE and the best RMSE belonging to a run
that registered 4 of 20 cameras.

Note on the wall-clock column: both rows use the parallel matcher, because the
switches below turn off only the *accuracy* fixes.  The column therefore shows
what the accuracy work costs, not the end-to-end speedup — for that, compare
`--match-workers 1` against the default.

Usage
-----
    python eval/fix_impact.py --image-dir <buddha_20> --gt-dir <buddha> \\
        [--results-dir eval_results] [--n-features 8000]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "eval"))

# Every accuracy fix, as the switch that turns it *off*.  The "before" row is
# the current binary with all of them off, so nothing but the fixes can
# explain a difference between the rows.
OFF = {
    "focal":  ["--no-focal-search"],           # keep the max(W, H) guess
    "ba":     ["--ba-no-param-scaling",        # old conditioning …
               "--ba-ftol", "1e-4",            # … and old tolerances
               "--ba-xtol", "1e-4",
               "--ba-gtol", "1e-4"],
    "tracks": ["--no-track-completion", "--no-global-tracks"],
    "loop":   ["--no-geometric-loop-closure"],
}
ALL_FIXES = list(OFF)


def flags_with(enabled) -> list:
    """Command-line flags that enable exactly the named fixes."""
    out: list = []
    for name in ALL_FIXES:
        if name not in enabled:
            out.extend(OFF[name])
    return out


# Rows measured by default: the old state, each fix alone, then all of them.
def default_configs() -> list:
    configs = [("before", [])]
    configs += [(f"+{name}", [name]) for name in ALL_FIXES]
    configs.append(("after", ALL_FIXES))
    return configs


def run(name: str, image_dir: str, results_dir: Path, extra: list,
        n_features: int, ckpt: Path) -> dict:
    cmd = [
        sys.executable, str(_ROOT / "eval" / "run_config.py"),
        "--name", name, "--image_dir", image_dir,
        "--results-dir", str(results_dir), "--",
        "--n_features", str(n_features),
        "--checkpoint-dir", str(ckpt),
        "--resume",
        *extra,
    ]
    subprocess.run(cmd, cwd=str(_ROOT), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((results_dir / f"{name}.run.json").read_text(encoding="utf-8"))


def score(cameras_json: Path, gt_dir: str) -> dict:
    """Ground-truth pose error, via the same evaluator the report uses."""
    from gt_pose_eval import evaluate  # noqa: E402

    return evaluate(cameras_json, Path(gt_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--gt-dir", required=True, help="Directory holding *_P.txt")
    ap.add_argument("--results-dir", default=str(_ROOT / "eval_results" / "fix_impact"))
    ap.add_argument("--n-features", type=int, default=8000)
    ap.add_argument(
        "--only", default=None,
        help="Comma-separated subset of the configuration names to run, "
             "e.g. 'before,after'.  Default runs the full ablation.",
    )
    ap.add_argument(
        "--extra", default=None,
        help="Extra flags appended to every configuration, comma-separated.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = default_configs()
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c[0] in wanted]

    rows = []
    for name, enabled in configs:
        flags = flags_with(enabled)
        # Separate checkpoint dirs: the two runs must not share cached state.
        if args.extra:
            flags = flags + args.extra.split(",")
        # One shared checkpoint for every configuration.  None of the switches
        # above touches feature extraction or matching — they change bundle
        # adjustment, focal initialisation, track handling and loop closure,
        # all of which run afterwards — so every row consumes byte-identical
        # features and matches.  That is not only faster, it also removes
        # matching noise as an explanation for any difference between rows.
        rec = run(name, args.image_dir, results_dir, flags, args.n_features,
                  results_dir / "ckpt_shared")
        cams = results_dir / f"{name}.cameras.json"
        row = {"config": name, "enabled": enabled, "flags": flags,
               "wall_s": rec.get("wall_s"),
               "peak_rss_mb": rec.get("peak_rss_mb"),
               "summary": rec.get("summary", {})}
        if cams.exists():
            row["gt"] = score(cams, args.gt_dir)
        rows.append(row)

    (results_dir / "fix_impact.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    def g(row, *path, default=None):
        cur = row
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    print()
    hdr = ("%-8s %6s %8s %8s %7s %8s %9s %9s %9s %8s" %
           ("config", "cams", "points", "obs", "track", "rmse px",
            "focal %", "rot med°", "pos med%", "wall s"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = r["summary"]
        # 'lsq' is the plain Umeyama fit; gt_pose_eval also reports a RANSAC
        # variant under 'ransac' in the same structure.
        print("%-8s %6s %8s %8s %7s %8s %9s %9s %9s %8s" % (
            r["config"],
            s.get("n_cameras_registered", "?"),
            s.get("n_points", "?"),
            s.get("n_observations", "?"),
            ("%.2f" % s["mean_track_length"]) if "mean_track_length" in s else "?",
            _fmt(g(r, "summary", "reprojection", "rmse_px")),
            _fmt(g(r, "gt", "focal", "error_pct")),
            _fmt(g(r, "gt", "alignment", "lsq", "rotation_err_deg", "median")),
            _fmt(g(r, "gt", "alignment", "lsq",
                   "position_err_pct_of_extent", "median")),
            r.get("wall_s", "?"),
        ))
    print()
    print(f"artefacts: {results_dir}")
    return 0


def _fmt(v) -> str:
    return "?" if v is None else ("%.3f" % v)


if __name__ == "__main__":
    raise SystemExit(main())
