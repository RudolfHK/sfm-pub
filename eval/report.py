#!/usr/bin/env python3
"""Aggregate every measured artefact into EVALUATION_RESULTS.md.

Reads only what previous stages wrote (`*.run.json`, `robustness.json`,
`checkpoint_test.json`) plus the ground-truth directory, so it never re-runs the
pipeline and never invents a number: anything missing is rendered as an explicit
`n/a`. The narrative half of the report — code audit, defect list, verdicts —
lives in `eval/findings.md` and is appended verbatim, which keeps hand-written
analysis and machine-measured tables in one reproducible document.

Usage
-----
    python eval/report.py --results-dir eval_results \
        --gt-dir .../buddha --gt-dir-mini6 .../buddha_mini6 \
        --out EVALUATION_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import gt_pose_eval
import ply_stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _f(v, spec: str = ".2f", dash: str = "n/a"):
    """Format a number, or return a dash when it is missing."""
    if v is None:
        return dash
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(results_dir.glob("*.run.json")):
        with open(f, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def enrich(runs: list[dict], gt_dir: Path, gt_mini: Path) -> None:
    """Attach GT pose metrics and point-cloud stats to each run in place."""
    for r in runs:
        gt = gt_mini if "mini6" in r["image_dir"].replace("\\", "/") else gt_dir
        r["gt_dir_used"] = str(gt)
        if r.get("cameras_json") and Path(r["cameras_json"]).exists():
            try:
                r["gt"] = gt_pose_eval.evaluate(Path(r["cameras_json"]), gt)
            except Exception as exc:  # a broken artefact must not kill the report
                r["gt"] = None
                r["gt_error"] = str(exc)
        else:
            r["gt"] = None
        if r.get("ply") and Path(r["ply"]).exists():
            try:
                r["ply_stats"] = ply_stats.analyse(Path(r["ply"]))
            except Exception as exc:
                r["ply_stats"] = None
                r["ply_error"] = str(exc)
        else:
            r["ply_stats"] = None


# ── table builders ───────────────────────────────────────────────────────────

def tbl_efficiency(runs: list[dict]) -> str:
    out = [
        "| Config | Images | Wall (s) | Features (s) | Matching (s) | Verify (s) "
        "| Recon+BA (s) | Export (s) | Dense (s) | Mesh (s) | Peak RSS (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        s = r.get("summary", {}) or {}
        st = s.get("stage_times_s", {}) or {}
        out.append(
            f"| `{r['name']}` | {s.get('n_images', 'n/a')} | {_f(r.get('wall_s'), '.1f')} "
            f"| {_f(st.get('features'), '.1f')} | {_f(st.get('matching'), '.1f')} "
            f"| {_f(st.get('verification'), '.1f')} | {_f(st.get('reconstruction'), '.1f')} "
            f"| {_f(st.get('export'), '.1f')} | {_f(st.get('dense'), '.1f', '—')} "
            f"| {_f(st.get('mesh'), '.1f', '—')} | {_f(r.get('peak_rss_mb'), '.0f')} |"
        )
    return "\n".join(out)


def tbl_scaling(runs: list[dict]) -> str:
    by_name = {r["name"]: r for r in runs}
    rows = [("n06_base", 6), ("n20_base", 20), ("n67_base", 67)]
    out = [
        "| Images | Pairs (N·(N−1)/2) | Matching (s) | s / pair | Total (s) "
        "| s / image | Peak RSS (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, n in rows:
        r = by_name.get(name)
        if not r:
            continue
        st = (r.get("summary", {}) or {}).get("stage_times_s", {}) or {}
        pairs = n * (n - 1) // 2
        m = st.get("matching")
        out.append(
            f"| {n} | {pairs} | {_f(m, '.1f')} | {_f(m / pairs if m else None, '.3f')} "
            f"| {_f(r.get('wall_s'), '.1f')} | {_f(r['wall_s'] / n, '.1f')} "
            f"| {_f(r.get('peak_rss_mb'), '.0f')} |"
        )
    return "\n".join(out)


def tbl_quality(runs: list[dict]) -> str:
    out = [
        "| Config | Registered | Points | Mean track | Reproj RMSE (px) | "
        "Focal err | Pos err med (% extent) | Pos err max | Rot err med (°) | Rot err max (°) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        s = r.get("summary", {}) or {}
        g = r.get("gt")
        rep = (s.get("reprojection") or {})
        cells = [
            f"`{r['name']}`",
            f"{s.get('n_cameras_registered', 'n/a')}/{s.get('n_images', 'n/a')}",
            f"{s.get('n_points', 'n/a'):,}" if s.get("n_points") else "n/a",
            _f(s.get("mean_track_length")),
            _f(rep.get("rmse_px"), ".3f"),
        ]
        if g and g.get("focal"):
            cells.append(f"{g['focal']['error_pct']:+.1f}%")
        else:
            cells.append("n/a")
        a = (g or {}).get("alignment")
        if a:
            r_ = a["ransac"]
            cells += [
                _f(r_["position_err_pct_of_extent"]["median"]),
                _f(r_["position_err_pct_of_extent"]["max"]),
                _f(r_["rotation_err_deg"]["median"]),
                _f(r_["rotation_err_deg"]["max"]),
            ]
        else:
            cells += ["n/a"] * 4
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("Pose columns use the RANSAC Sim(3) alignment; `n/a` means fewer "
               "than three cameras matched ground truth, so no similarity "
               "transform is defined.")
    return "\n".join(out)


def tbl_cloud(runs: list[dict]) -> str:
    out = [
        "| Config | Points | Bytes/pt | NN dist median | NN p95 | Duplicates | "
        "Flyers (NN>10×med) | Outliers (r>5×med) | Fallback grey |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        p = r.get("ply_stats")
        if not p:
            continue
        nn = p.get("nn_distance") or {}
        col = p.get("color") or {}
        out.append(
            f"| `{r['name']}` | {p['n_points']:,} | {_f(p.get('bytes_per_point'), '.1f')} "
            f"| {_f(nn.get('median'), '.4f')} | {_f(nn.get('p95'), '.4f')} "
            f"| {_f(100 * p['duplicate_fraction'] if p.get('duplicate_fraction') is not None else None, '.2f')}% "
            f"| {_f(100 * p['flyer_fraction_nn_gt_10x_median'] if p.get('flyer_fraction_nn_gt_10x_median') is not None else None, '.2f')}% "
            f"| {_f(100 * p['outlier_fraction_r_gt_5x_median'] if p.get('outlier_fraction_r_gt_5x_median') is not None else None, '.2f')}% "
            f"| {_f(100 * col['fallback_grey_fraction'] if col.get('fallback_grey_fraction') is not None else None, '.2f')}% |"
        )
    return "\n".join(out)


def tbl_reproducibility(runs: list[dict]) -> tuple[str, dict]:
    """Runs sharing an identical command line, compared against each other."""
    group = [r for r in runs
             if r["name"] in {"n20_base", "n20_repeat", "n20_repeat2", "n20_repeat3"}]
    out = ["| Run | Registered | Points | Reproj RMSE (px) |",
           "|---|---:|---:|---:|"]
    cams, pts = [], []
    for r in group:
        s = r.get("summary", {}) or {}
        cams.append(s.get("n_cameras_registered"))
        pts.append(s.get("n_points"))
        out.append(
            f"| `{r['name']}` | {s.get('n_cameras_registered')}/{s.get('n_images')} "
            f"| {s.get('n_points'):,} "
            f"| {_f((s.get('reprojection') or {}).get('rmse_px'), '.3f')} |"
        )
    cams = [c for c in cams if c is not None]
    pts = [p for p in pts if p is not None]
    stats = {
        "n_runs": len(group),
        "cameras": cams,
        "points": pts,
        "camera_spread_pct": (100 * (max(cams) - min(cams)) / max(cams)) if cams else None,
        "point_spread_pct": (100 * (max(pts) - min(pts)) / max(pts)) if pts else None,
    }
    return "\n".join(out), stats


def tbl_robustness(results_dir: Path) -> str:
    f = results_dir / "robustness.json"
    if not f.exists():
        return "_Not run (`eval/robustness.py` produced no `robustness.json`)._"
    with open(f, encoding="utf-8") as fh:
        cases = json.load(fh)
    out = ["| Case | Verdict | Observed |", "|---|---|---|"]
    for name, c in cases.items():
        out.append(f"| `{name}` | **{c['verdict']}** | {c['why']} |")
    n_pass = sum(1 for c in cases.values() if c["verdict"] == "PASS")
    out.append("")
    out.append(f"**{n_pass}/{len(cases)} cases pass.**")
    return "\n".join(out)


def sec_checkpoint(results_dir: Path) -> str:
    f = results_dir / "checkpoint_test.json"
    if not f.exists():
        return "_Not run (`eval/checkpoint_test.py` produced no `checkpoint_test.json`)._"
    with open(f, encoding="utf-8") as fh:
        d = json.load(fh)
    steps, v = d["steps"], d["verdict"]
    out = [
        "| Step | Wall (s) | Features cached | Matches cached | Cameras | Points |",
        "|---|---:|:---:|:---:|---:|---:|",
    ]
    for k, s in steps.items():
        out.append(
            f"| `{k}` | {_f(s.get('wall_s'), '.1f')} "
            f"| {'yes' if s.get('features_from_cache') else 'no'} "
            f"| {'yes' if s.get('matches_from_cache') else 'no'} "
            f"| {s.get('cameras')} | {s.get('points')} |"
        )
    out += [
        "",
        f"- `--resume` speedup: **{_f(v.get('speedup_resume'), '.2f')}×**",
        f"- On-disk checkpoint size: **{v.get('checkpoint_bytes', 0) / 1e6:.1f} MB**",
        f"- Cache hit on resume: **{v.get('cache_hit_warm')}**",
        f"- Match cache invalidated by `--ratio` change: **{v.get('invalidation_on_ratio_change')}**",
        f"- Feature cache invalidated by image-set change: **{v.get('invalidation_on_image_change')}**",
        f"- Resumed run reproduces the cold run: **{v.get('resume_reproduces_cold')}**",
        f"- Two resumed runs agree with each other: **{v.get('two_resumes_agree')}**",
        "",
        v.get("determinism_note", ""),
    ]
    return "\n".join(out)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--gt-dir-mini6", default=None)
    ap.add_argument("--out", default="EVALUATION_RESULTS.md")
    ap.add_argument("--findings", default=str(_HERE / "findings.md"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    gt_dir = Path(args.gt_dir)
    gt_mini = Path(args.gt_dir_mini6) if args.gt_dir_mini6 else gt_dir

    runs = load_runs(results_dir)
    if not runs:
        print(f"No *.run.json in {results_dir}", file=sys.stderr)
        return 1
    enrich(runs, gt_dir, gt_mini)

    repro_tbl, repro_stats = tbl_reproducibility(runs)
    n_gt = len(gt_pose_eval.load_gt(gt_dir))

    import cv2
    import numpy as np
    import scipy

    doc = f"""# `sfm-pub` — Measured Evaluation Results

Generated by `python eval/report.py`. Every number below comes from an artefact
in `{results_dir}/`; nothing is estimated. See
[EVALUATION_PROMPT.md](EVALUATION_PROMPT.md) for the evaluation brief and
`eval/` for the harness.

## Environment

| Item | Value |
|---|---|
| Platform | {platform.platform()} |
| CPU | {platform.processor() or 'n/a'} |
| Python | {platform.python_version()} |
| OpenCV | {cv2.__version__} |
| NumPy / SciPy | {np.__version__} / {scipy.__version__} |
| Absent optional deps | torch, kornia, pyceres, poselib, faiss, COLMAP binary |
| Dataset | AliceVision Buddha, {n_gt} images @ 2736×1540, ground-truth `P` per image |
| Ground truth | `{gt_dir}` (byte-identical images to the supplied `imgs_only`) |

Because torch/kornia/pyceres/poselib/COLMAP are unavailable, the `superpoint`,
`disk`, `loftr`, `dinov2`, `pyceres`, `poselib` and all COLMAP paths are
**untested by environment**; they appear below only in the robustness table,
which checks that they fail comprehensibly rather than silently.

---

## A. Efficiency

### A.1 Per-stage wall time and peak memory

{tbl_efficiency(runs)}

### A.2 Scaling with image count

{tbl_scaling(runs)}

### A.3 Checkpointing and `--resume`

{sec_checkpoint(results_dir)}

---

## B. Quality of results

### B.1 Completeness and accuracy against ground truth

Camera poses are aligned to ground truth with a Sim(3) (Umeyama) fit on camera
centres before any error is computed; monocular SfM is scale-free, so an
unaligned comparison would be meaningless. Position errors are normalised by the
full ground-truth trajectory extent so runs that register different numbers of
cameras stay comparable.

{tbl_quality(runs)}

### B.2 Point-cloud health

{tbl_cloud(runs)}

### B.3 Reproducibility — identical command, repeated

{repro_tbl}

Camera count spread across {repro_stats['n_runs']} identical runs:
**{_f(repro_stats['camera_spread_pct'], '.0f')}%**
({repro_stats['cameras']}). Point count spread:
**{_f(repro_stats['point_spread_pct'], '.0f')}%** ({repro_stats['points']}).

### B.4 Robustness and degenerate inputs

{tbl_robustness(results_dir)}

---
"""

    findings = Path(args.findings)
    if findings.exists():
        doc += "\n" + findings.read_text(encoding="utf-8")
    else:
        doc += f"\n_(No narrative findings file at {findings}.)_\n"

    Path(args.out).write_text(doc, encoding="utf-8")
    print(f"wrote {args.out} ({len(doc):,} chars) from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
