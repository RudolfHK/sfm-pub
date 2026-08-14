#!/usr/bin/env python3
"""Does the pipeline still work when the scene is a plane?

`paper/paper_l.md` §7 listed planar scenes as an unhandled limit: when a
homography explains more than 85 % of the inliers, the fundamental matrix is
ill-conditioned, and the verifier used to drop the pair. Dropping avoids a wrong
pose but loses the data, and on a scene that is mostly planar it disconnects the
graph entirely.

The Buddha set contains no planar pair, so this failure cannot be exercised on
it. This script builds a synthetic planar scene with known poses, runs the real
`GeometricVerifier`, and reports whether the pair survives and how accurate the
recovered rotation is. Both the old behaviour (`planar_fallback=False`) and the
new one are measured, so the difference is visible rather than asserted.

    python eval/planar_check.py [--json eval_results/planar_check.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sfm.geometric_verification import GeometricVerifier  # noqa: E402

logging.basicConfig(level=logging.WARNING)

W, H = 1600, 1200
F = 1400.0
NOISE_PX = 0.3


def build_scene(kind: str, n_points: int, rng):
    """Points on a plane, or in a slab of given thickness, plus two cameras."""
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], float)

    x = rng.uniform(-1.0, 1.0, n_points)
    y = rng.uniform(-0.75, 0.75, n_points)
    if kind == "planar":
        z = np.zeros(n_points)
    else:                       # a volumetric control scene
        z = rng.uniform(-0.35, 0.35, n_points)
    X = np.column_stack([x, y, z + 4.0])

    R1 = np.eye(3)
    t1 = np.zeros(3)
    angle = np.radians(12.0)
    R2 = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]])
    C2 = np.array([0.9, 0.05, 0.0])
    t2 = -R2 @ C2

    def project(R, t):
        Xc = (R @ X.T).T + t
        uv = (K @ Xc.T).T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    p1, z1 = project(R1, t1)
    p2, z2 = project(R2, t2)
    ok = (z1 > 0) & (z2 > 0) \
        & (p1[:, 0] > 0) & (p1[:, 0] < W) & (p1[:, 1] > 0) & (p1[:, 1] < H) \
        & (p2[:, 0] > 0) & (p2[:, 0] < W) & (p2[:, 1] > 0) & (p2[:, 1] < H)
    p1 = p1[ok] + rng.normal(0, NOISE_PX, (ok.sum(), 2))
    p2 = p2[ok] + rng.normal(0, NOISE_PX, (ok.sum(), 2))

    matches = np.column_stack([np.arange(len(p1)), np.arange(len(p2))]).astype(np.int32)
    return K, p1, p2, matches, R2, C2


def rot_err_deg(R_est, R_true) -> float:
    d = R_est @ R_true.T
    return float(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=str(_ROOT / "eval_results" / "planar_check.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for kind in ("planar", "volumetric"):
        K, p1, p2, matches, R_true, C_true = build_scene(kind, args.points, rng)
        for fallback in (False, True):
            cv2.setRNGSeed(args.seed)
            np.random.seed(args.seed)
            verifier = GeometricVerifier(
                ransac_threshold=1.0, min_inliers=15, planar_fallback=fallback,
            )
            result = verifier.verify_pair(p1.astype(np.float32),
                                          p2.astype(np.float32), matches, K)
            row = {
                "scene": kind,
                "planar_fallback": fallback,
                "accepted": result is not None,
            }
            if result is not None:
                row["n_inliers"] = int(result["n_inliers"])
                row["planar_flag"] = bool(result.get("planar", False))
                row["rotation_error_deg"] = rot_err_deg(result["R"], R_true)
                # Translation is only known up to scale; compare directions.
                t = np.asarray(result["t"], float).ravel()
                t_true = (-R_true @ C_true)
                cos = abs(float(t @ t_true / (np.linalg.norm(t) * np.linalg.norm(t_true))))
                row["translation_dir_error_deg"] = float(
                    np.degrees(np.arccos(np.clip(cos, -1, 1))))
            rows.append(row)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    hdr = "%-11s %-9s %-9s %8s %11s %11s" % (
        "scene", "fallback", "accepted", "inliers", "rot err °", "t dir err °")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-11s %-9s %-9s %8s %11s %11s" % (
            r["scene"], r["planar_fallback"], r["accepted"],
            r.get("n_inliers", "-"),
            ("%.2f" % r["rotation_error_deg"]) if "rotation_error_deg" in r else "-",
            ("%.2f" % r["translation_dir_error_deg"])
            if "translation_dir_error_deg" in r else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
