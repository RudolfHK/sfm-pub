#!/usr/bin/env python3
"""Score estimated camera poses against ground-truth projection matrices.

Monocular SfM recovers geometry only up to a similarity transform, so the
estimate is first aligned to ground truth with a Sim(3) (Umeyama, scale
included) fit on camera centres; a RANSAC variant is also reported because a
single grossly misplaced camera can drag a least-squares fit and disguise an
otherwise good reconstruction.

Ground truth: one `<stem>_P.txt` per image holding a 3x4 P = K[R|t], as shipped
with the AliceVision Buddha dataset. Estimated poses: the JSON written by
`run_sfm.py --export-cameras`.

Usage
-----
    python eval/gt_pose_eval.py est.cameras.json --gt-dir .../buddha \
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ── ground truth ─────────────────────────────────────────────────────────────

def load_gt(gt_dir: Path) -> dict[str, dict]:
    """stem -> {K, R, C}. `stem` is the image name without its extension."""
    gt: dict[str, dict] = {}
    for f in sorted(gt_dir.glob("*_P.txt")):
        P = np.loadtxt(f)
        if P.shape != (3, 4):
            continue
        K, R, Ch, *_ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]
        C = (Ch[:3] / Ch[3]).ravel()
        # "00001_P.txt" -> "00001"; images are named "00001._c.png"
        gt[f.name[: -len("_P.txt")]] = {"K": K, "R": R, "C": C}
    return gt


def _stem_key(image_name: str) -> str:
    """'00001._c.png' -> '00001' (the key used by the GT filenames)."""
    return image_name.split(".")[0]


# ── alignment ────────────────────────────────────────────────────────────────

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Least-squares similarity transform mapping src -> dst (Umeyama 1991).

    Returns (s, R, t) such that dst ~= s * R @ src.T + t.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (s_c**2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def umeyama_ransac(src: np.ndarray, dst: np.ndarray, iters: int = 500,
                   seed: int = 0) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC-robust Sim(3). Inlier threshold is 10 % of the GT scene extent.

    Returns (s, R, t, inlier_mask). Falls back to the plain fit when fewer than
    four correspondences are available (the minimum for a meaningful consensus).
    """
    n = len(src)
    if n < 4:
        s, R, t = umeyama(src, dst)
        return s, R, t, np.ones(n, dtype=bool)

    extent = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    thr = 0.10 * extent
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        resid = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
        mask = resid < thr
        if mask.sum() > best_mask.sum():
            best_mask = mask

    if best_mask.sum() < 3:
        s, R, t = umeyama(src, dst)
        return s, R, t, np.ones(n, dtype=bool)
    s, R, t = umeyama(src[best_mask], dst[best_mask])
    return s, R, t, best_mask


def rot_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, c)))))


# ── evaluation ───────────────────────────────────────────────────────────────

def _stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"median": None, "mean": None, "max": None, "n": 0}
    return {
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "max": float(np.max(v)),
        "n": int(v.size),
    }


def evaluate(est_path: Path, gt_dir: Path) -> dict:
    with open(est_path, encoding="utf-8") as fh:
        est = json.load(fh)
    gt = load_gt(gt_dir)

    matched = [c for c in est["cameras"] if _stem_key(c["image_name"]) in gt]
    result: dict = {
        "est_file": str(est_path),
        "gt_dir": str(gt_dir),
        "n_images_input": est.get("n_images"),
        "n_registered": est.get("n_cameras_registered"),
        "n_matched_to_gt": len(matched),
        "completeness_pct": (
            100.0 * est.get("n_cameras_registered", 0) / len(gt) if gt else None
        ),
        "reprojection": est.get("reprojection"),
        "n_points": est.get("n_points"),
        "mean_track_length": est.get("mean_track_length"),
    }

    # ── intrinsics: estimated focal vs GT focal ───────────────────────────
    gt_f = float(np.mean([g["K"][0, 0] for g in gt.values()]))
    est_K = np.array(est["shared_K"], dtype=float)
    result["focal"] = {
        "gt_px": gt_f,
        "est_px": float(est_K[0, 0]),
        "error_pct": 100.0 * (float(est_K[0, 0]) - gt_f) / gt_f,
        "gt_cx_cy": [float(np.mean([g["K"][0, 2] for g in gt.values()])),
                     float(np.mean([g["K"][1, 2] for g in gt.values()]))],
        "est_cx_cy": [float(est_K[0, 2]), float(est_K[1, 2])],
    }

    if len(matched) < 3:
        result["alignment"] = None
        result["note"] = (
            f"Only {len(matched)} camera(s) matched to ground truth; a Sim(3) fit "
            "needs at least 3 non-collinear centres, so pose accuracy is not "
            "computable for this run."
        )
        return result

    C_est = np.array([c["center"] for c in matched], dtype=float)
    R_est = [np.array(c["R"], dtype=float) for c in matched]
    C_gt = np.array([gt[_stem_key(c["image_name"])]["C"] for c in matched])
    R_gt = [gt[_stem_key(c["image_name"])]["R"] for c in matched]

    gt_extent = float(np.linalg.norm(C_gt.max(0) - C_gt.min(0)))
    # Extent of the FULL gt trajectory, so numbers stay comparable between runs
    # that registered different numbers of cameras.
    C_gt_all = np.array([g["C"] for g in gt.values()])
    gt_extent_all = float(np.linalg.norm(C_gt_all.max(0) - C_gt_all.min(0)))

    out_align = {}
    for tag, (s, R_a, t_a, mask) in {
        "lsq": (*umeyama(C_est, C_gt), np.ones(len(C_est), dtype=bool)),
        "ransac": umeyama_ransac(C_est, C_gt),
    }.items():
        C_aligned = (s * (R_a @ C_est.T).T) + t_a
        pos_err = np.linalg.norm(C_aligned - C_gt, axis=1)
        rot_err = np.array([
            rot_angle_deg(R_gt[i] @ (R_est[i] @ R_a.T).T) for i in range(len(R_est))
        ])
        out_align[tag] = {
            "scale": float(s),
            "n_inliers": int(mask.sum()),
            "position_err_world": _stats(pos_err),
            "position_err_pct_of_extent": _stats(100.0 * pos_err / gt_extent_all),
            "rotation_err_deg": _stats(rot_err),
            "per_camera": [
                {
                    "image": matched[i]["image_name"],
                    "pos_err_pct": float(100.0 * pos_err[i] / gt_extent_all),
                    "rot_err_deg": float(rot_err[i]),
                    "inlier": bool(mask[i]),
                }
                for i in range(len(matched))
            ],
        }

    result["gt_scene_extent"] = gt_extent_all
    result["gt_extent_registered_subset"] = gt_extent
    result["alignment"] = out_align
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("est", help="cameras.json from run_sfm.py --export-cameras")
    ap.add_argument("--gt-dir", required=True, help="directory holding *_P.txt")
    ap.add_argument("--json", default=None, help="write full result here")
    args = ap.parse_args()

    res = evaluate(Path(args.est), Path(args.gt_dir))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"registered      : {res['n_registered']}/{res['n_images_input']} "
          f"({res['completeness_pct']:.1f}% of GT cameras)")
    f = res["focal"]
    print(f"focal           : est {f['est_px']:.1f} px vs GT {f['gt_px']:.1f} px "
          f"({f['error_pct']:+.1f}%)")
    if res.get("alignment") is None:
        print(f"pose accuracy   : n/a — {res.get('note')}")
        return 0
    for tag, a in res["alignment"].items():
        p = a["position_err_pct_of_extent"]
        r = a["rotation_err_deg"]
        print(f"[{tag:6s}] scale={a['scale']:.4f} inliers={a['n_inliers']}/{p['n']}  "
              f"pos median {p['median']:.2f}% max {p['max']:.2f}%  |  "
              f"rot median {r['median']:.2f}° max {r['max']:.2f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
