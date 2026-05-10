#!/usr/bin/env python3
"""
Integration test — SfM pipeline stages 3-5 with synthetic ground-truth data.

SIFT-based matching (stages 1-2) is replaced by analytically-projected
correspondences so that geometric verification and reconstruction receive
well-conditioned, noise-free input.  All downstream pipeline code (geometric
verification, incremental SfM, bundle adjustment, point-cloud export) is
exercised via its public API.

Pass criteria
-------------
  * ≥ MIN_CAMERAS cameras registered   (out of N_CAMERAS)
  * ≥ MIN_POINTS  3-D points output    (after outlier filtering)
  * mean reprojection error  < MAX_REPROJ_PX  for reconstructed cameras
"""

import logging
import struct
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Pass thresholds ─────────────────────────────────────────────────────────
MIN_CAMERAS   = 4        # at least 4 out of 6
MIN_POINTS    = 50       # 3-D points reconstructed
MAX_REPROJ_PX = 2.0      # pixel RMSE across registered cameras

# ── Scene constants ─────────────────────────────────────────────────────────
W, H           = 640, 480
N_WORLD_PTS    = 250
CAM_RADIUS     = 4.0
N_CAMERAS      = 6
NOISE_SIGMA_PX = 0.5     # keypoint position noise (pixels)

rng = np.random.default_rng(42)
WORLD_PTS = rng.uniform(-1.5, 1.5, (N_WORLD_PTS, 3))


# ── Camera helpers ──────────────────────────────────────────────────────────

def build_K() -> np.ndarray:
    return np.array(
        [[float(max(W, H)), 0.0, W / 2.0],
         [0.0, float(max(W, H)), H / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def look_at(
    eye: np.ndarray,
    target: np.ndarray = np.zeros(3),
    world_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> tuple:
    """
    (R, t) for a pinhole camera at *eye* gazing toward *target*.

    Standard OpenCV convention: x-right, y-DOWN, z-forward.
    Row ordering of R: [right, −world_up, forward].
    """
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, world_up)
    if np.linalg.norm(r) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, world_up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    u /= np.linalg.norm(u)
    R = np.vstack([r, -u, f])
    t = (-R @ eye).reshape(3, 1)
    return R, t


def project(R: np.ndarray, t: np.ndarray, K: np.ndarray,
            pts: np.ndarray) -> tuple:
    """Project (N,3) world points; return ((N,2) pixels, (N,) depths)."""
    X_cam = (R @ pts.T + t).T
    z = X_cam[:, 2]
    safe_z = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * X_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * X_cam[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), z


# ── Synthetic feature / match generation ────────────────────────────────────

def make_synthetic_dataset(K: np.ndarray, gt_poses: list) -> tuple:
    """
    Build feature dicts and pairwise match arrays from ground-truth cameras.

    Each camera's "keypoints" are the projected world points (with small
    Gaussian noise added so the RANSAC solvers have realistic input).
    Matches are exact ground-truth correspondences between shared world points.

    Returns
    -------
    features   : dict  img_idx → {keypoints, descriptors, image_path, …}
    all_matches: dict  (i,j)   → (M, 2) int32  match arrays
    world_index: dict  img_idx → (M_i,) int array  world-pt idx per keypoint
    """
    margin = 4     # pixels: exclude world pts projecting too close to border

    features = {}
    world_index = {}

    for cam_idx, (R, t) in enumerate(gt_poses):
        pxs, depths = project(R, t, K, WORLD_PTS)
        vis = (
            (depths > 0.3) &
            (pxs[:, 0] >= margin) & (pxs[:, 0] < W - margin) &
            (pxs[:, 1] >= margin) & (pxs[:, 1] < H - margin)
        )
        wids = np.where(vis)[0]
        kps  = pxs[vis].astype(np.float64)
        kps += rng.normal(0.0, NOISE_SIGMA_PX, kps.shape)   # realistic noise

        features[cam_idx] = {
            "keypoints":   kps.astype(np.float32),
            "descriptors": np.zeros((len(kps), 128), dtype=np.float32),
            "image_path":  Path(f"synthetic_{cam_idx:03d}.png"),
            "image_shape": (H, W, 3),
        }
        world_index[cam_idx] = wids

    all_matches = {}
    for i in range(len(gt_poses)):
        for j in range(i + 1, len(gt_poses)):
            wi = world_index[i]
            wj = world_index[j]
            common = np.intersect1d(wi, wj)
            if len(common) < 8:
                continue
            # Local keypoint index = position of the world-pt idx in each array
            wi_inv = {int(w): k for k, w in enumerate(wi)}
            wj_inv = {int(w): k for k, w in enumerate(wj)}
            matches = np.array(
                [[wi_inv[int(w)], wj_inv[int(w)]] for w in common],
                dtype=np.int32,
            )
            all_matches[(i, j)] = matches

    return features, all_matches, world_index


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = logging.getLogger("integration_test")

    K = build_K()

    # ── 1. Ground-truth cameras ───────────────────────────────────────────
    gt_poses = []
    for i in range(N_CAMERAS):
        angle = np.radians(i * (360 // N_CAMERAS))
        eye = np.array([
            CAM_RADIUS * np.cos(angle),
            0.0,
            CAM_RADIUS * np.sin(angle),
        ])
        gt_poses.append(look_at(eye))

    # Quick convention check: point *above* origin should project above centre
    R0, t0 = gt_poses[0]
    above = np.array([[0.0, 1.0, 0.0]])
    pxs_ab, _ = project(R0, t0, K, above)
    assert pxs_ab[0, 1] < H / 2.0, (
        f"y-DOWN convention check failed: v={pxs_ab[0,1]:.1f} should be < {H/2}"
    )
    log.info("Camera convention check passed (y-DOWN ✓)")

    # ── 2. Synthetic features + ground-truth matches ──────────────────────
    features, all_matches, world_index = make_synthetic_dataset(K, gt_poses)

    total_kps = sum(len(f["keypoints"]) for f in features.values())
    log.info(
        f"Scene: {N_CAMERAS} cameras, {N_WORLD_PTS} world pts, "
        f"{total_kps} synthetic keypoints, {len(all_matches)} matchable pairs"
    )
    for (i, j), m in all_matches.items():
        log.info(f"  pair ({i},{j}): {len(m)} ground-truth matches")

    # ── 3. Geometric verification ─────────────────────────────────────────
    from sfm.geometric_verification import GeometricVerifier

    verifier = GeometricVerifier(
        ransac_threshold=2.0,
        min_inliers=8,
        confidence=0.999,
    )
    verified = verifier.verify_all(features, all_matches, K)
    log.info(f"Verified pairs: {len(verified)}/{len(all_matches)}")

    if not verified:
        log.error("No pairs verified — aborting.")
        return 1

    # ── 4. Incremental reconstruction + BA ───────────────────────────────
    from sfm.reconstruction import IncrementalSfM

    sfm = IncrementalSfM(
        features=features,
        verified_pairs=verified,
        K=K,
        max_reproj_error=4.0,
        ba_interval=3,
    )
    cameras_out, pts_out, observations, kp_to_3d = sfm.reconstruct()

    n_cams = len(cameras_out)
    n_pts  = len(pts_out)
    log.info(f"Reconstruction: {n_cams}/{N_CAMERAS} cameras, {n_pts} 3-D points")

    # ── 5. Reprojection-error check ───────────────────────────────────────
    from sfm.utils import reprojection_error

    errs = []
    for img_idx, pt_idx, x, y in observations:
        if img_idx not in cameras_out or pt_idx >= len(pts_out):
            continue
        cam = cameras_out[img_idx]
        e = reprojection_error(
            pts_out[pt_idx], np.array([x, y]),
            K, cam["R"], cam["t"],
        )
        if np.isfinite(e):
            errs.append(e)

    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2))) if errs else np.inf
    log.info(f"Reprojection RMSE: {rmse:.3f} px  ({len(errs)} observations)")

    # ── 6. PLY export ─────────────────────────────────────────────────────
    from sfm.point_cloud import PointCloudExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = PointCloudExporter(max_reproj_error=4.0)
        pts_filt, obs_filt, _ = exporter.filter_outliers(
            pts_out, observations, cameras_out, K
        )

        # Colorize with grey (no real images — all colours default to 128)
        colors = np.full((len(pts_filt), 3), 128, dtype=np.uint8)

        ply_path = str(Path(tmpdir) / "output.ply")
        exporter.save_ply(ply_path, pts_filt, colors)

        # Verify PLY header
        n_ply_pts = 0
        with open(ply_path, "rb") as f:
            for line in f:
                decoded = line.decode("ascii", errors="ignore").strip()
                if decoded.startswith("element vertex"):
                    n_ply_pts = int(decoded.split()[-1])
                    break
        log.info(f"PLY: {n_ply_pts} vertices written")

    # ── 7. Pass / fail ────────────────────────────────────────────────────
    ok = True

    if n_cams < MIN_CAMERAS:
        log.error(f"FAIL cameras: {n_cams} < {MIN_CAMERAS}")
        ok = False

    if n_ply_pts < MIN_POINTS:
        log.error(f"FAIL points: {n_ply_pts} < {MIN_POINTS}")
        ok = False

    if rmse > MAX_REPROJ_PX:
        log.error(f"FAIL reprojection RMSE: {rmse:.3f} > {MAX_REPROJ_PX}")
        ok = False

    if ok:
        log.info("=== INTEGRATION TEST PASSED ===")
        return 0
    else:
        log.error("=== INTEGRATION TEST FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
