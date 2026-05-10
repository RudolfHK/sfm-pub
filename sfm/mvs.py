"""
Stage 6b — Multi-View Stereo (MVS) densification.

Produces a dense point cloud from the sparse SfM reconstruction by running
Semi-Global Block Matching (SGBM) on rectified stereo pairs.

Pipeline per pair
-----------------
1. cv2.stereoRectify        — compute rectification transforms R1/R2/P1/P2/Q
2. cv2.initUndistortRectifyMap + cv2.remap — warp images to common epipolar plane
3. cv2.StereoSGBM_create   — semi-global disparity computation
4. cv2.reprojectImageTo3D  — lift disparity → 3-D in rectified cam-1 frame
5. Coordinate transform    — rectified cam-1 frame → world frame
6. Filtering               — invalid disparity, negative depth, distance outliers

Pairs are selected from consecutive registered cameras (nearest neighbours by
sorted index).  At most `max_pairs_per_image` neighbours are processed per
image to bound runtime.

The output point cloud is merged from all successful stereo pairs and
subsampled to at most `max_dense_pts` points if necessary.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import camera_center, load_image

logger = logging.getLogger(__name__)


class MVSDensifier:
    """
    Dense reconstruction via Semi-Global Block Matching.

    Parameters
    ----------
    min_baseline_fraction : Minimum baseline as a fraction of scene scale.
                            Pairs with smaller baselines are skipped.
    num_disparities       : Must be divisible by 16.  Larger → slower but handles
                            larger depth ranges.
    block_size            : SGBM block size (odd, ≥ 1).  Smaller = finer detail.
    max_pairs_per_image   : Neighbour window size when selecting stereo pairs.
    max_dense_pts         : Cap on total output points (random subsampling).
    """

    def __init__(
        self,
        min_baseline_fraction: float = 0.02,
        num_disparities: int = 256,
        block_size: int = 5,
        max_pairs_per_image: int = 2,
        max_dense_pts: int = 500_000,
    ) -> None:
        self.min_baseline_fraction = min_baseline_fraction
        self.num_disparities       = (num_disparities // 16) * 16  # ensure multiple of 16
        self.block_size            = block_size
        self.max_pairs_per_image   = max_pairs_per_image
        self.max_dense_pts         = max_dense_pts

    # ── public API ────────────────────────────────────────────────────────

    def densify(
        self,
        cameras: dict,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
        image_paths: Dict[int, Path],
        max_reproj_error: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run dense reconstruction for all valid stereo pairs.

        Parameters
        ----------
        cameras      : {img_idx: {'R':(3,3), 't':(3,1)}}  — registered cameras
        K            : (3,3) shared intrinsics
        dist_coeffs  : (4,) or (5,) distortion; None → zero distortion
        image_paths  : {img_idx: Path}  — paths to the original images
        max_reproj_error : (unused; kept for API symmetry with other stages)

        Returns
        -------
        pts    : (N, 3) float64  world-space 3-D points
        colors : (N, 3) uint8    RGB colours
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()

        cam_list = sorted(cameras.keys())
        n        = len(cam_list)
        if n < 2:
            logger.warning("MVS: need ≥ 2 registered cameras — skipping densification.")
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

        # Scene scale: median camera-centre spread
        centres   = np.array([camera_center(cameras[c]["R"], cameras[c]["t"]) for c in cam_list])
        spread    = np.linalg.norm(centres - centres.mean(0), axis=1)
        scene_scale = float(np.median(spread)) if len(spread) > 1 else 1.0
        min_base  = self.min_baseline_fraction * scene_scale

        all_pts, all_colors = [], []
        n_pairs_used = 0

        for r, i in enumerate(cam_list):
            neighbours = cam_list[r + 1 : r + 1 + self.max_pairs_per_image]
            for j in neighbours:
                Ri = cameras[i]["R"]
                ti = cameras[i]["t"].flatten()
                Rj = cameras[j]["R"]
                tj = cameras[j]["t"].flatten()

                # Relative rotation & translation (j expressed in i's frame)
                R_rel = Rj @ Ri.T
                t_rel = (tj - R_rel @ ti).reshape(3, 1)
                baseline = float(np.linalg.norm(t_rel))

                if baseline < min_base:
                    logger.debug(
                        f"MVS: pair ({i},{j}) baseline {baseline:.4f} < {min_base:.4f} — skip"
                    )
                    continue

                if i not in image_paths or j not in image_paths:
                    logger.debug(f"MVS: pair ({i},{j}) missing image path — skip")
                    continue

                pts, colors = self._process_pair(
                    Ri, ti, R_rel, t_rel,
                    K, dist_coeffs,
                    image_paths[i], image_paths[j],
                )

                if len(pts) > 0:
                    all_pts.append(pts)
                    all_colors.append(colors)
                    n_pairs_used += 1
                    logger.info(f"MVS: pair ({i},{j}) → {len(pts):,} dense points")

        if not all_pts:
            logger.warning("MVS: no dense points generated.")
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

        pts_merged    = np.vstack(all_pts)
        colors_merged = np.vstack(all_colors)

        if len(pts_merged) > self.max_dense_pts:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(pts_merged), self.max_dense_pts, replace=False)
            pts_merged    = pts_merged[sel]
            colors_merged = colors_merged[sel]

        logger.info(
            f"MVS densification: {n_pairs_used} stereo pairs → "
            f"{len(pts_merged):,} dense points total"
        )
        return pts_merged, colors_merged

    # ── internals ─────────────────────────────────────────────────────────

    def _process_pair(
        self,
        Ri: np.ndarray,
        ti: np.ndarray,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        path_i: Path,
        path_j: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SGBM on one rectified stereo pair; return world pts + RGB colors."""
        try:
            img_i = load_image(str(path_i))
            img_j = load_image(str(path_j))
        except Exception as exc:
            logger.warning(f"MVS: could not load images: {exc}")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        h, w = img_i.shape[:2]
        size = (w, h)
        D    = dist_coeffs.astype(np.float64)

        # Stereo rectification
        try:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K, D, K, D, size, R_rel, t_rel,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
        except cv2.error as exc:
            logger.warning(f"MVS: stereoRectify failed: {exc}")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1)

        img1_rect = cv2.remap(img_i, map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img_j, map2x, map2y, cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)

        # Semi-global block matching
        bs     = self.block_size
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=bs,
            P1=8  * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

        # Reproject to 3-D in rectified camera-1 frame
        pts3d_rect = cv2.reprojectImageTo3D(disp, Q)   # (H, W, 3)

        # Valid mask: positive disparity + finite depth
        valid = (
            (disp > 1.0)
            & np.isfinite(pts3d_rect[:, :, 2])
            & (pts3d_rect[:, :, 2] > 0)
            & (pts3d_rect[:, :, 2] < 1e6)
        )

        pts_rect  = pts3d_rect[valid]           # (N, 3) in rectified cam-1 frame
        colors_bgr = img1_rect[valid]           # (N, 3) BGR
        colors_rgb = colors_bgr[:, ::-1]        # → RGB

        if len(pts_rect) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        # Transform: rectified cam-1 frame → original cam-i frame
        # Rectification applies R1, so: X_cam_i = R1.T @ X_rect
        pts_cam_i = (R1.T @ pts_rect.T).T      # (N, 3)

        # Transform: cam-i frame → world frame
        # X_cam_i = Ri @ X_world + ti  →  X_world = Ri.T @ (X_cam_i − ti)
        pts_world = (Ri.T @ (pts_cam_i - ti).T).T   # (N, 3)

        # Filter: positive depth in camera i
        depth_valid = pts_cam_i[:, 2] > 0.01
        pts_world   = pts_world[depth_valid]
        colors_rgb  = colors_rgb[depth_valid]

        if len(pts_world) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        # Filter: distance outliers (keep within 10× median distance from origin)
        dists  = np.linalg.norm(pts_world, axis=1)
        median = float(np.median(dists))
        dist_valid = dists < max(median * 10.0, 1e-3)
        pts_world  = pts_world[dist_valid]
        colors_rgb = colors_rgb[dist_valid]

        return pts_world.astype(np.float64), colors_rgb
