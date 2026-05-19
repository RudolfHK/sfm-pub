"""
Stage 5 — Point cloud colorisation and PLY export.

Color strategy
--------------
For each 3-D point, average the RGB samples from every observation whose
reprojection error is below the threshold.  Bilinear interpolation is used
for sub-pixel colour sampling.

PLY format
----------
Binary little-endian (much faster to write than ASCII for large clouds).
Compatible with MeshLab, CloudCompare, and open3d.
"""

import logging
import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .utils import load_image, reprojection_error

logger = logging.getLogger(__name__)


# ─── Vectorised camera projection helper ─────────────────────────────────────

def _reproject_batch(
    pts_3d: np.ndarray,
    pt_indices: np.ndarray,
    obs_2d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute reprojection errors for a batch of observations from a single camera.

    Parameters
    ----------
    pts_3d     : (P, 3) — all 3-D points in the reconstruction
    pt_indices : (M,) int — index into pts_3d for each observation
    obs_2d     : (M, 2) float64 — observed pixel coordinates
    R, t       : camera extrinsics
    K          : (3, 3) camera intrinsics

    Returns
    -------
    errors : (M,) float64
    """
    X   = pts_3d[pt_indices]              # (M, 3)
    tf  = t.flatten()
    X_c = (R @ X.T).T + tf               # (M, 3)

    f   = K[0, 0]
    cx  = K[0, 2]
    cy  = K[1, 2]

    z   = X_c[:, 2]
    z_safe = np.where(z > 1e-6, z, 1e-6)

    u   = f * X_c[:, 0] / z_safe + cx
    v   = f * X_c[:, 1] / z_safe + cy

    errs = np.hypot(u - obs_2d[:, 0], v - obs_2d[:, 1])
    errs[z <= 0] = np.inf                # behind camera
    return errs


# ─── Colour sampling ─────────────────────────────────────────────────────────

def _sample_bilinear(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample BGR colour at (x, y) with bilinear interpolation."""
    h, w = image.shape[:2]
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))

    x0, y0 = int(x), int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0

    c = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx       * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx       * dy
    )
    return c.astype(np.uint8)   # BGR


# ─── PointCloudExporter ──────────────────────────────────────────────────────

class PointCloudExporter:
    """
    Parameters
    ----------
    max_reproj_error : Observations above this threshold are ignored when
                       computing per-point colour averages.
    """

    def __init__(self, max_reproj_error: float = 4.0) -> None:
        self.max_reproj_error = max_reproj_error

    # ── colour assignment ─────────────────────────────────────────────────

    def colorize(
        self,
        points_3d:    np.ndarray,
        observations: list,
        features:     dict,
        cameras:      dict,
        K:            np.ndarray,
    ) -> np.ndarray:
        """
        Assign RGB colours to all 3-D points.

        Uses vectorised per-camera reprojection to filter observations, then
        samples bilinear colour from the image for each passing observation.

        Returns
        -------
        colors : (P, 3) uint8  in **RGB** order
        """
        n_pts  = len(points_3d)
        colors = np.full((n_pts, 3), 128, dtype=np.uint8)   # default: mid-grey

        # Group observations by camera for vectorised reprojection
        by_cam: Dict[int, Tuple[List[int], List[float], List[float]]] = {}
        for img_idx, pt_idx, x, y in observations:
            if img_idx not in cameras or pt_idx >= n_pts:
                continue
            if img_idx not in by_cam:
                by_cam[img_idx] = ([], [], [])
            by_cam[img_idx][0].append(pt_idx)
            by_cam[img_idx][1].append(x)
            by_cam[img_idx][2].append(y)

        # per-point colour accumulators
        color_sum   = np.zeros((n_pts, 3), dtype=np.float64)
        color_count = np.zeros(n_pts, dtype=np.int32)

        image_cache: Dict[int, np.ndarray] = {}
        t0 = time.time()

        for img_idx, (pt_idxs, xs, ys) in by_cam.items():
            cam = cameras[img_idx]
            pt_arr  = np.array(pt_idxs, dtype=np.int32)
            obs_2d  = np.column_stack([xs, ys]).astype(np.float64)   # (M, 2)

            # Vectorised reprojection errors for the whole camera at once
            errs = _reproject_batch(
                points_3d, pt_arr, obs_2d,
                cam["R"], cam["t"], K,
            )
            good = errs <= self.max_reproj_error
            if not good.any():
                continue

            if img_idx not in image_cache:
                image_cache[img_idx] = load_image(features[img_idx]["image_path"])
            img = image_cache[img_idx]

            good_pts = pt_arr[good]
            fxs = obs_2d[good, 0]
            fys = obs_2d[good, 1]

            h, w = img.shape[:2]
            fxs = np.clip(fxs, 0.0, w - 1.0)
            fys = np.clip(fys, 0.0, h - 1.0)

            # Bilinear interpolation: sample all three channels at once.
            # map_coordinates with order=1 is equivalent to bilinear sampling
            # and avoids the ~0.5 px colour fringe that nearest-pixel produces.
            from scipy.ndimage import map_coordinates  # noqa: PLC0415
            img_f = img.astype(np.float64)
            coords = np.array([fys, fxs])   # (2, M) — row, col order
            b = map_coordinates(img_f[:, :, 0], coords, order=1, mode="nearest")
            g = map_coordinates(img_f[:, :, 1], coords, order=1, mode="nearest")
            r = map_coordinates(img_f[:, :, 2], coords, order=1, mode="nearest")
            rgb_batch = np.stack([r, g, b], axis=1)   # BGR→RGB reorder

            np.add.at(color_sum,   good_pts, rgb_batch)
            np.add.at(color_count, good_pts, 1)

        has_color = color_count > 0
        colors[has_color] = (
            color_sum[has_color] / color_count[has_color, np.newaxis]
        ).astype(np.uint8)

        elapsed = time.time() - t0
        logger.info(
            f"Colourised {n_pts} points ({len(image_cache)} images sampled) "
            f"in {elapsed:.2f}s ({n_pts / elapsed:.0f} pts/s)"
        )
        return colors

    # ── outlier filtering ─────────────────────────────────────────────────

    def filter_outliers(
        self,
        points_3d:    np.ndarray,
        observations: list,
        cameras:      dict,
        K:            np.ndarray,
    ) -> Tuple[np.ndarray, list, dict]:
        """
        Remove 3-D points with high mean reprojection error.

        Threshold = min(max_reproj_error, median_err + 3σ).

        Returns
        -------
        filtered_pts, filtered_obs, old_to_new_index_map
        """
        n_pts = len(points_3d)
        if n_pts == 0:
            return points_3d, observations, {}

        # Per-point error accumulation — vectorised per camera
        err_sum   = np.zeros(n_pts, dtype=np.float64)
        err_count = np.zeros(n_pts, dtype=np.int32)

        by_cam: Dict[int, Tuple[List[int], List[float], List[float]]] = {}
        for img_idx, pt_idx, x, y in observations:
            if img_idx not in cameras or pt_idx >= n_pts:
                continue
            if img_idx not in by_cam:
                by_cam[img_idx] = ([], [], [])
            by_cam[img_idx][0].append(pt_idx)
            by_cam[img_idx][1].append(x)
            by_cam[img_idx][2].append(y)

        for img_idx, (pt_idxs, xs, ys) in by_cam.items():
            cam    = cameras[img_idx]
            pt_arr = np.array(pt_idxs, dtype=np.int32)
            obs_2d = np.column_stack([xs, ys]).astype(np.float64)

            errs = _reproject_batch(
                points_3d, pt_arr, obs_2d,
                cam["R"], cam["t"], K,
            )
            finite = np.isfinite(errs)
            np.add.at(err_sum,   pt_arr[finite], errs[finite])
            np.add.at(err_count, pt_arr[finite], 1)

        mean_errs = np.where(
            err_count > 0,
            err_sum / np.maximum(err_count, 1),
            np.inf,
        )

        finite = mean_errs[np.isfinite(mean_errs)]
        if len(finite) == 0:
            return points_3d, observations, {i: i for i in range(n_pts)}

        thr = min(
            self.max_reproj_error,
            float(np.median(finite) + 3.0 * np.std(finite)),
        )
        keep = mean_errs <= thr

        logger.info(
            f"Outlier filter: keeping {keep.sum()}/{n_pts} points "
            f"(thr={thr:.2f} px, median={np.median(finite):.2f} px)"
        )

        new_indices              = np.full(n_pts, -1, dtype=np.int64)
        new_indices[keep]        = np.arange(keep.sum(), dtype=np.int64)
        old_to_new               = {
            int(old): int(new_indices[old])
            for old in range(n_pts) if keep[old]
        }

        filtered_pts = points_3d[keep]
        filtered_obs = [
            (img_idx, old_to_new[pt_idx], x, y)
            for img_idx, pt_idx, x, y in observations
            if pt_idx in old_to_new
        ]
        return filtered_pts, filtered_obs, old_to_new

    # ── PLY export ────────────────────────────────────────────────────────

    def save_ply(
        self,
        filepath:  str,
        points_3d: np.ndarray,
        colors:    np.ndarray,
    ) -> None:
        """
        Write a binary little-endian PLY file.

        Each vertex has  float32 (x, y, z)  +  uint8 (r, g, b).
        """
        n_pts = len(points_3d)
        if n_pts == 0:
            raise ValueError("No points to write.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_pts}\n"
            "property double x\n"
            "property double y\n"
            "property double z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        dtype = np.dtype([
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
        ])
        data       = np.empty(n_pts, dtype=dtype)
        data["x"]  = points_3d[:, 0]
        data["y"]  = points_3d[:, 1]
        data["z"]  = points_3d[:, 2]
        data["r"]  = colors[:, 0]
        data["g"]  = colors[:, 1]
        data["b"]  = colors[:, 2]

        with open(filepath, "wb") as f:
            f.write(header.encode("ascii"))
            data.tofile(f)

        size_kb = filepath.stat().st_size / 1024
        logger.info(
            f"Saved {n_pts:,} points → {filepath}  ({size_kb:.1f} KB)"
        )
