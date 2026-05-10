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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .utils import load_image, reprojection_error

logger = logging.getLogger(__name__)


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

        Returns
        -------
        colors : (P, 3) uint8  in **RGB** order
        """
        n_pts  = len(points_3d)
        colors = np.full((n_pts, 3), 128, dtype=np.uint8)   # default: mid-grey

        # Group observations by 3-D point index
        obs_by_pt: Dict[int, List[Tuple[int, float, float]]] = {}
        for img_idx, pt_idx, x, y in observations:
            obs_by_pt.setdefault(pt_idx, []).append((img_idx, x, y))

        # Load images lazily
        image_cache: Dict[int, np.ndarray] = {}

        for pt_idx, obs_list in obs_by_pt.items():
            if pt_idx >= n_pts:
                continue
            X = points_3d[pt_idx]
            samples: List[np.ndarray] = []

            for img_idx, x, y in obs_list:
                if img_idx not in cameras:
                    continue
                cam = cameras[img_idx]
                err = reprojection_error(X, np.array([x, y]), K, cam["R"], cam["t"])
                if err > self.max_reproj_error:
                    continue

                if img_idx not in image_cache:
                    image_cache[img_idx] = load_image(features[img_idx]["image_path"])

                bgr = _sample_bilinear(image_cache[img_idx], x, y)
                samples.append(bgr[::-1].astype(np.float64))   # BGR → RGB

            if samples:
                colors[pt_idx] = np.mean(samples, axis=0).astype(np.uint8)

        logger.info(
            f"Colourised {n_pts} points "
            f"({len(image_cache)} images sampled)"
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

        # Per-point error accumulation
        err_acc = [[] for _ in range(n_pts)]
        for img_idx, pt_idx, x, y in observations:
            if img_idx not in cameras or pt_idx >= n_pts:
                continue
            cam = cameras[img_idx]
            e = reprojection_error(
                points_3d[pt_idx], np.array([x, y]), K, cam["R"], cam["t"]
            )
            if np.isfinite(e):
                err_acc[pt_idx].append(e)

        mean_errs = np.array(
            [np.mean(e) if e else np.inf for e in err_acc], dtype=np.float64
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

        # Build compact index mapping
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
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        # Build binary payload in one shot using a structured numpy array
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
        ])
        data       = np.empty(n_pts, dtype=dtype)
        data["x"]  = points_3d[:, 0].astype(np.float32)
        data["y"]  = points_3d[:, 1].astype(np.float32)
        data["z"]  = points_3d[:, 2].astype(np.float32)
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
