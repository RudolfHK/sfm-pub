"""
Per-camera intrinsics: dataclass + EXIF-based estimation.

Used when ``--per-camera-intrinsics`` is enabled.  Each image gets its own
``CameraIntrinsics`` instance initialised from EXIF focal length when
available; falls back to the pipeline's shared estimate otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 35 mm sensor diagonal (mm) — constant used in focal-length conversion
_SENSOR_DIAG_35MM = float(np.sqrt(36.0**2 + 24.0**2))


@dataclass
class CameraIntrinsics:
    """
    Pinhole + 2-parameter radial distortion model for a single camera.

    Attributes
    ----------
    fx, fy : focal lengths in pixels (may differ for non-square pixels)
    cx, cy : principal point in pixels
    k1, k2 : Brown-Conrady radial distortion coefficients
    width, height : image dimensions in pixels
    """

    fx:     float
    fy:     float
    cx:     float
    cy:     float
    k1:     float = 0.0
    k2:     float = 0.0
    width:  int   = 0
    height: int   = 0

    # ── conversions ───────────────────────────────────────────────────────

    def to_K(self) -> np.ndarray:
        """Return (3, 3) camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dist(self) -> np.ndarray:
        """Return (4,) OpenCV-style distortion coefficients [k1, k2, 0, 0]."""
        return np.array([self.k1, self.k2, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def from_K_and_dist(
        K: np.ndarray,
        dist: np.ndarray,
        width: int = 0,
        height: int = 0,
    ) -> "CameraIntrinsics":
        """Construct from a (3,3) K matrix and (N,) dist vector."""
        dist_flat = np.asarray(dist, dtype=np.float64).ravel()
        k1 = float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
        k2 = float(dist_flat[1]) if len(dist_flat) > 1 else 0.0
        return CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            k1=k1,
            k2=k2,
            width=width,
            height=height,
        )

    def update_from_ba(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        k1: Optional[float] = None,
        k2: Optional[float] = None,
    ) -> "CameraIntrinsics":
        """Return a new instance with updated values from BA results."""
        return CameraIntrinsics(
            fx=fx if fx is not None else self.fx,
            fy=fy if fy is not None else self.fy,
            cx=cx if cx is not None else self.cx,
            cy=cy if cy is not None else self.cy,
            k1=k1 if k1 is not None else self.k1,
            k2=k2 if k2 is not None else self.k2,
            width=self.width,
            height=self.height,
        )


# ── per-image estimation ──────────────────────────────────────────────────────

def estimate_per_image(
    features: dict,
    shared_K: np.ndarray,
    shared_dist: np.ndarray,
) -> Dict[int, CameraIntrinsics]:
    """
    Build a per-image ``CameraIntrinsics`` dict.

    For each image, attempt to read the EXIF focal length
    (``FocalLengthIn35mmFilm`` tag).  If EXIF is unavailable, fall back to
    the shared focal from ``shared_K``.  Principal point is always
    ``(W/2, H/2)`` using the image's own dimensions.

    Parameters
    ----------
    features   : feature dict from ``FeatureExtractor.extract_all()``
    shared_K   : (3, 3) fall-back camera matrix
    shared_dist: (N,) fall-back distortion coefficients

    Returns
    -------
    per_cam : {img_idx: CameraIntrinsics}
    """
    per_cam: Dict[int, CameraIntrinsics] = {}

    for img_idx, feat in features.items():
        h, w = feat["image_shape"][:2]
        path = feat.get("image_path")

        focal = _read_exif_focal(path, h, w) if path is not None else None
        if focal is None:
            focal = float(shared_K[0, 0])

        dist_flat = np.asarray(shared_dist, dtype=np.float64).ravel()
        k1 = float(dist_flat[0]) if len(dist_flat) > 0 else 0.0
        k2 = float(dist_flat[1]) if len(dist_flat) > 1 else 0.0

        per_cam[img_idx] = CameraIntrinsics(
            fx=focal, fy=focal,
            cx=w / 2.0, cy=h / 2.0,
            k1=k1, k2=k2,
            width=w, height=h,
        )

        if path is not None:
            logger.debug(
                f"  img {img_idx} ({Path(path).name}): "
                f"fx={focal:.0f} cx={w/2:.0f} cy={h/2:.0f}"
            )

    return per_cam


def _read_exif_focal(path, h: int, w: int) -> Optional[float]:
    """Focal length in pixels from EXIF, all routes, or None.

    Shares :mod:`sfm.exif_focal` with the shared-intrinsics path so that
    per-camera and shared estimates can never disagree about what a file says.
    """
    from .exif_focal import focal_from_exif

    est = focal_from_exif(path, (h, w))
    return est.focal_px if est is not None else None
