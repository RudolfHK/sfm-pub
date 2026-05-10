"""Shared utilities: logging, I/O, camera math."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ─── Image I/O ───────────────────────────────────────────────────────────────

def list_images(image_dir: str) -> list:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    images = sorted(
        [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")
    logger.info(f"Found {len(images)} images in {p}")
    return images


def load_image(path) -> np.ndarray:
    """Load image as BGR uint8 array."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


# ─── EXIF reading ─────────────────────────────────────────────────────────────

def read_exif_focal_px(path, image_shape: tuple) -> Optional[float]:
    """
    Read 35 mm-equivalent focal length from EXIF and convert to pixels.
    Returns None when EXIF data is absent or incomplete.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(str(path)) as img:
            exif = img._getexif()
            if exif is None:
                return None
            focal_35 = exif.get(0xA405)   # FocalLengthIn35mmFilm
            if focal_35 is None or focal_35 == 0:
                return None
    except Exception:
        return None

    h, w = image_shape[:2]
    sensor_diag_35mm = float(np.sqrt(36.0 ** 2 + 24.0 ** 2))
    image_diag_px    = float(np.sqrt(h ** 2 + w ** 2))
    return float(focal_35) / sensor_diag_35mm * image_diag_px


# ─── Camera maths ────────────────────────────────────────────────────────────

def estimate_intrinsics(
    image_shape: tuple,
    image_path=None,
) -> np.ndarray:
    """
    Estimate a plausible pinhole K from image dimensions (+ optional EXIF).

    Tries FocalLengthIn35mmFilm from EXIF first; falls back to
    focal ≈ max(W, H), which is ~53° diagonal FoV.
    """
    h, w = image_shape[:2]
    focal: Optional[float] = None

    if image_path is not None:
        focal = read_exif_focal_px(image_path, image_shape)
        if focal is not None:
            logger.info(f"EXIF focal length: {focal:.0f} px")

    if focal is None:
        focal = float(max(h, w))
        logger.debug(f"Focal estimated from image size: {focal:.0f} px")

    K = np.array(
        [[focal, 0.0,   w / 2.0],
         [0.0,   focal, h / 2.0],
         [0.0,   0.0,   1.0   ]],
        dtype=np.float64,
    )
    logger.debug(f"K: f={focal:.0f}, cx={w/2:.1f}, cy={h/2:.1f}")
    return K


def undistort_points(
    pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    Undistort 2-D image points using OpenCV's undistortPoints.

    Parameters
    ----------
    pts  : (N, 2) float64
    K    : (3, 3) camera matrix
    dist : (4,) or (5,) distortion coefficients [k1, k2, p1, p2[, k3]]

    Returns
    -------
    undistorted : (N, 2) float64
    """
    if pts.shape[0] == 0:
        return pts.copy()
    pts_ud = cv2.undistortPoints(
        pts.astype(np.float64).reshape(-1, 1, 2),
        K,
        dist.astype(np.float64),
        P=K,
    )
    return pts_ud.reshape(-1, 2).astype(np.float64)


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3×4 projection matrix  P = K [R | t]."""
    return K @ np.hstack([R, t.reshape(3, 1)])


def reprojection_error(
    X: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """Pixel reprojection error for one 3-D ↔ 2-D correspondence."""
    t_flat = t.reshape(3)
    X_cam = R @ X.reshape(3) + t_flat
    if X_cam[2] <= 0.0:
        return np.inf
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_proj = fx * X_cam[0] / X_cam[2] + cx
    y_proj = fy * X_cam[1] / X_cam[2] + cy
    return float(np.sqrt((x_proj - x_obs[0]) ** 2 + (y_proj - x_obs[1]) ** 2))


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera centre  C = -R^T t."""
    return (-R.T @ t.reshape(3)).flatten()
