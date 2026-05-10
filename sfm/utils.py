"""Shared utilities: logging, I/O, camera math."""

import logging
from pathlib import Path

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


# ─── Camera maths ────────────────────────────────────────────────────────────

def estimate_intrinsics(image_shape: tuple) -> np.ndarray:
    """
    Estimate a plausible pinhole K from image dimensions.

    Heuristic: focal ≈ max(W, H), which corresponds to ~53° diagonal FoV —
    reasonable for typical phone / DSLR imagery.
    """
    h, w = image_shape[:2]
    focal = float(max(h, w))
    K = np.array(
        [[focal, 0.0,   w / 2.0],
         [0.0,   focal, h / 2.0],
         [0.0,   0.0,   1.0   ]],
        dtype=np.float64,
    )
    logger.debug(f"Estimated K: f={focal:.0f}, cx={w/2:.1f}, cy={h/2:.1f}")
    return K


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
