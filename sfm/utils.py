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
    """Load image as BGR uint8 array, applying EXIF orientation when present.

    cv2.imread ignores the EXIF orientation tag on most platforms, causing
    portrait-mode mobile images to load sideways.  We use PIL + exif_transpose
    (Pillow is already a required dependency) and fall back to cv2.imread when
    PIL cannot open the file.
    """
    try:
        from PIL import Image as _PIL, ImageOps as _IOP
        with _PIL.open(str(path)) as pil_img:
            pil_img = _IOP.exif_transpose(pil_img)
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()   # RGB → BGR
    except Exception:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Could not read image: {path}")
        return img


# ─── EXIF reading ─────────────────────────────────────────────────────────────

def read_exif_focal_px(path, image_shape: tuple) -> Optional[float]:
    """
    Focal length in pixels from EXIF, or None when EXIF says nothing usable.

    Delegates to :mod:`sfm.exif_focal`, which tries the 35 mm equivalent, the
    focal-plane resolution and a sensor-width table in that order.  Only the
    first of those was consulted before, which made cameras that write
    ``FocalLength`` but no 35 mm equivalent look uncalibrated.
    """
    est = read_exif_focal(path, image_shape)
    return est.focal_px if est is not None else None


def read_exif_focal(path, image_shape: tuple):
    """Same as :func:`read_exif_focal_px` but keeps the provenance."""
    from .exif_focal import focal_from_exif
    return focal_from_exif(path, image_shape)


# ─── Camera maths ────────────────────────────────────────────────────────────

def estimate_intrinsics(
    image_shape: tuple,
    image_path=None,
    return_source: bool = False,
):
    """
    Estimate a plausible pinhole K from image dimensions (+ optional EXIF).

    Tries every EXIF route in :mod:`sfm.exif_focal`; falls back to
    focal = max(W, H), which is ~53° diagonal FoV and, on a camera whose true
    field of view differs, wrong by whatever that difference happens to be.
    The caller is told which of the two happened via ``return_source`` so it
    can start a focal search instead of trusting the fallback.
    """
    h, w = image_shape[:2]
    focal: Optional[float] = None
    source = "fallback_max_wh"
    detail = "no usable EXIF; focal = max(W, H)"

    if image_path is not None:
        est = read_exif_focal(image_path, image_shape)
        if est is not None:
            focal, source, detail = est.focal_px, est.source, est.detail
            logger.info("EXIF focal length: %.0f px (%s)", focal, detail)

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
    if return_source:
        return K, {"source": source, "detail": detail, "focal_px": float(focal)}
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


def check_scene_graph_connectivity(
    verified_pairs: dict,
    all_image_indices: list,
    min_inliers: int = 1,
) -> list:
    """
    Find connected components in the scene graph using union-find.

    Parameters
    ----------
    verified_pairs     : {(i, j): {'n_inliers': int, ...}} from geometric verification
    all_image_indices  : all image indices (nodes), including isolated ones
    min_inliers        : minimum inlier count for an edge to count as connected

    Returns
    -------
    components : list of sets, each set is one connected component of image indices,
                 sorted largest component first
    """
    parent = {idx: idx for idx in all_image_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (i, j), data in verified_pairs.items():
        if data.get("n_inliers", 0) >= min_inliers:
            if i in parent and j in parent:
                union(i, j)

    groups: dict = {}
    for idx in all_image_indices:
        root = find(idx)
        groups.setdefault(root, set()).add(idx)

    return sorted(groups.values(), key=len, reverse=True)
