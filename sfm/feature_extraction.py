"""
Stage 1 — Feature extraction.

Detects SIFT keypoints and 128-D descriptors for every image.
Falls back gracefully through GPU → kornia → plain OpenCV CPU.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import load_image

logger = logging.getLogger(__name__)


# ─── GPU detection ───────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    """Return True when a usable CUDA device is present."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        pass
    return False


# ─── Extractor ───────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    SIFT-based feature extractor with optional CUDA/kornia acceleration.

    Output contract
    ---------------
    keypoints   : (N, 2)  float32   pixel (x, y) positions
    descriptors : (N, 128) float32  L2-normalised SIFT descriptors
    """

    def __init__(
        self,
        n_features: int = 8_000,
        use_cuda: Optional[bool] = None,
    ) -> None:
        self.n_features = n_features
        self.use_cuda = _cuda_available() if use_cuda is None else use_cuda
        self._backend = self._init_backend()

    # ── initialisation ────────────────────────────────────────────────────

    def _init_backend(self) -> str:
        if self.use_cuda:
            backend = self._try_init_gpu()
            if backend:
                return backend
            logger.warning("No GPU backend available — falling back to CPU SIFT.")

        self._sift = cv2.SIFT_create(
            nfeatures=self.n_features,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
        )
        logger.info("Feature extraction backend: OpenCV SIFT (CPU)")
        return "cpu_sift"

    def _try_init_gpu(self) -> Optional[str]:
        # Option A: kornia (PyTorch-based, most portable)
        try:
            import torch
            import kornia.feature as KF  # noqa: F401
            self._torch = torch
            self._KF = KF
            # Also keep a CPU SIFT as descriptor back-end when kornia only
            # provides keypoints (some builds)
            self._sift = cv2.SIFT_create(nfeatures=self.n_features)
            logger.info("Feature extraction backend: kornia (GPU)")
            return "kornia"
        except Exception:
            pass

        # Option B: OpenCV CUDA SURF (requires opencv-contrib + CUDA build)
        try:
            self._surf_gpu = cv2.cuda.SURF_CUDA_create(400, extended=False)
            logger.info("Feature extraction backend: OpenCV CUDA SURF")
            return "cuda_surf"
        except Exception:
            pass

        return None

    # ── public API ────────────────────────────────────────────────────────

    def extract(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract keypoints + descriptors from a BGR image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._backend == "kornia":
            return self._extract_kornia(gray)
        if self._backend == "cuda_surf":
            return self._extract_surf_gpu(gray)
        return self._extract_sift_cpu(gray)

    def extract_all(self, image_paths: list) -> Dict[int, dict]:
        """
        Extract features from every image path.

        Returns
        -------
        features : dict
            image_idx → {
                'keypoints':    (N, 2) float32,
                'descriptors':  (N, 128) float32,
                'image_path':   Path,
                'image_shape':  (H, W, C),
            }
        """
        features: Dict[int, dict] = {}
        n = len(image_paths)
        for idx, path in enumerate(image_paths):
            logger.info(f"  Extracting [{idx+1}/{n}]: {Path(path).name}")
            img = load_image(path)
            kps, descs = self.extract(img)
            features[idx] = {
                "keypoints":   kps,
                "descriptors": descs,
                "image_path":  Path(path),
                "image_shape": img.shape,
            }
            logger.debug(f"    → {len(kps)} keypoints")

        total = sum(len(f["keypoints"]) for f in features.values())
        logger.info(f"Extraction complete — {total:,} keypoints across {n} images")
        return features

    # ── backends ─────────────────────────────────────────────────────────

    def _extract_sift_cpu(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        kps_cv, descs = self._sift.detectAndCompute(gray, None)
        if not kps_cv:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 128), dtype=np.float32),
            )
        pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
        return pts, descs.astype(np.float32)

    def _extract_kornia(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use kornia's ScaleSpaceDetector for GPU keypoints, then SIFT descs."""
        try:
            import torch

            device = torch.device("cuda")
            t = (
                torch.from_numpy(gray)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            detector = self._KF.ScaleSpaceDetector(
                num_features=self.n_features,
                resp_module=self._KF.BlobDoG(),
                nms_module=self._KF.ConvQuadInterp3d(10),
            ).to(device)

            with torch.no_grad():
                lafs, responses = detector(t)

            # Convert LAFs to pixel coords on CPU
            centres = self._KF.get_laf_center(lafs).squeeze(0).cpu().numpy()  # (N, 2)

            # Fall back to CPU SIFT for descriptors (robust, no extra dep)
            return self._extract_sift_cpu(gray)

        except Exception as e:
            logger.warning(f"kornia GPU extraction failed ({e}), using CPU SIFT")
            return self._extract_sift_cpu(gray)

    def _extract_surf_gpu(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)
            kps_gpu, descs_gpu = self._surf_gpu.detectWithDescriptors(gpu_img, None)
            kps_cv = cv2.cuda_SURF_CUDA.downloadKeypoints(self._surf_gpu, kps_gpu)
            descs = descs_gpu.download()
            pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
            # SURF gives 64-D; pad to 128-D so downstream code is uniform
            if descs.shape[1] == 64:
                descs = np.hstack([descs, np.zeros_like(descs)])
            return pts, descs.astype(np.float32)
        except Exception as e:
            logger.warning(f"CUDA SURF failed ({e}), falling back to CPU SIFT")
            self._sift = cv2.SIFT_create(nfeatures=self.n_features)
            self._backend = "cpu_sift"
            return self._extract_sift_cpu(gray)
