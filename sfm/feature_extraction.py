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

from .device import get_device, has_gpu
from .utils import load_image

logger = logging.getLogger(__name__)


# ─── GPU detection ───────────────────────────────────────────────────────────


def _cuda_available() -> bool:
    """Return True when a usable CUDA device is present."""
    if has_gpu():
        return True
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
        sift_contrast_threshold: float = 0.02,
        sift_edge_threshold: float = 10.0,
        sift_n_octave_layers: int = 3,
        sift_sigma: float = 1.6,
        use_cuda: Optional[bool] = None,
    ) -> None:
        self.n_features = n_features
        self.sift_contrast_threshold = sift_contrast_threshold
        self.sift_edge_threshold = sift_edge_threshold
        self.sift_n_octave_layers = sift_n_octave_layers
        self.sift_sigma = sift_sigma
        self.use_cuda = _cuda_available() if use_cuda is None else use_cuda
        self._backend = self._init_backend()

    def _sift_kwargs(self) -> dict:
        return {
            "nfeatures": self.n_features,
            "nOctaveLayers": self.sift_n_octave_layers,
            "contrastThreshold": self.sift_contrast_threshold,
            "edgeThreshold": self.sift_edge_threshold,
            "sigma": self.sift_sigma,
        }

    # ── initialisation ────────────────────────────────────────────────────

    def _init_backend(self) -> str:
        if self.use_cuda:
            backend = self._try_init_gpu()
            if backend:
                return backend
            logger.warning("No GPU backend available — falling back to CPU SIFT.")

        self._sift = cv2.SIFT_create(**self._sift_kwargs())
        logger.info("Feature extraction backend: OpenCV SIFT (CPU)")
        return "cpu_sift"

    def _try_init_gpu(self) -> Optional[str]:
        # Option A: kornia (PyTorch-based, most portable)
        try:
            import torch
            import kornia.feature as KF  # noqa: F401

            self._torch = torch
            self._KF = KF
            # CPU SIFT computes descriptors at GPU-detected keypoint locations.
            self._sift = cv2.SIFT_create(**self._sift_kwargs())

            device = get_device()
            self._kornia_detector = KF.ScaleSpaceDetector(
                num_features=self.n_features,
                resp_module=KF.BlobDoG(),
                nms_module=KF.ConvQuadInterp3d(10),
            ).to(device)
            logger.info(
                "Feature extraction backend: kornia (GPU keypoints + CPU SIFT descriptors) "
                f"— detector initialized once on {device}"
            )
            return "kornia"
        except Exception:
            pass

        # Option B: OpenCV CUDA SURF (requires opencv-contrib + CUDA build)
        try:
            self._surf_gpu = cv2.cuda.SURF_CUDA_create(400, extended=True)
            logger.info("Feature extraction backend: OpenCV CUDA SURF")
            return "cuda_surf"
        except Exception:
            pass

        return None

    # ── public API ────────────────────────────────────────────────────────

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                "keypoints": kps,
                "descriptors": descs,
                "image_path": Path(path),
                "image_shape": img.shape,
            }
            logger.debug(f"    → {len(kps)} keypoints")

        total = sum(len(f["keypoints"]) for f in features.values())
        logger.info(f"Extraction complete — {total:,} keypoints across {n} images")
        return features

    # ── backends ─────────────────────────────────────────────────────────

    def _extract_sift_cpu(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kps_cv, descs = self._sift.detectAndCompute(gray, None)
        if not kps_cv:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 128), dtype=np.float32),
            )
        pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
        return pts, descs.astype(np.float32)

    def _extract_kornia(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU keypoint detection via kornia ScaleSpaceDetector.
        Descriptors are computed by OpenCV SIFT at the GPU-detected positions.
        Previously, the GPU keypoints were computed but discarded; this version
        actually uses them.
        """
        try:
            import torch

            device = get_device()
            if device is None or device.type == "cpu":
                return self._extract_sift_cpu(gray)

            t = (
                torch.from_numpy(gray)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                lafs, _ = self._kornia_detector(t)

            # lafs: (1, N, 2, 3)  — Local Affine Frames
            lafs_cpu = lafs.squeeze(0).cpu()  # (N, 2, 3)

            if lafs_cpu.shape[0] == 0:
                return (
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 128), dtype=np.float32),
                )

            # Extract pixel centres: last column of each 2×3 LAF matrix
            centres = (
                self._KF.get_laf_center(lafs).squeeze(0).cpu().numpy()
            )  # (N, 2)  x, y

            # Derive scale from the 2×2 linear sub-block (det gives area)
            A = lafs_cpu[:, :, :2].numpy()  # (N, 2, 2)
            det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
            scales = np.sqrt(np.abs(det).clip(1e-6))  # (N,)

            # Build cv2.KeyPoint list so SIFT can compute 128-D descriptors
            kp_list = [
                cv2.KeyPoint(
                    x=float(centres[k, 0]),
                    y=float(centres[k, 1]),
                    size=float(scales[k]) * 6.0,  # SIFT uses diameter
                )
                for k in range(len(centres))
            ]

            kp_out, descs = self._sift.compute(gray, kp_list)
            if descs is None or len(descs) == 0:
                return (
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 128), dtype=np.float32),
                )

            pts = np.array([kp.pt for kp in kp_out], dtype=np.float32)
            return pts, descs.astype(np.float32)

        except Exception as e:
            logger.warning(f"kornia GPU extraction failed ({e}), using CPU SIFT")
            return self._extract_sift_cpu(gray)

    def _extract_surf_gpu(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)
            kps_gpu, descs_gpu = self._surf_gpu.detectWithDescriptors(gpu_img, None)
            kps_cv = cv2.cuda_SURF_CUDA.downloadKeypoints(self._surf_gpu, kps_gpu)
            descs = descs_gpu.download()
            pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
            return pts, descs.astype(np.float32)
        except Exception as e:
            logger.warning(f"CUDA SURF failed ({e}), falling back to CPU SIFT")
            self._sift = cv2.SIFT_create(**self._sift_kwargs())
            self._backend = "cpu_sift"
            return self._extract_sift_cpu(gray)


# ─── SuperPoint extractor ─────────────────────────────────────────────────────

class SuperPointExtractor:
    """
    Feature extractor using SuperPoint (kornia.feature.SuperPointDescriptor).

    Produces (N, 2) float32 keypoints and (N, 256) float32 descriptors.
    Requires ``torch`` and ``kornia>=0.7`` (included in the ``gpu`` extra).

    Activated by ``--feature-backend superpoint`` in run_sfm.py.  Use with
    ``LightGlueMatcher`` in feature_matching.py for best quality.

    Parameters
    ----------
    n_features : Max keypoints to keep per image (score-sorted).
    use_cuda   : Force GPU/CPU; None = auto-detect.
    """

    DESC_DIM = 256

    def __init__(
        self,
        n_features: int = 8_000,
        use_cuda: Optional[bool] = None,
    ) -> None:
        self.n_features = n_features
        self.use_cuda   = _cuda_available() if use_cuda is None else use_cuda
        self._model     = None   # lazy-loaded

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            import kornia.feature as KF

            model = KF.SuperPoint(num_features=self.n_features).eval()
            device = get_device()
            if device is not None and self.use_cuda:
                model = model.to(device)
            self._model = model
            logger.info(
                f"SuperPoint loaded on {device if device else 'cpu'} "
                f"(n_features={self.n_features})"
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"SuperPoint could not be initialised: {exc}.  "
                "Install kornia>=0.7 and torch."
            ) from exc

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SuperPoint keypoints + 256-D descriptors from a BGR image."""
        import torch

        model  = self._load_model()
        device = get_device() if self.use_cuda else None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        t = (
            torch.from_numpy(gray)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if device is not None:
            t = t.to(device)

        try:
            with torch.no_grad():
                out = model(t)
        except Exception as exc:
            logger.warning(f"SuperPoint inference failed ({exc}), returning empty")
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, self.DESC_DIM), dtype=np.float32),
            )

        # kornia SuperPoint output: dict with 'keypoints', 'scores', 'descriptors'
        kps_t   = out["keypoints"][0].cpu()       # (N, 2)  x, y in pixels
        descs_t = out["descriptors"][0].cpu()     # (256, N) — needs transpose

        if kps_t.shape[0] == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, self.DESC_DIM), dtype=np.float32),
            )

        kps   = kps_t.numpy().astype(np.float32)         # (N, 2)
        descs = descs_t.T.numpy().astype(np.float32)     # (N, 256)
        return kps, descs

    def extract_all(self, image_paths: list) -> Dict[int, dict]:
        """Same interface as FeatureExtractor.extract_all()."""
        features: Dict[int, dict] = {}
        n = len(image_paths)
        for idx, path in enumerate(image_paths):
            logger.info(f"  SuperPoint [{idx+1}/{n}]: {Path(path).name}")
            img = load_image(str(path))
            kps, descs = self.extract(img)
            features[idx] = {
                "keypoints":   kps,
                "descriptors": descs,
                "image_path":  Path(path),
                "image_shape": img.shape,
            }
            logger.debug(f"    → {len(kps)} keypoints")

        total = sum(len(f["keypoints"]) for f in features.values())
        logger.info(
            f"SuperPoint extraction complete — {total:,} keypoints across {n} images"
        )
        return features
