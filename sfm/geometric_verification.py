"""
Stage 3 — Geometric verification.

For each matched pair:
  1. Hartley-normalize pixel coordinates, estimate F via USAC_MAGSAC,
     then de-normalize F back to pixel space.
  2. Run findEssentialMat (also USAC_MAGSAC) on the undistorted F-inlier
     subset to refine the E-inlier set.
  3. Decompose E → (R, t) via cv2.recoverPose (cheirality check included).
  4. Reject pairs with too few inliers or near-zero baseline.

Hartley normalization reference
--------------------------------
Hartley, R. (1997). In defense of the eight-point algorithm.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(6), 580–593.

USAC_MAGSAC reference
---------------------
Barath, D., Matas, J., & Noskova, J. (2020). MAGSAC++: A fast, reliable and
  accurate robust estimator. CVPR 2020.

Output: verified pair dict containing F, E, R, t, and the inlier match indices.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import undistort_points

logger = logging.getLogger(__name__)

# USAC_MAGSAC: available since OpenCV 4.5.  More robust than FM_RANSAC and
# avoids the crash seen in FM_RANSAC on certain distributions in OpenCV ≥4.10.
_HAS_USAC_MAGSAC = hasattr(cv2, "USAC_MAGSAC")
_F_METHOD = cv2.USAC_MAGSAC if _HAS_USAC_MAGSAC else cv2.FM_RANSAC
_E_METHOD = cv2.USAC_MAGSAC if _HAS_USAC_MAGSAC else cv2.RANSAC


def _hartley_normalize(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotropic Hartley normalization: translate centroid to origin, scale so
    that the mean distance of points from the origin equals √2.

    Dramatically reduces the condition number of the DLT system used inside
    findFundamentalMat, improving F/E estimate accuracy on high-resolution
    images where raw pixel values span thousands of units.

    Parameters
    ----------
    pts : (N, 2) float64  — input pixel coordinates

    Returns
    -------
    pts_norm : (N, 2)  normalized coordinates
    T        : (3, 3)  similarity transform so that pts_norm_h = T @ pts_h
    """
    centroid  = pts.mean(axis=0)
    shifted   = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    scale     = np.sqrt(2.0) / max(float(mean_dist), 1e-9)
    T = np.array(
        [
            [scale, 0.0,   -scale * centroid[0]],
            [0.0,   scale, -scale * centroid[1]],
            [0.0,   0.0,    1.0               ],
        ],
        dtype=np.float64,
    )
    return shifted * scale, T

VerifiedDict = Dict[Tuple[int, int], dict]


class GeometricVerifier:
    """
    Parameters
    ----------
    ransac_threshold : RANSAC inlier threshold in pixels.
    min_inliers      : Minimum inliers required to keep a pair.
    confidence       : RANSAC confidence level.
    """

    def __init__(
        self,
        ransac_threshold: float = 1.0,
        min_inliers: int = 15,
        confidence: float = 0.999,
    ) -> None:
        self.ransac_threshold = ransac_threshold
        self.min_inliers      = min_inliers
        self.confidence       = confidence

        if _HAS_USAC_MAGSAC:
            logger.debug("Geometric verifier: USAC_MAGSAC for both F and E.")
        else:
            logger.debug("Geometric verifier: FM_RANSAC / RANSAC (fallback).")

    # ── public API ────────────────────────────────────────────────────────

    def verify_pair(
        self,
        kps1: np.ndarray,
        kps2: np.ndarray,
        matches: np.ndarray,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Verify one matched pair.

        Returns
        -------
        dict with keys {F, E, R, t, inlier_matches, n_inliers}
        or None if verification fails.
        """
        if len(matches) < self.min_inliers:
            return None

        # Shuffle before RANSAC — avoids a secondary ordering-triggered crash
        # seen in some OpenCV 4.x builds on specific datasets.
        rng_idx = np.random.permutation(len(matches))
        matches = matches[rng_idx]

        pts1 = kps1[matches[:, 0]].astype(np.float64)  # (M, 2)
        pts2 = kps2[matches[:, 1]].astype(np.float64)

        # Undistort before RANSAC when distortion coefficients are available
        if dist_coeffs is not None and np.any(dist_coeffs != 0):
            pts1 = undistort_points(pts1, K, dist_coeffs)
            pts2 = undistort_points(pts2, K, dist_coeffs)

        # ── Fundamental matrix (Hartley-normalized) ───────────────────────
        # Isotropic normalization maps pts to a well-conditioned range before
        # the DLT solver inside findFundamentalMat.  F is de-normalized after.
        pts1_norm, T1 = _hartley_normalize(pts1)
        pts2_norm, T2 = _hartley_normalize(pts2)

        try:
            F_norm, mask_f = cv2.findFundamentalMat(
                pts1_norm, pts2_norm,
                _F_METHOD,
                self.ransac_threshold,
                self.confidence,
            )
        except cv2.error as exc:
            logger.debug(f"  findFundamentalMat error: {exc}")
            return None

        if F_norm is None or mask_f is None:
            logger.debug("  F estimation failed (no solution)")
            return None

        # De-normalize: if x̃ = T·x, then x̃₂ᵀ F_norm x̃₁ = x₂ᵀ (T₂ᵀ F_norm T₁) x₁
        F = T2.T @ F_norm @ T1

        mask_f = mask_f.ravel().astype(bool)
        if mask_f.sum() < self.min_inliers:
            logger.debug(f"  Too few F-inliers: {mask_f.sum()}")
            return None

        inlier_matches = matches[mask_f]
        # Use the already-undistorted pts subset — more consistent than re-fetching kps
        pts1_fin = pts1[mask_f]   # (N1, 2) undistorted F-inliers
        pts2_fin = pts2[mask_f]

        # ── Essential matrix (USAC_MAGSAC, same quality level as F step) ──
        try:
            E, mask_e = cv2.findEssentialMat(
                pts1_fin.reshape(-1, 1, 2),
                pts2_fin.reshape(-1, 1, 2),
                K,
                _E_METHOD,
                self.confidence,
                self.ransac_threshold,
            )
        except cv2.error as exc:
            # Analytic fallback: E = Kᵀ F K is exact when F is correct
            logger.debug(f"  findEssentialMat error ({exc}), computing E = K^T F K")
            E = K.T @ F @ K
            mask_e = np.ones(len(inlier_matches), dtype=np.uint8).reshape(-1, 1)

        if E is None or mask_e is None:
            logger.debug("  E estimation failed")
            return None

        mask_e = mask_e.ravel().astype(bool)
        if mask_e.sum() < self.min_inliers:
            logger.debug(f"  Too few E-inliers: {mask_e.sum()}")
            return None

        inlier_matches = inlier_matches[mask_e]
        pts1_ein = pts1_fin[mask_e]   # (N2, 2) undistorted E-inliers
        pts2_ein = pts2_fin[mask_e]

        # ── Pose recovery (cheirality selects the correct (R, t) out of 4) ──
        try:
            n_pos, R, t, mask_pose = cv2.recoverPose(
                E,
                pts1_ein.reshape(-1, 1, 2),
                pts2_ein.reshape(-1, 1, 2),
                K,
            )
        except cv2.error as exc:
            logger.debug(f"  recoverPose error: {exc}")
            return None

        if n_pos < self.min_inliers:
            logger.debug(f"  Too few positive-depth pts: {n_pos}")
            return None

        mask_pose = mask_pose.ravel().astype(bool)
        inlier_matches = inlier_matches[mask_pose]

        # ── Reject near-degenerate (nearly-pure rotation) pairs ───────────
        baseline = float(np.linalg.norm(t))
        if baseline < 1e-4:
            logger.debug(f"  Near-zero baseline ({baseline:.2e}), skipped")
            return None

        return {
            "F": F,
            "E": E,
            "R": R,
            "t": t.flatten(),
            "inlier_matches": inlier_matches,
            "n_inliers": len(inlier_matches),
        }

    def verify_all(
        self,
        features: dict,
        all_matches: dict,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> VerifiedDict:
        """
        Verify every matched pair.

        Returns
        -------
        verified : dict  (i, j) → verification result
        """
        total = len(all_matches)
        logger.info(f"Geometric verification of {total} pairs…")

        verified: VerifiedDict = {}
        for (i, j), matches in all_matches.items():
            result = self.verify_pair(
                features[i]["keypoints"],
                features[j]["keypoints"],
                matches,
                K,
                dist_coeffs=dist_coeffs,
            )
            if result is not None:
                verified[(i, j)] = result
                logger.debug(
                    f"  ({i},{j}) ✓  {result['n_inliers']} inliers"
                )
            else:
                logger.debug(f"  ({i},{j}) ✗  failed")

        logger.info(
            f"Verification done: {len(verified)}/{total} pairs accepted "
            f"(min_inliers={self.min_inliers})"
        )
        return verified
