"""
Stage 3 — Geometric verification.

For each matched pair:
  1. Estimate the Fundamental matrix F via RANSAC.
  2. Derive the Essential matrix E = K^T F K and run a second RANSAC pass
     via findEssentialMat to refine the inlier set.
  3. Decompose E → (R, t) via cv2.recoverPose (cheirality check included).
  4. Reject pairs with too few inliers or near-zero baseline.

OpenCV ≥4.10 has a bug in FM_RANSAC / FM_LMEDS that crashes on certain
distributions of real SIFT keypoints.  We use USAC_MAGSAC for F estimation
(more robust and avoids the crash) and fall back to FM_RANSAC for E.

Output: verified pair dict containing F, E, R, t, and the inlier match indices.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pick best available F-estimation method.
# USAC_MAGSAC is available since OpenCV 4.5 and is both more robust and
# avoids the crash seen with FM_RANSAC in OpenCV ≥4.10.
_HAS_USAC_MAGSAC = hasattr(cv2, "USAC_MAGSAC")
_F_METHOD = cv2.USAC_MAGSAC if _HAS_USAC_MAGSAC else cv2.FM_RANSAC

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
            logger.debug("Geometric verifier using USAC_MAGSAC (robust, no crash).")
        else:
            logger.debug("Geometric verifier using FM_RANSAC (fallback).")

    # ── public API ────────────────────────────────────────────────────────

    def verify_pair(
        self,
        kps1: np.ndarray,
        kps2: np.ndarray,
        matches: np.ndarray,
        K: np.ndarray,
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

        # ── Fundamental matrix ────────────────────────────────────────────
        try:
            F, mask_f = cv2.findFundamentalMat(
                pts1, pts2,
                _F_METHOD,
                self.ransac_threshold,
                self.confidence,
            )
        except cv2.error as exc:
            logger.debug(f"  findFundamentalMat error: {exc}")
            return None

        if F is None or mask_f is None:
            logger.debug("  F estimation failed (no solution)")
            return None

        mask_f = mask_f.ravel().astype(bool)
        if mask_f.sum() < self.min_inliers:
            logger.debug(f"  Too few F-inliers: {mask_f.sum()}")
            return None

        inlier_matches = matches[mask_f]
        pts1_in = kps1[inlier_matches[:, 0]].astype(np.float64)
        pts2_in = kps2[inlier_matches[:, 1]].astype(np.float64)

        # ── Essential matrix + second RANSAC pass ─────────────────────────
        # findEssentialMat on the already-filtered inlier set (small, fast).
        try:
            E, mask_e = cv2.findEssentialMat(
                pts1_in.reshape(-1, 1, 2),
                pts2_in.reshape(-1, 1, 2),
                K,
                cv2.RANSAC,
                self.confidence,
                self.ransac_threshold,
            )
        except cv2.error as exc:
            # Fall back to computing E analytically from F
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
        pts1_final = kps1[inlier_matches[:, 0]].astype(np.float64).reshape(-1, 1, 2)
        pts2_final = kps2[inlier_matches[:, 1]].astype(np.float64).reshape(-1, 1, 2)

        # ── Pose recovery (cheirality selects the correct (R, t) out of 4) ──
        try:
            n_pos, R, t, mask_pose = cv2.recoverPose(E, pts1_final, pts2_final, K)
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
