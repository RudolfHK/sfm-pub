"""
Focal-length initialisation by search over candidate values.

Why this exists
---------------
Without EXIF, `estimate_intrinsics` falls back to `focal = max(W, H)` — a ~53°
FoV guess.  On the AliceVision Buddha set that guess is 2736 px against a true
1860.9 px: **+47 %**, and it propagates into every triangulation, every PnP and
the seed pose, costing ~6° of median rotation error.  COLMAP, starting from the
same position, recovers the focal to within 0.2 %.

What to score on
----------------
The controlled sweep in `paper/scripts/focal_experiment.py` (Tab. in
`paper/paper_l.md` §6.5) measured reconstruction quality against assumed focal
with everything else held fixed:

    focal error   −30 %   −15 %    0 %    +15 %   +30 %   +47 %   +70 %
    observations  2 310   4 866   8 304   5 310   2 914   3 953   2 002
    3-D points      611     869   1 214     890     666     790     517
    reproj RMSE    6.12    6.33    2.86    5.50    5.67    6.11    6.39

The count of surviving observations peaks sharply and unimodally at the true
focal — 1.7× the nearest wrong candidate.  Reprojection error separates right
from wrong but does **not** order the wrong values (6.12 at −30 % vs 5.50 at
+15 %), which is the same blindness §6.6 documents for the pipeline's own
quality metric.  So this search maximises accepted correspondences and ignores
reprojection error as a ranking signal.

Cost
----
Verification and two-view triangulation on a bounded subset of the strongest
pairs, per candidate.  With the defaults (12 pairs × 13 candidates) this is a
few seconds even on the 67-image set, against a 1000 s+ matching stage.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Multiplicative sweep around the initial guess.  Spans roughly a 3× focal
# range, which brackets both the max(W,H) fallback being far too long (the
# Buddha case, true/guess = 0.68) and too short.
DEFAULT_FACTORS: Tuple[float, ...] = (
    0.45, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0, 1.15, 1.3, 1.5, 1.75,
)


def _K_with_focal(K: np.ndarray, focal: float) -> np.ndarray:
    """Copy of K with both focal entries replaced; principal point untouched."""
    K_new = np.asarray(K, dtype=np.float64).copy()
    K_new[0, 0] = focal
    K_new[1, 1] = focal
    return K_new


def _select_pairs(
    matches: Dict[Tuple[int, int], np.ndarray], n_pairs: int
) -> List[Tuple[int, int]]:
    """
    The `n_pairs` pairs with the most raw matches, with each image used at most
    twice.

    Spreading the sample over distinct images stops the score from being
    decided by one accidentally-rich cluster of near-duplicate views, which
    would make the peak reflect that cluster's geometry rather than the
    camera's.
    """
    ranked = sorted(matches.items(), key=lambda kv: len(kv[1]), reverse=True)
    used: Dict[int, int] = {}
    chosen: List[Tuple[int, int]] = []
    for (i, j), _m in ranked:
        if len(chosen) >= n_pairs:
            break
        if used.get(i, 0) >= 2 or used.get(j, 0) >= 2:
            continue
        chosen.append((i, j))
        used[i] = used.get(i, 0) + 1
        used[j] = used.get(j, 0) + 1
    # Fall back to plain ranking when the spread rule was too strict to fill.
    if len(chosen) < min(n_pairs, len(ranked)):
        for (i, j), _m in ranked:
            if len(chosen) >= n_pairs:
                break
            if (i, j) not in chosen:
                chosen.append((i, j))
    return chosen


def _score_focal(
    focal: float,
    K: np.ndarray,
    features: dict,
    matches: Dict[Tuple[int, int], np.ndarray],
    pairs: Sequence[Tuple[int, int]],
    ransac_threshold: float,
    min_inliers: int,
    max_reproj_error: float,
) -> dict:
    """
    Verify and two-view triangulate every sampled pair at this focal.

    Returns the total number of accepted 3-D points, which is the ranking
    signal, plus diagnostics that are logged but never ranked on.
    """
    # Imported here to avoid a circular import at module load.
    from .geometric_verification import GeometricVerifier
    from .reconstruction import _accept_batch, _triangulate_batch
    from .utils import camera_center, projection_matrix

    K_f = _K_with_focal(K, focal)
    verifier = GeometricVerifier(
        ransac_threshold=ransac_threshold, min_inliers=min_inliers,
    )

    n_points = 0
    n_inliers = 0
    n_verified = 0
    zero = np.zeros(4, dtype=np.float64)

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    P1 = projection_matrix(K_f, R1, t1)
    C1 = camera_center(R1, t1)

    for (i, j) in pairs:
        m = matches.get((i, j))
        if m is None or len(m) < min_inliers:
            continue
        res = verifier.verify_pair(
            features[i]["keypoints"], features[j]["keypoints"], m, K_f,
            dist_coeffs=zero,
        )
        if res is None:
            continue
        n_verified += 1
        n_inliers += int(res["n_inliers"])

        R2 = np.asarray(res["R"], dtype=np.float64)
        t2 = np.asarray(res["t"], dtype=np.float64).reshape(3, 1)
        P2 = projection_matrix(K_f, R2, t2)
        C2 = camera_center(R2, t2)

        inl = res["inlier_matches"]
        pts1 = features[i]["keypoints"][inl[:, 0]].astype(np.float64)
        pts2 = features[j]["keypoints"][inl[:, 1]].astype(np.float64)

        X3d, valid = _triangulate_batch(P1, P2, pts1, pts2)
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        acc = _accept_batch(
            X3d[idx], R1, t1, R2, t2, pts1[idx], pts2[idx],
            C1, C2, K_f, max_reproj_error, K2=K_f,
        )
        n_points += int(np.count_nonzero(acc))

    return {
        "focal": float(focal),
        "n_points": int(n_points),
        "n_inliers": int(n_inliers),
        "n_verified_pairs": int(n_verified),
    }


def search_focal(
    K: np.ndarray,
    features: dict,
    matches: Dict[Tuple[int, int], np.ndarray],
    factors: Sequence[float] = DEFAULT_FACTORS,
    n_pairs: int = 12,
    ransac_threshold: float = 1.0,
    min_inliers: int = 15,
    max_reproj_error: float = 4.0,
) -> Tuple[float, List[dict]]:
    """
    Choose the focal that maximises accepted two-view correspondences.

    Parameters
    ----------
    K        : initial camera matrix; only the focal entries are varied.
    features : the extracted-feature dict, as passed to matching.
    matches  : raw pairwise matches, before geometric verification.
    factors  : multiples of the initial focal to try.

    Returns
    -------
    best_focal : the winning focal in pixels (the initial value when the search
                 cannot decide, so the caller never has to special-case it).
    curve      : one score record per candidate, for logging and for the
                 `--export-cameras` payload.
    """
    f0 = float(K[0, 0])
    if not matches:
        logger.warning("Focal search: no matches available — keeping f=%.1f px", f0)
        return f0, []

    pairs = _select_pairs(matches, n_pairs)
    if not pairs:
        logger.warning("Focal search: no usable pairs — keeping f=%.1f px", f0)
        return f0, []

    t0 = time.time()
    logger.info(
        "Focal search: %d candidates × %d pairs (initial f=%.1f px)…",
        len(factors), len(pairs), f0,
    )

    curve: List[dict] = []
    for factor in factors:
        rec = _score_focal(
            f0 * factor, K, features, matches, pairs,
            ransac_threshold, min_inliers, max_reproj_error,
        )
        rec["factor"] = float(factor)
        curve.append(rec)
        logger.debug(
            "  f=%.1f px (×%.2f): %d points, %d inliers, %d/%d pairs verified",
            rec["focal"], factor, rec["n_points"], rec["n_inliers"],
            rec["n_verified_pairs"], len(pairs),
        )

    best = max(curve, key=lambda r: r["n_points"])
    elapsed = time.time() - t0

    if best["n_points"] == 0:
        logger.warning(
            "Focal search: no candidate produced any accepted point — "
            "keeping f=%.1f px", f0,
        )
        return f0, curve

    # Refine with a parabola through the peak and its two neighbours.  The
    # score is sampled on a coarse multiplicative grid, so the true optimum
    # generally sits between samples.
    best_i = curve.index(best)
    focal = best["focal"]
    if 0 < best_i < len(curve) - 1:
        x = np.array([curve[best_i - 1]["focal"], focal, curve[best_i + 1]["focal"]])
        y = np.array([
            curve[best_i - 1]["n_points"], best["n_points"],
            curve[best_i + 1]["n_points"],
        ], dtype=np.float64)
        denom = (y[0] - 2.0 * y[1] + y[2])
        if denom < -1e-9:      # strictly concave ⇒ a real interior maximum
            shift = 0.5 * (y[0] - y[2]) / denom
            vertex = x[1] + shift * (x[2] - x[0]) * 0.5
            if x[0] < vertex < x[2]:
                focal = float(vertex)

    logger.info(
        "Focal search: f=%.1f px chosen (%+.1f%% vs initial, %d accepted points, "
        "peak %.1f×) in %.1fs",
        focal, 100.0 * (focal - f0) / f0, best["n_points"],
        best["factor"], elapsed,
    )
    return float(focal), curve
