"""
Stage 4 — Incremental Structure from Motion.

Algorithm outline
-----------------
1.  Select seed pair   : largest number of verified inliers.
2.  Initialise         : camera_0 = Identity; camera_1 from E decomposition.
                         Triangulate inlier matches → initial 3-D point set.
3.  Register loop      : while unregistered images remain:
      a. Find the image with the most 2-D ↔ 3-D correspondences.
      b. Register via PnP + RANSAC (undistorted keypoints).
      c. Triangulate new 3-D points with every already-registered neighbour.
      d. Run Bundle Adjustment every `ba_interval` new cameras.
4.  Final BA.

Data structures
---------------
cameras    : {img_idx: {'R':(3,3), 't':(3,1), 'K':(3,3)}}
points_3d  : list of (3,) float64 arrays      (index = pt_3d_idx)
kp_to_3d   : {(img_idx, kp_idx): pt_3d_idx}
observations: list of (img_idx, pt_3d_idx, x, y)   used by BA
             x, y are the ORIGINAL (distorted) pixel coordinates so that
             the BA projection model can apply and refine radial distortion.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .bundle_adjustment import BundleAdjuster, PyceresBundleAdjuster
from .intrinsics import CameraIntrinsics
from .utils import camera_center, projection_matrix, reprojection_error, undistort_points

logger = logging.getLogger(__name__)

# Tuning constants
_MIN_TRIANGULATION_ANGLE_DEG = 2.0
_MAX_POINT_DISTANCE_RATIO    = 50.0   # reject pts > ratio × median dist from origin


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _triangulate_dlt(
    P1: np.ndarray,
    P2: np.ndarray,
    pt1: np.ndarray,
    pt2: np.ndarray,
) -> Optional[np.ndarray]:
    """DLT triangulation via cv2.triangulatePoints. Returns (3,) or None."""
    X_h = cv2.triangulatePoints(
        P1, P2,
        pt1.reshape(2, 1).astype(np.float64),
        pt2.reshape(2, 1).astype(np.float64),
    )
    w = X_h[3, 0]
    if abs(w) < 1e-10:
        return None
    return (X_h[:3, 0] / w).astype(np.float64)


def _triangulate_batch(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate M point pairs in a single cv2.triangulatePoints call.

    Parameters
    ----------
    pts1, pts2 : (M, 2) float64 — undistorted image points

    Returns
    -------
    X3d   : (M, 3) float64 — 3-D points (zero-filled where w ≈ 0)
    valid : (M,) bool      — True where triangulation produced a finite point
    """
    M = len(pts1)
    if M == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)

    X_h = cv2.triangulatePoints(
        P1, P2,
        pts1.T.astype(np.float64),   # (2, M)
        pts2.T.astype(np.float64),
    )  # (4, M)

    w     = X_h[3]                              # (M,)
    valid = np.abs(w) > 1e-10
    X3d   = np.zeros((M, 3), dtype=np.float64)
    if valid.any():
        X3d[valid] = (X_h[:3, valid] / w[valid]).T
    return X3d, valid


def _accept_batch(
    X3d:    np.ndarray,   # (M, 3)
    R1:     np.ndarray,
    t1:     np.ndarray,   # (3,) or (3,1)
    R2:     np.ndarray,
    t2:     np.ndarray,
    pts1:   np.ndarray,   # (M, 2) undistorted
    pts2:   np.ndarray,
    C1:     np.ndarray,   # (3,)
    C2:     np.ndarray,
    K1:     np.ndarray,   # intrinsics for camera 1
    max_reproj_err: float,
    K2:     Optional[np.ndarray] = None,   # intrinsics for camera 2; defaults to K1
) -> np.ndarray:
    """
    Vectorised acceptance filter for a batch of triangulated 3-D points.

    Checks (in order):
      1. Positive depth in both cameras.
      2. Triangulation angle ≥ _MIN_TRIANGULATION_ANGLE_DEG.
      3. Reprojection error ≤ max_reproj_err in both cameras (pinhole model).

    Returns
    -------
    accept : (M,) bool
    """
    if K2 is None:
        K2 = K1
    t1f = t1.flatten()
    t2f = t2.flatten()
    f1, cx1, cy1 = K1[0, 0], K1[0, 2], K1[1, 2]
    f2, cx2, cy2 = K2[0, 0], K2[0, 2], K2[1, 2]

    X_cam1 = (R1 @ X3d.T).T + t1f   # (M, 3)
    X_cam2 = (R2 @ X3d.T).T + t2f

    depth_ok = (X_cam1[:, 2] > 0) & (X_cam2[:, 2] > 0)

    # Triangulation angle
    r1  = X3d - C1                                                    # (M, 3)
    r2  = X3d - C2
    n1  = np.linalg.norm(r1, axis=1, keepdims=True).clip(min=1e-10)
    n2  = np.linalg.norm(r2, axis=1, keepdims=True).clip(min=1e-10)
    cos = np.clip((r1 / n1 * r2 / n2).sum(axis=1), -1.0, 1.0)
    angle_ok = np.degrees(np.arccos(cos)) >= _MIN_TRIANGULATION_ANGLE_DEG

    # Reprojection — camera 1
    z1   = X_cam1[:, 2].clip(min=1e-6)
    u1   = f1 * X_cam1[:, 0] / z1 + cx1
    v1   = f1 * X_cam1[:, 1] / z1 + cy1
    err1 = np.hypot(u1 - pts1[:, 0], v1 - pts1[:, 1])

    # Reprojection — camera 2
    z2   = X_cam2[:, 2].clip(min=1e-6)
    u2   = f2 * X_cam2[:, 0] / z2 + cx2
    v2   = f2 * X_cam2[:, 1] / z2 + cy2
    err2 = np.hypot(u2 - pts2[:, 0], v2 - pts2[:, 1])

    reproj_ok = (err1 <= max_reproj_err) & (err2 <= max_reproj_err)

    return depth_ok & angle_ok & reproj_ok


def _triangulation_angle_ok(
    X: np.ndarray, C1: np.ndarray, C2: np.ndarray
) -> bool:
    """Return True when the bearing-angle at X is ≥ the minimum threshold."""
    r1 = X - C1
    r2 = X - C2
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-10 or n2 < 1e-10:
        return False
    cos_a = np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a))) >= _MIN_TRIANGULATION_ANGLE_DEG


# ─── IncrementalSfM ──────────────────────────────────────────────────────────

class IncrementalSfM:
    """Incremental SfM pipeline."""

    def __init__(
        self,
        features: dict,
        verified_pairs: dict,
        K: np.ndarray,
        max_reproj_error: float = 4.0,
        ba_interval: int = 5,
        dist_coeffs: Optional[np.ndarray] = None,
        refine_intrinsics: bool = True,
        fix_principal_point: bool = False,
        visualizer=None,
        merge_tracks: bool = False,
        per_camera_intrinsics: Optional[Dict[int, "CameraIntrinsics"]] = None,
        ba_backend: str = "scipy",
        local_ba_window: int = 0,
        ba_separate_focal: bool = False,
        pnp_backend: str = "cv2",
    ) -> None:
        self.features            = features
        self.verified_pairs      = verified_pairs
        self.K                   = K.copy()
        self.max_reproj_err      = max_reproj_error
        self.ba_interval         = ba_interval
        self.refine_intrinsics   = refine_intrinsics
        self.fix_principal_point = fix_principal_point
        self.merge_tracks        = merge_tracks
        self.viz                 = visualizer
        self._ba_step            = 0
        self._local_ba_window    = local_ba_window
        self._pnp_backend        = pnp_backend
        # Per-camera intrinsics mode: each image may have its own K
        self._per_cam_intr: Optional[Dict[int, CameraIntrinsics]] = per_camera_intrinsics

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(4, dtype=np.float64)
        else:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()

        # Pre-compute undistorted keypoints for PnP / triangulation.
        # Observations for BA keep the ORIGINAL distorted pixel coordinates.
        self._undist_kps: Dict[int, np.ndarray] = {}
        for img_idx, feat in features.items():
            kps = feat["keypoints"].astype(np.float64)
            K_i = self._get_K(img_idx)
            d_i = self._get_dist(img_idx)
            if np.any(d_i != 0):
                self._undist_kps[img_idx] = undistort_points(kps, K_i, d_i)
            else:
                self._undist_kps[img_idx] = kps

        # Reconstruction state
        self.cameras:      Dict[int, dict]            = {}
        self.points_3d:    List[np.ndarray]           = []
        self.observations: List[Tuple]                = []
        self.kp_to_3d:     Dict[Tuple[int,int], int]  = {}

        # Covisibility graph: img_idx → {img_idx, ...} for all images that share
        # at least one triangulated 3-D point. Maintained incrementally in _add_obs.
        # Reduces _count_corr and _triangulate_new_points from O(N_pairs) to O(degree).
        self.covisibility: Dict[int, Set[int]] = defaultdict(set)
        # Reverse map: 3-D point index → set of image indices that observe it.
        self._pt_observers: Dict[int, Set[int]] = defaultdict(set)
        # Pair index: img_idx → list of (i, j) verified pairs involving that image.
        # Built once from verified_pairs and used to avoid O(N²) pair scanning.
        self._pairs_by_img: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for pair_key in self.verified_pairs:
            i, j = pair_key
            self._pairs_by_img[i].append(pair_key)
            self._pairs_by_img[j].append(pair_key)

        if ba_backend == "pyceres":
            self._ba = PyceresBundleAdjuster(fix_principal_point=self.fix_principal_point)
        else:
            self._ba = BundleAdjuster(
                fix_principal_point=self.fix_principal_point,
                separate_focal=ba_separate_focal,
            )
        self._cams_since_ba   = 0
        self._registration_order: List[int] = []   # for local BA window

    # ── public entry point ────────────────────────────────────────────────

    def reconstruct(self) -> Tuple[dict, np.ndarray, list, dict]:
        """
        Run the full incremental SfM pipeline.

        Returns
        -------
        cameras      : {img_idx: {'R','t','K'}}
        points_3d    : (P, 3) float64
        observations : list of (img_idx, pt_idx, x, y)
        kp_to_3d     : {(img_idx, kp_idx): pt_3d_idx}
        """
        if not self.verified_pairs:
            raise ValueError("No verified pairs — cannot reconstruct.")

        if self.merge_tracks:
            self._merge_tracks()

        # ── Seed ──────────────────────────────────────────────────────────
        seed_i, seed_j = self._select_seed_pair()
        logger.info(f"Seed pair: images {seed_i} ↔ {seed_j}")
        if self.viz is not None:
            self.viz.on_seed_pair_selected(seed_i, seed_j)
        self._initialise_from_pair(seed_i, seed_j)
        logger.info(
            f"Initialised: {len(self.points_3d)} 3-D points "
            f"from {len(self.cameras)} cameras"
        )
        if self.viz is not None:
            pts0 = np.array(self.points_3d, dtype=np.float64) if self.points_3d else np.zeros((0, 3))
            csnap = {k: {"R": v["R"].copy(), "t": v["t"].copy()} for k, v in self.cameras.items()}
            self.viz.on_camera_registered((seed_i, seed_j), csnap, pts0, len(pts0))

        # ── Incremental registration ───────────────────────────────────────
        registered   = {seed_i, seed_j}
        unregistered = set(self.features.keys()) - registered

        while unregistered:
            pick = self._pick_next_image(unregistered)
            if pick is None:
                logger.warning("No more images can be registered — stopping.")
                break

            img_idx, n_corr = pick
            logger.info(
                f"Registering image {img_idx}  "
                f"({n_corr} 2D-3D correspondences)…"
            )

            ok = self._register_image(img_idx)
            unregistered.remove(img_idx)

            if not ok:
                logger.warning(f"  ✗ Registration failed for image {img_idx}")
                continue

            registered.add(img_idx)
            self._registration_order.append(img_idx)
            n_new = self._triangulate_new_points(img_idx)
            logger.info(
                f"  ✓ Registered. New 3-D pts: {n_new}, "
                f"total: {len(self.points_3d)}, cameras: {len(self.cameras)}"
            )
            if self.viz is not None:
                pts_snap = np.array(self.points_3d, dtype=np.float64) if self.points_3d else np.zeros((0, 3))
                csnap    = {k: {"R": v["R"].copy(), "t": v["t"].copy()} for k, v in self.cameras.items()}
                self.viz.on_camera_registered(img_idx, csnap, pts_snap, n_new)

            # Local BA: optimize last W cameras + their visible points after every registration
            if self._local_ba_window > 0 and len(self._registration_order) >= 2:
                local_win = self._registration_order[-self._local_ba_window :]
                self._run_local_ba(local_win)

            self._cams_since_ba += 1
            if self._cams_since_ba >= self.ba_interval:
                self._run_ba()
                self._cams_since_ba = 0

        # ── Final BA + cleanup ─────────────────────────────────────────────
        logger.info("Running final bundle adjustment…")
        self._run_ba()
        self._remove_outlier_points()

        pts_array = np.array(self.points_3d, dtype=np.float64) if self.points_3d else np.zeros((0, 3))
        logger.info(
            f"Reconstruction complete — "
            f"{len(self.cameras)}/{len(self.features)} cameras, "
            f"{len(pts_array)} 3-D points"
        )
        return self.cameras, pts_array, self.observations, self.kp_to_3d

    # ── seed selection ────────────────────────────────────────────────────

    def _select_seed_pair(self) -> Tuple[int, int]:
        """
        Pick the seed pair by score = baseline × inlier_count, enforcing a
        minimum median triangulation angle of 5°.  Falls back to max-inlier
        selection if no pair clears the angle threshold.
        """
        _MIN_ANGLE_DEG = 5.0

        best_key = None
        best_score = -1.0
        fallback_key = None
        fallback_score = -1.0

        for pair_key, data in self.verified_pairs.items():
            n_inliers = data["n_inliers"]
            t_vec = data["t"]
            baseline = float(np.linalg.norm(t_vec))
            score = baseline * n_inliers

            # Fallback: track best by inliers alone (no angle requirement)
            if n_inliers > fallback_score:
                fallback_score = n_inliers
                fallback_key = pair_key

            # Estimate median triangulation angle for this pair
            i, j = pair_key
            R_j, t_j = data["R"], data["t"].reshape(3, 1)
            R_i, t_i = np.eye(3), np.zeros((3, 1))
            C_i = camera_center(R_i, t_i)
            C_j = camera_center(R_j, t_j)
            P_i = projection_matrix(self._get_K(i), R_i, t_i)
            P_j = projection_matrix(self._get_K(j), R_j, t_j)

            matches = data["inlier_matches"]
            if len(matches) == 0:
                continue

            kps_i = self._undist_kps[i][matches[:, 0].astype(int)]
            kps_j = self._undist_kps[j][matches[:, 1].astype(int)]
            X3d, valid = _triangulate_batch(P_i, P_j, kps_i, kps_j)
            if not valid.any():
                continue

            X_valid = X3d[valid]
            r1 = X_valid - C_i
            r2 = X_valid - C_j
            n1 = np.linalg.norm(r1, axis=1, keepdims=True).clip(min=1e-10)
            n2 = np.linalg.norm(r2, axis=1, keepdims=True).clip(min=1e-10)
            cos_a = np.clip((r1 / n1 * r2 / n2).sum(axis=1), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos_a))
            median_angle = float(np.median(angles_deg))

            if median_angle < _MIN_ANGLE_DEG:
                continue

            if score > best_score:
                best_score = score
                best_key = pair_key
                _best_angle = median_angle
                _best_baseline = baseline
                _best_inliers = n_inliers

        if best_key is None:
            logger.warning(
                "[SEED] No pair met the %.1f° angle threshold — "
                "falling back to max-inlier selection.", _MIN_ANGLE_DEG
            )
            best_key = fallback_key

        i, j = best_key
        data = self.verified_pairs[best_key]
        logger.info(
            "[SEED] Selected pair (%d, %d): score=%.1f "
            "(baseline=%.3f, inliers=%d, median_angle=%.1f°)",
            i, j,
            best_score if best_key != fallback_key else fallback_score,
            float(np.linalg.norm(data["t"])),
            data["n_inliers"],
            _best_angle if best_key != fallback_key else 0.0,
        )
        return best_key

    # ── initialisation ────────────────────────────────────────────────────

    def _initialise_from_pair(self, i: int, j: int) -> None:
        data  = self.verified_pairs[(i, j)]
        R_j   = data["R"]
        t_j   = data["t"].reshape(3, 1)

        K_i = self._get_K(i)
        K_j = self._get_K(j)

        R_i, t_i = np.eye(3), np.zeros((3, 1))
        self.cameras[i] = {"R": R_i, "t": t_i, "K": K_i}
        self.cameras[j] = {"R": R_j, "t": t_j, "K": K_j}

        P_i = projection_matrix(K_i, R_i, t_i)
        P_j = projection_matrix(K_j, R_j, t_j)
        C_i = camera_center(R_i, t_i)
        C_j = camera_center(R_j, t_j)

        kps_i_ud   = self._undist_kps[i]
        kps_j_ud   = self._undist_kps[j]
        kps_i_orig = self.features[i]["keypoints"]
        kps_j_orig = self.features[j]["keypoints"]

        matches    = data["inlier_matches"]
        ki_arr     = matches[:, 0].astype(np.int32)
        kj_arr     = matches[:, 1].astype(np.int32)

        pts1_ud = kps_i_ud[ki_arr]   # (M, 2)
        pts2_ud = kps_j_ud[kj_arr]

        # Batch triangulation — one cv2 call instead of M calls
        X3d, valid_w = _triangulate_batch(P_i, P_j, pts1_ud, pts2_ud)

        if not valid_w.any():
            return

        # Vectorised acceptance (depth + angle + reproj) for all valid pts
        accept = np.zeros(len(matches), dtype=bool)
        vw_idx = np.where(valid_w)[0]
        accept[vw_idx] = _accept_batch(
            X3d[vw_idx],
            R_i, t_i, R_j, t_j,
            pts1_ud[vw_idx], pts2_ud[vw_idx],
            C_i, C_j, K_i, self.max_reproj_err, K2=K_j,
        )

        for k in np.where(accept)[0]:
            ki, kj = int(ki_arr[k]), int(kj_arr[k])
            X          = X3d[k]
            orig_pt1   = kps_i_orig[ki].astype(np.float64)
            orig_pt2   = kps_j_orig[kj].astype(np.float64)
            idx = self._add_point(X, i, ki, orig_pt1)
            self._link_kp(j, kj, idx)
            self._add_obs(j, idx, orig_pt2)

    # ── next-image selection ──────────────────────────────────────────────

    def _pick_next_image(
        self, unregistered: Set[int]
    ) -> Optional[Tuple[int, int]]:
        """Return (img_idx, n_correspondences) for the best candidate."""
        best_idx, best_n = None, 5   # minimum 6 required for PnP

        for img_idx in unregistered:
            n = self._count_corr(img_idx)
            if n > best_n:
                best_n, best_idx = n, img_idx

        return (best_idx, best_n) if best_idx is not None else None

    def _count_corr(self, img_idx: int) -> int:
        """Count 2D-3D correspondences available for img_idx.

        Uses _pairs_by_img to iterate only O(degree) pairs instead of
        all O(N²) verified pairs, giving O(degree × N_matches) complexity.
        """
        seen: Set[int] = set()
        for pair_key in self._pairs_by_img[img_idx]:
            i, j = pair_key
            data = self.verified_pairs[pair_key]
            other, self_col, other_col = self._pair_roles(img_idx, i, j)
            if other is None or other not in self.cameras:
                continue
            for m in data["inlier_matches"]:
                ko = int(m[other_col])
                if (other, ko) in self.kp_to_3d:
                    pt3d = self.kp_to_3d[(other, ko)]
                    if pt3d not in seen:
                        seen.add(pt3d)
        return len(seen)

    # ── image registration ────────────────────────────────────────────────

    def _register_poselib(
        self,
        pts2d_ud: np.ndarray,
        pts3d: np.ndarray,
        K: np.ndarray,
    ):
        """PnP via PoseLib p3p RANSAC; returns (R, t, inlier_indices) or (None, None, None)."""
        try:
            import poselib
        except ImportError:
            logger.warning(
                "poselib not installed — falling back to cv2.solvePnPRansac.  "
                "Install with: pip install poselib"
            )
            _no_dist = np.zeros(4, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d.reshape(-1, 1, 3),
                pts2d_ud.reshape(-1, 1, 2),
                K,
                _no_dist,
                confidence=0.999,
                reprojectionError=self.max_reproj_err,
                iterationsCount=1000,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 6:
                return None, None, None
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3, 1), inliers

        camera = {
            "model":  "PINHOLE",
            "width":  0,
            "height": 0,
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        }
        ransac_opts = poselib.RansacOptions()
        ransac_opts.max_reproj_error = self.max_reproj_err

        try:
            pose, info = poselib.estimate_absolute_pose(
                pts2d_ud.astype(np.float64),
                pts3d.astype(np.float64),
                camera,
                ransac_options=ransac_opts,
            )
        except Exception as exc:
            logger.debug(f"PoseLib p3p failed ({exc})")
            return None, None, None

        inliers_mask = np.array(info.get("inliers", []), dtype=bool)
        if inliers_mask.sum() < 6:
            return None, None, None

        R = np.array(pose.R, dtype=np.float64)
        t = np.array(pose.t, dtype=np.float64).reshape(3, 1)
        inlier_indices = np.where(inliers_mask)[0].reshape(-1, 1).astype(np.int32)
        return R, t, inlier_indices

    def _register_image(self, img_idx: int) -> bool:
        # PnP uses undistorted 2D points with the camera matrix (no dist needed)
        pts2d_ud, pts3d, kp_idxs, pt3d_idxs = self._get_corr(img_idx)
        if len(pts2d_ud) < 6:
            logger.debug(f"  Insufficient correspondences: {len(pts2d_ud)}")
            return False

        K_i  = self._get_K(img_idx)
        # pts2d_ud are already undistorted; pass zero dist to PnP so OpenCV
        # does not apply distortion correction a second time (W-02 fix).
        _no_dist = np.zeros(4, dtype=np.float64)

        if self._pnp_backend == "poselib":
            R, t, inliers = self._register_poselib(pts2d_ud, pts3d, K_i)
            if R is None:
                return False
        else:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d.reshape(-1, 1, 3),
                pts2d_ud.reshape(-1, 1, 2),
                K_i,
                _no_dist,
                confidence=0.999,
                reprojectionError=self.max_reproj_err,
                iterationsCount=1000,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 6:
                return False

            # LM refinement from EPnP initial estimate (~0.5 px improvement at <1 ms cost)
            inlier_pts3d = pts3d[inliers.flatten()]
            inlier_pts2d = pts2d_ud[inliers.flatten()]
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_pts3d.reshape(-1, 1, 3),
                    inlier_pts2d.reshape(-1, 1, 2),
                    K_i,
                    _no_dist,   # points are pre-undistorted
                    rvec,
                    tvec,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        20,
                        1e-6,
                    ),
                )
            except cv2.error:
                pass  # fall back to EPnP result if refinement fails

            R, _ = cv2.Rodrigues(rvec)
            t    = tvec.reshape(3, 1)

        # Reject wildly placed cameras (> 100× median scene extent)
        if self.cameras:
            scene_pts = np.array(self.points_3d)
            med_dist  = np.median(np.linalg.norm(scene_pts, axis=1)) + 1e-6
            cam_dist  = float(np.linalg.norm(t))
            if cam_dist > _MAX_POINT_DISTANCE_RATIO * med_dist:
                logger.warning(
                    f"  Suspicious camera position (dist={cam_dist:.1f}), rejected."
                )
                return False

        self.cameras[img_idx] = {"R": R, "t": t, "K": K_i}

        orig_kps = self.features[img_idx]["keypoints"]
        for ci in inliers.flatten():
            ki      = kp_idxs[ci]
            pt3d_i  = pt3d_idxs[ci]
            orig_pt = orig_kps[ki].astype(np.float64)
            if (img_idx, ki) not in self.kp_to_3d:
                self.kp_to_3d[(img_idx, ki)] = pt3d_i
            self._add_obs(img_idx, pt3d_i, orig_pt)

        logger.debug(f"  PnP inliers: {len(inliers)}/{len(pts2d_ud)}")
        return True

    def _get_corr(
        self, img_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Collect 2-D (undistorted) ↔ 3-D correspondences for PnP.

        Returns pts2d_ud (M,2), pts3d (M,3), kp_indices, pt3d_indices.
        """
        pts2d_l, pts3d_l, kp_l, pt3d_l = [], [], [], []
        seen_pt3d: Set[int] = set()
        kps_ud = self._undist_kps[img_idx]

        for pair_key in self._pairs_by_img[img_idx]:
            i, j = pair_key
            data = self.verified_pairs[pair_key]
            other, self_col, other_col = self._pair_roles(img_idx, i, j)
            if other is None or other not in self.cameras:
                continue
            for m in data["inlier_matches"]:
                ks = int(m[self_col])
                ko = int(m[other_col])
                if (other, ko) not in self.kp_to_3d:
                    continue
                pt3d_idx = self.kp_to_3d[(other, ko)]
                if pt3d_idx in seen_pt3d:
                    continue
                seen_pt3d.add(pt3d_idx)
                pts2d_l.append(kps_ud[ks].astype(np.float64))
                pts3d_l.append(self.points_3d[pt3d_idx])
                kp_l.append(ks)
                pt3d_l.append(pt3d_idx)

        if not pts2d_l:
            return (
                np.zeros((0, 2)),
                np.zeros((0, 3)),
                [],
                [],
            )
        return (
            np.array(pts2d_l, dtype=np.float64),
            np.array(pts3d_l, dtype=np.float64),
            kp_l,
            pt3d_l,
        )

    # ── triangulation ─────────────────────────────────────────────────────

    def _triangulate_new_points(self, new_idx: int) -> int:
        """
        Triangulate 3-D points visible from `new_idx` and any registered
        neighbour.  Uses batched triangulation per pair.
        """
        R_n = self.cameras[new_idx]["R"]
        t_n = self.cameras[new_idx]["t"]
        K_n = self._get_K(new_idx)
        P_n = projection_matrix(K_n, R_n, t_n)
        C_n = camera_center(R_n, t_n)
        kps_n_ud   = self._undist_kps[new_idx]
        kps_n_orig = self.features[new_idx]["keypoints"]
        n_new = 0

        for pair_key in self._pairs_by_img[new_idx]:
            i, j = pair_key
            data = self.verified_pairs[pair_key]
            other, self_col, other_col = self._pair_roles(new_idx, i, j)
            if other is None or other not in self.cameras:
                continue

            R_o  = self.cameras[other]["R"]
            t_o  = self.cameras[other]["t"]
            K_o  = self._get_K(other)
            P_o  = projection_matrix(K_o, R_o, t_o)
            C_o  = camera_center(R_o, t_o)
            kps_o_ud   = self._undist_kps[other]
            kps_o_orig = self.features[other]["keypoints"]

            matches = data["inlier_matches"]

            # First pass: propagate existing 3-D point links
            link_new: List[Tuple[int, int, int]] = []   # (ks, ko, pt3d_idx)
            to_tri_ks: List[int] = []
            to_tri_ko: List[int] = []

            for m in matches:
                ks = int(m[self_col])
                ko = int(m[other_col])

                if (other, ko) in self.kp_to_3d:
                    if (new_idx, ks) not in self.kp_to_3d:
                        pt3d_idx = self.kp_to_3d[(other, ko)]
                        link_new.append((ks, ko, pt3d_idx))
                    continue

                if (new_idx, ks) not in self.kp_to_3d:
                    to_tri_ks.append(ks)
                    to_tri_ko.append(ko)

            # Apply propagated links
            for ks, ko, pt3d_idx in link_new:
                self._link_kp(new_idx, ks, pt3d_idx)
                self._add_obs(new_idx, pt3d_idx, kps_n_orig[ks].astype(np.float64))

            if not to_tri_ks:
                continue

            # Batch triangulate the remaining unassigned matches
            ks_arr = np.array(to_tri_ks, dtype=np.int32)
            ko_arr = np.array(to_tri_ko, dtype=np.int32)
            pts_n  = kps_n_ud[ks_arr]   # (M, 2)
            pts_o  = kps_o_ud[ko_arr]

            X3d, valid_w = _triangulate_batch(P_n, P_o, pts_n, pts_o)

            if not valid_w.any():
                continue

            vw_idx = np.where(valid_w)[0]
            accept = np.zeros(len(ks_arr), dtype=bool)
            accept[vw_idx] = _accept_batch(
                X3d[vw_idx],
                R_n, t_n, R_o, t_o,
                pts_n[vw_idx], pts_o[vw_idx],
                C_n, C_o, K_n, self.max_reproj_err, K2=K_o,
            )

            for k in np.where(accept)[0]:
                ks, ko   = int(ks_arr[k]), int(ko_arr[k])
                X        = X3d[k]
                pt_n_orig = kps_n_orig[ks].astype(np.float64)
                pt_o_orig = kps_o_orig[ko].astype(np.float64)
                idx = self._add_point(X, new_idx, ks, pt_n_orig)
                self._link_kp(other, ko, idx)
                self._add_obs(other, idx, pt_o_orig)
                n_new += 1

        return n_new

    # ── point quality checks ──────────────────────────────────────────────

    def _accept_point(
        self,
        X: np.ndarray,
        R1: np.ndarray, t1: np.ndarray,
        R2: np.ndarray, t2: np.ndarray,
        pt1: np.ndarray, pt2: np.ndarray,
        C1: np.ndarray, C2: np.ndarray,
    ) -> bool:
        """Return True when X passes all geometric sanity checks (single point)."""
        t1f, t2f = t1.flatten(), t2.flatten()

        if (R1 @ X + t1f)[2] <= 0 or (R2 @ X + t2f)[2] <= 0:
            return False

        if not _triangulation_angle_ok(X, C1, C2):
            return False

        if (
            reprojection_error(X, pt1, self.K, R1, t1) > self.max_reproj_err
            or reprojection_error(X, pt2, self.K, R2, t2) > self.max_reproj_err
        ):
            return False

        return True

    # ── outlier removal ───────────────────────────────────────────────────

    def _remove_outlier_points(self) -> None:
        """
        Remove 3-D points whose mean reprojection error exceeds the threshold.
        Rebuilds self.points_3d, self.observations, and self.kp_to_3d with
        remapped indices.
        """
        n = len(self.points_3d)
        if n == 0:
            return

        # Accumulate per-point errors
        errors = [[] for _ in range(n)]
        for img_idx, pt_idx, x, y in self.observations:
            if img_idx not in self.cameras or pt_idx >= n:
                continue
            cam = self.cameras[img_idx]
            e = reprojection_error(
                self.points_3d[pt_idx],
                np.array([x, y]),
                self._get_K(img_idx), cam["R"], cam["t"],
            )
            if np.isfinite(e):
                errors[pt_idx].append(e)

        mean_errs = np.array(
            [np.mean(e) if e else np.inf for e in errors], dtype=np.float64
        )

        finite = mean_errs[np.isfinite(mean_errs)]
        if len(finite) == 0:
            return
        thr = min(
            self.max_reproj_err * 2.0,
            float(np.median(finite) + 3.0 * np.std(finite)),
        )
        keep = mean_errs <= thr

        n_removed = int((~keep).sum())
        if n_removed == 0:
            return
        logger.info(
            f"Outlier removal: dropped {n_removed}/{n} points "
            f"(threshold={thr:.2f} px)"
        )

        new_idx_map = np.full(n, -1, dtype=np.int64)
        new_idx_map[keep] = np.arange(keep.sum(), dtype=np.int64)

        self.points_3d = [self.points_3d[i] for i in range(n) if keep[i]]
        self.observations = [
            (img_idx, int(new_idx_map[pt_idx]), x, y)
            for img_idx, pt_idx, x, y in self.observations
            if pt_idx < n and keep[pt_idx]
        ]
        self.kp_to_3d = {
            key: int(new_idx_map[old])
            for key, old in self.kp_to_3d.items()
            if old < n and keep[old]
        }

    # ── track merging (union-find) ────────────────────────────────────────

    def _merge_tracks(self) -> None:
        """
        Build consistent feature tracks across all verified pairs using a
        union-find (disjoint-set union) structure, then rewrite verified_pairs
        so that each keypoint participates in at most one track.

        This pre-processing step prevents the incremental SfM loop from creating
        duplicate 3-D points for the same physical scene point when a keypoint is
        matched to more than one image independently.

        The approach:
        1.  Assign a global node ID to every (img_idx, kp_idx) observation.
        2.  Union matched pairs across all verified pairs.
        3.  Replace inlier_matches with the transitive-closure consistent subset:
            for each pair (i, j), only keep matches where both keypoints agree
            with their track root (no contradictory assignments).
        """
        # Collect all (img_idx, kp_idx) nodes
        node_to_id: Dict[Tuple[int, int], int] = {}
        id_ctr = [0]

        def node_id(img: int, kp: int) -> int:
            key = (img, kp)
            if key not in node_to_id:
                node_to_id[key] = id_ctr[0]
                id_ctr[0] += 1
            return node_to_id[key]

        for pair_key, data in self.verified_pairs.items():
            i, j = pair_key
            for m in data["inlier_matches"]:
                node_id(i, int(m[0]))
                node_id(j, int(m[1]))

        n_nodes = id_ctr[0]
        if n_nodes == 0:
            return

        parent = list(range(n_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Union all matched pairs
        for pair_key, data in self.verified_pairs.items():
            i, j = pair_key
            for m in data["inlier_matches"]:
                union(node_id(i, int(m[0])), node_id(j, int(m[1])))

        # Filter inlier_matches: within each pair, drop matches where the two
        # keypoints would create a conflicting track assignment (same root but
        # different source keypoints merged to the same track).
        n_removed = 0
        for pair_key, data in self.verified_pairs.items():
            i, j = pair_key
            kept = []
            # Track which roots have already been assigned in this pair
            used_roots_i: Set[int] = set()
            used_roots_j: Set[int] = set()
            for m in data["inlier_matches"]:
                ki, kj = int(m[0]), int(m[1])
                ri = find(node_id(i, ki))
                rj = find(node_id(j, kj))
                if ri in used_roots_i or rj in used_roots_j:
                    n_removed += 1
                    continue
                used_roots_i.add(ri)
                used_roots_j.add(rj)
                kept.append(m)
            if kept:
                data["inlier_matches"] = np.array(kept, dtype=data["inlier_matches"].dtype)
                data["n_inliers"] = len(kept)
            else:
                data["inlier_matches"] = np.zeros((0, 2), dtype=np.int32)
                data["n_inliers"] = 0

        logger.info(
            f"Track merging: {n_nodes} nodes, "
            f"{n_removed} conflicting matches removed"
        )

    # ── bundle adjustment wrapper ─────────────────────────────────────────

    def _run_local_ba(self, local_cam_idxs: List[int]) -> None:
        """
        Run BA over a sliding window of recently-registered cameras plus all
        3-D points visible from any of those cameras.

        Only the local cameras' poses are extracted from the result; all 3-D
        points in the window's field of view are updated.  Non-local cameras
        are left unchanged.  This O(W²·P_local) call is 10-50× cheaper than
        global BA for large scenes.
        """
        local_set = {idx for idx in local_cam_idxs if idx in self.cameras}
        if len(local_set) < 2:
            return

        local_cams = {idx: self.cameras[idx] for idx in local_set}
        pts_arr    = np.array(self.points_3d, dtype=np.float64)
        local_obs  = [
            (img, pt, x, y)
            for img, pt, x, y in self.observations
            if img in local_set and pt < len(pts_arr)
        ]
        if len(local_obs) < 8:
            return

        updated_cams, updated_pts, K_ref, dist_ref = self._ba.adjust(
            local_cams, pts_arr, local_obs, self.K,
            dist_coeffs=self.dist_coeffs,
            refine_intrinsics=self.refine_intrinsics,
        )

        for idx in local_set:
            if idx in updated_cams:
                self.cameras[idx] = updated_cams[idx]

        self.points_3d = [updated_pts[i] for i in range(len(updated_pts))]

        if K_ref is not None:
            self.K = K_ref
            self.dist_coeffs = dist_ref
            if self._per_cam_intr is not None:
                k1_new = float(dist_ref[0]) if len(dist_ref) > 0 else 0.0
                k2_new = float(dist_ref[1]) if len(dist_ref) > 1 else 0.0
                for img_idx in self._per_cam_intr:
                    self._per_cam_intr[img_idx] = (
                        self._per_cam_intr[img_idx].update_from_ba(k1=k1_new, k2=k2_new)
                    )

    def _run_ba(self) -> None:
        if len(self.cameras) < 2 or len(self.points_3d) < 10:
            return

        pts_arr      = np.array(self.points_3d, dtype=np.float64)
        n_pts_before = len(pts_arr)
        rmse_before  = self._quick_rmse(pts_arr) if self.viz is not None else 0.0

        updated_cams, updated_pts, K_ref, dist_ref = self._ba.adjust(
            self.cameras, pts_arr, self.observations, self.K,
            dist_coeffs=self.dist_coeffs,
            refine_intrinsics=self.refine_intrinsics,
        )
        self.cameras   = updated_cams
        self.points_3d = [updated_pts[i] for i in range(len(updated_pts))]

        if self.viz is not None:
            pts_after   = np.array(self.points_3d, dtype=np.float64)
            rmse_after  = self._quick_rmse(pts_after)
            self._ba_step += 1
            self.viz.on_bundle_adjustment_run(
                self._ba_step, rmse_before, rmse_after,
                len(self.cameras), n_pts_before, len(pts_after),
            )

        if K_ref is not None:
            self.K = K_ref
            self.dist_coeffs = dist_ref
            # Propagate refined k1/k2 into per-camera intrinsics so subsequent
            # _get_dist() calls return the updated distortion (W-03 fix).
            if self._per_cam_intr is not None:
                k1_new = float(dist_ref[0]) if len(dist_ref) > 0 else 0.0
                k2_new = float(dist_ref[1]) if len(dist_ref) > 1 else 0.0
                for img_idx in self._per_cam_intr:
                    self._per_cam_intr[img_idx] = (
                        self._per_cam_intr[img_idx].update_from_ba(k1=k1_new, k2=k2_new)
                    )
            if np.any(dist_ref != 0):
                for img_idx in self.features:
                    kps = self.features[img_idx]["keypoints"].astype(np.float64)
                    self._undist_kps[img_idx] = undistort_points(
                        kps, self._get_K(img_idx), self._get_dist(img_idx)
                    )

        self._retriangulate_after_ba()

    def _retriangulate_after_ba(self) -> None:
        """
        After BA refines camera poses, re-triangulate keypoint pairs whose 3-D
        track was not yet established.  Only registered camera pairs that share a
        verified match are considered.  New points go through the standard
        acceptance filter (depth + angle + reprojection) so they are consistent
        with the updated geometry.
        """
        if len(self.cameras) < 2:
            return

        n_new_total = 0
        registered  = set(self.cameras.keys())

        for pair_key, data in self.verified_pairs.items():
            i, j = pair_key
            if i not in registered or j not in registered:
                continue

            R_i = self.cameras[i]["R"];  t_i = self.cameras[i]["t"]
            R_j = self.cameras[j]["R"];  t_j = self.cameras[j]["t"]
            K_i = self._get_K(i);  K_j = self._get_K(j)
            P_i = projection_matrix(K_i, R_i, t_i)
            P_j = projection_matrix(K_j, R_j, t_j)
            C_i = camera_center(R_i, t_i)
            C_j = camera_center(R_j, t_j)

            kps_i_ud   = self._undist_kps[i]
            kps_j_ud   = self._undist_kps[j]
            kps_i_orig = self.features[i]["keypoints"]
            kps_j_orig = self.features[j]["keypoints"]

            matches = data["inlier_matches"]
            # Collect match pairs that still lack a 3-D assignment on BOTH sides
            free_ki, free_kj = [], []
            for m in matches:
                ki, kj = int(m[0]), int(m[1])
                if (i, ki) not in self.kp_to_3d and (j, kj) not in self.kp_to_3d:
                    free_ki.append(ki)
                    free_kj.append(kj)

            if not free_ki:
                continue

            pts1_ud = kps_i_ud[free_ki]
            pts2_ud = kps_j_ud[free_kj]
            X3d, valid_w = _triangulate_batch(P_i, P_j, pts1_ud, pts2_ud)

            if not valid_w.any():
                continue

            vw_idx = np.where(valid_w)[0]
            accept = np.zeros(len(free_ki), dtype=bool)
            accept[vw_idx] = _accept_batch(
                X3d[vw_idx],
                R_i, t_i, R_j, t_j,
                pts1_ud[vw_idx], pts2_ud[vw_idx],
                C_i, C_j, K_i, self.max_reproj_err, K2=K_j,
            )

            for k in np.where(accept)[0]:
                ki = free_ki[k]; kj = free_kj[k]
                orig_pt_i = kps_i_orig[ki].astype(np.float64)
                orig_pt_j = kps_j_orig[kj].astype(np.float64)
                idx = self._add_point(X3d[k], i, ki, orig_pt_i)
                self._link_kp(j, kj, idx)
                self._add_obs(j, idx, orig_pt_j)
                n_new_total += 1

        if n_new_total:
            logger.info(f"  Re-triangulation after BA: +{n_new_total} new 3-D points")

    # ── small utilities ───────────────────────────────────────────────────

    @staticmethod
    def _pair_roles(
        img_idx: int, i: int, j: int
    ) -> Tuple[Optional[int], int, int]:
        """Return (other_img, self_col, other_col) for a pair (i,j)."""
        if i == img_idx:
            return j, 0, 1
        if j == img_idx:
            return i, 1, 0
        return None, -1, -1

    def _add_point(
        self, X: np.ndarray, img_idx: int, kp_idx: int, pt2d: np.ndarray
    ) -> int:
        """Append a new 3-D point, link it to img_idx/kp_idx, add observation."""
        idx = len(self.points_3d)
        self.points_3d.append(X)
        self._link_kp(img_idx, kp_idx, idx)
        self._add_obs(img_idx, idx, pt2d)
        return idx

    def _get_K(self, img_idx: int) -> np.ndarray:
        """Return the (3,3) camera matrix for img_idx (per-cam or shared)."""
        if self._per_cam_intr is not None and img_idx in self._per_cam_intr:
            return self._per_cam_intr[img_idx].to_K()
        return self.K

    def _get_dist(self, img_idx: int) -> np.ndarray:
        """Return distortion coefficients for img_idx (per-cam or shared)."""
        if self._per_cam_intr is not None and img_idx in self._per_cam_intr:
            return self._per_cam_intr[img_idx].to_dist()
        return self.dist_coeffs

    def _quick_rmse(self, pts_arr: np.ndarray) -> float:
        """Fast vectorised reprojection RMSE over all current observations."""
        if not self.cameras or not self.observations:
            return 0.0
        sq: List[float] = []
        for img_idx, pt_idx, x, y in self.observations:
            if img_idx not in self.cameras or pt_idx >= len(pts_arr):
                continue
            K_i = self._get_K(img_idx)
            f = K_i[0, 0]; cx = K_i[0, 2]; cy = K_i[1, 2]
            cam = self.cameras[img_idx]
            X_c = cam["R"] @ pts_arr[pt_idx] + cam["t"].flatten()
            if X_c[2] > 1e-6:
                u = f * X_c[0] / X_c[2] + cx
                v = f * X_c[1] / X_c[2] + cy
                sq.append((u - x) ** 2 + (v - y) ** 2)
        return float(np.sqrt(np.mean(sq))) if sq else 0.0

    def _link_kp(self, img_idx: int, kp_idx: int, pt3d_idx: int) -> None:
        self.kp_to_3d[(img_idx, kp_idx)] = pt3d_idx

    def _add_obs(
        self, img_idx: int, pt3d_idx: int, pt2d: np.ndarray
    ) -> None:
        if img_idx in self._pt_observers[pt3d_idx]:
            return
        self.observations.append(
            (img_idx, pt3d_idx, float(pt2d[0]), float(pt2d[1]))
        )
        for other_img in self._pt_observers[pt3d_idx]:
            self.covisibility[img_idx].add(other_img)
            self.covisibility[other_img].add(img_idx)
        self._pt_observers[pt3d_idx].add(img_idx)
