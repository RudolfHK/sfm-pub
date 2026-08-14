"""
Stage 6b — Multi-View Stereo (MVS) densification.

Produces a dense point cloud from the sparse SfM reconstruction by running
Semi-Global Block Matching (SGBM) on rectified stereo pairs.

Pipeline per pair
-----------------
1. cv2.stereoRectify        — compute rectification transforms R1/R2/P1/P2/Q
2. cv2.initUndistortRectifyMap + cv2.remap — warp images to common epipolar plane
3. cv2.StereoSGBM_create   — semi-global disparity computation
4. cv2.reprojectImageTo3D  — lift disparity → 3-D in rectified cam-1 frame
5. Coordinate transform    — rectified cam-1 frame → world frame
6. Filtering               — invalid disparity, negative depth, distance outliers

Pairs are selected from consecutive registered cameras (nearest neighbours by
sorted index).  At most `max_pairs_per_image` neighbours are processed per
image to bound runtime.

The output point cloud is merged from all successful stereo pairs and
subsampled to at most `max_dense_pts` points if necessary.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import camera_center, load_image

logger = logging.getLogger(__name__)


class MVSDensifier:
    """
    Dense reconstruction via Semi-Global Block Matching.

    Parameters
    ----------
    min_baseline_fraction : Minimum baseline as a fraction of scene scale.
                            Pairs with smaller baselines are skipped.
    num_disparities       : Must be divisible by 16.  Larger → slower but handles
                            larger depth ranges.
    block_size            : SGBM block size (odd, ≥ 1).  Smaller = finer detail.
    max_pairs_per_image   : Neighbour window size when selecting stereo pairs.
    max_dense_pts         : Cap on total output points (random subsampling).
    """

    def __init__(
        self,
        min_baseline_fraction: float = 0.02,
        num_disparities: int = 256,
        block_size: int = 5,
        max_pairs_per_image: int = 2,
        max_dense_pts: int = 500_000,
        mvs_fusion: bool = False,
        fusion_min_views: int = 2,
    ) -> None:
        self.min_baseline_fraction = min_baseline_fraction
        self.num_disparities       = (num_disparities // 16) * 16  # ensure multiple of 16
        self.block_size            = block_size
        self.max_pairs_per_image   = max_pairs_per_image
        self.max_dense_pts         = max_dense_pts
        self.mvs_fusion            = mvs_fusion
        self.fusion_min_views      = fusion_min_views
        # Upper bound on the searched window.  SGBM allocates proportionally to
        # it, so an unbounded range derived from a bad depth estimate could
        # exhaust memory.
        # SGBM allocates several buffers proportional to numDisparities times
        # the image area; 512 keeps a full-resolution 2736 x 1540 pair inside a
        # few hundred megabytes.  Pairs needing more are scaled down instead.
        self._max_num_disparities  = max(self.num_disparities, 512)
        # width x height x disparity range that the matcher may cost.  SGBM
        # holds several int16 buffers of that size, so 6e7 corresponds to a few
        # hundred megabytes.  A full-resolution wide-baseline pair on this
        # dataset would ask for 4e9 and simply fails to allocate.
        self._sgbm_cost_budget     = 6.0e7
        # Largest baseline, relative to the camera-to-object distance, that
        # still yields a disparity range fitting inside the image.  With
        # f ~ 0.7 x width, a ratio of 0.35 puts the nearest surface at roughly
        # a quarter of the image width even after rectification zoom.
        self.max_baseline_ratio    = 0.35

    # ── public API ────────────────────────────────────────────────────────

    def densify(
        self,
        cameras: dict,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
        image_paths: Dict[int, Path],
        max_reproj_error: float = 2.0,
        covisibility_counts: Optional[Dict[Tuple[int, int], int]] = None,
        points_3d: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run dense reconstruction for all valid stereo pairs.

        Parameters
        ----------
        cameras             : {img_idx: {'R':(3,3), 't':(3,1)}}
        K                   : (3,3) shared intrinsics
        dist_coeffs         : (4,) or (5,) distortion; None → zero distortion
        image_paths         : {img_idx: Path}
        max_reproj_error    : (unused; kept for API symmetry with other stages)
        covisibility_counts : {(i,j): shared_3d_point_count, i<j} — when provided,
                              pairs are selected by descending shared-point count
                              instead of consecutive index proximity.
        points_3d           : the sparse cloud.  Used to derive the disparity
                              search range per pair, and to reject dense points
                              that fall far outside the reconstructed scene.
                              Without it the stage falls back to a fixed window,
                              which is what produced unusable depth (see the
                              note on the disparity range below).

        Returns
        -------
        pts    : (N, 3) float64  world-space 3-D points
        colors : (N, 3) uint8    RGB colours
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()

        cam_list = sorted(cameras.keys())
        n        = len(cam_list)
        if n < 2:
            logger.warning("MVS: need ≥ 2 registered cameras — skipping densification.")
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

        # Scene scale: median camera-centre spread
        centres   = np.array([camera_center(cameras[c]["R"], cameras[c]["t"]) for c in cam_list])
        spread    = np.linalg.norm(centres - centres.mean(0), axis=1)
        scene_scale = float(np.median(spread)) if len(spread) > 1 else 1.0
        min_base  = self.min_baseline_fraction * scene_scale

        # Typical distance from a camera to the object; the baseline is judged
        # against it because disparity is proportional to baseline over depth.
        if points_3d is not None and len(points_3d) >= 8:
            obj_centre = np.median(np.asarray(points_3d, dtype=np.float64), axis=0)
        else:
            obj_centre = centres.mean(0)
        scene_depth = float(np.median(np.linalg.norm(centres - obj_centre, axis=1)))

        # Build per-image neighbour lists
        if covisibility_counts:
            # Covisibility-based: for each image, sort candidates by shared 3-D point count
            cam_set = set(cam_list)
            neighbours_map: Dict[int, list] = {}
            for i in cam_list:
                candidates = []
                for (a, b), count in covisibility_counts.items():
                    if a == i and b in cam_set:
                        candidates.append((b, count))
                    elif b == i and a in cam_set:
                        candidates.append((a, count))
                # Rectified stereo needs the two views similar enough that the
                # disparity range fits inside the image: OpenCV sizes its
                # buffers from width - (minDisparity + numDisparities).  The
                # ratio of disparity to width does not change with the working
                # resolution, so a pair that does not fit cannot be rescued by
                # scaling it down.  Candidates are therefore filtered by the
                # predicted disparity of the nearest scene point before
                # covisibility decides the order.
                Ci = camera_center(cameras[i]["R"], cameras[i]["t"])
                usable = []
                for cand, count in candidates:
                    Cc = camera_center(cameras[cand]["R"], cameras[cand]["t"])
                    base = float(np.linalg.norm(Cc - Ci))
                    if base < min_base:
                        continue
                    if scene_depth > 0 and base / scene_depth > self.max_baseline_ratio:
                        continue
                    usable.append((cand, count))
                usable.sort(key=lambda x: x[1], reverse=True)
                neighbours_map[i] = [c for c, _ in usable[: self.max_pairs_per_image]]
            logger.info(
                "[MVS] Pair selection: covisibility-based "
                f"(top-{self.max_pairs_per_image} per image)"
            )
        else:
            # Fallback: consecutive sorted index (original behaviour)
            neighbours_map = {}
            for r, i in enumerate(cam_list):
                neighbours_map[i] = cam_list[r + 1 : r + 1 + self.max_pairs_per_image]
            logger.warning(
                "[MVS] No covisibility data — using consecutive-index pair selection. "
                "Pass covisibility_counts from IncrementalSfM for better results."
            )

        all_pts, all_colors = [], []
        n_pairs_used = 0

        for i in cam_list:
            for j in neighbours_map[i]:
                Ri = cameras[i]["R"]
                ti = cameras[i]["t"].flatten()
                Rj = cameras[j]["R"]
                tj = cameras[j]["t"].flatten()

                # Relative rotation & translation (j expressed in i's frame)
                R_rel = Rj @ Ri.T
                t_rel = (tj - R_rel @ ti).reshape(3, 1)
                baseline = float(np.linalg.norm(t_rel))

                if baseline < min_base:
                    logger.debug(
                        f"MVS: pair ({i},{j}) baseline {baseline:.4f} < {min_base:.4f} — skip"
                    )
                    continue

                if i not in image_paths or j not in image_paths:
                    logger.debug(f"MVS: pair ({i},{j}) missing image path — skip")
                    continue

                # SGBM requires the left camera to be to the left (positive X) of
                # the right camera so that valid disparities are ≥ minDisparity=0.
                # When t_rel[0] < 0, camera j is to the LEFT of camera i in i's
                # rectified frame — swap roles so disparity is always positive.
                if float(t_rel[0, 0]) < 0:
                    Ri_proc, ti_proc = Rj, tj
                    R_rel_proc = Ri @ Rj.T
                    t_rel_proc = (ti - R_rel_proc @ tj).reshape(3, 1)
                    path_left, path_right = image_paths[j], image_paths[i]
                else:
                    Ri_proc, ti_proc = Ri, ti
                    R_rel_proc, t_rel_proc = R_rel, t_rel
                    path_left, path_right = image_paths[i], image_paths[j]

                pts, colors = self._process_pair(
                    Ri_proc, ti_proc, R_rel_proc, t_rel_proc,
                    K, dist_coeffs,
                    path_left, path_right,
                    points_3d=points_3d,
                )

                if len(pts) > 0:
                    all_pts.append(pts)
                    all_colors.append(colors)
                    n_pairs_used += 1
                    cov_str = ""
                    if covisibility_counts:
                        key = (min(i, j), max(i, j))
                        cnt = covisibility_counts.get(key, 0)
                        cov_str = f"  [{cnt} shared pts]"
                    logger.info(f"MVS: pair ({i},{j}){cov_str} → {len(pts):,} dense points")

        if not all_pts:
            logger.warning("MVS: no dense points generated.")
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

        pts_merged    = np.vstack(all_pts)
        colors_merged = np.vstack(all_colors)

        if self.mvs_fusion and len(cameras) >= self.fusion_min_views:
            pts_merged, colors_merged = self._depth_consistency_filter(
                pts_merged, colors_merged, cameras, K,
            )

        if len(pts_merged) > self.max_dense_pts:
            # Announce the cap.  Silently truncating made the reported point
            # count a property of this constant rather than of the scene: two
            # different dense runs both returned exactly 500 000 points with
            # nothing in the log to say so.
            logger.warning(
                "Dense cloud truncated: %d points subsampled to the "
                "--max-dense-points cap of %d. The reported point count is set "
                "by this cap, not by the scene; raise it to keep more.",
                len(pts_merged), self.max_dense_pts,
            )
            rng = np.random.default_rng(0)
            sel = rng.choice(len(pts_merged), self.max_dense_pts, replace=False)
            pts_merged    = pts_merged[sel]
            colors_merged = colors_merged[sel]

        logger.info(
            f"MVS densification: {n_pairs_used} stereo pairs → "
            f"{len(pts_merged):,} dense points total"
        )
        return pts_merged, colors_merged

    def _depth_consistency_filter(
        self,
        pts: np.ndarray,
        colors: np.ndarray,
        cameras: dict,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove world-space points that are not geometrically consistent across
        at least ``fusion_min_views`` registered cameras.

        A point is considered consistent in a camera when:
          1. Its depth in that camera is positive (> 0.01).
          2. Its projected pixel (u, v) lies within the image bounds.

        The image bounds are estimated from K as [0, 2*cx] × [0, 2*cy].
        This is more discriminating than a pure positive-depth check: points
        from degenerate stereo patches that happen to project behind a wall
        in the camera coordinate frame but outside the image frame are
        discarded even if their depth is technically positive.

        Activated by ``--mvs-fusion``.
        """
        n = len(pts)
        if n == 0:
            return pts, colors

        cam_list = sorted(cameras.keys())
        n_cams   = len(cam_list)
        n_req    = min(self.fusion_min_views, n_cams)

        # Batch: project all N points into all C cameras at once
        R_stack = np.stack([cameras[c]["R"] for c in cam_list], axis=0)   # (C, 3, 3)
        t_stack = np.stack([cameras[c]["t"].flatten() for c in cam_list], axis=0)  # (C, 3)

        # X_cam[c, :, n] = R_stack[c] @ pts[n] + t_stack[c]
        X_cam = (R_stack @ pts.T[None]) + t_stack[:, :, None]  # (C, 3, N)

        depths = X_cam[:, 2, :]   # (C, N) — Z in each camera
        z_safe = np.where(depths > 1e-6, depths, 1e-6)

        # Project to pixel coords using shared K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        # Image extents estimated from principal point
        w_est = cx * 2.0; h_est = cy * 2.0

        u = fx * X_cam[:, 0, :] / z_safe + cx   # (C, N)
        v = fy * X_cam[:, 1, :] / z_safe + cy   # (C, N)

        # Consistent = positive depth AND projected pixel inside image
        in_bounds = (
            (depths > 0.01)
            & (u >= 0) & (u < w_est)
            & (v >= 0) & (v < h_est)
        )   # (C, N)

        views_per_pt = in_bounds.sum(axis=0)   # (N,)
        keep = views_per_pt >= n_req
        n_removed = int((~keep).sum())

        if n_removed > 0:
            logger.info(
                f"[MVS fusion] Removed {n_removed}/{n} points "
                f"with < {n_req} consistent views (depth + in-bounds projection)"
            )

        return pts[keep].astype(np.float64), colors[keep]

    # ── internals ─────────────────────────────────────────────────────────



    def _required_disparity(
        self, K: np.ndarray, Ri: np.ndarray, ti: np.ndarray,
        t_rel: np.ndarray, points_3d: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        """Approximate (smallest, largest) disparity of the scene for this pair."""
        if points_3d is None or len(points_3d) < 8:
            return None
        pts = np.asarray(points_3d, dtype=np.float64)
        z = (Ri @ pts.T).T[:, 2] + np.asarray(ti, dtype=np.float64).ravel()[2]
        z = z[np.isfinite(z) & (z > 1e-6)]
        if z.size < 8:
            return None
        z_near = float(np.percentile(z, 2)) * 0.7
        z_far = float(np.percentile(z, 98)) * 1.6
        baseline = float(np.linalg.norm(np.asarray(t_rel, dtype=np.float64)))
        f = float(K[0, 0])
        if baseline <= 0 or z_near <= 0 or z_far <= z_near:
            return None
        return f * baseline / z_far, f * baseline / z_near

    def _resolution_for_pair(
        self, K: np.ndarray, Ri: np.ndarray, ti: np.ndarray,
        R_rel: np.ndarray, t_rel: np.ndarray, points_3d: Optional[np.ndarray],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> float:
        """
        Working scale for one pair, from the memory the matcher would need.

        SGBM allocates buffers proportional to width x height x disparity
        range.  Scaling the pair by s shrinks the area by s squared and the
        disparity range by s, so the cost falls with the cube of the scale.
        The factor is therefore chosen as the cube root of the ratio between
        the budget and the full-resolution cost, which is the largest scale
        that still fits.
        """
        rng = self._required_disparity(K, Ri, ti, t_rel, points_3d)
        if rng is None or image_shape is None:
            return 1.0
        d_min, d_max = rng
        span = max(d_max - d_min, 16.0)
        h, w = image_shape
        cost = float(w) * float(h) * span
        if cost <= self._sgbm_cost_budget:
            return 1.0
        scale = (self._sgbm_cost_budget / cost) ** (1.0 / 3.0)
        return float(np.clip(scale, 0.1, 1.0))

    def _disparity_range(
        self,
        P1: np.ndarray,
        P2: np.ndarray,
        Ri: np.ndarray,
        ti: np.ndarray,
        points_3d: Optional[np.ndarray],
    ) -> Tuple[int, int]:
        """
        Disparity window (minDisparity, numDisparities) for one rectified pair.

        Derived from the depth range of the sparse cloud as seen by this
        camera.  Returns the configured fixed window when no cloud is
        available, and logs that this is a guess.
        """
        fallback = (0, self.num_disparities)
        if points_3d is None or len(points_3d) < 8:
            return fallback

        pts = np.asarray(points_3d, dtype=np.float64)
        z = (Ri @ pts.T).T[:, 2] + np.asarray(ti, dtype=np.float64).ravel()[2]
        z = z[np.isfinite(z) & (z > 1e-6)]
        if z.size < 8:
            return fallback

        # Robust range, widened so points slightly outside the sparse cloud
        # still fall inside the search window.
        z_near = float(np.percentile(z, 2)) * 0.7
        z_far = float(np.percentile(z, 98)) * 1.6
        if not (0 < z_near < z_far):
            return fallback

        f_rect = float(P1[0, 0])
        baseline = abs(float(P2[0, 3]) / f_rect) if f_rect else 0.0
        if baseline <= 0.0:
            return fallback

        d_far = f_rect * baseline / z_far        # smallest disparity
        d_near = f_rect * baseline / z_near      # largest disparity

        min_disp = int(np.floor(d_far / 16.0) * 16)
        span = d_near - min_disp
        num_disp = int(np.ceil(span / 16.0) * 16)
        num_disp = int(np.clip(num_disp, 64, self._max_num_disparities))
        if min_disp < 0:
            min_disp = 0
        logger.debug(
            "  MVS disparity window: %d .. %d px (depth %.3f .. %.3f, "
            "baseline %.4f)", min_disp, min_disp + num_disp, z_near, z_far, baseline,
        )
        return min_disp, num_disp

    def _process_pair(
        self,
        Ri: np.ndarray,
        ti: np.ndarray,
        R_rel: np.ndarray,
        t_rel: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        path_i: Path,
        path_j: Path,
        points_3d: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SGBM on one rectified stereo pair; return world pts + RGB colors."""
        try:
            img_i = load_image(str(path_i))
            img_j = load_image(str(path_j))
        except Exception as exc:
            logger.warning(f"MVS: could not load images: {exc}")
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        h, w = img_i.shape[:2]
        D    = dist_coeffs.astype(np.float64)

        # ── Working resolution and disparity window ───────────────────────
        # These are wide-baseline pairs: with a baseline-to-depth ratio near a
        # half, the object's true disparity runs into the thousands of pixels
        # at full resolution.  SGBM allocates buffers proportional to
        # width x height x disparity range, so such a window cannot even be
        # allocated, and the fixed 0..256 window used before simply never
        # searched where the matches are.  Scaling image and intrinsics by the
        # same factor scales disparity by that factor and leaves the recovered
        # geometry unchanged.
        #
        # The window can only be measured after rectification, because
        # rectifying with alpha=0 changes the focal length.  So the scale is
        # chosen, the pair is rectified, the true window is measured, and if it
        # still does not fit the budget the step is repeated at a smaller
        # scale.
        K_full, w_full, h_full = K.copy(), w, h
        scale = 1.0
        rect = None
        for _attempt in range(4):
            size = (max(64, int(round(w_full * scale))),
                    max(64, int(round(h_full * scale))))
            K_s = K_full.copy()
            K_s[0, :] *= size[0] / w_full
            K_s[1, :] *= size[1] / h_full
            try:
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    K_s, D, K_s, D, size, R_rel, t_rel,
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                )
            except cv2.error as exc:
                logger.warning(f"MVS: stereoRectify failed: {exc}")
                return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

            min_disp, num_disp = self._disparity_range(P1, P2, Ri, ti, points_3d)
            cost = float(size[0]) * float(size[1]) * float(num_disp)
            rect = (size, K_s, R1, R2, P1, P2, Q, min_disp, num_disp)
            if cost <= self._sgbm_cost_budget or scale <= 0.1:
                break
            scale = max(0.1, scale * (self._sgbm_cost_budget / cost) ** (1.0 / 3.0) * 0.95)

        size, K_s, R1, R2, P1, P2, Q, min_disp, num_disp = rect

        # The searched window has to fit inside the image: OpenCV computes its
        # buffers from width - (minDisparity + numDisparities), which turns
        # negative otherwise and fails the allocation.  The ratio of disparity
        # to width does not change with the working resolution, since both
        # shrink together, so a pair that does not fit cannot be rescued by
        # scaling and is skipped.  That is the honest outcome: such a pair is
        # too wide-baseline for rectified block matching.
        if min_disp + num_disp >= 0.6 * size[0]:
            logger.debug(
                "  MVS pair skipped: disparity window %d..%d px does not fit a "
                "%d px wide image (baseline too wide for rectified stereo)",
                min_disp, min_disp + num_disp, size[0],
            )
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        if size != (w_full, h_full):
            img_i = cv2.resize(img_i, size, interpolation=cv2.INTER_AREA)
            img_j = cv2.resize(img_j, size, interpolation=cv2.INTER_AREA)
        K = K_s
        w, h = size
        logger.debug(
            "  MVS pair at %dx%d (scale %.3f), disparity window %d..%d px",
            size[0], size[1], size[0] / w_full, min_disp, min_disp + num_disp,
        )

        map1x, map1y = cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1)

        img1_rect = cv2.remap(img_i, map1x, map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img_j, map2x, map2y, cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)

        # Semi-global block matching
        bs     = self.block_size
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=bs,
            P1=8  * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

        # Reproject to 3-D in rectified camera-1 frame
        pts3d_rect = cv2.reprojectImageTo3D(disp, Q)   # (H, W, 3)

        # Valid mask: positive disparity + finite depth
        valid = (
            (disp > min_disp + 0.5)
            & np.isfinite(pts3d_rect[:, :, 2])
            & (pts3d_rect[:, :, 2] > 0)
            & (pts3d_rect[:, :, 2] < 1e6)
        )

        pts_rect  = pts3d_rect[valid]           # (N, 3) in rectified cam-1 frame
        colors_bgr = img1_rect[valid]           # (N, 3) BGR
        colors_rgb = colors_bgr[:, ::-1]        # → RGB

        if len(pts_rect) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        # Transform: rectified cam-1 frame → original cam-i frame
        # Rectification applies R1, so: X_cam_i = R1.T @ X_rect
        pts_cam_i = (R1.T @ pts_rect.T).T      # (N, 3)

        # Transform: cam-i frame → world frame
        # X_cam_i = Ri @ X_world + ti  →  X_world = Ri.T @ (X_cam_i − ti)
        pts_world = (Ri.T @ (pts_cam_i - ti).T).T   # (N, 3)

        # Filter: positive depth in camera i
        depth_valid = pts_cam_i[:, 2] > 0.01
        pts_world   = pts_world[depth_valid]
        colors_rgb  = colors_rgb[depth_valid]

        if len(pts_world) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        # Filter: keep what lies inside the reconstructed scene.
        # The old test measured distance from the *world origin*, which is the
        # first camera and not the object, and compared against the median of
        # the same distances — so when most points were wrong, the median was
        # wrong too and the filter kept them.  The sparse cloud is the trusted
        # geometry, so the bound comes from it.
        if points_3d is not None and len(points_3d) >= 8:
            sparse = np.asarray(points_3d, dtype=np.float64)
            centre = np.median(sparse, axis=0)
            radius = float(np.percentile(np.linalg.norm(sparse - centre, axis=1), 98))
            keep = np.linalg.norm(pts_world - centre, axis=1) <= max(3.0 * radius, 1e-6)
        else:
            dists = np.linalg.norm(pts_world, axis=1)
            median = float(np.median(dists))
            keep = dists < max(median * 10.0, 1e-3)
        pts_world  = pts_world[keep]
        colors_rgb = colors_rgb[keep]

        return pts_world.astype(np.float64), colors_rgb
