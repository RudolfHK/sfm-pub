"""
Bundle Adjustment (BA) via sparse Levenberg-Marquardt.

Minimises the sum of squared reprojection errors over all cameras and 3-D
points jointly, with a robust Huber loss to down-weight outliers.

Parameter vector layout
-----------------------
  refine_intrinsics=True, fix_principal_point=False (default):
    [ f(1)  k1(1)  k2(1)  cx(1)  cy(1) |
      cam_0_rvec(3)  cam_0_tvec(3) | … | cam_{C-1}_rvec  cam_{C-1}_tvec |
      X_0(3)  X_1(3)  …  X_{P-1}(3) ]

    Total length = 5 + 6*C + 3*P   (N_SHARED = 5)

  refine_intrinsics=True, fix_principal_point=True:
    [ f(1)  k1(1)  k2(1) |
      cam_0_rvec(3)  cam_0_tvec(3) | … | cam_{C-1}_rvec  cam_{C-1}_tvec |
      X_0(3)  X_1(3)  …  X_{P-1}(3) ]

    Total length = 3 + 6*C + 3*P   (N_SHARED = 3)

  refine_intrinsics=False:
    [ cam_0_rvec(3)  cam_0_tvec(3) | … | X_0(3) … ]

    Total length = 6*C + 3*P        (N_SHARED = 0)

Projection model
----------------
  Radial distortion (Brown–Conrady, 2 coefficients):

    X_cam = R * X_world + t
    xn = X_cam[0] / X_cam[2],  yn = X_cam[1] / X_cam[2]
    r² = xn² + yn²
    u  = f * xn * (1 + k1*r² + k2*r⁴) + cx
    v  = f * yn * (1 + k1*r² + k2*r⁴) + cy

  When fix_principal_point=False, cx and cy are optimized (default).
  When k1 = k2 = 0 this reduces to the standard pinhole model.

Observation convention
-----------------------
  (camera_idx, point_idx, x_obs, y_obs)  — camera_idx and point_idx are the
  *consecutive* indices used inside BA (not the original image indices).
  x_obs / y_obs are the *original distorted* pixel coordinates.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


# ─── Vectorised Rodrigues rotation ───────────────────────────────────────────

def _rodrigues_rotate_batch(rvecs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply N independent Rodrigues rotations to N points.

    Parameters
    ----------
    rvecs : (N, 3)  axis-angle rotation vectors
    pts   : (N, 3)  3-D points

    Returns
    -------
    rotated : (N, 3)
    """
    thetas = np.linalg.norm(rvecs, axis=1, keepdims=True)   # (N,1)
    small = thetas.flatten() < 1e-10

    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(thetas > 1e-10, rvecs / thetas, rvecs)  # (N,3) unit axes

    cos_t = np.cos(thetas)                                    # (N,1)
    sin_t = np.sin(thetas)                                    # (N,1)
    k_dot_p = np.einsum("ni,ni->n", k, pts)[:, np.newaxis]   # (N,1)
    k_cross_p = np.cross(k, pts)                              # (N,3)

    rotated = cos_t * pts + sin_t * k_cross_p + (1.0 - cos_t) * k_dot_p * k
    rotated[small] = pts[small]
    return rotated


# ─── Distortion-aware projection ─────────────────────────────────────────────

def _project_distorted(
    cam_params: np.ndarray,
    pts_3d: np.ndarray,
    f: float,
    k1: float,
    k2: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Project N 3-D points through N cameras with radial distortion.

    Parameters
    ----------
    cam_params : (N, 6)  [rvec(3) | tvec(3)] per observation
    pts_3d     : (N, 3)
    f, k1, k2  : shared intrinsics (single focal; k1/k2 Brown–Conrady)
    cx, cy     : fixed principal point

    Returns
    -------
    projected : (N, 2)
    """
    rvecs = cam_params[:, :3]
    tvecs = cam_params[:, 3:6]

    X_cam = _rodrigues_rotate_batch(rvecs, pts_3d) + tvecs   # (N,3)

    z_safe = np.where(X_cam[:, 2] > 1e-6, X_cam[:, 2], 1e-6)

    xn = X_cam[:, 0] / z_safe
    yn = X_cam[:, 1] / z_safe
    r2 = xn ** 2 + yn ** 2
    dist_factor = 1.0 + k1 * r2 + k2 * r2 ** 2

    u = f * xn * dist_factor + cx
    v = f * yn * dist_factor + cy

    return np.stack([u, v], axis=1)                          # (N,2)


# ─── Residual function ───────────────────────────────────────────────────────

def _residuals_v2(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    cam_indices: np.ndarray,
    pt_indices: np.ndarray,
    pts_2d: np.ndarray,
    cx_fixed: float,
    cy_fixed: float,
    refine_intrinsics: bool,
    fix_principal_point: bool,
    f_fixed: float,
    k1_fixed: float,
    k2_fixed: float,
) -> np.ndarray:
    if refine_intrinsics:
        f, k1, k2 = params[0], params[1], params[2]
        if fix_principal_point:
            cx, cy = cx_fixed, cy_fixed
            off = 3
        else:
            cx, cy = params[3], params[4]
            off = 5
    else:
        f, k1, k2 = f_fixed, k1_fixed, k2_fixed
        cx, cy = cx_fixed, cy_fixed
        off = 0

    cam_params = params[off : off + n_cameras * 6].reshape(n_cameras, 6)
    pts3d      = params[off + n_cameras * 6 :].reshape(n_points, 3)

    obs_cam = cam_params[cam_indices]
    obs_pts = pts3d[pt_indices]

    projected = _project_distorted(obs_cam, obs_pts, f, k1, k2, cx, cy)
    return (projected - pts_2d).ravel()


# ─── Sparsity pattern ────────────────────────────────────────────────────────

def _build_sparsity_v2(
    n_cameras: int,
    n_points: int,
    cam_indices: np.ndarray,
    pt_indices: np.ndarray,
    refine_intrinsics: bool,
    fix_principal_point: bool = True,
) -> "scipy.sparse.csr_matrix":
    n_obs    = len(cam_indices)
    if refine_intrinsics:
        n_shared = 3 if fix_principal_point else 5
    else:
        n_shared = 0
    n_params = n_shared + n_cameras * 6 + n_points * 3
    n_res    = n_obs * 2

    J = lil_matrix((n_res, n_params), dtype=np.int8)

    for k in range(n_obs):
        c = int(cam_indices[k])
        p = int(pt_indices[k])

        row_x = 2 * k
        row_y = 2 * k + 1

        # Shared intrinsics (dense columns — every residual touches these)
        if refine_intrinsics:
            J[row_x, 0:n_shared] = 1
            J[row_y, 0:n_shared] = 1

        # Camera pose block (6 params)
        cam_start = n_shared + c * 6
        J[row_x, cam_start : cam_start + 6] = 1
        J[row_y, cam_start : cam_start + 6] = 1

        # 3-D point block (3 params)
        pt_start = n_shared + n_cameras * 6 + p * 3
        J[row_x, pt_start : pt_start + 3] = 1
        J[row_y, pt_start : pt_start + 3] = 1

    return J.tocsr()


# ─── BundleAdjuster class ────────────────────────────────────────────────────

class BundleAdjuster:
    """
    Sparse bundle adjustment using scipy's TRF least-squares solver.

    Parameters
    ----------
    max_nfev            : Maximum number of function evaluations.
    ftol/gtol/xtol      : Convergence tolerances.
    loss                : Robust loss name ('huber', 'cauchy', 'linear', …).
    f_scale             : Scale of the robust loss (pixels).
    fix_principal_point : When False (default), cx and cy are included in the
                          optimized parameter vector.  Set True to keep the
                          principal point fixed (e.g. for small datasets where
                          cx/cy optimization may not converge).
    """

    def __init__(
        self,
        max_nfev: int = 200,
        ftol: float = 1e-4,
        gtol: float = 1e-4,
        xtol: float = 1e-4,
        loss: str = "huber",
        f_scale: float = 2.0,
        fix_principal_point: bool = False,
    ) -> None:
        self.max_nfev            = max_nfev
        self.ftol                = ftol
        self.gtol                = gtol
        self.xtol                = xtol
        self.loss                = loss
        self.f_scale             = f_scale
        self.fix_principal_point = fix_principal_point

    def adjust(
        self,
        cameras: dict,
        points_3d: np.ndarray,
        observations: list,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        refine_intrinsics: bool = True,
    ) -> Tuple[dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run bundle adjustment.

        Parameters
        ----------
        cameras           : {img_idx: {'R':(3,3), 't':(3,1), 'K':(3,3)}}
        points_3d         : (P, 3) float64
        observations      : list of (img_idx, pt_3d_idx, x_obs, y_obs)
                            x_obs/y_obs are the original (distorted) pixel coords.
        K                 : (3, 3) shared intrinsics
        dist_coeffs       : (4,) or (5,) [k1, k2, p1, p2[, k3]]; None → zeros
        refine_intrinsics : If True, jointly optimise f, k1, k2 with pose/points.

        Returns
        -------
        updated_cameras   : same structure as input
        updated_points_3d : (P, 3) float64
        K_refined         : updated K if refine_intrinsics else None
        dist_refined      : updated (4,) dist if refine_intrinsics else None
        """
        if not cameras or len(points_3d) == 0 or not observations:
            logger.warning("BA: nothing to adjust.")
            return cameras, points_3d, None, None

        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        k1_init = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
        k2_init = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0

        cx_init = float(K[0, 2])
        cy_init = float(K[1, 2])
        f_init  = float(K[0, 0])
        # Convenience alias — used as fixed values when fix_principal_point=True
        cx = cx_init
        cy = cy_init

        # Build consecutive index maps
        cam_list   = sorted(cameras.keys())
        cam_to_idx = {c: i for i, c in enumerate(cam_list)}
        n_cameras  = len(cam_list)
        n_points   = len(points_3d)

        # Filter observations that reference valid cameras / points
        valid_obs = [
            obs for obs in observations
            if obs[0] in cam_to_idx and 0 <= obs[1] < n_points
        ]
        if len(valid_obs) < 8:
            logger.warning(f"BA: only {len(valid_obs)} valid observations — skipped.")
            return cameras, points_3d, None, None

        cam_indices = np.array([cam_to_idx[o[0]] for o in valid_obs], dtype=np.int32)
        pt_indices  = np.array([o[1]             for o in valid_obs], dtype=np.int32)
        pts_2d      = np.array([[o[2], o[3]]     for o in valid_obs], dtype=np.float64)

        # Pack initial parameter vector
        cam_params_init = np.zeros((n_cameras, 6), dtype=np.float64)
        for c_key, c_idx in cam_to_idx.items():
            rvec, _ = cv2.Rodrigues(cameras[c_key]["R"].astype(np.float64))
            cam_params_init[c_idx, :3] = rvec.flatten()
            cam_params_init[c_idx, 3:] = cameras[c_key]["t"].flatten()

        fix_pp = self.fix_principal_point
        if refine_intrinsics:
            if fix_pp:
                x0 = np.concatenate([
                    [f_init, k1_init, k2_init],
                    cam_params_init.ravel(),
                    points_3d.ravel(),
                ])
                lb = np.full_like(x0, -np.inf)
                ub = np.full_like(x0,  np.inf)
                lb[0] = 0.5 * f_init;  ub[0] = 2.0 * f_init
                lb[1] = -2.0;          ub[1] = 2.0
                lb[2] = -2.0;          ub[2] = 2.0
            else:
                # Include cx, cy in optimized params (indices 3 and 4)
                x0 = np.concatenate([
                    [f_init, k1_init, k2_init, cx_init, cy_init],
                    cam_params_init.ravel(),
                    points_3d.ravel(),
                ])
                # Bounds: cx/cy within ±10% of image dimensions from initial
                lb = np.full_like(x0, -np.inf)
                ub = np.full_like(x0,  np.inf)
                lb[0] = 0.5 * f_init;  ub[0] = 2.0 * f_init
                lb[1] = -2.0;          ub[1] = 2.0
                lb[2] = -2.0;          ub[2] = 2.0
                # Infer image dims from initial principal point (cx ≈ W/2, cy ≈ H/2)
                _w = cx_init * 2.0
                _h = cy_init * 2.0
                lb[3] = cx_init - 0.1 * _w;  ub[3] = cx_init + 0.1 * _w
                lb[4] = cy_init - 0.1 * _h;  ub[4] = cy_init + 0.1 * _h
            bounds = (lb, ub)
        else:
            x0 = np.concatenate([cam_params_init.ravel(), points_3d.ravel()])
            bounds = (-np.inf, np.inf)

        def fun(params):
            return _residuals_v2(
                params, n_cameras, n_points, cam_indices, pt_indices, pts_2d,
                cx_init, cy_init, refine_intrinsics, fix_pp, f_init, k1_init, k2_init,
            )

        res_init  = fun(x0)
        rmse_init = float(np.sqrt(np.nanmean(res_init ** 2)))
        logger.info(
            f"  BA  init RMSE: {rmse_init:.3f} px  "
            f"({n_cameras} cams, {n_points} pts, {len(valid_obs)} obs)"
        )

        # Adaptive Huber scale: use 1.4826 × MAD of initial residuals.
        # 1.4826 is the consistency factor for a Gaussian distribution
        # (MAD → σ conversion), so f_scale ≈ 1 σ of the inlier noise.
        # This adapts to the actual reprojection noise rather than using
        # a fixed 2.0 px threshold.  Clamp to [0.5, 10.0] for safety.
        # Reference: Hampel et al. (1986) "Robust Statistics: The Approach
        #   Based on Influence Functions." Wiley.
        abs_res = np.abs(res_init)
        mad     = float(np.median(abs_res))
        f_scale_adaptive = float(np.clip(1.4826 * mad, 0.5, 10.0))
        f_scale_used = f_scale_adaptive if self.f_scale == 2.0 else self.f_scale
        logger.debug(f"  BA  Huber f_scale: {f_scale_used:.3f} px (MAD={mad:.3f})")

        J_sparse = _build_sparsity_v2(
            n_cameras, n_points, cam_indices, pt_indices,
            refine_intrinsics, fix_pp,
        )

        try:
            result = least_squares(
                fun,
                x0,
                bounds=bounds,
                jac_sparsity=J_sparse,
                method="trf",
                loss=self.loss,
                f_scale=f_scale_used,
                max_nfev=self.max_nfev * len(x0),
                ftol=self.ftol,
                gtol=self.gtol,
                xtol=self.xtol,
                verbose=0,
            )
        except Exception as exc:
            logger.error(f"BA optimisation failed: {exc}")
            return cameras, points_3d, None, None

        rmse_final = float(np.sqrt(np.nanmean(result.fun ** 2)))
        logger.info(
            f"  BA final RMSE: {rmse_final:.3f} px  "
            f"(cost={result.cost:.4f}, {result.message})"
        )

        # Guard: reject if BA diverged
        _div_ratio = rmse_final / rmse_init if rmse_init > 1e-10 else 1.0
        if _div_ratio > 1.5:
            logger.warning(
                "[BA] Divergence guard triggered: RMSE went from %.3fpx to %.3fpx "
                "(%.1f×). Rejecting BA result, keeping previous camera/point estimates.",
                rmse_init, rmse_final, _div_ratio,
            )
            return cameras, points_3d, None, None

        # Unpack optimised parameters
        opt = result.x
        if refine_intrinsics:
            f_opt, k1_opt, k2_opt = opt[0], opt[1], opt[2]
            if fix_pp:
                cx_opt, cy_opt = cx_init, cy_init
                off = 3
            else:
                cx_opt, cy_opt = opt[3], opt[4]
                off = 5
        else:
            f_opt, k1_opt, k2_opt = f_init, k1_init, k2_init
            cx_opt, cy_opt = cx_init, cy_init
            off = 0

        opt_cam = opt[off : off + n_cameras * 6].reshape(n_cameras, 6)
        opt_pts = opt[off + n_cameras * 6 :].reshape(n_points, 3)

        updated_cameras = dict(cameras)
        for c_key, c_idx in cam_to_idx.items():
            rvec = opt_cam[c_idx, :3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            # Project R back onto SO(3) to correct floating-point drift.
            # After TRF update steps det(R) can deviate from 1 by ~1e-6.
            # SVD projection is the exact nearest orthogonal matrix.
            # Reference: Grassia (1998) "Practical parameterization of
            #   rotations using the exponential map." J. Graphics Tools 3(3).
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            t    = opt_cam[c_idx, 3:].reshape(3, 1)
            updated_cameras[c_key] = {
                "R": R,
                "t": t,
                "K": cameras[c_key]["K"],
            }

        # Build refined K and dist if intrinsics were optimised
        K_refined   = None
        dist_refined = None
        if refine_intrinsics:
            K_refined = K.copy()
            K_refined[0, 0] = f_opt
            K_refined[1, 1] = f_opt
            K_refined[0, 2] = cx_opt
            K_refined[1, 2] = cy_opt
            dist_out = np.zeros(max(len(dist_coeffs), 4), dtype=np.float64)
            dist_out[0] = k1_opt
            dist_out[1] = k2_opt
            dist_refined = dist_out
            if fix_pp:
                logger.info(
                    f"  BA refined: f={f_opt:.1f} (Δ{f_opt - f_init:+.2f})  "
                    f"k1={k1_opt:.5f}  k2={k2_opt:.5f}  cx/cy fixed"
                )
            else:
                logger.info(
                    f"  BA refined: f={f_opt:.1f} (Δ{f_opt - f_init:+.2f})  "
                    f"k1={k1_opt:.5f}  k2={k2_opt:.5f}  "
                    f"cx={cx_opt:.1f} (Δ{cx_opt - cx_init:+.2f})  "
                    f"cy={cy_opt:.1f} (Δ{cy_opt - cy_init:+.2f})"
                )

        return updated_cameras, opt_pts.astype(np.float64), K_refined, dist_refined
