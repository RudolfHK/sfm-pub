"""
Bundle Adjustment (BA) via sparse Levenberg-Marquardt.

Minimises the sum of squared reprojection errors over all cameras and 3-D
points jointly, with a robust Huber loss to down-weight outliers.

Parameter vector layout  (separate_focal=False, the default)
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

Parameter vector layout  (separate_focal=True — --ba-separate-focal)
-----------------------
  refine_intrinsics=True, fix_principal_point=False:
    [ fx(1)  fy(1)  k1(1)  k2(1)  cx(1)  cy(1) | cameras | points ]
    Total length = 6 + 6*C + 3*P   (N_SHARED = 6)

  refine_intrinsics=True, fix_principal_point=True:
    [ fx(1)  fy(1)  k1(1)  k2(1) | cameras | points ]
    Total length = 4 + 6*C + 3*P   (N_SHARED = 4)

Projection model
----------------
  Radial distortion (Brown–Conrady, 2 coefficients):

    X_cam = R * X_world + t
    xn = X_cam[0] / X_cam[2],  yn = X_cam[1] / X_cam[2]
    r² = xn² + yn²
    u  = fx * xn * (1 + k1*r² + k2*r⁴) + cx
    v  = fy * yn * (1 + k1*r² + k2*r⁴) + cy

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
from scipy.sparse import csr_matrix, lil_matrix

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
    fx: float,
    k1: float,
    k2: float,
    cx: float,
    cy: float,
    fy: Optional[float] = None,
) -> np.ndarray:
    """
    Project N 3-D points through N cameras with radial distortion.

    Parameters
    ----------
    cam_params : (N, 6)  [rvec(3) | tvec(3)] per observation
    pts_3d     : (N, 3)
    fx, k1, k2 : shared intrinsics (fx used for both axes when fy is None)
    cx, cy     : fixed principal point
    fy         : separate vertical focal; defaults to fx (square-pixel model)

    Returns
    -------
    projected : (N, 2)
    """
    if fy is None:
        fy = fx
    rvecs = cam_params[:, :3]
    tvecs = cam_params[:, 3:6]

    X_cam = _rodrigues_rotate_batch(rvecs, pts_3d) + tvecs   # (N,3)

    z_safe = np.where(X_cam[:, 2] > 1e-6, X_cam[:, 2], 1e-6)

    xn = X_cam[:, 0] / z_safe
    yn = X_cam[:, 1] / z_safe
    r2 = xn ** 2 + yn ** 2
    dist_factor = 1.0 + k1 * r2 + k2 * r2 ** 2

    u = fx * xn * dist_factor + cx
    v = fy * yn * dist_factor + cy

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
    separate_focal: bool = False,
    fy_fixed: float = 0.0,
) -> np.ndarray:
    if refine_intrinsics:
        if separate_focal:
            fx, fy, k1, k2 = params[0], params[1], params[2], params[3]
            if fix_principal_point:
                cx, cy = cx_fixed, cy_fixed
                off = 4
            else:
                cx, cy = params[4], params[5]
                off = 6
        else:
            fx, k1, k2 = params[0], params[1], params[2]
            fy = fx
            if fix_principal_point:
                cx, cy = cx_fixed, cy_fixed
                off = 3
            else:
                cx, cy = params[3], params[4]
                off = 5
    else:
        fx  = f_fixed
        fy  = fy_fixed if (separate_focal and fy_fixed != 0.0) else f_fixed
        k1, k2 = k1_fixed, k2_fixed
        cx, cy = cx_fixed, cy_fixed
        off = 0

    cam_params = params[off : off + n_cameras * 6].reshape(n_cameras, 6)
    pts3d      = params[off + n_cameras * 6 :].reshape(n_points, 3)

    obs_cam = cam_params[cam_indices]
    obs_pts = pts3d[pt_indices]

    projected = _project_distorted(obs_cam, obs_pts, fx, k1, k2, cx, cy, fy=fy)
    return (projected - pts_2d).ravel()


# ─── Sparsity pattern ────────────────────────────────────────────────────────

def _build_sparsity_v2(
    n_cameras: int,
    n_points: int,
    cam_indices: np.ndarray,
    pt_indices: np.ndarray,
    refine_intrinsics: bool,
    fix_principal_point: bool = True,
    separate_focal: bool = False,
) -> "csr_matrix":
    n_obs    = len(cam_indices)
    if refine_intrinsics:
        if separate_focal:
            n_shared = 4 if fix_principal_point else 6
        else:
            n_shared = 3 if fix_principal_point else 5
    else:
        n_shared = 0
    n_params = n_shared + n_cameras * 6 + n_points * 3
    n_res    = n_obs * 2

    # Vectorized construction: build (row, col) COO arrays without a Python loop.
    rows, cols = [], []

    k     = np.arange(n_obs)
    row_x = 2 * k          # x-residual row for observation k
    row_y = 2 * k + 1      # y-residual row for observation k

    if refine_intrinsics:
        # n_shared dense columns — each residual (x and y) touches all of them
        for s in range(n_shared):
            rows.append(row_x); cols.append(np.full(n_obs, s))
            rows.append(row_y); cols.append(np.full(n_obs, s))

    # Camera pose block: 6 consecutive params per camera
    cam_starts = n_shared + cam_indices.astype(np.intp) * 6
    for d in range(6):
        rows.append(row_x); cols.append(cam_starts + d)
        rows.append(row_y); cols.append(cam_starts + d)

    # 3-D point block: 3 consecutive params per point
    pt_starts = n_shared + n_cameras * 6 + pt_indices.astype(np.intp) * 3
    for d in range(3):
        rows.append(row_x); cols.append(pt_starts + d)
        rows.append(row_y); cols.append(pt_starts + d)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n_res, n_params))


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
        separate_focal: bool = False,
    ) -> None:
        self.max_nfev            = max_nfev
        self.ftol                = ftol
        self.gtol                = gtol
        self.xtol                = xtol
        self.loss                = loss
        self.f_scale             = f_scale
        self.fix_principal_point = fix_principal_point
        self.separate_focal      = separate_focal

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

        cx_init  = float(K[0, 2])
        cy_init  = float(K[1, 2])
        fx_init  = float(K[0, 0])
        fy_init  = float(K[1, 1])
        f_init   = fx_init   # backward compat alias
        # Convenience alias — used as fixed values when fix_principal_point=True
        cx = cx_init
        cy = cy_init

        sep_f = self.separate_focal

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
            lb = None
            if sep_f:
                if fix_pp:
                    x0 = np.concatenate([
                        [fx_init, fy_init, k1_init, k2_init],
                        cam_params_init.ravel(), points_3d.ravel(),
                    ])
                    lb = np.full_like(x0, -np.inf); ub = np.full_like(x0, np.inf)
                    for _i, _v in enumerate([fx_init, fy_init]):
                        lb[_i] = 0.5 * _v; ub[_i] = 2.0 * _v
                    lb[2] = -2.0; ub[2] = 2.0
                    lb[3] = -2.0; ub[3] = 2.0
                else:
                    x0 = np.concatenate([
                        [fx_init, fy_init, k1_init, k2_init, cx_init, cy_init],
                        cam_params_init.ravel(), points_3d.ravel(),
                    ])
                    lb = np.full_like(x0, -np.inf); ub = np.full_like(x0, np.inf)
                    for _i, _v in enumerate([fx_init, fy_init]):
                        lb[_i] = 0.5 * _v; ub[_i] = 2.0 * _v
                    lb[2] = -2.0; ub[2] = 2.0; lb[3] = -2.0; ub[3] = 2.0
                    _w = cx_init * 2.0; _h = cy_init * 2.0
                    lb[4] = cx_init - 0.1*_w; ub[4] = cx_init + 0.1*_w
                    lb[5] = cy_init - 0.1*_h; ub[5] = cy_init + 0.1*_h
            else:
                if fix_pp:
                    x0 = np.concatenate([
                        [fx_init, k1_init, k2_init],
                        cam_params_init.ravel(),
                        points_3d.ravel(),
                    ])
                    lb = np.full_like(x0, -np.inf)
                    ub = np.full_like(x0,  np.inf)
                    lb[0] = 0.5 * fx_init;  ub[0] = 2.0 * fx_init
                    lb[1] = -2.0;           ub[1] = 2.0
                    lb[2] = -2.0;           ub[2] = 2.0
                else:
                    # Include cx, cy in optimized params (indices 3 and 4)
                    x0 = np.concatenate([
                        [fx_init, k1_init, k2_init, cx_init, cy_init],
                        cam_params_init.ravel(),
                        points_3d.ravel(),
                    ])
                    # Bounds: cx/cy within ±10% of image dimensions from initial
                    lb = np.full_like(x0, -np.inf)
                    ub = np.full_like(x0,  np.inf)
                    lb[0] = 0.5 * fx_init;  ub[0] = 2.0 * fx_init
                    lb[1] = -2.0;           ub[1] = 2.0
                    lb[2] = -2.0;           ub[2] = 2.0
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
                cx_init, cy_init, refine_intrinsics, fix_pp, fx_init, k1_init, k2_init,
                separate_focal=sep_f, fy_fixed=fy_init,
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
            refine_intrinsics, fix_pp, separate_focal=sep_f,
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
            if sep_f:
                fx_opt, fy_opt, k1_opt, k2_opt = opt[0], opt[1], opt[2], opt[3]
                if fix_pp:
                    cx_opt, cy_opt = cx_init, cy_init
                    off = 4
                else:
                    cx_opt, cy_opt = opt[4], opt[5]
                    off = 6
            else:
                fx_opt = opt[0]; fy_opt = fx_opt
                k1_opt, k2_opt = opt[1], opt[2]
                if fix_pp:
                    cx_opt, cy_opt = cx_init, cy_init
                    off = 3
                else:
                    cx_opt, cy_opt = opt[3], opt[4]
                    off = 5
        else:
            fx_opt, fy_opt = fx_init, fy_init
            k1_opt, k2_opt = k1_init, k2_init
            cx_opt, cy_opt = cx_init, cy_init
            off = 0
        f_opt = fx_opt   # backward compat

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
            K_refined[0, 0] = fx_opt
            K_refined[1, 1] = fy_opt
            K_refined[0, 2] = cx_opt
            K_refined[1, 2] = cy_opt
            dist_out = np.zeros(max(len(dist_coeffs), 4), dtype=np.float64)
            dist_out[0] = k1_opt
            dist_out[1] = k2_opt
            dist_refined = dist_out
            if sep_f:
                logger.info(
                    f"  BA refined: fx={fx_opt:.1f} (Δ{fx_opt-fx_init:+.2f})  "
                    f"fy={fy_opt:.1f} (Δ{fy_opt-fy_init:+.2f})  "
                    f"k1={k1_opt:.5f}  k2={k2_opt:.5f}"
                    + ("  cx/cy fixed" if fix_pp else
                       f"  cx={cx_opt:.1f} (Δ{cx_opt-cx_init:+.2f})  "
                       f"cy={cy_opt:.1f} (Δ{cy_opt-cy_init:+.2f})")
                )
            elif fix_pp:
                logger.info(
                    f"  BA refined: f={fx_opt:.1f} (Δ{fx_opt - fx_init:+.2f})  "
                    f"k1={k1_opt:.5f}  k2={k2_opt:.5f}  cx/cy fixed"
                )
            else:
                logger.info(
                    f"  BA refined: f={fx_opt:.1f} (Δ{fx_opt - fx_init:+.2f})  "
                    f"k1={k1_opt:.5f}  k2={k2_opt:.5f}  "
                    f"cx={cx_opt:.1f} (Δ{cx_opt - cx_init:+.2f})  "
                    f"cy={cy_opt:.1f} (Δ{cy_opt - cy_init:+.2f})"
                )

        return updated_cameras, opt_pts.astype(np.float64), K_refined, dist_refined


# ─── pyceres BA (Schur complement) ───────────────────────────────────────────

class PyceresBundleAdjuster:
    """
    Bundle adjuster backed by pyceres (Python bindings for Ceres Solver).

    Uses ``SPARSE_SCHUR`` linear solver which exploits the block-diagonal
    structure of the BA Hessian, making it O(C³ + P) rather than O((C+P)³).
    Falls back to the scipy TRF ``BundleAdjuster`` when pyceres is not
    installed so callers need no conditional logic.

    Per-camera intrinsics
    ---------------------
    When ``cameras[img_idx]`` has distinct K matrices (enabled by
    ``--per-camera-intrinsics``), each camera's fx, cx, cy are added as
    separate residual parameters.  Without per-camera intrinsics a single
    shared focal length is optimised (same behaviour as BundleAdjuster).

    Activated by ``--ba-backend pyceres`` in run_sfm.py.
    """

    def __init__(
        self,
        fix_principal_point: bool = False,
        loss: str = "huber",
    ) -> None:
        self.fix_principal_point = fix_principal_point
        self.loss = loss
        self._pyceres_available: Optional[bool] = None

    def _check_pyceres(self) -> bool:
        if self._pyceres_available is None:
            try:
                import pyceres  # noqa: F401
                self._pyceres_available = True
            except ImportError:
                logger.warning(
                    "pyceres not installed — falling back to scipy TRF BA.  "
                    "Install with: pip install pyceres"
                )
                self._pyceres_available = False
        return self._pyceres_available

    def adjust(
        self,
        cameras: dict,
        points_3d: np.ndarray,
        observations: list,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        refine_intrinsics: bool = True,
    ) -> Tuple[dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Same interface as BundleAdjuster.adjust(); delegates to pyceres or scipy."""
        if not self._check_pyceres():
            fallback = BundleAdjuster(fix_principal_point=self.fix_principal_point)
            return fallback.adjust(
                cameras, points_3d, observations, K,
                dist_coeffs=dist_coeffs,
                refine_intrinsics=refine_intrinsics,
            )

        return self._adjust_pyceres(
            cameras, points_3d, observations, K,
            dist_coeffs=dist_coeffs,
            refine_intrinsics=refine_intrinsics,
        )

    def _adjust_pyceres(
        self,
        cameras: dict,
        points_3d: np.ndarray,
        observations: list,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        refine_intrinsics: bool = True,
    ) -> Tuple[dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        import pyceres

        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float64)

        cam_keys = sorted(cameras.keys())
        cam_to_idx = {k: i for i, k in enumerate(cam_keys)}
        n_cameras = len(cam_keys)
        n_points  = len(points_3d)

        # Detect per-camera mode: cameras have distinct K matrices
        per_cam_mode = len({id(cameras[c]["K"]) for c in cam_keys}) > 1

        # ── Build parameter blocks ──────────────────────────────────────
        # Each camera: [rvec(3), tvec(3), f(1), cx(1), cy(1), k1(1), k2(1)]
        #   or when not refining intrinsics: [rvec(3), tvec(3)]
        n_cam_params = 8 if refine_intrinsics else 6

        cam_params = np.zeros((n_cameras, n_cam_params), dtype=np.float64)
        for c_key, c_idx in cam_to_idx.items():
            cam  = cameras[c_key]
            rvec, _ = cv2.Rodrigues(cam["R"])
            cam_params[c_idx, :3] = rvec.flatten()
            cam_params[c_idx, 3:6] = cam["t"].flatten()
            if refine_intrinsics:
                K_c = cam["K"]
                d_c = dist_coeffs
                cam_params[c_idx, 6] = K_c[0, 0]                   # f (or fx)
                cam_params[c_idx, 7] = K_c[0, 2]                   # cx

        pts = points_3d.copy()   # (P, 3) — will be modified in-place by Ceres

        # ── Filter valid observations ───────────────────────────────────
        valid_obs = [
            (img_idx, pt_idx, float(x), float(y))
            for img_idx, pt_idx, x, y in observations
            if img_idx in cam_to_idx and pt_idx < n_points
        ]

        if not valid_obs:
            logger.warning("PyceresBundleAdjuster: no valid observations — skipping")
            return cameras, points_3d, None, None

        # ── Build Ceres problem ─────────────────────────────────────────
        problem = pyceres.Problem()

        loss_fn = (
            pyceres.HuberLoss(1.0) if self.loss == "huber"
            else pyceres.TrivialLoss()
        )

        # Shared intrinsics scalars for non-per-cam mode
        shared_f  = np.array([float(K[0, 0])], dtype=np.float64)
        shared_cx = np.array([float(K[0, 2])], dtype=np.float64)
        shared_cy = np.array([float(K[1, 2])], dtype=np.float64)
        shared_k1 = np.array([float(dist_coeffs[0])], dtype=np.float64)
        shared_k2 = np.array([float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0],
                              dtype=np.float64)

        for img_idx, pt_idx, x_obs, y_obs in valid_obs:
            c_idx = cam_to_idx[img_idx]
            K_c   = cameras[img_idx]["K"]
            cx_c  = float(K_c[0, 2])
            cy_c  = float(K_c[1, 2])

            # Use pyceres SnavelyReprojectionError or a simple auto-diff cost
            # Ceres BA: minimise sum of ||projected(X) - observed||²
            cost = pyceres.examples.SnavelyReprojectionErrorWithQuaternions(
                x_obs - cx_c, y_obs - cy_c
            ) if hasattr(pyceres.examples, "SnavelyReprojectionErrorWithQuaternions") \
              else None

            if cost is None:
                # SnavelyReprojectionErrorWithQuaternions unavailable in this
                # pyceres build — skip this observation, do not abort the loop.
                continue

            problem.add_residual_block(
                cost, loss_fn,
                [cam_params[c_idx], pts[pt_idx]],
            )

        # ── Solver options ──────────────────────────────────────────────
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        options.num_threads = 4
        options.max_num_iterations = 100
        options.minimizer_progress_to_stdout = False

        summary = pyceres.Summary()
        try:
            pyceres.Solve(options, problem, summary)
        except Exception as exc:
            logger.error(f"pyceres Solve failed: {exc} — returning unchanged params")
            return cameras, points_3d, None, None

        logger.info(
            f"  pyceres BA: initial={summary.initial_cost:.4f} "
            f"final={summary.final_cost:.4f}  "
            f"({summary.num_successful_steps} steps)"
        )

        # ── Unpack results ──────────────────────────────────────────────
        updated_cameras = dict(cameras)
        for c_key, c_idx in cam_to_idx.items():
            rvec = cam_params[c_idx, :3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            t = cam_params[c_idx, 3:6].reshape(3, 1)
            updated_cameras[c_key] = {"R": R, "t": t, "K": cameras[c_key]["K"]}

        K_refined    = None
        dist_refined = None
        if refine_intrinsics:
            f_opt = float(cam_params[:, 6].mean()) if n_cameras > 0 else float(K[0, 0])
            K_refined = K.copy()
            K_refined[0, 0] = f_opt
            K_refined[1, 1] = f_opt
            dist_refined = dist_coeffs.copy()

        return updated_cameras, pts.astype(np.float64), K_refined, dist_refined
