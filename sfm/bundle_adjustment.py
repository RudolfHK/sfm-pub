"""
Bundle Adjustment (BA) via sparse Levenberg-Marquardt.

Minimises the sum of squared reprojection errors over all cameras and 3-D
points jointly, with a robust Huber loss to down-weight outliers.

Parameter vector layout
-----------------------
  [  cam_0_rvec(3)  cam_0_tvec(3)  |  ...  |  cam_{C-1}_rvec  cam_{C-1}_tvec  |
     X_0(3)  X_1(3)  ...  X_{P-1}(3)  ]

Total length = 6*C + 3*P

Observation convention
-----------------------
  (camera_idx, point_idx, x_obs, y_obs)   — camera_idx and point_idx are the
  *consecutive* indices used inside BA (not the original image indices).
"""

import logging
from typing import Dict, Tuple

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


# ─── Vectorised projection ───────────────────────────────────────────────────

def _project_batch(
    cam_params: np.ndarray,
    pts_3d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Project N 3-D points through N cameras (one camera per point).

    Parameters
    ----------
    cam_params : (N, 6)  [rvec(3) | tvec(3)] per observation
    pts_3d     : (N, 3)
    K          : (3, 3)  shared intrinsics

    Returns
    -------
    projected : (N, 2)   — uses z_safe to avoid ±inf during optimisation
    """
    rvecs = cam_params[:, :3]
    tvecs = cam_params[:, 3:6]

    X_cam = _rodrigues_rotate_batch(rvecs, pts_3d) + tvecs   # (N,3)

    z_safe = np.where(X_cam[:, 2] > 1e-6, X_cam[:, 2], 1e-6)

    x_proj = K[0, 0] * X_cam[:, 0] / z_safe + K[0, 2]
    y_proj = K[1, 1] * X_cam[:, 1] / z_safe + K[1, 2]

    return np.stack([x_proj, y_proj], axis=1)                # (N,2)


# ─── Residual function ───────────────────────────────────────────────────────

def _residuals(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    cam_indices: np.ndarray,
    pt_indices: np.ndarray,
    pts_2d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    cam_params = params[: n_cameras * 6].reshape(n_cameras, 6)
    pts3d      = params[n_cameras * 6 :].reshape(n_points, 3)

    obs_cam = cam_params[cam_indices]   # (n_obs, 6)
    obs_pts = pts3d[pt_indices]         # (n_obs, 3)

    projected = _project_batch(obs_cam, obs_pts, K)          # (n_obs, 2)
    return (projected - pts_2d).ravel()                      # (2*n_obs,)


# ─── Sparsity pattern ────────────────────────────────────────────────────────

def _build_sparsity(
    n_cameras: int,
    n_points: int,
    cam_indices: np.ndarray,
    pt_indices: np.ndarray,
) -> "scipy.sparse.csr_matrix":
    n_obs    = len(cam_indices)
    n_params = n_cameras * 6 + n_points * 3
    n_res    = n_obs * 2

    J = lil_matrix((n_res, n_params), dtype=np.int8)

    rows = np.repeat(np.arange(n_obs), 2)   # [0,0,1,1,2,2,...] for pairs
    # Actually it's easier with explicit loops for clarity; still O(n_obs)
    for k in range(n_obs):
        c = int(cam_indices[k])
        p = int(pt_indices[k])

        # Camera pose (6 params)
        J[2 * k,     c * 6 : c * 6 + 6] = 1
        J[2 * k + 1, c * 6 : c * 6 + 6] = 1

        # 3-D point (3 params)
        base = n_cameras * 6 + p * 3
        J[2 * k,     base : base + 3] = 1
        J[2 * k + 1, base : base + 3] = 1

    return J.tocsr()


# ─── BundleAdjuster class ────────────────────────────────────────────────────

class BundleAdjuster:
    """
    Sparse bundle adjustment using scipy's TRF least-squares solver.

    Parameters
    ----------
    max_nfev   : Maximum number of function evaluations.
    ftol/gtol/xtol : Convergence tolerances.
    loss       : Robust loss name ('huber', 'cauchy', 'linear', …).
    f_scale    : Scale of the robust loss (pixels).
    """

    def __init__(
        self,
        max_nfev: int = 200,
        ftol: float = 1e-4,
        gtol: float = 1e-4,
        xtol: float = 1e-4,
        loss: str = "huber",
        f_scale: float = 2.0,
    ) -> None:
        self.max_nfev = max_nfev
        self.ftol     = ftol
        self.gtol     = gtol
        self.xtol     = xtol
        self.loss     = loss
        self.f_scale  = f_scale

    def adjust(
        self,
        cameras: dict,
        points_3d: np.ndarray,
        observations: list,
        K: np.ndarray,
    ) -> Tuple[dict, np.ndarray]:
        """
        Run bundle adjustment.

        Parameters
        ----------
        cameras      : {img_idx: {'R':(3,3), 't':(3,1), 'K':(3,3)}}
        points_3d    : (P, 3) float64
        observations : list of (img_idx, pt_3d_idx, x_obs, y_obs)
        K            : (3, 3) shared intrinsics

        Returns
        -------
        updated_cameras, updated_points_3d
        """
        if not cameras or len(points_3d) == 0 or not observations:
            logger.warning("BA: nothing to adjust.")
            return cameras, points_3d

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
            return cameras, points_3d

        cam_indices = np.array([cam_to_idx[o[0]] for o in valid_obs], dtype=np.int32)
        pt_indices  = np.array([o[1]             for o in valid_obs], dtype=np.int32)
        pts_2d      = np.array([[o[2], o[3]]     for o in valid_obs], dtype=np.float64)

        # Pack initial parameter vector
        cam_params_init = np.zeros((n_cameras, 6), dtype=np.float64)
        for c_key, c_idx in cam_to_idx.items():
            rvec, _ = cv2.Rodrigues(cameras[c_key]["R"].astype(np.float64))
            cam_params_init[c_idx, :3] = rvec.flatten()
            cam_params_init[c_idx, 3:] = cameras[c_key]["t"].flatten()

        x0 = np.concatenate([cam_params_init.ravel(), points_3d.ravel()])

        res_init = _residuals(x0, n_cameras, n_points, cam_indices, pt_indices, pts_2d, K)
        rmse_init = float(np.sqrt(np.nanmean(res_init ** 2)))
        logger.info(
            f"  BA  init RMSE: {rmse_init:.3f} px  "
            f"({n_cameras} cams, {n_points} pts, {len(valid_obs)} obs)"
        )

        J_sparse = _build_sparsity(n_cameras, n_points, cam_indices, pt_indices)

        def fun(params):
            return _residuals(
                params, n_cameras, n_points, cam_indices, pt_indices, pts_2d, K
            )

        try:
            result = least_squares(
                fun,
                x0,
                jac_sparsity=J_sparse,
                method="trf",
                loss=self.loss,
                f_scale=self.f_scale,
                max_nfev=self.max_nfev * len(x0),
                ftol=self.ftol,
                gtol=self.gtol,
                xtol=self.xtol,
                verbose=0,
            )
        except Exception as exc:
            logger.error(f"BA optimisation failed: {exc}")
            return cameras, points_3d

        res_final = result.fun
        rmse_final = float(np.sqrt(np.nanmean(res_final ** 2)))
        logger.info(
            f"  BA final RMSE: {rmse_final:.3f} px  "
            f"(cost={result.cost:.4f}, {result.message})"
        )

        # Guard: reject if BA diverged
        if rmse_final > rmse_init * 3.0:
            logger.warning("BA diverged — reverting to initial parameters.")
            return cameras, points_3d

        # Unpack optimised parameters
        opt = result.x
        opt_cam  = opt[: n_cameras * 6].reshape(n_cameras, 6)
        opt_pts  = opt[n_cameras * 6 :].reshape(n_points, 3)

        updated_cameras = dict(cameras)
        for c_key, c_idx in cam_to_idx.items():
            rvec = opt_cam[c_idx, :3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            t    = opt_cam[c_idx, 3:].reshape(3, 1)
            updated_cameras[c_key] = {
                "R": R,
                "t": t,
                "K": cameras[c_key]["K"],
            }

        return updated_cameras, opt_pts.astype(np.float64)
