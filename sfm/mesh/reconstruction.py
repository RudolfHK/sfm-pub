"""
sfm/mesh/reconstruction.py — Surface reconstruction methods.

Three methods are supported:
  poisson — Screened Poisson surface reconstruction (default, best for smooth objects)
  bpa     — Ball-Pivoting Algorithm (better for thin/open surfaces)
  alpha   — Alpha shapes (fast, good for convex/simple objects)

References
----------
Kazhdan, M. & Hoppe, H. (2013). Screened Poisson Surface Reconstruction.
  ACM Transactions on Graphics, 32(3), 29.

Bernardini, F. et al. (1999). The Ball-Pivoting Algorithm for Surface
  Reconstruction. IEEE Transactions on Visualization and Computer Graphics, 5(4), 349–359.

Edelsbrunner, H. & Mücke, E.P. (1994). Three-dimensional alpha shapes.
  ACM Transactions on Graphics, 13(1), 43–72.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def reconstruct_surface(
    pcd,
    method: str,
    preset: Dict[str, Any],
    depth_override: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Reconstruct a triangle mesh from a prepared point cloud.

    Parameters
    ----------
    pcd            : open3d.geometry.PointCloud  (must have normals for Poisson)
    method         : 'poisson' | 'bpa' | 'alpha'
    preset         : Quality preset dict from pipeline.QUALITY_PRESETS.
    depth_override : Poisson octree depth override; None = use preset value.

    Returns
    -------
    mesh  : open3d.geometry.TriangleMesh
    stats : Dict with reconstruction statistics.
    """
    logger.info("[MESH RECON] Method: %s", method)

    if method == "bpa":
        mesh, stats = _bpa(pcd, preset)
    elif method == "alpha":
        mesh, stats = _alpha(pcd, preset)
    else:
        if method != "poisson":
            logger.warning("[MESH RECON] Unknown method %r — falling back to 'poisson'", method)
        mesh, stats = _poisson(pcd, preset, depth_override)

    if mesh is not None:
        logger.info(
            "[MESH RECON] Done: %d faces, %d vertices",
            len(mesh.triangles),
            len(mesh.vertices),
        )

    return mesh, stats


# ── Poisson ────────────────────────────────────────────────────────────────────

def _poisson(
    pcd,
    preset: Dict[str, Any],
    depth_override: Optional[int],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Screened Poisson surface reconstruction with density-based phantom trimming.

    Poisson always produces a watertight mesh including extrapolated "phantom"
    surfaces in regions not covered by the point cloud.  The density array
    returned by open3d is used to remove these low-support vertices.

    Reference: Kazhdan & Hoppe (2013), ACM TOG 32(3).
    """
    import open3d as o3d

    depth = depth_override if depth_override is not None else preset["poisson_depth"]
    logger.info("[MESH RECON] Poisson depth=%d", depth)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
        n_threads=-1,
    )
    densities = np.asarray(densities)

    density_threshold = preset["density_threshold"]
    quantile_val = float(np.quantile(densities, density_threshold))
    vertices_to_remove = densities < quantile_val
    n_phantom = int(vertices_to_remove.sum())

    logger.info(
        "[MESH RECON] Density  min=%.4f  max=%.4f  mean=%.4f — "
        "removing %d phantom vertices (bottom %.0f%%)",
        float(densities.min()),
        float(densities.max()),
        float(densities.mean()),
        n_phantom,
        density_threshold * 100,
    )

    mesh.remove_vertices_by_mask(vertices_to_remove)

    stats = {
        "density_min":     float(densities.min()),
        "density_max":     float(densities.max()),
        "density_mean":    float(densities.mean()),
        "phantom_removed": n_phantom,
        "depth":           depth,
    }
    return mesh, stats


# ── BPA ────────────────────────────────────────────────────────────────────────

def _bpa(pcd, preset: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Ball-Pivoting Algorithm.  Auto-estimates ball radii from the average
    nearest-neighbour distance.

    BPA works best on well-sampled, uniformly dense point clouds.  A warning
    is logged if coverage falls below 50 % of input points.

    Reference: Bernardini et al. (1999), IEEE TVCG 5(4).
    """
    import open3d as o3d

    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist = float(np.mean(dists)) if len(dists) > 0 else 0.01
    radii = [avg_dist * m for m in [1.0, 2.0, 4.0, 8.0]]

    if avg_dist < 1e-9:
        logger.warning(
            "[MESH RECON] BPA: average NN distance is near zero — "
            "point cloud may be degenerate."
        )

    logger.info(
        "[MESH RECON] BPA radii: [%s]",
        ", ".join(f"{r:.5f}" for r in radii),
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    n_pts = len(pcd.points)
    coverage = len(mesh.vertices) / max(n_pts, 1)
    if coverage < 0.5:
        logger.warning(
            "[MESH RECON] BPA covered only %.1f%% of points — "
            "cloud may be too sparse or non-uniform. "
            "Consider --mesh-method poisson instead.",
            coverage * 100,
        )

    return mesh, {"coverage": float(coverage)}


# ── Alpha shapes ───────────────────────────────────────────────────────────────

def _alpha(pcd, preset: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Alpha shapes reconstruction.

    Alpha is estimated from the bounding-box diagonal and a quality-dependent
    factor.  Larger alpha = coarser shape; smaller alpha = more detail but
    more holes.

    Reference: Edelsbrunner & Mücke (1994), ACM TOG 13(1).
    """
    import open3d as o3d

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())
    diagonal = float(np.linalg.norm(extent))

    # Map Poisson depth to alpha factor: deeper preset → finer alpha
    depth = preset["poisson_depth"]
    alpha_factors = {8: 0.05, 9: 0.03, 10: 0.02, 11: 0.01}
    alpha_factor = alpha_factors.get(depth, 0.03)
    alpha = diagonal * alpha_factor

    logger.info(
        "[MESH RECON] Alpha=%.5f  (diagonal=%.4f, factor=%.3f)",
        alpha, diagonal, alpha_factor,
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh, {"alpha": float(alpha), "diagonal": float(diagonal)}
