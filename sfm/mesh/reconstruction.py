"""
sfm/mesh/reconstruction.py — Surface reconstruction methods.

Three methods are supported:
  poisson — Screened Poisson surface reconstruction (default, best for smooth objects)
  bpa     — Ball-Pivoting Algorithm (better for thin/open surfaces)
  alpha   — Alpha shapes (fast, good for convex/simple objects)

All method parameters are derived from the cloud's measured point spacing rather
than from absolute constants or bounding-box fractions, so the same preset
behaves identically at any reconstruction scale.

Phantom-surface trimming
------------------------
Screened Poisson solves for an indicator function over the whole octree domain
and always extracts a closed iso-surface.  Wherever the cloud has no support the
solver still produces geometry — an enclosing "bubble" or webbing between
disconnected parts.  Three complementary criteria remove it:

  1. low sample density  — open3d's per-vertex density output, thresholded at a
     quantile.  Catches thinly-supported regions but is scale-relative, so on a
     cloud with uniformly good coverage it still deletes the configured
     fraction, and on a cloud with a huge phantom bubble it deletes too little.
  2. distance to the input cloud — a vertex further than k·spacing from every
     input point was invented by the solver.  This is the criterion that
     actually removes the bubble, and it is absolute rather than relative.
  3. bounding-box crop — a cheap backstop for anything the first two miss.

The three masks are combined and applied in a single ``remove_vertices_by_mask``
call, because each call rebuilds the vertex index and invalidates any mask
computed against the previous indexing.

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
    spacing: float,
    method: str,
    preset: Dict[str, Any],
    depth_override: Optional[int] = None,
    trim_override: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Reconstruct a triangle mesh from a prepared point cloud.

    Parameters
    ----------
    pcd            : open3d.geometry.PointCloud  (must have normals for Poisson)
    spacing        : Median nearest-neighbour distance of ``pcd``.
    method         : 'poisson' | 'bpa' | 'alpha'
    preset         : Quality preset dict from pipeline.QUALITY_PRESETS.
    depth_override : Poisson octree depth override; None = use preset value.
    trim_override  : Poisson distance-trim factor in multiples of spacing;
                     None = use preset value.

    Returns
    -------
    mesh  : open3d.geometry.TriangleMesh
    stats : Dict with reconstruction statistics.
    """
    logger.info("[MESH RECON] Method: %s  (spacing=%.6g)", method, spacing)

    if method == "bpa":
        mesh, stats = _bpa(pcd, spacing, preset)
    elif method == "alpha":
        mesh, stats = _alpha(pcd, spacing, preset)
    else:
        if method != "poisson":
            logger.warning("[MESH RECON] Unknown method %r — falling back to 'poisson'", method)
        mesh, stats = _poisson(pcd, spacing, preset, depth_override, trim_override)

    if mesh is not None:
        logger.info(
            "[MESH RECON] Done: %d faces, %d vertices",
            len(mesh.triangles), len(mesh.vertices),
        )

    return mesh, stats


# ── Poisson ────────────────────────────────────────────────────────────────────

def _poisson(
    pcd,
    spacing: float,
    preset: Dict[str, Any],
    depth_override: Optional[int],
    trim_override: Optional[float],
) -> Tuple[Any, Dict[str, Any]]:
    """Screened Poisson reconstruction followed by three-criterion trimming."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    if not pcd.has_normals():
        raise ValueError("Poisson reconstruction requires oriented normals on the point cloud")

    cloud_pts_all = np.asarray(pcd.points)
    bbox_diag = float(
        np.linalg.norm(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent()))
    )

    # ── Octree depth ──────────────────────────────────────────────────────
    #
    # Poisson's resolution is bbox-relative: the finest octree cell is
    # bbox_diagonal / 2^depth.  A fixed depth therefore means completely
    # different detail on different clouds — and on a cloud whose bounding box
    # is inflated by far-flung outliers it means no detail at all.  Measured on
    # a diverged reconstruction spanning 281 000 x its own point spacing, depth 9
    # gave a finest cell of 550 x spacing and Poisson returned 7 599 vertices
    # from 193 310 points.
    #
    # The preset depth is treated as a floor and raised, up to a cap, until the
    # finest cell is about `poisson_cell_factor` x the point spacing.  An
    # explicit --mesh-depth disables this entirely.
    linear_fit = bool(preset.get("poisson_linear_fit", True))
    cell_factor = float(preset.get("poisson_cell_factor", 1.5))
    max_auto_depth = int(preset.get("poisson_max_auto_depth", 12))

    if depth_override is not None:
        depth = int(depth_override)
        logger.info("[MESH RECON] Poisson: depth=%d (explicit)  linear_fit=%s",
                    depth, linear_fit)
    else:
        base_depth = int(preset["poisson_depth"])
        target_cell = max(cell_factor * spacing, 1e-12)
        needed = int(np.ceil(np.log2(max(bbox_diag / target_cell, 1.0))))
        depth = int(min(max(base_depth, needed), max_auto_depth))
        if depth != base_depth:
            logger.info(
                "[MESH RECON] Poisson depth auto-raised %d -> %d to reach a finest "
                "octree cell of ~%.1f x point spacing",
                base_depth, depth, cell_factor,
            )
        logger.info("[MESH RECON] Poisson: depth=%d  scale=1.1  linear_fit=%s",
                    depth, linear_fit)

    cell_size = bbox_diag / (2 ** depth) if depth > 0 else bbox_diag
    cell_over_spacing = cell_size / spacing if spacing > 0 else float("inf")
    logger.info(
        "[MESH RECON] Resolution: bbox diagonal %.6g = %.0f x spacing; "
        "finest cell %.6g = %.1f x spacing",
        bbox_diag, bbox_diag / max(spacing, 1e-12), cell_size, cell_over_spacing,
    )
    if cell_over_spacing > 4.0:
        logger.warning(
            "[MESH RECON] Poisson is resolution-starved: the finest octree cell is "
            "%.0f x the point spacing, so detail below that scale cannot be "
            "reconstructed. The cloud's bounding box (%.0f x spacing) is very large "
            "relative to its sampling — it almost certainly contains far-off outlier "
            "points, or the reconstruction diverged. Inspect the cloud, or raise "
            "--mesh-depth beyond the automatic cap of %d.",
            cell_over_spacing, bbox_diag / max(spacing, 1e-12), max_auto_depth,
        )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=linear_fit,
        n_threads=-1,
    )
    densities = np.asarray(densities)
    n_verts_raw = len(mesh.vertices)
    n_faces_raw = len(mesh.triangles)

    if n_verts_raw == 0:
        logger.error("[MESH RECON] Poisson produced no geometry")
        return mesh, {"depth": depth, "verts_raw": 0, "faces_raw": 0}

    vertices = np.asarray(mesh.vertices)
    cloud_pts = np.asarray(pcd.points)

    # ── Criterion 1: low sample density ───────────────────────────────────
    density_threshold = float(preset["density_threshold"])
    if density_threshold > 0 and len(densities) == n_verts_raw:
        quantile_val = float(np.quantile(densities, density_threshold))
        mask_density = densities < quantile_val
    else:
        quantile_val = float("nan")
        mask_density = np.zeros(n_verts_raw, dtype=bool)

    # ── Criterion 2: distance from the supporting cloud ───────────────────
    #
    # The threshold is *locally adaptive*.  A single global `trim_factor *
    # spacing` uses the median nearest-neighbour distance of the whole cloud,
    # which is right only where the cloud is uniformly dense.  MVS clouds are
    # not: coverage varies with baseline, texture and visibility, and in sparse
    # regions the true local spacing is several times the median.  A global
    # threshold there deletes genuine surface — measured on a real dense
    # reconstruction it removed 30 % of the Poisson vertices and shattered the
    # mesh into 581 components, dropping completeness to 73 %.
    #
    # Each vertex is therefore compared against the local spacing of the cloud
    # point nearest to it, floored at the global spacing so the test is never
    # *tighter* than the global rule — only more permissive where the data is
    # genuinely sparser.
    trim_factor = float(
        trim_override if trim_override is not None else preset["trim_distance_factor"]
    )
    if trim_factor > 0:
        tree = cKDTree(cloud_pts)
        dist_to_cloud, nearest_idx = tree.query(vertices, k=1, workers=-1)

        k_local = int(min(6, max(1, len(cloud_pts) - 1)))
        local_d, _ = tree.query(cloud_pts, k=k_local + 1, workers=-1)
        local_spacing = np.asarray(local_d)[:, -1] / k_local ** 0.5

        tau_per_vertex = trim_factor * np.maximum(local_spacing[nearest_idx], spacing)
        tau = trim_factor * spacing          # reported reference value
        mask_distance = dist_to_cloud > tau_per_vertex
    else:
        tau = float("inf")
        dist_to_cloud = np.zeros(n_verts_raw)
        mask_distance = np.zeros(n_verts_raw, dtype=bool)

    # ── Criterion 3: bounding-box crop ────────────────────────────────────
    margin = float(preset.get("bbox_margin_factor", 3.0)) * spacing
    bb_min = cloud_pts.min(axis=0) - margin
    bb_max = cloud_pts.max(axis=0) + margin
    mask_bbox = np.any((vertices < bb_min) | (vertices > bb_max), axis=1)

    remove = mask_density | mask_distance | mask_bbox
    n_remove = int(remove.sum())

    # A trim that deletes essentially everything means the criteria are
    # mis-scaled for this cloud; keeping a bad mesh beats returning none, so
    # fall back to density-only and say why.
    if n_remove >= n_verts_raw * 0.95:
        logger.warning(
            "[MESH RECON] Trim would remove %d/%d vertices (>=95%%) — the trim distance "
            "(%.6g = %.1f x spacing) is probably too tight for this cloud. "
            "Falling back to density-only trimming.",
            n_remove, n_verts_raw, tau, trim_factor,
        )
        remove = mask_density
        n_remove = int(remove.sum())

    mesh.remove_vertices_by_mask(remove)

    logger.info(
        "[MESH RECON] Trimmed %d/%d vertices (%.1f%%): "
        "density<%.4g -> %d | dist>%.1f x local spacing (>=%.6g) -> %d | "
        "outside bbox -> %d",
        n_remove, n_verts_raw, 100.0 * n_remove / max(n_verts_raw, 1),
        quantile_val, int(mask_density.sum()),
        trim_factor, tau, int(mask_distance.sum()),
        int(mask_bbox.sum()),
    )

    stats = {
        "depth":              depth,
        "linear_fit":         linear_fit,
        "bbox_diagonal":      bbox_diag,
        "cell_size":          float(cell_size),
        "cell_over_spacing":  float(cell_over_spacing),
        "verts_raw":          n_verts_raw,
        "faces_raw":          n_faces_raw,
        "density_min":        float(densities.min()),
        "density_max":        float(densities.max()),
        "density_mean":       float(densities.mean()),
        "density_quantile":   quantile_val,
        "trim_distance":      float(tau),
        "trim_factor":        trim_factor,
        "removed_density":    int(mask_density.sum()),
        "removed_distance":   int(mask_distance.sum()),
        "removed_bbox":       int(mask_bbox.sum()),
        "phantom_removed":    n_remove,
        "max_vertex_dist":    float(dist_to_cloud.max()) if trim_factor > 0 else None,
    }
    return mesh, stats


# ── BPA ────────────────────────────────────────────────────────────────────────

def _bpa(pcd, spacing: float, preset: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Ball-Pivoting Algorithm with radii derived from the measured point spacing.

    BPA interpolates the input points instead of approximating them, so it keeps
    the cloud's own accuracy and leaves genuine holes as holes — the right choice
    for thin or open surfaces where Poisson would invent a closed shell.  It
    needs reasonably uniform sampling: coverage below 50 % of input points means
    the radii never bridged the gaps.
    """
    import open3d as o3d

    factors = list(preset.get("bpa_radius_factors", (1.0, 2.0, 4.0, 8.0)))
    radii = [spacing * f for f in factors]
    logger.info(
        "[MESH RECON] BPA radii (x spacing %s): [%s]",
        ", ".join(f"{f:g}" for f in factors),
        ", ".join(f"{r:.6g}" for r in radii),
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    coverage = len(mesh.vertices) / max(len(pcd.points), 1)
    if coverage < 0.5:
        logger.warning(
            "[MESH RECON] BPA covered only %.1f%% of the input points — the cloud is "
            "too sparse or too non-uniform for ball pivoting. Use --mesh-method poisson.",
            coverage * 100,
        )

    return mesh, {"coverage": float(coverage), "radii": radii}


# ── Alpha shapes ───────────────────────────────────────────────────────────────

def _alpha(pcd, spacing: float, preset: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Alpha-shape reconstruction.

    Alpha is a *sampling-density* parameter, so it is set from the point spacing.
    Deriving it from the bounding-box diagonal (the earlier behaviour) couples it
    to the object's size instead of its sampling: the same alpha then behaves
    completely differently on a densely-sampled small object and a sparsely
    sampled large one.  Larger alpha = coarser hull; smaller alpha = more detail
    and more holes.
    """
    import open3d as o3d

    alpha_factor = float(preset.get("alpha_factor", 5.0))
    alpha = alpha_factor * spacing

    bbox_diag = float(np.linalg.norm(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())))
    alpha = min(alpha, bbox_diag * 0.5)

    logger.info(
        "[MESH RECON] Alpha=%.6g (%.1f x spacing, bbox diagonal=%.6g)",
        alpha, alpha / spacing, bbox_diag,
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh, {"alpha": float(alpha), "alpha_factor": alpha_factor, "diagonal": bbox_diag}
