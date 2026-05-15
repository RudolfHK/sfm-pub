"""
sfm/mesh/pointcloud_prep.py — Point cloud preparation for surface reconstruction.

Pipeline (in order):
  3a. Load and validate
  3b. Statistical outlier removal (SOR)
  3c. Radius outlier removal (ROR) with auto-estimated radius
  3d. Normal estimation + consistent orientation (when normals absent)
  3e. Voxel downsampling (only if point count exceeds quality-preset target)
  3f. Log preparation summary

All parameters come from the quality preset dict defined in pipeline.QUALITY_PRESETS.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def prepare_point_cloud(
    ply_path: "str | Path",
    preset: Dict[str, Any],
    quality: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a PLY point cloud and run the full preparation pipeline.

    Parameters
    ----------
    ply_path : Path to the input PLY file (XYZ + optional RGB).
    preset   : Quality preset dict from pipeline.QUALITY_PRESETS.
    quality  : Preset name string (used only for logging).

    Returns
    -------
    pcd   : Prepared open3d.geometry.PointCloud with normals.
    stats : Dict with point counts at each stage for reporting.
    """
    import open3d as o3d

    ply_path = Path(ply_path)

    # ── 3a: Load and validate ──────────────────────────────────────────────
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_input = len(pcd.points)
    has_normals = pcd.has_normals()
    has_colors = pcd.has_colors()

    logger.info(
        "[MESH PREP] Loaded point cloud: %d points, has_normals=%s, has_colors=%s",
        n_input,
        has_normals,
        has_colors,
    )

    if n_input < 1000:
        logger.warning(
            "[MESH PREP] Only %d points — mesh quality will be poor (recommend ≥ 1000)",
            n_input,
        )
    if n_input == 0:
        return None, {}

    stats: Dict[str, Any] = {
        "input": n_input,
        "has_normals": has_normals,
        "has_colors": has_colors,
    }

    # ── 3b: Statistical outlier removal (SOR) ─────────────────────────────
    nb_neighbors = preset["sor_neighbors"]
    std_ratio = preset["sor_std_ratio"]
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    n_after_sor = len(pcd.points)
    removed_sor = n_input - n_after_sor
    logger.info(
        "[MESH PREP] SOR: removed %d outlier points (%.1f%% of cloud)",
        removed_sor,
        100.0 * removed_sor / max(n_input, 1),
    )
    stats["after_sor"] = n_after_sor

    # ── 3c: Radius outlier removal (ROR) ──────────────────────────────────
    if n_after_sor >= 2:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        median_nn = float(np.median(dists)) if len(dists) > 0 else 1.0
        radius = median_nn * preset["ror_multiplier"]
        pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=max(radius, 1e-9))

    n_after_ror = len(pcd.points)
    removed_ror = n_after_sor - n_after_ror
    logger.info("[MESH PREP] ROR: removed %d sparse-region points", removed_ror)
    stats["after_ror"] = n_after_ror

    if n_after_ror == 0:
        logger.warning("[MESH PREP] All points removed by outlier filters — aborting prep")
        return None, stats

    # ── 3d: Normal estimation ──────────────────────────────────────────────
    if not pcd.has_normals():
        dists2 = np.asarray(pcd.compute_nearest_neighbor_distance())
        normal_radius = float(np.median(dists2)) * 2.0 if len(dists2) > 0 else 0.01
        max_nn = preset["normal_max_nn"]
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(normal_radius, 1e-9),
                max_nn=max_nn,
            )
        )
        # Orient normals consistently using the tangent plane approach;
        # this is the best available heuristic when camera positions are unknown.
        pcd.orient_normals_consistent_tangent_plane(k=15)
        logger.info("[MESH PREP] Estimated normals for %d points", n_after_ror)

        # Warn on poor normal consistency (low adjacent-normal dot-product mean).
        normals = np.asarray(pcd.normals)
        if len(normals) > 10:
            normal_tree = o3d.geometry.KDTreeFlann(pcd)
            sample_size = min(500, len(normals))
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(normals), sample_size, replace=False)
            dot_products = []
            for si in sample_idx:
                [_, knn_idx, _] = normal_tree.search_knn_vector_3d(pcd.points[si], 2)
                if len(knn_idx) > 1:
                    dot_products.append(
                        abs(float(np.dot(normals[si], normals[knn_idx[1]])))
                    )
            if dot_products and float(np.mean(dot_products)) < 0.7:
                logger.warning(
                    "[MESH PREP] Normal consistency is low (mean adjacent-dot=%.2f). "
                    "Mesh quality may be degraded — consider a denser point cloud.",
                    float(np.mean(dot_products)),
                )
    else:
        logger.info("[MESH PREP] Using existing normals from point cloud")

    # ── 3e: Voxel downsampling ─────────────────────────────────────────────
    target_points = preset["target_points"]
    n_before_ds = len(pcd.points)

    if target_points is not None and n_before_ds > target_points:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        volume = float(np.prod(np.where(extent > 0, extent, 1e-9)))
        voxel_size = (volume / target_points) ** (1.0 / 3.0)
        pcd = pcd.voxel_down_sample(voxel_size=max(voxel_size, 1e-9))
        n_after_ds = len(pcd.points)
        logger.info(
            "[MESH PREP] Voxel downsampled: %d → %d points (voxel_size=%.5f)",
            n_before_ds,
            n_after_ds,
            voxel_size,
        )
        stats["after_downsample"] = n_after_ds

        # Re-estimate normals after downsampling (voxel_down_sample drops them).
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2,
                    max_nn=preset["normal_max_nn"],
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        stats["after_downsample"] = n_before_ds
        if target_points is not None:
            logger.info(
                "[MESH PREP] No downsampling needed (%d ≤ target %d for '%s' preset)",
                n_before_ds,
                target_points,
                quality,
            )
        else:
            logger.info("[MESH PREP] No downsampling (quality='ultra')")

    # ── 3f: Summary ────────────────────────────────────────────────────────
    bbox = pcd.get_axis_aligned_bounding_box()
    bb_min = np.asarray(bbox.min_bound)
    bb_max = np.asarray(bbox.max_bound)

    after_ds = stats["after_downsample"]
    logger.info(
        "[MESH PREP] Preparation complete:\n"
        "  Input:          %d points\n"
        "  After SOR:      %d points (-%d, -%.1f%%)\n"
        "  After ROR:      %d points (-%d, -%.1f%%)\n"
        "  After downsamp: %d points\n"
        "  Has normals:    %s\n"
        "  Has colors:     %s\n"
        "  Bounding box:   [(%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)]\n"
        "  Ready for surface reconstruction.",
        n_input,
        stats["after_sor"],
        n_input - stats["after_sor"],
        100.0 * (n_input - stats["after_sor"]) / max(n_input, 1),
        stats["after_ror"],
        stats["after_sor"] - stats["after_ror"],
        100.0 * (stats["after_sor"] - stats["after_ror"]) / max(stats["after_sor"], 1),
        after_ds,
        "YES" if pcd.has_normals() else "NO",
        "YES" if pcd.has_colors() else "NO",
        bb_min[0], bb_min[1], bb_min[2],
        bb_max[0], bb_max[1], bb_max[2],
    )

    return pcd, stats
