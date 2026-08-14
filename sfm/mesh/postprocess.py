"""
sfm/mesh/postprocess.py — Optional mesh post-processing.

Steps (all optional, applied in this order):
  6a. Taubin smoothing — volume-preserving noise reduction
  6b. Quadric decimation — polygon count reduction
  6c. Vertex colour transfer — inverse-distance-weighted kNN from the point cloud

The order is not arbitrary: open3d's ``simplify_quadric_decimation`` discards
vertex colours, and both smoothing and decimation move or delete vertices, so
colour transfer has to be the last step or it would be sampling colours for
vertices that no longer exist where they were sampled.

Reference (Taubin smoothing):
  Taubin, G. (1995). A signal processing approach to fair surface design.
  SIGGRAPH 1995.  λ/μ smoothing alternates a shrinking and an expanding pass so
  the mesh keeps its volume, unlike plain Laplacian smoothing which collapses it.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def postprocess_mesh(
    mesh,
    pcd,
    smooth: bool = False,
    smooth_iterations: int = 5,
    decimate: bool = False,
    decimate_target: int = 100_000,
    texture: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply optional post-processing to a cleaned mesh.

    Parameters
    ----------
    mesh              : open3d.geometry.TriangleMesh
    pcd               : open3d.geometry.PointCloud  source for RGB colours
    smooth            : Apply Taubin smoothing (lossy — reduces fine detail).
    smooth_iterations : Number of smoothing iterations.
    decimate          : Reduce polygon count via quadric decimation.
    decimate_target   : Target face count after decimation.
    texture           : Transfer vertex colours from ``pcd`` to the mesh.

    Returns
    -------
    mesh  : Post-processed open3d.geometry.TriangleMesh.
    stats : Dict describing what was applied.
    """
    import open3d as o3d

    stats: Dict[str, Any] = {
        "smoothed": False,
        "decimated": False,
        "colored": False,
    }

    # ── 6a: Taubin smoothing ───────────────────────────────────────────────
    if smooth and len(mesh.triangles) > 0:
        logger.warning(
            "[MESH POST] Taubin smoothing: %d iterations. Smoothing is lossy — fine "
            "surface detail will be reduced. Use --mesh-smooth-iterations 3 for a "
            "subtler effect.",
            smooth_iterations,
        )
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=int(smooth_iterations),
            lambda_filter=0.5,
            mu=-0.53,
        )
        mesh.compute_vertex_normals()
        stats["smoothed"] = True
        stats["smooth_iterations"] = int(smooth_iterations)

    # ── 6b: Quadric decimation ─────────────────────────────────────────────
    if decimate and len(mesh.triangles) > 0:
        n_faces = len(mesh.triangles)
        if n_faces > decimate_target:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(decimate_target)
            )
            mesh.compute_vertex_normals()
            logger.info(
                "[MESH POST] Decimated: %d -> %d faces (%.1f%% reduction)",
                n_faces, len(mesh.triangles),
                100.0 * (n_faces - len(mesh.triangles)) / max(n_faces, 1),
            )
            stats["decimated"] = True
            stats["faces_before_decimation"] = n_faces
        else:
            logger.info(
                "[MESH POST] Decimation skipped: mesh already at or below target (%d <= %d)",
                n_faces, decimate_target,
            )

    # ── 6c: Vertex colour transfer ─────────────────────────────────────────
    if texture and pcd is not None and pcd.has_colors() and len(pcd.points) > 0:
        try:
            from scipy.spatial import cKDTree

            vertices = np.asarray(mesh.vertices)
            pcd_pts = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)

            if len(vertices) == 0:
                raise ValueError("mesh has no vertices to colour")

            k_nn = int(min(5, len(pcd_pts)))
            dists, indices = cKDTree(pcd_pts).query(vertices, k=k_nn, workers=-1)

            if k_nn == 1:
                vertex_colors = pcd_colors[np.asarray(indices).reshape(-1)]
                far = np.asarray(dists).reshape(-1)
            else:
                # Inverse-distance weighting; the clamp keeps an exact hit
                # (distance 0) from producing an infinite weight.
                weights = 1.0 / np.maximum(dists, 1e-10)
                weights /= weights.sum(axis=1, keepdims=True)
                vertex_colors = np.einsum("vk,vkc->vc", weights, pcd_colors[indices])
                far = dists[:, 0]

            vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            stats["colored"] = True
            stats["color_source_points"] = int(len(pcd_pts))
            stats["color_max_source_dist"] = float(np.max(far))
            logger.info(
                "[MESH POST] Vertex colours transferred from the cloud "
                "(%d vertices from %d points, k=%d, max source distance %.4g)",
                len(vertices), len(pcd_pts), k_nn, float(np.max(far)),
            )
        except Exception as exc:
            logger.warning("[MESH POST] Colour transfer failed: %s", exc)
    elif texture:
        logger.info("[MESH POST] Texture skipped: the point cloud has no colours")

    return mesh, stats
