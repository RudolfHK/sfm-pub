"""
sfm/mesh/postprocess.py — Optional mesh post-processing.

Steps (all optional):
  6a. Taubin smoothing — volume-preserving noise reduction
  6b. Quadric decimation — polygon count reduction
  6c. Vertex color transfer — nearest-neighbour RGB from point cloud

Reference (Taubin smoothing):
  Taubin, G. (1995). A signal processing approach to fair surface design.
  SIGGRAPH 1995.
"""

import logging

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
):
    """
    Apply optional post-processing to a cleaned mesh.

    Parameters
    ----------
    mesh              : open3d.geometry.TriangleMesh
    pcd               : open3d.geometry.PointCloud  source for RGB colors
    smooth            : Apply Taubin smoothing (lossy — reduces fine detail).
    smooth_iterations : Number of smoothing iterations (default 5).
    decimate          : Reduce polygon count via quadric decimation.
    decimate_target   : Target face count after decimation (default 100 000).
    texture           : Transfer vertex colors from pcd to mesh.

    Returns
    -------
    mesh : Post-processed open3d.geometry.TriangleMesh.
    """
    import open3d as o3d

    # ── 6a: Taubin smoothing (Taubin 1995, SIGGRAPH) ──────────────────────
    if smooth:
        logger.warning(
            "[MESH POST] Taubin smoothing: %d iterations. "
            "Smoothing is lossy — fine surface detail will be reduced. "
            "Use --mesh-smooth-iterations 3 for subtle smoothing.",
            smooth_iterations,
        )
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.5,
            mu=-0.53,
        )
        mesh.compute_vertex_normals()
        logger.info("[MESH POST] Taubin smoothing: %d iterations applied", smooth_iterations)

    # ── 6b: Quadric decimation ─────────────────────────────────────────────
    if decimate:
        n_faces = len(mesh.triangles)
        if n_faces > decimate_target:
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=decimate_target
            )
            mesh.compute_vertex_normals()
            logger.info(
                "[MESH POST] Decimated: %d → %d faces",
                n_faces,
                len(mesh.triangles),
            )
        else:
            logger.info(
                "[MESH POST] Decimation skipped: mesh already at or below target "
                "(%d ≤ %d)",
                n_faces,
                decimate_target,
            )

    # ── 6c: Vertex color transfer from point cloud ─────────────────────────
    if texture and pcd is not None and pcd.has_colors() and len(pcd.points) > 0:
        try:
            from scipy.spatial import cKDTree

            vertices   = np.asarray(mesh.vertices)   # (V, 3)
            pcd_pts    = np.asarray(pcd.points)       # (P, 3)
            pcd_colors = np.asarray(pcd.colors)       # (P, 3) float64 [0, 1]

            tree = cKDTree(pcd_pts)
            k_nn = min(5, len(pcd_pts))
            dists, indices = tree.query(vertices, k=k_nn, workers=-1)
            if k_nn == 1:
                vertex_colors = pcd_colors[indices]
            else:
                # inverse-distance weighting; guard against zero-distance hits
                weights = 1.0 / np.maximum(dists, 1e-10)   # (V, k)
                weights /= weights.sum(axis=1, keepdims=True)
                vertex_colors = np.einsum(
                    "vk,vkc->vc", weights, pcd_colors[indices]
                )                                           # (V, 3)

            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            logger.info(
                "[MESH POST] Vertex colors transferred from point cloud "
                "(%d vertices, %d cloud points)", len(vertices), len(pcd_pts)
            )
        except Exception as exc:
            logger.warning("[MESH POST] Color transfer failed: %s", exc)
    elif texture and (pcd is None or not pcd.has_colors()):
        logger.info("[MESH POST] Texture skipped: point cloud has no colors")

    return mesh
