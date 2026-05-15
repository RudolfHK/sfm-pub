"""
sfm/mesh/cleaning.py — Sequential mesh cleaning pipeline.

Steps (in order):
  5a. Remove degenerate / duplicate triangles and vertices
  5b. Remove non-manifold edges
  5c. Remove isolated small components (keep large connected component(s))
  5d. Fill holes (if requested)
  5e. Recompute vertex and triangle normals
  5f. Log cleaning summary
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def clean_mesh(
    mesh,
    preset: Dict[str, Any],
    fill_holes: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run the full sequential mesh cleaning pipeline.

    Parameters
    ----------
    mesh       : open3d.geometry.TriangleMesh  raw reconstruction output.
    preset     : Quality preset dict (controls min_component_ratio).
    fill_holes : Whether to attempt automatic hole filling.

    Returns
    -------
    mesh  : Cleaned open3d.geometry.TriangleMesh.
    stats : Dict with face/vertex counts at each stage.
    """
    n_faces_in = len(mesh.triangles)
    n_verts_in = len(mesh.vertices)
    logger.info("[MESH CLEAN] Starting: %d faces, %d vertices", n_faces_in, n_verts_in)

    stats: Dict[str, Any] = {"faces_in": n_faces_in, "verts_in": n_verts_in}

    # ── 5a: Degenerate / duplicate removal ────────────────────────────────
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    n_after_dedup = len(mesh.triangles)
    removed_dedup = n_faces_in - n_after_dedup
    logger.info(
        "[MESH CLEAN] Removed %d degenerate/duplicate triangles",
        removed_dedup,
    )
    stats["after_dedup"] = n_after_dedup

    # ── 5b: Non-manifold edges ─────────────────────────────────────────────
    n_before_manifold = len(mesh.triangles)
    mesh.remove_non_manifold_edges()
    n_after_manifold = len(mesh.triangles)
    logger.info(
        "[MESH CLEAN] Removed %d triangles (non-manifold edge cleaning)",
        n_before_manifold - n_after_manifold,
    )
    stats["after_manifold"] = n_after_manifold

    # ── 5c: Isolated small component removal ──────────────────────────────
    if len(mesh.triangles) > 0:
        triangle_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)

        total = int(len(mesh.triangles))
        min_ratio = preset["min_component_ratio"]
        min_tris = max(100, int(total * min_ratio))

        small_mask = cluster_n_tris[triangle_clusters] < min_tris
        n_removed_fragments = int(small_mask.sum())
        n_small_components = int((cluster_n_tris < min_tris).sum())

        if n_removed_fragments > 0:
            mesh.remove_triangles_by_mask(small_mask)
            mesh.remove_unreferenced_vertices()

        large_counts = sorted(
            cluster_n_tris[cluster_n_tris >= min_tris].tolist(),
            reverse=True,
        )
        logger.info(
            "[MESH CLEAN] Components: %d large (%s faces) + "
            "%d small fragment(s) removed (%d faces)",
            len(large_counts),
            ", ".join(str(c) for c in large_counts[:3]),
            n_small_components,
            n_removed_fragments,
        )
        stats["after_components"] = len(mesh.triangles)
        stats["fragments_removed"] = n_removed_fragments

    # ── 5d: Hole filling ───────────────────────────────────────────────────
    if fill_holes and len(mesh.triangles) > 0:
        n_before_fill = len(mesh.triangles)
        try:
            bbox = mesh.get_axis_aligned_bounding_box()
            extent = np.asarray(bbox.get_extent())
            hole_size = float(np.max(extent)) * 0.1
            mesh = mesh.fill_holes(hole_size=hole_size)
            n_added = len(mesh.triangles) - n_before_fill
            if n_added > 0:
                logger.info("[MESH CLEAN] Hole filling: added %d triangles", n_added)
            else:
                logger.info("[MESH CLEAN] No fillable holes found (or holes too large for auto-fill)")
        except Exception as exc:
            logger.warning(
                "[MESH CLEAN] Hole filling failed (%s). "
                "For complex holes use MeshLab (Filters → Remeshing → Close Holes) or Blender.",
                exc,
            )
        stats["after_fill"] = len(mesh.triangles)

    # ── 5e: Recompute normals ──────────────────────────────────────────────
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ── 5f: Summary ────────────────────────────────────────────────────────
    n_faces_out = len(mesh.triangles)
    n_verts_out = len(mesh.vertices)
    is_manifold = mesh.is_edge_manifold() and mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()

    logger.info(
        "[MESH CLEAN] Cleaning complete:\n"
        "  Before:    %d faces, %d vertices\n"
        "  After:     %d faces, %d vertices\n"
        "  Removed:   %d faces (%.1f%%)\n"
        "  Manifold:  %s\n"
        "  Watertight: %s",
        n_faces_in, n_verts_in,
        n_faces_out, n_verts_out,
        n_faces_in - n_faces_out,
        100.0 * (n_faces_in - n_faces_out) / max(n_faces_in, 1),
        "YES" if is_manifold else "NO",
        "YES" if is_watertight else "NO",
    )

    stats["faces_out"] = n_faces_out
    stats["verts_out"] = n_verts_out
    stats["is_manifold"] = is_manifold
    stats["is_watertight"] = is_watertight

    return mesh, stats
