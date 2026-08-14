"""
sfm/mesh/cleaning.py — Sequential mesh cleaning pipeline.

Steps (in order):
  5a. Remove degenerate / duplicate triangles and vertices
  5b. Remove non-manifold edges
  5c. Remove self-intersecting triangles (optional, size-gated)
  5d. Remove isolated components (the largest is always kept)
  5e. Fill holes up to a spacing-derived size
  5f. Re-run dedup (hole filling and component removal both change topology)
  5g. Orient triangles consistently and recompute normals
  5h. Log a cleaning summary

Ordering rationale
------------------
Self-intersection removal punches new holes, so it runs before hole filling.
Component filtering runs after it, because deleting triangles can detach
fragments that were only connected through the removed ones.  A final dedup pass
follows hole filling because the filler introduces new triangles that may
duplicate existing ones.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def clean_mesh(
    mesh,
    preset: Dict[str, Any],
    spacing: float,
    fill_holes: bool = True,
    hole_size: float = None,
    keep_largest_only: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run the full sequential mesh cleaning pipeline.

    Parameters
    ----------
    mesh              : open3d.geometry.TriangleMesh  raw reconstruction output.
    preset            : Quality preset dict (component ratio, hole factor, gates).
    spacing           : Point spacing of the source cloud — the length unit for
                        hole sizing.
    fill_holes        : Whether to attempt automatic hole filling.
    hole_size         : Absolute maximum hole size to fill (scene units).
                        None = ``preset['hole_fill_factor'] * spacing``.
    keep_largest_only : Discard every connected component but the largest.

    Returns
    -------
    mesh  : Cleaned open3d.geometry.TriangleMesh.
    stats : Dict with face/vertex counts at each stage.
    """
    n_faces_in = len(mesh.triangles)
    n_verts_in = len(mesh.vertices)
    logger.info("[MESH CLEAN] Starting: %d faces, %d vertices", n_faces_in, n_verts_in)

    stats: Dict[str, Any] = {"faces_in": n_faces_in, "verts_in": n_verts_in}
    if n_faces_in == 0:
        logger.warning("[MESH CLEAN] Empty mesh — nothing to clean")
        stats.update(faces_out=0, verts_out=0, is_manifold=False, is_watertight=False)
        return mesh, stats

    # ── 5a: Degenerate / duplicate removal ────────────────────────────────
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    stats["after_dedup"] = len(mesh.triangles)
    logger.info(
        "[MESH CLEAN] Dedup: removed %d degenerate/duplicate triangles",
        n_faces_in - stats["after_dedup"],
    )

    # ── 5b: Non-manifold edges ─────────────────────────────────────────────
    n_before = len(mesh.triangles)
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    stats["after_manifold"] = len(mesh.triangles)
    logger.info(
        "[MESH CLEAN] Non-manifold edges: removed %d triangles",
        n_before - stats["after_manifold"],
    )

    # ── 5c: Self-intersections (optional, size-gated) ─────────────────────
    #
    # Self-intersection detection is a broad-phase/narrow-phase triangle pair
    # test; on meshes of a few hundred thousand faces it dominates the whole
    # mesh stage, so it is gated on face count rather than run unconditionally.
    max_faces = int(preset.get("self_intersection_max_faces", 250_000))
    if preset.get("remove_self_intersections", True) and len(mesh.triangles) > 0:
        if len(mesh.triangles) <= max_faces:
            n_before = len(mesh.triangles)
            try:
                mesh = _remove_self_intersections(mesh)
                removed = n_before - len(mesh.triangles)
                logger.info(
                    "[MESH CLEAN] Self-intersections: removed %d triangles", removed
                )
                stats["self_intersections_removed"] = removed
            except Exception as exc:
                logger.warning("[MESH CLEAN] Self-intersection removal failed: %s", exc)
        else:
            logger.info(
                "[MESH CLEAN] Self-intersection check skipped: %d faces exceeds the "
                "%d-face gate (would dominate runtime)",
                len(mesh.triangles), max_faces,
            )
    stats["after_selfint"] = len(mesh.triangles)

    # ── 5d: Connected components ───────────────────────────────────────────
    if len(mesh.triangles) > 0:
        clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        cluster_sizes = np.asarray(cluster_sizes)

        if len(cluster_sizes) > 0:
            largest = int(np.argmax(cluster_sizes))
            largest_size = int(cluster_sizes[largest])

            if keep_largest_only:
                keep_cluster = np.zeros(len(cluster_sizes), dtype=bool)
            else:
                # Threshold relative to the largest component, not to the total
                # face count: with many mid-sized components a total-relative
                # threshold keeps fragments, and a fixed floor can delete a
                # small mesh outright.
                min_tris = max(
                    int(largest_size * float(preset["min_component_ratio"])),
                    int(preset.get("min_component_faces", 50)),
                )
                keep_cluster = cluster_sizes >= min_tris
            keep_cluster[largest] = True   # the largest component always survives

            remove_mask = ~keep_cluster[clusters]
            n_removed = int(remove_mask.sum())
            if n_removed > 0:
                mesh.remove_triangles_by_mask(remove_mask)
                mesh.remove_unreferenced_vertices()

            kept = sorted(cluster_sizes[keep_cluster].tolist(), reverse=True)
            logger.info(
                "[MESH CLEAN] Components: %d of %d kept (%s%s faces) — "
                "removed %d fragment(s), %d faces",
                int(keep_cluster.sum()), len(cluster_sizes),
                ", ".join(str(c) for c in kept[:3]),
                ", ..." if len(kept) > 3 else "",
                len(cluster_sizes) - int(keep_cluster.sum()), n_removed,
            )
            stats["n_components_before"] = int(len(cluster_sizes))
            stats["n_components_kept"] = int(keep_cluster.sum())
            stats["fragments_removed"] = n_removed
            stats["largest_component_frac"] = largest_size / max(len(clusters), 1)

    stats["after_components"] = len(mesh.triangles)

    # ── 5e: Hole filling ───────────────────────────────────────────────────
    if fill_holes and len(mesh.triangles) > 0:
        if hole_size is None:
            hole_size = float(preset.get("hole_fill_factor", 20.0)) * spacing
        n_before = len(mesh.triangles)
        try:
            mesh = _fill_holes(mesh, hole_size)
            n_added = len(mesh.triangles) - n_before
            logger.info(
                "[MESH CLEAN] Hole filling (max size %.6g = %.0f x spacing): "
                "added %d triangles",
                hole_size, hole_size / spacing if spacing > 0 else 0.0, n_added,
            )
        except Exception as exc:
            logger.warning(
                "[MESH CLEAN] Hole filling failed (%s). The mesh is still valid, just "
                "not closed; for complex holes use MeshLab (Filters -> Remeshing -> "
                "Close Holes) or Blender.",
                exc,
            )
    stats["after_fill"] = len(mesh.triangles)

    # ── 5f: Post-fill dedup ────────────────────────────────────────────────
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # ── 5g: Consistent winding + normals ───────────────────────────────────
    try:
        oriented = mesh.orient_triangles()
        if not oriented:
            logger.info(
                "[MESH CLEAN] Triangle orientation left unchanged (mesh is not orientable)"
            )
    except Exception as exc:
        logger.warning("[MESH CLEAN] orient_triangles failed: %s", exc)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ── 5h: Summary ────────────────────────────────────────────────────────
    n_faces_out = len(mesh.triangles)
    n_verts_out = len(mesh.vertices)
    is_manifold = mesh.is_edge_manifold() and mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()

    logger.info(
        "[MESH CLEAN] Cleaning complete:\n"
        "  Before:     %d faces, %d vertices\n"
        "  After:      %d faces, %d vertices\n"
        "  Net change: %+d faces (%.1f%%)\n"
        "  Manifold:   %s\n"
        "  Watertight: %s",
        n_faces_in, n_verts_in,
        n_faces_out, n_verts_out,
        n_faces_out - n_faces_in,
        100.0 * (n_faces_out - n_faces_in) / max(n_faces_in, 1),
        "YES" if is_manifold else "NO",
        "YES" if is_watertight else "NO",
    )

    stats["faces_out"] = n_faces_out
    stats["verts_out"] = n_verts_out
    stats["is_manifold"] = is_manifold
    stats["is_watertight"] = is_watertight
    return mesh, stats


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fill_holes(mesh, hole_size: float):
    """
    Close boundary loops up to ``hole_size`` across.

    open3d exposes hole filling only on the *tensor* geometry API
    (``o3d.t.geometry.TriangleMesh.fill_holes``); the legacy ``TriangleMesh`` has
    no such method, so calling it there raises AttributeError and — when that
    exception is swallowed — hole filling silently never runs.  Convert, fill,
    convert back.  The round trip drops vertex colours, which is harmless here:
    colours are transferred from the point cloud in post-processing, after
    cleaning.
    """
    import open3d as o3d

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tmesh.fill_holes(hole_size=float(hole_size))
    out = filled.to_legacy()
    out.compute_vertex_normals()
    return out


def _remove_self_intersections(mesh):
    """
    Drop triangles that intersect another triangle of the same mesh.

    Uses open3d's dedicated remover when the installed version provides it, and
    otherwise builds the removal mask from the reported intersecting pairs.
    """
    import open3d as o3d  # noqa: F401

    if hasattr(mesh, "remove_self_intersecting_triangles"):
        mesh.remove_self_intersecting_triangles()
        mesh.remove_unreferenced_vertices()
        return mesh

    pairs = np.asarray(mesh.get_self_intersecting_triangles())
    if pairs.size == 0:
        return mesh

    mask = np.zeros(len(mesh.triangles), dtype=bool)
    mask[np.unique(pairs.reshape(-1))] = True
    mesh.remove_triangles_by_mask(mask)
    mesh.remove_unreferenced_vertices()
    return mesh
