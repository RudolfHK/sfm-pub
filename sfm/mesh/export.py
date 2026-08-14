"""
sfm/mesh/export.py — Multi-format mesh export.

Supported formats (determined by output path extension):
  .obj  — Wavefront OBJ with vertex normals and colors
  .ply  — Binary compressed PLY
  .glb  — GL Transmission Format (web/Three.js friendly)
  .stl  — Stereolithography (3-D printing; no color support)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = frozenset({".obj", ".ply", ".glb", ".stl"})


def export_mesh(mesh, output_path: "str | Path", verbose: bool = True) -> dict:
    """
    Write the mesh to disk in the format determined by the output path extension.

    Parameters
    ----------
    mesh        : open3d.geometry.TriangleMesh
    output_path : Destination file path; extension determines format.
    verbose     : Log detailed mesh statistics after writing.

    Returns
    -------
    Dict with the resolved path, format and file size.

    Raises
    ------
    ValueError  if the output extension is not in _SUPPORTED_FORMATS.
    OSError     if open3d reports a write failure.
    """
    import open3d as o3d

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported mesh format {suffix!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if suffix == ".obj":
        success = o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=False,
        )
    elif suffix == ".ply":
        success = o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            compressed=True,
        )
    elif suffix == ".stl":
        if mesh.has_vertex_colors():
            logger.warning(
                "[MESH EXPORT] STL format does not support vertex colors — "
                "colors will not be saved. Use .obj or .ply to preserve colors."
            )
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
    else:  # .glb
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)

    if not success:
        raise OSError(f"open3d failed to write mesh to {output_path}")

    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    size_mb = size_bytes / (1024 * 1024)

    if verbose and output_path.exists():
        logger.info(
            "[MESH EXPORT] Mesh saved: %s\n"
            "  Faces:       %d\n"
            "  Vertices:    %d\n"
            "  Has normals: %s\n"
            "  Has colors:  %s\n"
            "  File size:   %.1f MB (%s)\n"
            "\n"
            "  View with: MeshLab, Blender, CloudCompare, or open3d viewer",
            output_path,
            len(mesh.triangles),
            len(mesh.vertices),
            "YES" if mesh.has_vertex_normals() else "NO",
            "YES" if mesh.has_vertex_colors() else "NO",
            size_mb,
            "ASCII — use .ply for a ~5x smaller binary file" if suffix == ".obj"
            else "binary",
        )

    return {
        "path": str(output_path),
        "format": suffix,
        "size_bytes": int(size_bytes),
    }
