"""
Standalone mesh reconstruction CLI.

    python -m sfm.mesh dense.ply -o mesh.ply --quality high

Meshes an existing point cloud without re-running Structure from Motion.  The
full pipeline (``run_sfm.py --mesh``) recomputes features, matches and poses
before it ever reaches the mesh stage, which makes iterating on meshing
parameters impractical; this entry point takes minutes off that loop and is what
the mesh stage should be tuned and verified with.

Camera poses are optional but recommended: pass the JSON written by
``run_sfm.py --export-cameras`` and normals are oriented towards the cameras that
observed the surface instead of by the tangent-plane heuristic.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _load_camera_centers(path: "str | Path") -> "np.ndarray | None":
    """
    Read world-space camera centres from a ``--export-cameras`` JSON file.

    Falls back to computing C = -Rᵀt when the file predates the ``center`` field.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger(__name__).warning("Could not read camera file %s: %s", path, exc)
        return None

    centers = []
    for cam in payload.get("cameras", []):
        if "center" in cam:
            centers.append(np.asarray(cam["center"], dtype=np.float64).reshape(3))
        elif "R" in cam and "t" in cam:
            R = np.asarray(cam["R"], dtype=np.float64).reshape(3, 3)
            t = np.asarray(cam["t"], dtype=np.float64).reshape(3)
            centers.append(-R.T @ t)

    if not centers:
        logging.getLogger(__name__).warning("No camera poses found in %s", path)
        return None
    return np.vstack(centers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sfm.mesh",
        description="Reconstruct a triangle mesh from an SfM/MVS point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pointcloud", help="Input point cloud (.ply)")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output mesh path; extension picks the format (.ply .obj .glb .stl). "
             "Default: <input_stem>_mesh.ply",
    )
    p.add_argument(
        "--cameras", default=None, metavar="JSON",
        help="Camera poses from 'run_sfm.py --export-cameras'. Used to orient normals.",
    )

    g = p.add_argument_group("reconstruction")
    g.add_argument("--method", dest="mesh_method", choices=["poisson", "bpa", "alpha"],
                   default="poisson", help="Surface reconstruction algorithm")
    g.add_argument("--quality", dest="mesh_quality",
                   choices=["low", "medium", "high", "ultra"], default="medium",
                   help="Quality preset")
    g.add_argument("--depth", dest="mesh_depth", type=int, default=None, metavar="N",
                   help="Poisson octree depth override (typical 8-12)")
    g.add_argument("--trim", dest="mesh_trim", type=float, default=None, metavar="F",
                   help="Phantom-surface trim distance in multiples of point spacing")
    g.add_argument("--target-points", dest="mesh_target_points", type=int, default=None,
                   metavar="N", help="Downsample target point count (0 = keep all)")

    c = p.add_argument_group("cleaning")
    c.add_argument("--no-clean", dest="mesh_no_clean", action="store_true",
                   help="Skip the cleaning pipeline entirely")
    c.add_argument("--no-fill-holes", dest="mesh_fill_holes", action="store_false",
                   default=True, help="Disable hole filling")
    c.add_argument("--hole-size", dest="mesh_hole_size", type=float, default=None,
                   metavar="D", help="Max fillable hole size in scene units")
    c.add_argument("--keep-largest", dest="mesh_keep_largest", action="store_true",
                   help="Keep only the largest connected component")
    c.add_argument("--self-intersections", dest="mesh_self_intersections",
                   action="store_true",
                   help="Detect and remove self-intersecting triangles, and report them "
                        "in the quality check. Expensive; off by default because a "
                        "Poisson iso-surface cannot self-intersect by construction")

    o = p.add_argument_group("post-processing and output")
    o.add_argument("--smooth", dest="mesh_smooth", action="store_true",
                   help="Apply Taubin smoothing (lossy)")
    o.add_argument("--smooth-iterations", dest="mesh_smooth_iterations", type=int, default=5,
                   metavar="N")
    o.add_argument("--decimate", dest="mesh_decimate", action="store_true",
                   help="Reduce the polygon count")
    o.add_argument("--decimate-target", dest="mesh_decimate_target", type=int,
                   default=100_000, metavar="N")
    o.add_argument("--no-texture", dest="mesh_no_texture", action="store_true",
                   help="Skip vertex-colour transfer from the cloud")
    o.add_argument("--no-keep-pointcloud", dest="mesh_keep_pointcloud", action="store_false",
                   default=True, help="Do not write the prepared point cloud")
    o.add_argument("--no-validate", dest="mesh_no_validate", action="store_true",
                   help="Skip the quality report")
    o.add_argument("--preview", dest="mesh_preview", action="store_true",
                   help="Open the interactive open3d viewer when finished")
    o.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # A Windows console defaults to cp1252 and raises UnicodeEncodeError inside
    # logging when a message contains characters outside it, which turns a
    # cosmetic glyph into a lost log line.  The report itself is ASCII; this
    # covers everything else.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    src = Path(args.pointcloud)
    if not src.exists():
        print(f"Error: point cloud not found: {src}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else src.parent / f"{src.stem}_mesh.ply"

    camera_centers = _load_camera_centers(args.cameras) if args.cameras else None

    from .pipeline import MeshPipeline

    result = MeshPipeline(args).run(src, out, camera_centers=camera_centers)

    if not result.success:
        print(f"\nMesh reconstruction failed: {result.error}", file=sys.stderr)
        return 1

    print(f"\nMesh    : {result.output_path}  ({result.face_count:,} faces)")
    if result.cloud_path:
        print(f"Cloud   : {result.cloud_path}")
    if result.report_path:
        print(f"Report  : {result.report_path}")
    if result.verdict:
        print(f"Verdict : {result.verdict}")

    # FAIL is a quality verdict, not a crash: the mesh is on disk either way, but
    # a non-zero exit lets scripted sweeps notice without parsing the log.
    return 2 if result.verdict == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
