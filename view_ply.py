#!/usr/bin/env python3
"""View a PLY point cloud with Open3D.

Usage:
    python view_ply.py output.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display a PLY point cloud in an interactive Open3D window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ply_file",
        nargs="?",
        default="output.ply",
        help="Path to the PLY file to display.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Rendered point size in pixels.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import open3d as o3d
    except ImportError:
        print(
            "Open3D is not installed. Install it with: pip install open3d",
            file=sys.stderr,
        )
        return 1

    ply_path = Path(args.ply_file)
    if not ply_path.exists():
        print(f"PLY file not found: {ply_path}", file=sys.stderr)
        return 1

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    if point_cloud.is_empty():
        print(f"No points could be read from: {ply_path}", file=sys.stderr)
        return 1

    window_title = f"PLY Viewer - {ply_path.name}"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)
    vis.add_geometry(point_cloud)

    render_options = vis.get_render_option()
    if render_options is not None:
        render_options.point_size = args.point_size
        render_options.background_color = np.asarray([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
