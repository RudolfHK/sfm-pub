#!/usr/bin/env python3
"""Render two PLY point clouds side-by-side under identical viewpoint and scale.

Produces the central results figure (Abb. 8): eigene Pipeline vs. COLMAP on the
same scene. Reads the repo's binary-little-endian PLY (float x/y/z + uchar rgb),
also handles ASCII and double-precision variants. Matplotlib only — no Open3D.

Usage
-----
    python paper/scripts/compare_ply.py own.ply colmap.ply -o abb8_vergleich.pdf
    python paper/scripts/compare_ply.py own.ply colmap.ply \
        --labels "Eigene Pipeline" "COLMAP" --elev 20 --azim -60 --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# numpy dtype for each PLY scalar type token
_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (points Nx3 float, colors Nx3 uint8 or None)."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        n_vert = 0
        props: list[tuple[str, str]] = []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("element"):
                # another element after vertices — stop collecting vertex props
                pass
            elif line.startswith("property") and "list" not in line:
                _, ptype, pname = line.split()[:3]
                props.append((pname, _PLY_TYPES.get(ptype, "<f4")))
            elif line == "end_header":
                break
            elif not line:
                raise ValueError(f"{path}: malformed header (no end_header)")
        names = [p[0] for p in props]

        if fmt == "ascii":
            data = np.loadtxt(fh, max_rows=n_vert)
            cols = {n: data[:, i] for i, n in enumerate(names)}
        else:  # binary_little_endian / binary_big_endian
            dtype = np.dtype(props if fmt == "binary_little_endian"
                             else [(n, t.replace("<", ">")) for n, t in props])
            arr = np.frombuffer(fh.read(dtype.itemsize * n_vert), dtype=dtype)
            cols = {n: arr[n].astype(np.float64) for n in names}

    pts = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float64)
    if {"red", "green", "blue"} <= set(names):
        rgb = np.column_stack([cols["red"], cols["green"], cols["blue"]])
        colors = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        colors = None
    return pts, colors


def _center_and_scale(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Center on median, return (centered, robust half-extent)."""
    c = np.median(pts, axis=0)
    p = pts - c
    # robust extent: 98th percentile radius, avoids outlier-driven zoom-out
    half = float(np.percentile(np.abs(p), 98)) or 1.0
    return p, half


def render(paths, labels, out, elev, azim, dpi, max_points, point_size,
           normalize=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(11, 5.2))
    halves = []
    clouds = []
    for path in paths:
        pts, colors = read_ply(Path(path))
        if len(pts) > max_points:
            idx = rng.choice(len(pts), max_points, replace=False)
            pts, colors = pts[idx], (colors[idx] if colors is not None else None)
        p, half = _center_and_scale(pts)
        if normalize:
            # Monocular SfM is scale-free, so two reconstructions of the same
            # scene differ by an arbitrary factor. Normalising each cloud to its
            # own robust extent compares shape and completeness instead of that
            # meaningless factor.
            p, half = p / half, 1.0
        clouds.append((p, colors))
        halves.append(half)
    shared = max(halves)  # identical axis range for both panels

    for i, ((p, colors), label, path) in enumerate(zip(clouds, labels, paths)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        c = (colors / 255.0) if colors is not None else "0.4"
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=point_size, c=c,
                   marker=".", linewidths=0, depthshade=True)
        ax.set_xlim(-shared, shared)
        ax.set_ylim(-shared, shared)
        ax.set_zlim(-shared, shared)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"{label}\n{len(p):,} Punkte", fontsize=13,
                     color="#1F3864", fontweight="bold", pad=0)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[compare_ply] wrote {out}  ({', '.join(str(p) for p in paths)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", nargs=2, help="two PLY files: own.ply colmap.ply")
    ap.add_argument("-o", "--output", default="abb8_vergleich.pdf")
    ap.add_argument("--labels", nargs=2, default=["Eigene Pipeline", "COLMAP"])
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--point-size", type=float, default=1.5)
    ap.add_argument("--normalize", action="store_true",
                    help="jede Wolke auf ihre eigene Ausdehnung normieren "
                         "(SfM ist skalenfrei; vergleicht Form statt Skala)")
    a = ap.parse_args(argv)
    try:
        render(a.ply, a.labels, a.output, a.elev, a.azim, a.dpi,
               a.max_points, a.point_size, a.normalize)
    except (FileNotFoundError, ValueError) as e:
        print(f"[compare_ply] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
