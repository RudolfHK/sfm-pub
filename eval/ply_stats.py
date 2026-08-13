#!/usr/bin/env python3
"""Structural health metrics for an output point cloud.

There is no ground-truth surface for the Buddha dataset, so this measures
internal cloud health rather than absolute geometric accuracy: density,
isolated flyers, exact duplicates, and whether colours were really sampled from
the images or left at the fallback grey.

Usage
-----
    python eval/ply_stats.py cloud.ply [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return (points Nx3 float64, colors Nx3 uint8 or None, format token)."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, n_vert, props = None, 0, []
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("truncated PLY header")
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("property") and "list" not in line:
                _, ptype, pname = line.split()[:3]
                props.append((pname, ptype))
            elif line == "end_header":
                break

        names = [p for p, _ in props]
        if fmt == "ascii":
            arr = np.loadtxt(fh, max_rows=n_vert)
            cols = {n: arr[:, i] for i, n in enumerate(names)}
        else:
            dt = np.dtype([(n, _PLY_TYPES[t]) for n, t in props])
            if fmt == "binary_big_endian":
                dt = dt.newbyteorder(">")
            data = np.frombuffer(fh.read(dt.itemsize * n_vert), dtype=dt, count=n_vert)
            cols = {n: data[n] for n in names}

    pts = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    colors = None
    if {"red", "green", "blue"} <= set(names):
        colors = np.stack(
            [cols["red"], cols["green"], cols["blue"]], axis=1
        ).astype(np.uint8)
    return pts, colors, fmt or "unknown"


def analyse(path: Path, sample: int = 200_000, seed: int = 0) -> dict:
    pts, colors, fmt = read_ply(path)
    n = len(pts)
    out: dict = {
        "file": str(path),
        "format": fmt,
        "bytes": path.stat().st_size,
        "n_points": int(n),
        "bytes_per_point": round(path.stat().st_size / n, 1) if n else None,
    }
    if n == 0:
        return out

    finite = np.isfinite(pts).all(axis=1)
    out["n_non_finite"] = int((~finite).sum())
    pts = pts[finite]
    if colors is not None:
        colors = colors[finite]

    bbox = pts.max(0) - pts.min(0)
    out["bbox_extent"] = bbox.tolist()
    out["bbox_diagonal"] = float(np.linalg.norm(bbox))
    out["centroid"] = pts.mean(0).tolist()

    # Neighbour statistics on a random subsample — full KD-trees on dense
    # clouds are slow and the distribution is what matters.
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(pts), sample, replace=False)
           if len(pts) > sample else np.arange(len(pts)))
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, _ = tree.query(pts[idx], k=2, workers=-1)
        nn = d[:, 1]
        out["nn_distance"] = {
            "median": float(np.median(nn)),
            "mean": float(np.mean(nn)),
            "p95": float(np.percentile(nn, 95)),
            "max": float(np.max(nn)),
        }
        out["duplicate_fraction"] = float(np.mean(nn < 1e-9))
        # A flyer is a point whose nearest neighbour is far away relative to the
        # typical spacing: isolated debris rather than surface.
        med = float(np.median(nn))
        out["flyer_fraction_nn_gt_10x_median"] = (
            float(np.mean(nn > 10 * med)) if med > 0 else None
        )
    except ImportError:
        out["nn_distance"] = None

    # Distance from centroid: catches the long tails that wreck bbox/meshing.
    r = np.linalg.norm(pts - pts.mean(0), axis=1)
    out["radius_from_centroid"] = {
        "median": float(np.median(r)),
        "p95": float(np.percentile(r, 95)),
        "p99": float(np.percentile(r, 99)),
        "max": float(np.max(r)),
    }
    out["outlier_fraction_r_gt_5x_median"] = float(np.mean(r > 5 * np.median(r)))

    if colors is not None:
        grey = np.all(colors == 128, axis=1)
        black = np.all(colors == 0, axis=1)
        out["color"] = {
            "fallback_grey_fraction": float(np.mean(grey)),
            "pure_black_fraction": float(np.mean(black)),
            "mean_rgb": colors.mean(0).tolist(),
            "unique_colors": int(len(np.unique(colors, axis=0))),
        }
    else:
        out["color"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = analyse(Path(args.ply))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
