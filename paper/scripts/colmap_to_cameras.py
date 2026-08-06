#!/usr/bin/env python3
"""Convert a COLMAP TXT model into the cameras.json that eval/gt_pose_eval.py reads.

That makes COLMAP scoreable against the same ground truth, with the same Sim(3)
alignment, as the runs of this pipeline. COLMAP stores world-to-camera rotation
and translation, which is the convention `--export-cameras` uses as well, so the
conversion is a straight rename plus C = -R^T t.

Usage
-----
    colmap model_converter --input_path <ws>/sparse/0 \
        --output_path <ws>/sparse_txt --output_type TXT
    python paper/scripts/colmap_to_cameras.py <ws>/sparse_txt \
        -o paper_out/colmap.cameras.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qvec_to_rot(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_camera(path: Path) -> tuple[np.ndarray, int]:
    """First camera in cameras.txt as (K, n_cameras). Assumes a shared camera."""
    rows = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    if not rows:
        raise SystemExit(f"no camera in {path}")
    parts = rows[0].split()
    model, params = parts[1], [float(v) for v in parts[4:]]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise SystemExit(f"unsupported camera model: {model}")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, len(rows)


def read_images(path: Path) -> list[dict]:
    """Pose per registered image; COLMAP writes two lines per image."""
    rows = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    cameras = []
    for line in rows[::2]:                       # odd lines hold the 2-D points
        p = line.split()
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        R = qvec_to_rot(q)
        cameras.append({
            "image_name": p[9],
            "R": R.tolist(),
            "t": t.tolist(),
            "center": (-R.T @ t).tolist(),
        })
    return sorted(cameras, key=lambda c: c["image_name"])


def read_points(path: Path) -> tuple[int, float]:
    """(number of 3-D points, mean track length)."""
    n, obs = 0, 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        n += 1
        obs += (len(line.split()) - 8) // 2      # TRACK[] holds (image_id, pt2d_id)
    return n, (obs / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="COLMAP model in TXT format")
    ap.add_argument("-o", "--output", default="colmap.cameras.json")
    ap.add_argument("--n-images", type=int, default=None,
                    help="images offered to COLMAP (default: registered count)")
    args = ap.parse_args()

    model = Path(args.model_dir)
    K, n_cams = read_camera(model / "cameras.txt")
    cameras = read_images(model / "images.txt")
    n_points, track = read_points(model / "points3D.txt")
    if n_cams > 1:
        print(f"warning: {n_cams} camera models in the model, using the first")

    out = {
        "convention": "world_to_camera (x_cam = R @ x_world + t); C = -R.T @ t",
        "source": "COLMAP",
        "n_images": args.n_images if args.n_images else len(cameras),
        "n_cameras_registered": len(cameras),
        "n_points": n_points,
        "mean_track_length": track,
        "reprojection": None,     # COLMAP does not export a residual summary
        "shared_K": K.tolist(),
        "cameras": cameras,
    }
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"geschrieben: {args.output}  "
          f"({len(cameras)} Kameras, {n_points:,} Punkte, f={K[0, 0]:.1f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
