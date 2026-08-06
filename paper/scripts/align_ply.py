#!/usr/bin/env python3
"""Bring one reconstruction into the coordinate frame of another.

Two SfM reconstructions of the same scene live in arbitrary, unrelated frames:
rotation, translation and scale are all free. Rendering them side by side under
one viewing angle therefore shows two different sides of the object. This script
estimates the Sim(3) that maps the source reconstruction onto the reference from
the centres of cameras with the same image name, applies it to the source point
cloud and writes the result, so one viewpoint shows the same view of both.

Usage
-----
    python paper/scripts/align_ply.py own.ply own.cameras.json ref.cameras.json \
        -o own_aligned.ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.scripts.compare_ply import read_ply  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float((sig * np.diag(W)).sum() / var_s) if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def centres(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["image_name"]: np.asarray(c["center"], float) for c in data["cameras"]}


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray | None) -> None:
    """Binary little-endian PLY, float xyz plus optional uchar rgb."""
    n = len(pts)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=np.dtype(fields))
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is not None:
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(arr.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", help="point cloud to transform")
    ap.add_argument("src_cameras", help="cameras.json belonging to that cloud")
    ap.add_argument("ref_cameras", help="cameras.json of the target frame")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    src, ref = centres(Path(args.src_cameras)), centres(Path(args.ref_cameras))
    shared = sorted(set(src) & set(ref))
    if len(shared) < 3:
        print(f"only {len(shared)} shared cameras, need 3", file=sys.stderr)
        return 1

    s, R, t = umeyama(np.array([src[k] for k in shared]),
                      np.array([ref[k] for k in shared]))
    resid = np.linalg.norm(
        np.array([s * R @ src[k] + t for k in shared]) - np.array([ref[k] for k in shared]),
        axis=1,
    )

    pts, rgb = read_ply(Path(args.ply))
    write_ply(Path(args.output), (s * (R @ pts.T).T + t), rgb)

    print(f"geschrieben: {args.output}  ({len(pts):,} Punkte)")
    print(f"  gemeinsame Kameras : {len(shared)}")
    print(f"  Skalenfaktor       : {s:.4f}")
    print(f"  Restfehler Zentren : Median {np.median(resid):.4f}, "
          f"Max {resid.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
