#!/usr/bin/env python3
"""Evaluate the reconstructed surface against what this dataset can prove.

What counts as ground truth here
--------------------------------
The Buddha dataset ships 67 images and one projection matrix per image. It does
*not* ship a reference surface, so no distance-to-truth in millimetres can be
computed and none is reported. What the ground truth does allow is a
photometric test: the poses are known exactly, so a surface point can be
projected into every view that sees it and the colours compared. A point lying
on the true surface shows the same colour from every direction on a matte,
diffusely lit object; a point floating in front of or behind the surface samples
different parts of the scene in each view and the colours disagree.

That gives three families of measurement, in decreasing order of what they
prove:

1. **Photometric consistency under ground-truth poses.** For each sampled
   surface point, the standard deviation of its colour across the cameras that
   see it. Reported together with a control: the same statistic for the same
   points displaced along their normals. Without that control the number has no
   scale, since a textured object always produces some spread.

2. **Agreement with a reference reconstruction.** The pipeline is run once with
   the ground-truth calibration supplied, which removes the largest error source
   the paper measured, and the resulting cloud serves as the reference geometry.
   Distances from the evaluated mesh to that cloud separate meshing error from
   reconstruction error.

3. **Fidelity to its own input and topology.** Distance to the cloud the mesh
   was built from, coverage of that cloud, watertightness, manifoldness,
   component count, triangle quality, and the fraction of surface area that
   Poisson invented where no data supports it.

Usage
-----
    python eval/mesh_eval.py --mesh out_mesh.obj --cloud out_mesh_prepared_cloud.ply \\
        --cameras out.cameras.json --gt-dir <buddha> --image-dir <buddha_67> \\
        --out eval_results/mesh_eval.json [--reference ref.ply]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "eval"))

from gt_pose_eval import load_gt, umeyama  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mesh_eval")


# ── alignment ─────────────────────────────────────────────────────────────────

def sim3_to_gt(cameras_json: Path, gt_dir: Path):
    """
    Similarity transform taking the reconstruction into the ground-truth frame.

    Estimated from the camera centres of images present in both, which is the
    only correspondence available: monocular SfM fixes rotation, translation and
    scale arbitrarily.
    """
    gt = load_gt(gt_dir)
    est = json.loads(Path(cameras_json).read_text(encoding="utf-8"))
    rows = [c for c in est["cameras"] if c["image_name"].split(".")[0] in gt]
    if len(rows) < 3:
        raise SystemExit("Fewer than three shared cameras; no Sim(3) is defined.")
    rows.sort(key=lambda c: c["image_name"])
    names = [c["image_name"].split(".")[0] for c in rows]
    C_est = np.array([c["center"] for c in rows], dtype=np.float64)
    C_gt = np.array([gt[n]["C"] for n in names], dtype=np.float64)
    s, R, t = umeyama(C_est, C_gt)
    log.info("Sim(3) to ground truth from %d cameras: scale %.4f", len(rows), s)
    return s, R, t, gt, names


def apply_sim3(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ np.asarray(pts, dtype=np.float64).T).T) + t


# ── photometric consistency ───────────────────────────────────────────────────

def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample an HxWx3 uint8 image at floating-point pixel positions."""
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    f = img.astype(np.float32)
    return (f[y0, x0] * (1 - dx) * (1 - dy) + f[y0, x0 + 1] * dx * (1 - dy)
            + f[y0 + 1, x0] * (1 - dx) * dy + f[y0 + 1, x0 + 1] * dx * dy)


def photometric_consistency(
    mesh, gt: Dict[str, dict], image_dir: Path, names: List[str],
    n_samples: int = 20_000, offset_factors: Tuple[float, ...] = (10.0, 30.0),
    spacing: float = 1.0, seed: int = 0,
) -> Dict[str, object]:
    """
    Colour agreement across ground-truth views, with a displaced control.

    A sample is used when at least three cameras see it unoccluded. Occlusion is
    resolved on the mesh itself with a ray cast from the camera to the sample.
    """
    import cv2
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    try:
        sampled = mesh.sample_points_uniformly(number_of_points=n_samples, seed=seed)
    except TypeError:
        sampled = mesh.sample_points_uniformly(number_of_points=n_samples)
    P = np.asarray(sampled.points, dtype=np.float64)
    N = np.asarray(sampled.normals, dtype=np.float64)
    if N.size == 0:
        sampled.estimate_normals()
        N = np.asarray(sampled.normals, dtype=np.float64)

    # Every variant is measured on the same samples so the comparison is paired.
    variants = {"surface": P}
    for f in offset_factors:
        variants[f"offset_{f:g}x"] = P + N * (f * spacing)

    images: Dict[str, np.ndarray] = {}
    for name in names:
        hits = list(Path(image_dir).glob(name + ".*"))
        if hits:
            img = cv2.imread(str(hits[0]))
            if img is not None:
                images[name] = img[:, :, ::-1].copy()   # BGR -> RGB
    if not images:
        raise SystemExit(f"No images of the evaluated cameras found in {image_dir}")
    log.info("Photometric test: %d samples, %d images", len(P), len(images))

    out: Dict[str, object] = {"n_samples": int(len(P)), "n_images": len(images)}
    for label, pts in variants.items():
        sums = np.zeros((len(pts), 3), dtype=np.float64)
        sqs = np.zeros((len(pts), 3), dtype=np.float64)
        counts = np.zeros(len(pts), dtype=np.int32)

        for name in names:
            img = images.get(name)
            if img is None:
                continue
            g = gt[name]
            h, w = img.shape[:2]
            X_cam = (g["R"] @ pts.T).T + (-g["R"] @ g["C"])
            z = X_cam[:, 2]
            in_front = z > 1e-6
            if not in_front.any():
                continue
            uvw = (g["K"] @ X_cam.T).T
            uv = uvw[:, :2] / np.where(uvw[:, 2:3] == 0, 1e-9, uvw[:, 2:3])
            inside = (in_front & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1))
            idx = np.nonzero(inside)[0]
            if idx.size == 0:
                continue

            # Occlusion: the first surface along the ray must be this sample.
            origins = np.repeat(g["C"][None, :], idx.size, axis=0)
            dirs = pts[idx] - origins
            dist = np.linalg.norm(dirs, axis=1)
            dirs = dirs / np.maximum(dist, 1e-12)[:, None]
            rays = o3d.core.Tensor(
                np.hstack([origins, dirs]).astype(np.float32), dtype=o3d.core.Dtype.Float32)
            hit = scene.cast_rays(rays)["t_hit"].numpy()
            visible = np.isfinite(hit) & (hit > dist * 0.98 - 1e-9)
            idx = idx[visible]
            if idx.size == 0:
                continue

            cols = _bilinear(img, uv[idx])
            sums[idx] += cols
            sqs[idx] += cols.astype(np.float64) ** 2
            counts[idx] += 1

        usable = counts >= 3
        if not usable.any():
            out[label] = {"n_usable": 0}
            continue
        c = counts[usable][:, None].astype(np.float64)
        mean = sums[usable] / c
        var = np.maximum(sqs[usable] / c - mean ** 2, 0.0)
        std = np.sqrt(var).mean(axis=1)          # mean over the three channels
        out[label] = {
            "n_usable": int(usable.sum()),
            "mean_views": float(counts[usable].mean()),
            "median_std": float(np.median(std)),
            "mean_std": float(std.mean()),
            "p90_std": float(np.percentile(std, 90)),
        }
        log.info("  %-12s %6d samples, %.1f views, median colour spread %.2f",
                 label, int(usable.sum()), float(counts[usable].mean()),
                 float(np.median(std)))
    return out


# ── geometry against a reference cloud ────────────────────────────────────────

def compare_to_reference(mesh, reference_pts: np.ndarray, spacing: float) -> Dict[str, float]:
    """Distances between the mesh surface and a reference point cloud."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(reference_pts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    d_ref_to_mesh = scene.compute_distance(query).numpy().astype(np.float64)

    try:
        s = mesh.sample_points_uniformly(number_of_points=100_000, seed=0)
    except TypeError:
        s = mesh.sample_points_uniformly(number_of_points=100_000)
    d_mesh_to_ref, _ = cKDTree(reference_pts).query(
        np.asarray(s.points), k=1, workers=-1)

    def summary(d):
        return {
            "median": float(np.median(d)), "mean": float(d.mean()),
            "p95": float(np.percentile(d, 95)), "max": float(d.max()),
            "median_in_spacing": float(np.median(d) / spacing) if spacing else None,
        }

    return {
        "reference_to_mesh": summary(d_ref_to_mesh),
        "mesh_to_reference": summary(d_mesh_to_ref),
        "completeness_1x": float((d_ref_to_mesh <= spacing).mean()),
        "completeness_3x": float((d_ref_to_mesh <= 3 * spacing).mean()),
        "extrapolated_area_frac": float((d_mesh_to_ref > 3 * spacing).mean()),
        "n_reference_points": int(len(reference_pts)),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cloud", required=True, help="Prepared cloud the mesh was built from")
    ap.add_argument("--cameras", required=True, help="cameras.json of the same run")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--reference", default=None,
                    help="Reference PLY (e.g. a run with the GT calibration supplied)")
    ap.add_argument("--reference-cameras", default=None,
                    help="cameras.json belonging to --reference, for its own Sim(3)")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--out", default=str(_ROOT / "eval_results" / "mesh_eval.json"))
    args = ap.parse_args()

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.triangles) == 0:
        raise SystemExit(f"No triangles in {args.mesh}")
    mesh.compute_vertex_normals()
    cloud = o3d.io.read_point_cloud(args.cloud)
    cloud_pts = np.asarray(cloud.points, dtype=np.float64)
    log.info("Mesh: %d vertices, %d faces.  Prepared cloud: %d points",
             len(mesh.vertices), len(mesh.triangles), len(cloud_pts))

    # Point spacing of the input cloud is the natural length unit.
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    sub = cloud_pts[rng.choice(len(cloud_pts), min(20000, len(cloud_pts)), replace=False)]
    nn, _ = cKDTree(cloud_pts).query(sub, k=2, workers=-1)
    spacing = float(np.median(nn[:, 1]))
    log.info("Point spacing: %.6f (reconstruction units)", spacing)

    report: Dict[str, object] = {
        "mesh": str(args.mesh), "cloud": str(args.cloud),
        "n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.triangles)),
        "n_cloud_points": int(len(cloud_pts)), "spacing": spacing,
    }

    # ── 3. fidelity to its own input, topology ───────────────────────────
    from sfm.mesh.validation import validate_mesh
    report["self"] = validate_mesh(mesh, cloud, spacing, n_samples=100_000)

    # ── alignment into the ground-truth frame ────────────────────────────
    s, R, t, gt, names = sim3_to_gt(Path(args.cameras), Path(args.gt_dir))
    mesh_gt = o3d.geometry.TriangleMesh(mesh)
    mesh_gt.vertices = o3d.utility.Vector3dVector(
        apply_sim3(np.asarray(mesh.vertices), s, R, t))
    mesh_gt.compute_vertex_normals()
    report["sim3_scale"] = float(s)

    # ── 1. photometric consistency under ground-truth poses ──────────────
    report["photometric"] = photometric_consistency(
        mesh_gt, gt, Path(args.image_dir), names,
        n_samples=args.samples, spacing=spacing * s,
    )

    # ── 2. agreement with a reference reconstruction ─────────────────────
    if args.reference:
        ref = o3d.io.read_point_cloud(args.reference)
        ref_pts = np.asarray(ref.points, dtype=np.float64)
        if args.reference_cameras:
            s2, R2, t2, _, _ = sim3_to_gt(Path(args.reference_cameras), Path(args.gt_dir))
            ref_pts = apply_sim3(ref_pts, s2, R2, t2)
        report["reference"] = compare_to_reference(mesh_gt, ref_pts, spacing * s)
        report["reference_path"] = str(args.reference)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("written: %s", out)

    # Console summary
    self_rep = report["self"]
    print()
    print(f"verdict           : {self_rep.get('verdict')}")
    topo = self_rep.get("topology", {})
    print(f"faces             : {topo.get('n_faces')}")
    print(f"watertight        : {topo.get('is_watertight')}")
    print(f"components        : {topo.get('n_components')} "
          f"(largest {100 * topo.get('largest_component_frac', 0):.1f} %)")
    acc = self_rep.get("accuracy", {})
    print(f"mesh -> cloud     : median {acc.get('median', float('nan')):.5f} "
          f"({acc.get('median_in_spacing', float('nan')):.2f} spacings)")
    ph = report["photometric"]
    for key in ("surface", "offset_10x", "offset_30x"):
        if key in ph and ph[key].get("n_usable"):
            print(f"colour spread {key:9s}: median {ph[key]['median_std']:.2f} "
                  f"over {ph[key]['mean_views']:.1f} views")
    if "reference" in report:
        r = report["reference"]
        print(f"vs reference      : mesh->ref median "
              f"{r['mesh_to_reference']['median']:.5f}, completeness@3sp "
              f"{100 * r['completeness_3x']:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
