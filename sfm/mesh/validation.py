"""
sfm/mesh/validation.py — Quantitative mesh quality assessment.

A reconstruction is only as trustworthy as the numbers that describe it.  This
module measures a finished mesh against the point cloud it came from and against
its own topology, then issues explicit PASS / WARN / FAIL verdicts.  The result
is logged and written next to the mesh as JSON.

Metrics
-------
accuracy      (mesh -> cloud)  How far the reconstructed surface strays from the
                               measured points.  The mesh is sampled uniformly by
                               area — sampling its *vertices* instead would bias
                               the statistic towards dense regions and ignore the
                               interiors of large phantom triangles entirely.

completeness  (cloud -> mesh)  How much of the measured data the surface actually
                               explains.  Distances are exact point-to-triangle
                               distances from a BVH raycasting scene, not
                               distances to the nearest vertex.  Coverage is the
                               fraction of cloud points within tau = k * spacing.

These two are the standard pair used to evaluate MVS/meshing output (they are
what the Middlebury and ETH3D benchmarks report as accuracy/completeness); one
alone is trivially gameable — a single well-placed triangle is perfectly
accurate, and a huge blob is perfectly complete.

topology      Manifoldness, watertightness, orientability, boundary and
              non-manifold edge counts, component count, Euler characteristic,
              genus where it is defined.

geometry      Surface area, enclosed volume when watertight, edge-length and
              normalized triangle aspect-ratio distributions, degenerate faces.

All distances are reported in scene units *and* normalized by the cloud's point
spacing, because SfM reconstructions have no absolute scale — "0.004 units of
error" is meaningless on its own, "0.4 x the point spacing" is not.

Calibration of the accuracy thresholds
--------------------------------------
The accuracy metric has a non-zero floor that has nothing to do with surface
error.  Comparing surface samples against a *finite* point cloud measures, in the
best case, the nearest-neighbour distance between two independent uniform samples
of the same surface.  Measured on an exact analytic sphere meshed at four
sampling densities (5 k … 200 k points), that floor is scale-invariant:

    mean ≈ 1.07 × spacing    median ≈ 1.00 × spacing    p95 ≈ 2.09 × spacing

A perfect mesh therefore scores p95 ≈ 2.1, and any PASS threshold below that is
unreachable by construction.  The gates below (PASS ≤ 3, WARN ≤ 6) are set
relative to that measured floor, not chosen by eye.  ACCURACY_FLOOR_P95 records
it so the numbers stay explainable.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Empirical p95 accuracy of a perfect mesh against a finite sample of its own
# surface, in multiples of point spacing (see the module docstring).  The PASS and
# WARN gates are multiples of this floor rather than free parameters.
ACCURACY_FLOOR_P95 = 2.1
ACCURACY_PASS_P95 = 3.0
ACCURACY_WARN_P95 = 6.0


@dataclass
class Check:
    """A single named quality gate with its verdict."""

    name: str
    level: str
    detail: str

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "level": self.level, "detail": self.detail}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dist_summary(d: np.ndarray, spacing: float) -> Dict[str, float]:
    """Mean/median/RMS/p95/max of a distance array, absolute and spacing-relative."""
    if d.size == 0:
        return {}
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    s = max(spacing, 1e-12)
    return {
        "mean":       float(np.mean(d)),
        "median":     float(np.median(d)),
        "rms":        float(np.sqrt(np.mean(d ** 2))),
        "p95":        float(np.percentile(d, 95)),
        "max":        float(np.max(d)),
        "mean_rel":   float(np.mean(d) / s),
        "median_rel": float(np.median(d) / s),
        "p95_rel":    float(np.percentile(d, 95) / s),
    }


def _edge_topology(triangles: np.ndarray) -> Dict[str, int]:
    """
    Count unique, boundary and non-manifold edges from the triangle array.

    An edge shared by exactly one triangle is a boundary edge (a hole rim); by
    more than two, a non-manifold edge.  Doing this with numpy is far cheaper
    than open3d's per-edge map for meshes of this size.
    """
    if len(triangles) == 0:
        return {"n_edges": 0, "n_boundary_edges": 0, "n_nonmanifold_edges": 0}

    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "n_edges":             int(counts.size),
        "n_boundary_edges":    int(np.sum(counts == 1)),
        "n_nonmanifold_edges": int(np.sum(counts > 2)),
    }


def _triangle_quality(vertices: np.ndarray, triangles: np.ndarray) -> Dict[str, float]:
    """
    Edge-length and normalized aspect-ratio statistics.

    Aspect ratio is circumradius / (2 * inradius), which equals 1.0 exactly for an
    equilateral triangle and grows without bound as a triangle degenerates into a
    sliver.  Slivers are what break downstream normal interpolation, simulation
    and 3-D printing, so their tail matters more than the mean.
    """
    if len(triangles) == 0:
        return {}

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)

    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    degenerate = area <= 1e-20

    with np.errstate(divide="ignore", invalid="ignore"):
        circum = (a * b * c) / (4.0 * area)
        inradius = (2.0 * area) / (a + b + c)
        aspect = circum / (2.0 * inradius)

    good = np.isfinite(aspect) & ~degenerate
    edges = np.concatenate([a, b, c])

    out = {
        "n_degenerate_faces": int(np.sum(degenerate)),
        "edge_len_min":       float(np.min(edges)),
        "edge_len_median":    float(np.median(edges)),
        "edge_len_max":       float(np.max(edges)),
        "face_area_total":    float(np.sum(area)),
    }
    if np.any(good):
        out.update({
            "aspect_median": float(np.median(aspect[good])),
            "aspect_p95":    float(np.percentile(aspect[good], 95)),
            "aspect_max":    float(np.max(aspect[good])),
            "sliver_frac":   float(np.mean(aspect[good] > 10.0)),
        })
    return out


def _sample_surface(mesh, n_samples: int):
    """Uniform-by-area surface sampling, with a deterministic seed where supported."""
    try:
        return mesh.sample_points_uniformly(number_of_points=n_samples, seed=0)
    except TypeError:
        # Older open3d builds have no seed parameter.
        return mesh.sample_points_uniformly(number_of_points=n_samples)


# ── Main entry point ───────────────────────────────────────────────────────────

def validate_mesh(
    mesh,
    pcd,
    spacing: float,
    n_samples: int = 200_000,
    tau_factor: float = 3.0,
    trim_factor: Optional[float] = None,
    check_self_intersections: bool = False,
    self_intersection_max_faces: int = 100_000,
) -> Dict[str, Any]:
    """
    Measure mesh quality against its source point cloud and its own topology.

    Parameters
    ----------
    mesh                        : open3d.geometry.TriangleMesh (final mesh).
    pcd                         : open3d.geometry.PointCloud the mesh was built from.
    spacing                     : Median NN distance of ``pcd`` — the length unit.
    n_samples                   : Surface samples for the accuracy metric.
    tau_factor                  : Coverage tolerance in multiples of spacing.
    trim_factor                 : The trim distance the reconstructor was run
                                  with, in multiples of spacing.  Relaxes the
                                  accuracy gates to match: trimming declares that
                                  surface within k x spacing of the data is
                                  wanted, so failing the mesh for containing it
                                  would contradict the configuration.
    check_self_intersections    : Run the self-intersection test.  Off by default:
                                  it costs ~3 s at 32 k faces and grows sharply
                                  from there, and a Poisson iso-surface cannot
                                  self-intersect by construction, so the test
                                  only earns its keep on BPA/alpha or hole-filled
                                  meshes.
    self_intersection_max_faces : Skip that test above this face count even when
                                  it is enabled.

    Returns
    -------
    Dict with 'topology', 'geometry', 'accuracy', 'completeness', 'checks' and
    an overall 'verdict' (PASS / WARN / FAIL).
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    report: Dict[str, Any] = {"spacing": float(spacing), "tau_factor": float(tau_factor)}
    checks: List[Check] = []

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_verts, n_faces = len(vertices), len(triangles)

    if n_faces == 0:
        report["topology"] = {"n_vertices": n_verts, "n_faces": 0}
        report["checks"] = [Check("mesh_non_empty", FAIL, "mesh has no triangles").as_dict()]
        report["verdict"] = FAIL
        return report

    # ── Topology ───────────────────────────────────────────────────────────
    edge_stats = _edge_topology(triangles)
    is_edge_manifold = bool(mesh.is_edge_manifold())
    is_vertex_manifold = bool(mesh.is_vertex_manifold())
    is_watertight = bool(mesh.is_watertight())
    is_orientable = bool(mesh.is_orientable())

    clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_sizes = np.asarray(cluster_sizes)
    n_components = int(len(cluster_sizes))
    largest_frac = float(cluster_sizes.max() / max(n_faces, 1)) if n_components else 0.0

    euler = n_verts - edge_stats["n_edges"] + n_faces
    topology: Dict[str, Any] = {
        "n_vertices":          n_verts,
        "n_faces":             n_faces,
        "is_edge_manifold":    is_edge_manifold,
        "is_vertex_manifold":  is_vertex_manifold,
        "is_watertight":       is_watertight,
        "is_orientable":       is_orientable,
        "n_components":        n_components,
        "largest_component_frac": largest_frac,
        "euler_characteristic": int(euler),
        **edge_stats,
    }
    if is_watertight and is_edge_manifold and is_orientable:
        topology["genus"] = int((2 - euler) // 2)

    if check_self_intersections and n_faces <= self_intersection_max_faces:
        try:
            pairs = np.asarray(mesh.get_self_intersecting_triangles())
            topology["n_self_intersecting_faces"] = int(np.unique(pairs).size) if pairs.size else 0
        except Exception as exc:
            logger.debug("[MESH CHECK] self-intersection test failed: %s", exc)
    report["topology"] = topology

    # ── Geometry ───────────────────────────────────────────────────────────
    geometry = _triangle_quality(vertices, triangles)
    geometry["surface_area"] = float(mesh.get_surface_area())
    if is_watertight:
        try:
            geometry["volume"] = float(mesh.get_volume())
        except Exception:
            pass
    bbox = mesh.get_axis_aligned_bounding_box()
    geometry["bbox_extent"] = np.asarray(bbox.get_extent()).tolist()
    geometry["has_vertex_colors"] = bool(mesh.has_vertex_colors())
    geometry["has_vertex_normals"] = bool(mesh.has_vertex_normals())
    report["geometry"] = geometry

    # ── Accuracy: mesh surface -> nearest cloud point ─────────────────────
    cloud_pts = np.asarray(pcd.points)
    if len(cloud_pts) > 0:
        n_s = int(min(n_samples, max(10_000, n_faces * 3)))
        sampled = np.asarray(_sample_surface(mesh, n_s).points)
        d_mesh_to_cloud, _ = cKDTree(cloud_pts).query(sampled, k=1, workers=-1)
        report["accuracy"] = _dist_summary(d_mesh_to_cloud, spacing)
        report["accuracy"]["n_samples"] = int(len(sampled))

    # ── Completeness: cloud point -> nearest mesh surface point ───────────
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(cloud_pts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    d_cloud_to_mesh = scene.compute_distance(query).numpy().astype(np.float64)

    tau = tau_factor * spacing
    completeness = _dist_summary(d_cloud_to_mesh, spacing)
    completeness["tau"] = float(tau)
    completeness["coverage"] = float(np.mean(d_cloud_to_mesh <= tau))
    completeness["coverage_1x"] = float(np.mean(d_cloud_to_mesh <= spacing))
    completeness["n_points"] = int(len(cloud_pts))
    report["completeness"] = completeness

    # ── Verdicts ───────────────────────────────────────────────────────────
    acc = report.get("accuracy", {})
    if acc:
        p95 = acc["p95_rel"]
        # The trim step already decided how far from the data a surface may sit.
        # Gating accuracy below that distance would fail a mesh for containing
        # exactly what the configuration asked to keep, so the gates never sit
        # tighter than the configured tolerance.
        pass_gate = ACCURACY_PASS_P95
        warn_gate = ACCURACY_WARN_P95
        if trim_factor is not None and trim_factor > 0:
            pass_gate = max(pass_gate, trim_factor + 1.0)
            warn_gate = max(warn_gate, 2.0 * trim_factor + 1.0)
        report["accuracy"]["pass_gate_rel"] = float(pass_gate)
        report["accuracy"]["warn_gate_rel"] = float(warn_gate)

        level = PASS if p95 <= pass_gate else WARN if p95 <= warn_gate else FAIL
        checks.append(Check(
            "accuracy",
            level,
            f"95 % of the surface lies within {p95:.2f} x spacing of the cloud "
            f"(median {acc['median_rel']:.2f} x; a perfect mesh floors at "
            f"{ACCURACY_FLOOR_P95:.1f} x, PASS gate {pass_gate:.1f} x)",
        ))

    cov = completeness["coverage"]
    level = PASS if cov >= 0.95 else WARN if cov >= 0.85 else FAIL
    checks.append(Check(
        "completeness",
        level,
        f"{cov * 100:.1f} % of cloud points are within {tau_factor:g} x spacing "
        f"of the surface",
    ))

    checks.append(Check(
        "edge_manifold",
        PASS if is_edge_manifold else WARN,
        f"{edge_stats['n_nonmanifold_edges']} non-manifold edge(s)",
    ))
    checks.append(Check(
        "watertight",
        PASS if is_watertight else WARN,
        "closed surface" if is_watertight
        else f"open surface, {edge_stats['n_boundary_edges']} boundary edge(s)",
    ))
    checks.append(Check(
        "single_component",
        PASS if largest_frac >= 0.95 else WARN if largest_frac >= 0.80 else FAIL,
        f"largest of {n_components} component(s) holds {largest_frac * 100:.1f} % of faces",
    ))

    n_degen = geometry.get("n_degenerate_faces", 0)
    checks.append(Check(
        "degenerate_faces",
        PASS if n_degen == 0 else WARN,
        f"{n_degen} zero-area face(s)",
    ))

    if "sliver_frac" in geometry:
        sf = geometry["sliver_frac"]
        checks.append(Check(
            "triangle_shape",
            PASS if sf <= 0.02 else WARN if sf <= 0.10 else FAIL,
            f"{sf * 100:.2f} % slivers (aspect > 10), median aspect "
            f"{geometry['aspect_median']:.2f}",
        ))

    n_si = topology.get("n_self_intersecting_faces")
    if n_si is not None:
        checks.append(Check(
            "self_intersections",
            PASS if n_si == 0 else WARN,
            f"{n_si} self-intersecting face(s)",
        ))

    report["checks"] = [c.as_dict() for c in checks]
    levels = [c.level for c in checks]
    report["verdict"] = FAIL if FAIL in levels else WARN if WARN in levels else PASS
    return report


# ── Reporting ──────────────────────────────────────────────────────────────────

def format_report(report: Dict[str, Any]) -> str:
    """Render a validation report as an aligned, log-friendly text block."""
    t = report.get("topology", {})
    g = report.get("geometry", {})
    a = report.get("accuracy", {})
    c = report.get("completeness", {})
    spacing = report.get("spacing", 0.0)

    lines = [
        "Mesh quality report",
        "-------------------------------------------------------------------",
        f"  Point spacing        {spacing:.6g} scene units",
        "",
        "  Topology",
        f"    Vertices / faces   {t.get('n_vertices', 0):,} / {t.get('n_faces', 0):,}",
        f"    Components         {t.get('n_components', 0)} "
        f"(largest holds {t.get('largest_component_frac', 0) * 100:.1f} %)",
        f"    Edge manifold      {'yes' if t.get('is_edge_manifold') else 'no'}"
        f"   ({t.get('n_nonmanifold_edges', 0)} non-manifold edges)",
        f"    Vertex manifold    {'yes' if t.get('is_vertex_manifold') else 'no'}",
        f"    Watertight         {'yes' if t.get('is_watertight') else 'no'}"
        f"   ({t.get('n_boundary_edges', 0)} boundary edges)",
        f"    Euler chi            {t.get('euler_characteristic', 0)}"
        + (f"   (genus {t['genus']})" if "genus" in t else ""),
    ]
    if "n_self_intersecting_faces" in t:
        lines.append(f"    Self-intersecting  {t['n_self_intersecting_faces']} faces")

    lines += [
        "",
        "  Geometry",
        f"    Surface area       {g.get('surface_area', 0):.6g}",
    ]
    if "volume" in g:
        lines.append(f"    Volume             {g['volume']:.6g}")
    lines += [
        f"    Edge length        min {g.get('edge_len_min', 0):.4g} | "
        f"median {g.get('edge_len_median', 0):.4g} | max {g.get('edge_len_max', 0):.4g}",
        f"    Aspect ratio       median {g.get('aspect_median', 0):.2f} | "
        f"p95 {g.get('aspect_p95', 0):.2f} | slivers "
        f"{g.get('sliver_frac', 0) * 100:.2f} %",
        f"    Degenerate faces   {g.get('n_degenerate_faces', 0)}",
        f"    Colors / normals   {'yes' if g.get('has_vertex_colors') else 'no'} / "
        f"{'yes' if g.get('has_vertex_normals') else 'no'}",
    ]

    if a:
        lines += [
            "",
            "  Accuracy  (mesh surface -> nearest cloud point)",
            f"    Mean / median      {a['mean']:.4g} / {a['median']:.4g}"
            f"   ({a['mean_rel']:.2f} / {a['median_rel']:.2f} x spacing)",
            f"    RMS / p95          {a['rms']:.4g} / {a['p95']:.4g}"
            f"   (p95 = {a['p95_rel']:.2f} x spacing)",
            f"    Samples            {a.get('n_samples', 0):,}",
        ]
    if c:
        lines += [
            "",
            "  Completeness  (cloud point -> nearest mesh surface)",
            f"    Mean / median      {c['mean']:.4g} / {c['median']:.4g}"
            f"   ({c['mean_rel']:.2f} / {c['median_rel']:.2f} x spacing)",
            f"    Coverage @ {report.get('tau_factor', 3):g}x      "
            f"{c['coverage'] * 100:.2f} %   (@ 1x spacing: {c['coverage_1x'] * 100:.2f} %)",
            f"    Cloud points       {c.get('n_points', 0):,}",
        ]

    lines += ["", "  Checks"]
    for chk in report.get("checks", []):
        lines.append(f"    [{chk['level']:4s}] {chk['name']:<20s} {chk['detail']}")
    lines += [
        "-------------------------------------------------------------------",
        f"  Overall verdict: {report.get('verdict', '?')}",
    ]
    return "\n".join(lines)


def write_report(report: Dict[str, Any], path: "str | Path") -> Optional[Path]:
    """Write the report as JSON; returns the path, or None if writing failed."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=False)
        return path
    except OSError as exc:
        logger.warning("[MESH CHECK] Could not write quality report to %s: %s", path, exc)
        return None
