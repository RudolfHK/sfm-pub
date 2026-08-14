"""
Tests for the mesh reconstruction stage (sfm/mesh/).

Synthetic geometry with analytically known properties is used throughout: a unit
sphere has surface area 4π, volume 4π/3, genus 0, no boundary edges, and every
outward normal is parallel to its position vector.  That makes it possible to
test the *metrics themselves* and not just that the code runs — a validation
module that reports plausible-looking numbers without being checked against known
answers is worth very little.

Run with:  pytest tests/test_mesh.py -v
"""

import argparse
import math

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="open3d is required for the mesh stage")

from sfm.mesh.cleaning import clean_mesh  # noqa: E402
from sfm.mesh.pipeline import QUALITY_PRESETS, MeshPipeline  # noqa: E402
from sfm.mesh.pointcloud_prep import (  # noqa: E402
    _orient_normals_towards_cameras,
    _voxel_size_for_target,
    estimate_spacing,
    prepare_point_cloud,
)
from sfm.mesh.reconstruction import reconstruct_surface  # noqa: E402
from sfm.mesh.validation import (  # noqa: E402
    ACCURACY_FLOOR_P95,
    _edge_topology,
    _triangle_quality,
    validate_mesh,
)

SPHERE_RADIUS = 1.0
N_SPHERE_POINTS = 30_000


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _sample(mesh, n, seed=0):
    """Uniform surface sampling; open3d only grew a `seed` parameter after 0.19."""
    try:
        return mesh.sample_points_uniformly(number_of_points=n, seed=seed)
    except TypeError:
        return mesh.sample_points_uniformly(number_of_points=n)


def _sphere_cloud(n=N_SPHERE_POINTS, radius=SPHERE_RADIUS, seed=0):
    """A coloured point cloud sampled uniformly over a sphere of known radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=60)
    mesh.compute_vertex_normals()
    pcd = _sample(mesh, n, seed)
    pts = np.asarray(pcd.points)
    colors = (pts - pts.min(0)) / np.ptp(pts, axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _camera_ring(n=12, radius=4.0):
    """Camera centres on a ring well outside the object."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)])


def _mesh_args(**overrides):
    """A Namespace with the same fields MeshPipeline reads from the CLI."""
    base = dict(
        mesh_method="poisson",
        mesh_quality="low",
        mesh_depth=None,
        mesh_trim=None,
        mesh_target_points=None,
        mesh_no_clean=False,
        mesh_fill_holes=True,
        mesh_hole_size=None,
        mesh_keep_largest=False,
        mesh_self_intersections=False,
        mesh_smooth=False,
        mesh_smooth_iterations=5,
        mesh_decimate=False,
        mesh_decimate_target=100_000,
        mesh_no_texture=False,
        mesh_keep_pointcloud=True,
        mesh_no_validate=False,
        mesh_preview=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def sphere_ply(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "sphere.ply"
    assert o3d.io.write_point_cloud(str(path), _sphere_cloud())
    return path


# ── Spacing and downsampling ───────────────────────────────────────────────────

def test_estimate_spacing_matches_regular_grid():
    """On a grid with known pitch, the median NN distance is that pitch."""
    g = np.arange(20) * 0.05
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    assert estimate_spacing(pts) == pytest.approx(0.05, rel=1e-6)


def test_estimate_spacing_handles_degenerate_input():
    assert estimate_spacing(np.zeros((1, 3))) == 0.0
    assert estimate_spacing(np.zeros((10, 3))) == 0.0


def test_voxel_size_hits_target_count():
    """
    Bisection must land near the target; the old volume-based formula could not,
    because points on a 2-manifold do not fill the bounding box.
    """
    pcd = _sphere_cloud(n=50_000)
    target = 10_000
    spacing = estimate_spacing(np.asarray(pcd.points))

    voxel = _voxel_size_for_target(pcd, target, spacing)
    assert voxel is not None
    n_after = len(pcd.voxel_down_sample(voxel_size=voxel).points)
    assert target * 0.7 <= n_after <= target * 1.3, f"got {n_after}, target {target}"


def test_voxel_size_returns_none_when_below_target():
    pcd = _sphere_cloud(n=1_000)
    assert _voxel_size_for_target(pcd, 10_000, estimate_spacing(np.asarray(pcd.points))) is None


# ── Normal orientation ─────────────────────────────────────────────────────────

def test_camera_orientation_makes_sphere_normals_outward():
    """
    Every normal on a sphere seen from outside must end up pointing outward,
    i.e. positively along its own position vector.

    The poles are the hard case: with cameras on an equatorial ring, the view ray
    grazes the surface there and sign(n · (c - p)) is negative for everything
    within ~11° of a pole.  Getting those right is exactly what the grazing-angle
    handling is for, so this asserts on *all* normals rather than most of them.
    """
    pcd = _sphere_cloud(n=5_000)
    pts = np.asarray(pcd.points)

    rng = np.random.default_rng(0)
    flipped = np.asarray(pcd.normals) * rng.choice([-1.0, 1.0], size=(len(pts), 1))
    pcd.normals = o3d.utility.Vector3dVector(flipped)

    frac, grazing = _orient_normals_towards_cameras(pcd, _camera_ring(radius=5.0))

    outward = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", np.asarray(pcd.normals), outward)
    assert np.all(dots > 0), (
        f"{int(np.sum(dots <= 0))} normals still point into the sphere"
    )
    assert 0.3 < frac < 0.7, "about half of randomly signed normals should flip"
    assert 0.0 < grazing < 0.2, "the grazing band should be a small minority"


# ── Preparation ────────────────────────────────────────────────────────────────

def test_prepare_point_cloud_produces_oriented_normals(sphere_ply):
    prepared = prepare_point_cloud(
        sphere_ply, QUALITY_PRESETS["low"], "low", camera_centers=_camera_ring()
    )
    assert prepared is not None
    assert prepared.pcd.has_normals()
    assert prepared.pcd.has_colors(), "colours must survive preparation"
    assert prepared.spacing > 0
    assert prepared.stats["normal_orientation"] == "cameras"
    # A well-sampled sphere is uniform, so the outlier filters must barely touch it.
    assert prepared.stats["after_ror"] > 0.9 * prepared.stats["input"]


def test_ror_removal_is_capped():
    """
    The radius filter must never delete more than its configured share of the
    cloud. A cloud with a genuinely sparse half used to lose ~36 % of its points.
    """
    rng = np.random.default_rng(0)
    dense = rng.uniform(0, 1, size=(20_000, 3)) * np.array([1.0, 1.0, 0.01])
    sparse = rng.uniform(2, 4, size=(2_000, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([dense, sparse])))

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mixed.ply"
        o3d.io.write_point_cloud(str(path), pcd)
        preset = dict(QUALITY_PRESETS["low"])
        preset["ror_max_removal"] = 0.10
        prepared = prepare_point_cloud(path, preset, "low", camera_centers=_camera_ring())

    assert prepared is not None
    removed_frac = prepared.stats["ror_removed"] / prepared.stats["after_sor"]
    assert removed_frac <= 0.101, f"ROR removed {removed_frac:.1%}, cap was 10%"


def test_extent_filter_drops_a_distant_outlier_blob(tmp_path):
    """
    A compact cluster far from the object passes SOR and ROR — its members are
    each other's neighbours — but inflates the bounding box, which is what caps
    Poisson's resolution.  On a real diverged cloud this cost a 5.9x bbox shrink.
    """
    rng = np.random.default_rng(0)
    body = np.asarray(_sphere_cloud(n=20_000).points)
    blob = rng.normal(0, 0.05, size=(400, 3)) + np.array([50.0, 0.0, 0.0])

    path = tmp_path / "blob.ply"
    o3d.io.write_point_cloud(
        str(path),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([body, blob]))),
    )

    prepared = prepare_point_cloud(
        path, QUALITY_PRESETS["low"], "low", camera_centers=_camera_ring()
    )
    assert prepared is not None
    assert prepared.stats["extent_filtered"], "the distant blob was not removed"
    assert prepared.stats["extent_shrink"] > 2.0

    kept = np.asarray(prepared.pcd.points)
    assert kept[:, 0].max() < 10.0, "blob survived the extent filter"
    assert len(kept) > 0.9 * len(body), "extent filter ate the object too"


def test_extent_filter_leaves_a_well_formed_cloud_alone(sphere_ply):
    prepared = prepare_point_cloud(
        sphere_ply, QUALITY_PRESETS["low"], "low", camera_centers=_camera_ring()
    )
    assert prepared is not None
    assert not prepared.stats["extent_filtered"]


def test_poisson_depth_adapts_to_bounding_box_scale():
    """
    Poisson resolution is bbox_diagonal / 2^depth, so a fixed depth means
    different detail on different clouds.  The depth must rise when the box is
    large relative to the point spacing, and stay put when it is not.
    """
    pcd = _sphere_cloud(n=20_000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    _orient_normals_towards_cameras(pcd, _camera_ring(radius=5.0))
    spacing = estimate_spacing(np.asarray(pcd.points))

    preset = dict(QUALITY_PRESETS["low"])
    _, stats = reconstruct_surface(pcd, spacing, "poisson", preset)
    assert stats["depth"] >= preset["poisson_depth"], "depth must never be lowered"
    assert stats["cell_over_spacing"] < 4.0, "cell should resolve the point spacing"

    # Ten times the spacing with the same extent => a coarser cloud needs less depth.
    sparse = pcd.random_down_sample(0.02)
    sparse.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    _orient_normals_towards_cameras(sparse, _camera_ring(radius=5.0))
    sparse_spacing = estimate_spacing(np.asarray(sparse.points))
    _, sparse_stats = reconstruct_surface(sparse, sparse_spacing, "poisson", preset)
    assert sparse_stats["depth"] <= stats["depth"]


def test_explicit_depth_overrides_auto_selection():
    pcd = _sphere_cloud(n=5_000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    _orient_normals_towards_cameras(pcd, _camera_ring(radius=5.0))
    spacing = estimate_spacing(np.asarray(pcd.points))

    _, stats = reconstruct_surface(
        pcd, spacing, "poisson", QUALITY_PRESETS["low"], depth_override=7
    )
    assert stats["depth"] == 7


def test_prepare_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        prepare_point_cloud("does_not_exist.ply", QUALITY_PRESETS["low"], "low")


# ── Validation metrics, checked against known geometry ─────────────────────────

def test_edge_topology_on_closed_and_open_meshes():
    closed = o3d.geometry.TriangleMesh.create_sphere(resolution=10)
    stats = _edge_topology(np.asarray(closed.triangles))
    assert stats["n_boundary_edges"] == 0
    assert stats["n_nonmanifold_edges"] == 0

    # Euler characteristic of a closed genus-0 surface is 2.
    assert len(closed.vertices) - stats["n_edges"] + len(closed.triangles) == 2

    open_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
    )
    assert _edge_topology(np.asarray(open_mesh.triangles))["n_boundary_edges"] == 3


def test_triangle_quality_aspect_ratio_of_equilateral_is_one():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    q = _triangle_quality(verts, tris)
    assert q["aspect_median"] == pytest.approx(1.0, rel=1e-6)
    assert q["n_degenerate_faces"] == 0


def test_triangle_quality_flags_slivers_and_degenerates():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-6, 0], [2, 0, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)   # sliver + zero-area
    q = _triangle_quality(verts, tris)
    assert q["n_degenerate_faces"] == 1
    assert q["aspect_max"] > 10.0


def test_validate_mesh_recovers_sphere_ground_truth():
    """
    Validation run on an exact sphere mesh with its own surface samples must
    report watertight, genus 0, full coverage and near-zero accuracy error.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS, resolution=40)
    mesh.compute_vertex_normals()
    pcd = _sample(mesh, 20_000)
    spacing = estimate_spacing(np.asarray(pcd.points))

    report = validate_mesh(
        mesh, pcd, spacing=spacing, n_samples=20_000, check_self_intersections=True
    )

    t, g = report["topology"], report["geometry"]
    assert t["is_watertight"] and t["is_edge_manifold"] and t["is_vertex_manifold"]
    assert t["n_components"] == 1
    assert t["euler_characteristic"] == 2
    assert t["genus"] == 0
    assert t["n_boundary_edges"] == 0

    # A resolution-40 triangulation slightly under-cuts the true sphere.
    assert g["surface_area"] == pytest.approx(4 * math.pi * SPHERE_RADIUS ** 2, rel=0.02)
    assert g["volume"] == pytest.approx(4 / 3 * math.pi * SPHERE_RADIUS ** 3, rel=0.03)

    assert t["n_self_intersecting_faces"] == 0

    assert report["completeness"]["coverage"] == pytest.approx(1.0, abs=1e-6)

    # The accuracy metric cannot reach zero: it measures the nearest-neighbour
    # distance between two independent samples of the same surface, which floors
    # at median ~1.0x and p95 ~2.1x spacing (see validation.ACCURACY_FLOOR_P95).
    # An exact mesh should sit at that floor, not below it.
    assert report["accuracy"]["median_rel"] == pytest.approx(1.0, abs=0.15)
    assert report["accuracy"]["p95_rel"] == pytest.approx(ACCURACY_FLOOR_P95, abs=0.3)
    assert report["verdict"] == "PASS"


def test_self_intersection_check_is_opt_in():
    """
    The test costs ~3 s at 32 k faces and grows sharply, so it must stay off
    unless asked for — and must actually run when it is.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    pcd = _sample(mesh, 2_000)
    spacing = estimate_spacing(np.asarray(pcd.points))

    off = validate_mesh(mesh, pcd, spacing=spacing, n_samples=2_000)
    assert "n_self_intersecting_faces" not in off["topology"]

    on = validate_mesh(
        mesh, pcd, spacing=spacing, n_samples=2_000, check_self_intersections=True
    )
    assert on["topology"]["n_self_intersecting_faces"] == 0


def test_accuracy_gate_respects_the_configured_trim_distance():
    """
    Trimming declares how far from the data a surface may sit.  A gate tighter
    than that would fail a mesh for containing exactly what it was told to keep.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    pcd = _sample(mesh, 5_000)
    spacing = estimate_spacing(np.asarray(pcd.points))

    strict = validate_mesh(mesh, pcd, spacing=spacing, n_samples=5_000)
    loose = validate_mesh(mesh, pcd, spacing=spacing, n_samples=5_000, trim_factor=6.0)

    assert loose["accuracy"]["pass_gate_rel"] > strict["accuracy"]["pass_gate_rel"]
    assert loose["accuracy"]["pass_gate_rel"] == pytest.approx(7.0)
    # A tolerance smaller than the default must not tighten the gate.
    tight = validate_mesh(mesh, pcd, spacing=spacing, n_samples=5_000, trim_factor=1.0)
    assert tight["accuracy"]["pass_gate_rel"] == strict["accuracy"]["pass_gate_rel"]


def test_quality_presets_form_a_monotone_ladder():
    """Higher presets must resolve finer: cell size per spacing strictly decreases."""
    factors = [QUALITY_PRESETS[q]["poisson_cell_factor"] for q in ("low", "medium", "high", "ultra")]
    assert factors == sorted(factors, reverse=True), factors

    depths = [QUALITY_PRESETS[q]["poisson_depth"] for q in ("low", "medium", "high", "ultra")]
    assert depths == sorted(depths), depths

    trims = [QUALITY_PRESETS[q]["trim_distance_factor"] for q in ("low", "medium", "high", "ultra")]
    assert trims == sorted(trims, reverse=True), trims


def test_validate_mesh_flags_empty_mesh():
    report = validate_mesh(
        o3d.geometry.TriangleMesh(), _sphere_cloud(n=100), spacing=0.1
    )
    assert report["verdict"] == "FAIL"


# ── Cleaning ───────────────────────────────────────────────────────────────────

def test_hole_filling_actually_closes_holes():
    """
    Regression test: open3d exposes fill_holes only on the tensor API, so the
    legacy call raised AttributeError and hole filling silently never ran.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    tris = np.asarray(mesh.triangles)
    keep = np.ones(len(tris), dtype=bool)
    keep[:12] = False                                  # punch a hole
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()

    before = _edge_topology(np.asarray(mesh.triangles))["n_boundary_edges"]
    assert before > 0 and not mesh.is_watertight()

    cleaned, stats = clean_mesh(
        mesh, QUALITY_PRESETS["low"], spacing=0.1, fill_holes=True, hole_size=10.0
    )
    after = _edge_topology(np.asarray(cleaned.triangles))["n_boundary_edges"]
    assert after < before, f"hole filling did nothing ({before} -> {after} boundary edges)"


def test_cleaning_keeps_largest_component():
    big = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    small = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=4)
    small.translate((5.0, 0, 0))
    combined = big + small

    cleaned, stats = clean_mesh(
        combined, QUALITY_PRESETS["low"], spacing=0.05,
        fill_holes=False, keep_largest_only=True,
    )
    assert stats["n_components_kept"] == 1
    assert len(cleaned.triangles) == len(big.triangles)


def test_cleaning_never_empties_a_tiny_mesh():
    """The largest component survives even when it is below every fixed floor."""
    tiny = o3d.geometry.TriangleMesh.create_tetrahedron()
    cleaned, stats = clean_mesh(tiny, QUALITY_PRESETS["low"], spacing=0.1, fill_holes=False)
    assert len(cleaned.triangles) > 0


def test_cleaning_handles_empty_mesh():
    cleaned, stats = clean_mesh(
        o3d.geometry.TriangleMesh(), QUALITY_PRESETS["low"], spacing=0.1
    )
    assert stats["faces_out"] == 0


# ── Reconstruction ─────────────────────────────────────────────────────────────

def test_poisson_trimming_removes_unsupported_geometry():
    """
    Poisson closes the surface over gaps.  With half the sphere removed, the
    distance trim must delete the invented half rather than keep the bubble.
    """
    pcd = _sphere_cloud(n=20_000)
    pts = np.asarray(pcd.points)
    keep = pts[:, 2] > -0.2
    half = pcd.select_by_index(np.flatnonzero(keep).tolist())
    half.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    _orient_normals_towards_cameras(half, _camera_ring(radius=5.0))

    spacing = estimate_spacing(np.asarray(half.points))
    preset = dict(QUALITY_PRESETS["low"])

    trimmed, stats = reconstruct_surface(half, spacing, "poisson", preset)
    assert stats["removed_distance"] > 0, "distance trim removed nothing"

    # Nothing should survive far below the cut plane.
    zs = np.asarray(trimmed.vertices)[:, 2]
    assert zs.min() > -0.6, f"phantom geometry remains at z={zs.min():.3f}"


def test_poisson_requires_normals():
    pcd = _sphere_cloud(n=1_000)
    pcd.normals = o3d.utility.Vector3dVector(np.empty((0, 3)))
    with pytest.raises(ValueError, match="normals"):
        reconstruct_surface(pcd, 0.05, "poisson", QUALITY_PRESETS["low"])


# ── End-to-end ─────────────────────────────────────────────────────────────────

def test_end_to_end_sphere_mesh_is_correct(sphere_ply, tmp_path):
    """The full pipeline on a sphere must recover its area, volume and topology."""
    out = tmp_path / "sphere_mesh.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())

    assert result.success, result.error
    assert out.exists()
    assert result.face_count > 1_000

    q = result.stats["quality"]
    assert q["topology"]["n_components"] == 1
    assert q["geometry"]["surface_area"] == pytest.approx(
        4 * math.pi * SPHERE_RADIUS ** 2, rel=0.10
    )
    assert q["completeness"]["coverage"] > 0.98
    assert q["accuracy"]["p95_rel"] < 5.0
    assert q["verdict"] in ("PASS", "WARN")


def test_pipeline_saves_prepared_cloud_by_default(sphere_ply, tmp_path):
    """The point cloud must remain a first-class output of the mesh stage."""
    out = tmp_path / "keep.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())

    assert result.cloud_path is not None and result.cloud_path.exists()
    cloud = o3d.io.read_point_cloud(str(result.cloud_path))
    assert len(cloud.points) > 0
    assert cloud.has_normals(), "the saved cloud must carry the normals Poisson used"

    # …and the original input is untouched.
    assert sphere_ply.exists()
    assert len(o3d.io.read_point_cloud(str(sphere_ply)).points) == N_SPHERE_POINTS


def test_pipeline_writes_quality_report(sphere_ply, tmp_path):
    import json

    out = tmp_path / "report.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())

    assert result.report_path is not None and result.report_path.exists()
    with open(result.report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    assert {"topology", "geometry", "accuracy", "completeness", "checks", "verdict"} <= set(report)


def test_pipeline_produces_colored_mesh(sphere_ply, tmp_path):
    out = tmp_path / "colored.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())
    assert result.stats["post"]["colored"]
    assert o3d.io.read_triangle_mesh(str(out)).has_vertex_colors()


def test_pipeline_decimation_reduces_faces(sphere_ply, tmp_path):
    out = tmp_path / "decimated.ply"
    args = _mesh_args(mesh_decimate=True, mesh_decimate_target=2_000)
    result = MeshPipeline(args).run(sphere_ply, out, camera_centers=_camera_ring())
    assert result.success, result.error
    assert result.face_count <= 2_200
    # Colour transfer runs after decimation, so colours must survive it.
    assert result.stats["post"]["colored"]


def test_pipeline_reports_failure_for_missing_input(tmp_path):
    result = MeshPipeline(_mesh_args()).run(tmp_path / "nope.ply", tmp_path / "out.ply")
    assert not result.success
    assert "not found" in (result.error or "").lower()


def test_pipeline_records_stage_timings(sphere_ply, tmp_path):
    out = tmp_path / "timed.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())
    timings = result.stats["timings"]
    assert {"prepare", "reconstruct", "clean", "postprocess", "export", "total"} <= set(timings)
    assert timings["total"] >= max(v for k, v in timings.items() if k != "total")


def test_visualizer_stats_contract_is_preserved(sphere_ply, tmp_path):
    """SfMVisualizer._render_mesh_cleaning_stats reads these exact keys."""
    out = tmp_path / "viz.ply"
    result = MeshPipeline(_mesh_args()).run(sphere_ply, out, camera_centers=_camera_ring())
    clean = result.stats["clean"]
    for key in ("faces_in", "after_dedup", "after_manifold", "after_components",
                "after_fill", "faces_out"):
        assert key in clean, f"visualizer key {key!r} missing from clean stats"


# ── CLI ────────────────────────────────────────────────────────────────────────

def test_cli_namespace_covers_pipeline_options(sphere_ply, tmp_path):
    """Whatever the CLI parses must be enough to drive the pipeline."""
    from sfm.mesh.__main__ import build_parser

    out = tmp_path / "cli.ply"
    args = build_parser().parse_args([str(sphere_ply), "-o", str(out), "--quality", "low"])
    result = MeshPipeline(args).run(args.pointcloud, out)
    assert result.success, result.error
    assert out.exists()
