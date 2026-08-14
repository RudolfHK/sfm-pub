"""
sfm/mesh/pointcloud_prep.py — Point cloud preparation for surface reconstruction.

Pipeline (in order):
  3a. Load, drop non-finite / duplicate points, validate
  3b. Estimate point spacing (median nearest-neighbour distance)
  3c. Statistical outlier removal (SOR)
  3d. Radius outlier removal (ROR) with a bounded removal fraction
  3e. Voxel downsampling to the preset target count (bisection on voxel size)
  3f. Normal estimation + camera-aware orientation
  3g. Final spacing estimate + summary

Length scale
------------
Every radius, trim distance and threshold downstream is expressed as a multiple
of ``spacing`` — the median nearest-neighbour distance of the cloud.  It is the
only intrinsic length scale a point cloud has: absolute constants break as soon
as the reconstruction scale changes (SfM output is scale-free), and bounding-box
fractions break as soon as the object is not roughly cubical.  Spacing is
measured once here and threaded through the rest of the mesh pipeline.

Normal orientation
------------------
Poisson reconstruction is only as good as its normals, and it needs them
*consistently oriented* (all pointing outwards).  The generic solution is
``orient_normals_consistent_tangent_plane``, an MST propagation that is both slow
and prone to flipping whole regions on noisy MVS clouds.  This pipeline does not
need it: SfM knows where the cameras were, and a surface point was necessarily
seen from the camera side.  Orienting each normal towards its nearest camera
centre is O(N log C), deterministic, and far more robust.  The MST heuristic is
kept as a fallback for clouds imported without camera information.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Prepared-cloud container ───────────────────────────────────────────────────

@dataclass
class PreparedCloud:
    """Point cloud ready for surface reconstruction, plus its length scale."""

    pcd: Any                                     # open3d.geometry.PointCloud
    spacing: float                               # median NN distance (scene units)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.pcd.points)


# ── Spacing estimation ─────────────────────────────────────────────────────────

def estimate_spacing(
    points: np.ndarray,
    sample: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Median nearest-neighbour distance of a point set.

    For clouds larger than ``sample`` the query set is randomly subsampled — the
    KD-tree is still built over every point, so each reported distance is a true
    nearest-neighbour distance; only the number of *queries* is bounded.  The
    median of a 50 000-point sample is more than accurate enough for a scale
    estimate and keeps preparation from being dominated by this one measurement.

    Returns 0.0 for degenerate inputs (fewer than two distinct points).
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n < 2:
        return 0.0

    tree = cKDTree(points)
    if n > sample:
        rng = np.random.default_rng(seed)
        query = points[rng.choice(n, sample, replace=False)]
    else:
        query = points

    dists, _ = tree.query(query, k=2, workers=-1)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 0.0


def _kth_neighbour_distance(points: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour (self excluded)."""
    from scipy.spatial import cKDTree

    n = len(points)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return np.zeros(n, dtype=np.float64)

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k_eff + 1, workers=-1)
    return np.ascontiguousarray(dists[:, -1])


# ── Voxel size search ──────────────────────────────────────────────────────────

def _voxel_size_for_target(
    pcd,
    target: int,
    spacing: float,
    tolerance: float = 0.05,
    max_iter: int = 16,
) -> Optional[float]:
    """
    Find a voxel size whose downsample lands within ``tolerance`` of ``target``.

    Bisection on the actual downsampled count, because there is no usable closed
    form: points from MVS lie on a 2-manifold embedded in 3-D, so the obvious
    ``(bbox_volume / target) ** (1/3)`` estimate models a solid and mispredicts
    the resulting count by orders of magnitude on anything flat or shell-like.
    Each probe is a single C++ voxel hash, so ~16 of them stay cheap even for
    multi-million-point clouds.

    Returns None when the cloud is already at or below the target.
    """
    n = len(pcd.points)
    if n <= target:
        return None

    bbox_diag = float(np.linalg.norm(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())))
    if bbox_diag <= 0:
        return None

    lo = max(spacing * 0.5, bbox_diag * 1e-6)   # count(lo) should be >= target
    hi = max(spacing * 4.0, lo * 2.0)

    # Grow the upper bound until it actually undershoots the target.
    for _ in range(max_iter):
        if len(pcd.voxel_down_sample(voxel_size=hi).points) <= target:
            break
        hi *= 2.0
        if hi > bbox_diag:
            hi = bbox_diag
            break
    else:
        return hi

    best = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        count = len(pcd.voxel_down_sample(voxel_size=mid).points)
        if count > target:
            lo = mid
        else:
            hi = mid
            best = mid
            if count >= target * (1.0 - tolerance):
                break
    return best


# ── Normal orientation ─────────────────────────────────────────────────────────

def _orient_normals_towards_cameras(
    pcd,
    camera_centers: np.ndarray,
    grazing_cos: float = 0.2,
) -> Tuple[float, float]:
    """
    Flip normals to face the cameras, resolving grazing-angle points separately.

    A point can only have been reconstructed from images that saw it, so the
    outward side of the surface is the side the cameras are on, and
    ``sign(n · (c - p))`` recovers the orientation directly.

    That rule is decisive only when the surface faces the camera reasonably
    head-on.  Where the viewing ray grazes the surface, ``n · (c - p)`` is near
    zero and its *sign* is dominated by geometry rather than visibility, so it
    can be wrong over a whole contiguous patch — for a sphere viewed from a ring
    of cameras, everything within ~11° of the poles is oriented inwards by the
    naive rule.  Those points are therefore treated as undecided and instead
    aligned with their nearest confidently-oriented neighbour, which propagates
    the correct sign into the grazing band.

    Returns
    -------
    (fraction flipped, fraction resolved by neighbour propagation).  A flipped
    fraction near 0.5 is expected and healthy — PCA normals come out with an
    arbitrary sign.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals).copy()
    if len(pts) == 0 or len(normals) != len(pts):
        return 0.0, 0.0

    _, nearest = cKDTree(camera_centers).query(pts, k=1, workers=-1)
    view = camera_centers[nearest] - pts                     # point → camera
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    unit_n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    cos = np.einsum("ij,ij->i", unit_n, view)
    flip = cos < 0.0
    normals[flip] *= -1.0

    # Grazing-angle points: let the confident majority decide their sign.
    confident = np.abs(cos) >= grazing_cos
    n_uncertain = int((~confident).sum())
    if 0 < n_uncertain < len(pts) and confident.any():
        conf_idx = np.flatnonzero(confident)
        unc_idx = np.flatnonzero(~confident)
        _, nn = cKDTree(pts[conf_idx]).query(pts[unc_idx], k=1, workers=-1)
        ref = normals[conf_idx[nn]]
        disagree = np.einsum("ij,ij->i", normals[unc_idx], ref) < 0.0
        normals[unc_idx[disagree]] *= -1.0

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return float(flip.mean()), float(n_uncertain / len(pts))


# ── Preparation ────────────────────────────────────────────────────────────────

def prepare_point_cloud(
    ply_path: "str | Path",
    preset: Dict[str, Any],
    quality: str,
    camera_centers: Optional[np.ndarray] = None,
) -> Optional[PreparedCloud]:
    """
    Load a PLY point cloud and run the full preparation pipeline.

    Parameters
    ----------
    ply_path       : Path to the input PLY file (XYZ + optional RGB/normals).
    preset         : Quality preset dict from pipeline.QUALITY_PRESETS.
    quality        : Preset name string (used only for logging).
    camera_centers : (C, 3) world-space camera centres from the SfM stage.  When
                     given, normals are oriented towards the nearest camera;
                     otherwise the tangent-plane MST fallback is used.

    Returns
    -------
    PreparedCloud with an oriented-normal point cloud, its spacing and per-stage
    statistics — or None if the cloud is empty or every point was filtered out.
    """
    import open3d as o3d

    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    # ── 3a: Load and validate ──────────────────────────────────────────────
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_raw = len(pcd.points)

    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    n_finite = len(pcd.points)
    if n_finite < n_raw:
        logger.warning(
            "[MESH PREP] Dropped %d non-finite point(s) (NaN/Inf)", n_raw - n_finite
        )

    has_normals_in = pcd.has_normals()
    has_colors_in = pcd.has_colors()

    logger.info(
        "[MESH PREP] Loaded %s: %d points, has_normals=%s, has_colors=%s",
        ply_path.name, n_finite, has_normals_in, has_colors_in,
    )

    if n_finite == 0:
        logger.error("[MESH PREP] Point cloud is empty — nothing to mesh")
        return None
    if n_finite < 1000:
        logger.warning(
            "[MESH PREP] Only %d points — mesh quality will be poor (recommend >= 1000). "
            "Run with --dense to densify before meshing.",
            n_finite,
        )

    stats: Dict[str, Any] = {
        "input": n_finite,
        "input_raw": n_raw,
        "has_normals": has_normals_in,
        "has_colors": has_colors_in,
    }

    # ── 3b: Spacing of the raw cloud ───────────────────────────────────────
    points = np.asarray(pcd.points)
    spacing = estimate_spacing(points)
    if spacing <= 0.0:
        logger.error(
            "[MESH PREP] Degenerate cloud: all points coincide (spacing = 0) — aborting prep"
        )
        return None
    stats["spacing_input"] = spacing
    logger.info("[MESH PREP] Point spacing (median NN distance): %.6g scene units", spacing)

    # ── 3c: Statistical outlier removal ────────────────────────────────────
    n_before = len(pcd.points)
    sor_neighbors = min(int(preset["sor_neighbors"]), max(n_before - 1, 1))
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=sor_neighbors,
        std_ratio=float(preset["sor_std_ratio"]),
    )
    n_after_sor = len(pcd.points)
    logger.info(
        "[MESH PREP] SOR (k=%d, std=%.1f): removed %d points (%.1f%%)",
        sor_neighbors, preset["sor_std_ratio"],
        n_before - n_after_sor,
        100.0 * (n_before - n_after_sor) / max(n_before, 1),
    )
    stats["after_sor"] = n_after_sor

    if n_after_sor == 0:
        logger.error("[MESH PREP] SOR removed every point — aborting prep")
        return None

    # ── 3d: Radius outlier removal, with a bounded removal fraction ────────
    #
    # Implemented as a k-th-nearest-neighbour distance threshold, which is
    # mathematically identical to a radius filter (a point has >= k neighbours
    # within r exactly when its k-th NN distance is <= r) but exposes the whole
    # distribution.  That lets the radius be clamped so the filter can never
    # delete more than `ror_max_removal` of the cloud: the previous fixed-radius
    # version silently removed 36 % of a real dense cloud, which is data loss,
    # not outlier rejection.
    points = np.asarray(pcd.points)
    ror_k = int(preset["ror_min_neighbors"])
    max_removal = float(preset["ror_max_removal"])
    n_ror_removed = 0

    if n_after_sor > ror_k + 1:
        d_k = _kth_neighbour_distance(points, ror_k)
        radius = float(preset["ror_multiplier"]) * spacing
        frac = float(np.mean(d_k > radius))

        if frac > max_removal:
            clamped = float(np.quantile(d_k, 1.0 - max_removal))
            logger.warning(
                "[MESH PREP] ROR radius %.6g would remove %.1f%% of the cloud "
                "(cap is %.0f%%) — relaxing radius to %.6g (%.1f x spacing). "
                "A large sparse fraction usually means uneven MVS coverage.",
                radius, frac * 100, max_removal * 100, clamped, clamped / spacing,
            )
            radius = clamped

        keep = d_k <= radius
        n_ror_removed = int((~keep).sum())
        if n_ror_removed:
            pcd = pcd.select_by_index(np.flatnonzero(keep).tolist())
        logger.info(
            "[MESH PREP] ROR (k=%d, r=%.6g = %.1f x spacing): removed %d points (%.1f%%)",
            ror_k, radius, radius / spacing, n_ror_removed,
            100.0 * n_ror_removed / max(n_after_sor, 1),
        )
    else:
        logger.info("[MESH PREP] ROR skipped: too few points (%d)", n_after_sor)

    n_after_ror = len(pcd.points)
    stats["after_ror"] = n_after_ror
    stats["ror_removed"] = n_ror_removed

    if n_after_ror == 0:
        logger.error("[MESH PREP] All points removed by outlier filters — aborting prep")
        return None

    # ── 3d-bis: Robust extent filter ───────────────────────────────────────
    #
    # SOR and ROR are *local density* tests: a compact cluster of points a long
    # way from the object passes both, because its members are each other's
    # neighbours.  Such a cluster is harmless for the statistics but ruinous for
    # Poisson, whose resolution is bounded by the bounding-box diagonal — one
    # distant blob divides the achievable detail everywhere else.
    #
    # This drops the outermost `extent_percentile` of points by distance from the
    # median centre, but only when doing so shrinks the bounding box
    # substantially; on a well-formed cloud the test simply does not fire.
    pcd, extent_stats = _apply_extent_filter(pcd, preset)
    stats.update(extent_stats)

    if len(pcd.points) == 0:
        logger.error("[MESH PREP] Extent filter removed every point — aborting prep")
        return None

    # ── 3e: Voxel downsampling ─────────────────────────────────────────────
    #
    # Runs before normal estimation, not after: normals cost far more per point
    # than a voxel hash, and estimating them on the decimated cloud both saves
    # the wasted work and produces normals at the scale Poisson will actually
    # see.  voxel_down_sample averages colours and normals of the points it
    # merges, so nothing is lost when the input already carries them.
    target_points = preset["target_points"]
    n_before_ds = len(pcd.points)

    if target_points is not None and n_before_ds > target_points:
        voxel_size = _voxel_size_for_target(pcd, int(target_points), spacing)
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            logger.info(
                "[MESH PREP] Voxel downsampled: %d -> %d points "
                "(voxel=%.6g = %.2f x spacing, target=%d)",
                n_before_ds, len(pcd.points), voxel_size, voxel_size / spacing, target_points,
            )
            stats["voxel_size"] = voxel_size
    elif target_points is None:
        logger.info("[MESH PREP] No downsampling (quality='%s' keeps every point)", quality)
    else:
        logger.info(
            "[MESH PREP] No downsampling needed (%d <= target %d for '%s')",
            n_before_ds, target_points, quality,
        )

    stats["after_downsample"] = len(pcd.points)

    # Spacing changed if we decimated — every downstream radius depends on it.
    spacing = estimate_spacing(np.asarray(pcd.points))
    if spacing <= 0.0:
        logger.error("[MESH PREP] Degenerate cloud after downsampling — aborting prep")
        return None
    stats["spacing"] = spacing

    # ── 3f: Normal estimation and orientation ──────────────────────────────
    n_pts = len(pcd.points)
    reuse_normals = has_normals_in and pcd.has_normals()

    if not reuse_normals:
        normal_radius = float(preset["normal_radius_factor"]) * spacing
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=int(preset["normal_max_nn"]),
            )
        )
        logger.info(
            "[MESH PREP] Estimated normals for %d points (r=%.6g = %.1f x spacing, max_nn=%d)",
            n_pts, normal_radius, normal_radius / spacing, preset["normal_max_nn"],
        )
    else:
        logger.info("[MESH PREP] Reusing %d normals from the input cloud", n_pts)

    if camera_centers is not None and len(camera_centers) > 0:
        camera_centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
        flipped, uncertain = _orient_normals_towards_cameras(pcd, camera_centers)
        stats["normal_orientation"] = "cameras"
        stats["normals_flipped_frac"] = flipped
        stats["normals_grazing_frac"] = uncertain
        logger.info(
            "[MESH PREP] Normals oriented towards %d camera centres "
            "(%.0f%% flipped, %.1f%% grazing-angle points resolved by neighbours)",
            len(camera_centers), flipped * 100, uncertain * 100,
        )
    else:
        logger.warning(
            "[MESH PREP] No camera centres available — falling back to tangent-plane "
            "normal orientation (slower, and may flip whole regions on noisy clouds)."
        )
        if n_pts > 400_000:
            logger.warning(
                "[MESH PREP] Tangent-plane orientation on %d points may take several "
                "minutes; lower --mesh-quality or supply camera poses to avoid it.",
                n_pts,
            )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        stats["normal_orientation"] = "tangent_plane"

    stats["normal_consistency"] = _normal_consistency(pcd)
    if stats["normal_consistency"] < 0.7:
        logger.warning(
            "[MESH PREP] Low normal consistency (mean adjacent |dot| = %.2f). "
            "Poisson will smear detail — a denser or less noisy cloud would help.",
            stats["normal_consistency"],
        )

    # ── 3g: Summary ────────────────────────────────────────────────────────
    bbox = pcd.get_axis_aligned_bounding_box()
    bb_min = np.asarray(bbox.min_bound)
    bb_max = np.asarray(bbox.max_bound)
    stats["bbox_min"] = bb_min.tolist()
    stats["bbox_max"] = bb_max.tolist()
    stats["bbox_diagonal"] = float(np.linalg.norm(bb_max - bb_min))

    logger.info(
        "[MESH PREP] Preparation complete:\n"
        "  Input:            %d points\n"
        "  After SOR:        %d points (-%d, -%.1f%%)\n"
        "  After ROR:        %d points (-%d, -%.1f%%)\n"
        "  After downsample: %d points\n"
        "  Spacing:          %.6g   Bbox diagonal: %.6g (%.0f x spacing)\n"
        "  Normals:          %s (%s)   Colors: %s\n"
        "  Bounding box:     [(%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)]",
        stats["input"],
        stats["after_sor"], stats["input"] - stats["after_sor"],
        100.0 * (stats["input"] - stats["after_sor"]) / max(stats["input"], 1),
        stats["after_ror"], stats["after_sor"] - stats["after_ror"],
        100.0 * (stats["after_sor"] - stats["after_ror"]) / max(stats["after_sor"], 1),
        stats["after_downsample"],
        spacing, stats["bbox_diagonal"], stats["bbox_diagonal"] / spacing,
        "YES" if pcd.has_normals() else "NO", stats["normal_orientation"],
        "YES" if pcd.has_colors() else "NO",
        bb_min[0], bb_min[1], bb_min[2],
        bb_max[0], bb_max[1], bb_max[2],
    )

    return PreparedCloud(pcd=pcd, spacing=spacing, stats=stats)


def _apply_extent_filter(pcd, preset: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Drop far-flung points when they dominate the bounding box.

    Contamination comes in two shapes and one cutoff rule does not catch both.
    A *detached blob* shows up as a large multiplicative jump between consecutive
    sorted distances — the empty space between the object and the blob — but is
    missed by a fixed quantile whenever the blob is larger than that quantile.  A
    *diverged tail*, by contrast, is a smooth spread of stray points with no jump
    at all, and is only caught by a quantile.  Both cutoffs are therefore
    considered, along with the trivial "remove nothing" option, and the one that
    removes the **fewest points** while bringing the bounding box under control
    wins.

    The bounding-box shrink requirement is the real safety gate: a cloud that is
    merely large, with no contamination, offers no candidate that shrinks it, so
    the filter does nothing at all.  Removal is additionally capped at
    ``extent_max_removal``.

    Returns the (possibly unchanged) cloud and stats describing what happened.
    """
    pts = np.asarray(pcd.points)
    stats: Dict[str, Any] = {"extent_filtered": False}
    if len(pts) < 100:
        return pcd, stats

    max_removal = float(preset.get("extent_max_removal", 0.05))
    min_gap = float(preset.get("extent_min_gap", 1.5))
    min_shrink = float(preset.get("extent_min_shrink", 2.0))
    if max_removal <= 0:
        return pcd, stats

    def _diag(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))

    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre, axis=1)
    order = np.sort(dist)
    diag_before = _diag(pts)

    candidates = []   # (n_removed, cutoff, label)

    # Candidate 1: the largest multiplicative jump in the outer tail.
    start = max(int(len(order) * (1.0 - max_removal)), 1)
    tail = order[start - 1:]
    if len(tail) >= 2:
        ratios = tail[1:] / np.maximum(tail[:-1], 1e-12)
        j = int(np.argmax(ratios))
        gap = float(ratios[j])
        stats["extent_gap"] = gap
        if gap >= min_gap:
            cutoff = float(tail[j])
            candidates.append((int(np.sum(dist > cutoff)), cutoff, f"{gap:.1f}x gap"))

    # Candidates 2..n: fixed quantiles, for smooth heavy tails with no jump.
    for frac in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        if frac > max_removal:
            break
        cutoff = float(np.quantile(dist, 1.0 - frac))
        candidates.append((int(np.sum(dist > cutoff)), cutoff, f"outer {frac * 100:g}%"))

    # Prefer the least destructive cutoff that actually tames the bounding box.
    for n_removed, cutoff, label in sorted(candidates):
        if n_removed == 0 or n_removed >= len(pts):
            continue
        keep = dist <= cutoff
        diag_after = _diag(pts[keep])
        shrink = diag_before / max(diag_after, 1e-12)
        if shrink < min_shrink:
            continue

        logger.warning(
            "[MESH PREP] Extent filter: dropped %d far-off point(s) (%.2f%% of the "
            "cloud, selected by %s) beyond %.6g from the centre. The bounding box "
            "shrinks %.1fx (%.6g -> %.6g) — Poisson resolution is bounded by that "
            "diagonal, so far-off points cost detail everywhere.",
            n_removed, 100.0 * n_removed / len(pts), label, cutoff,
            shrink, diag_before, diag_after,
        )
        stats.update(
            extent_filtered=True,
            extent_removed=n_removed,
            extent_rule=label,
            extent_shrink=shrink,
            extent_diag_before=diag_before,
            extent_diag_after=diag_after,
        )
        return pcd.select_by_index(np.flatnonzero(keep).tolist()), stats

    logger.debug("[MESH PREP] Extent filter not applied (no cutoff shrinks the bbox enough)")
    return pcd, stats


def _normal_consistency(pcd, sample: int = 2000, seed: int = 0) -> float:
    """
    Mean |dot| between each sampled normal and its nearest neighbour's normal.

    1.0 means locally parallel normals (a smooth, well-estimated field); values
    below ~0.7 indicate a noisy field that will blur the Poisson iso-surface.
    """
    from scipy.spatial import cKDTree

    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    if len(normals) < 10 or len(normals) != len(points):
        return 1.0

    rng = np.random.default_rng(seed)
    n_sample = min(sample, len(points))
    idx = rng.choice(len(points), n_sample, replace=False)

    _, nbr = cKDTree(points).query(points[idx], k=2, workers=-1)
    dots = np.abs(np.einsum("ij,ij->i", normals[idx], normals[nbr[:, 1]]))
    return float(np.mean(dots))
