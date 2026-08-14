"""
sfm/mesh/pipeline.py — Mesh reconstruction pipeline orchestrator.

Coordinates: point-cloud preparation → surface reconstruction → cleaning →
post-processing → export → validation.

All quality-preset parameters live in QUALITY_PRESETS so they are never scattered
across sub-modules.  Every length parameter is a *multiple of the cloud's point
spacing* rather than an absolute value, so a preset behaves the same on a
reconstruction scaled in metres and one scaled in arbitrary SfM units.

Contract with the rest of the pipeline
--------------------------------------
The mesh stage runs last and is strictly additive: the sparse PLY and, when
``--dense`` was used, the dense PLY are already on disk before it starts.  It
never modifies them, and it additionally writes the *prepared* cloud — the
filtered, downsampled, normal-carrying cloud that surface reconstruction actually
consumed — because without it a mesh cannot be reproduced or audited.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Quality presets ────────────────────────────────────────────────────────────
#
# Factor keys ending in `_factor`/`_multiplier` are multiples of point spacing.

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        # outlier removal
        "sor_neighbors":         10,
        "sor_std_ratio":         3.0,
        "ror_multiplier":        4.0,
        "ror_min_neighbors":     4,
        "ror_max_removal":       0.20,
        "extent_max_removal":    0.05,
        "extent_min_gap":        1.5,
        "extent_min_shrink":     2.0,
        # normals
        "normal_radius_factor":  3.0,
        "normal_max_nn":         20,
        # sampling
        "target_points":         50_000,
        # poisson
        "poisson_depth":         8,
        "poisson_max_auto_depth": 10,
        "poisson_cell_factor":   2.5,
        "poisson_linear_fit":    True,
        "density_threshold":     0.03,
        "trim_distance_factor":  4.0,
        "bbox_margin_factor":    4.0,
        # other reconstructors
        "bpa_radius_factors":    (1.0, 2.0, 4.0),
        "alpha_factor":          6.0,
        # cleaning
        "min_component_ratio":   0.02,
        "min_component_faces":   50,
        "hole_fill_factor":      10.0,
        "remove_self_intersections":   False,
        "self_intersection_max_faces": 150_000,
        # validation
        "validate_tau_factor":   3.0,
    },
    "medium": {
        "sor_neighbors":         20,
        "sor_std_ratio":         2.0,
        "ror_multiplier":        3.0,
        "ror_min_neighbors":     5,
        "ror_max_removal":       0.15,
        "extent_max_removal":    0.05,
        "extent_min_gap":        1.5,
        "extent_min_shrink":     2.0,
        "normal_radius_factor":  3.0,
        "normal_max_nn":         30,
        "target_points":         200_000,
        "poisson_depth":         9,
        "poisson_max_auto_depth": 11,
        "poisson_cell_factor":   1.5,
        "poisson_linear_fit":    True,
        "density_threshold":     0.03,
        "trim_distance_factor":  3.0,
        "bbox_margin_factor":    3.0,
        "bpa_radius_factors":    (1.0, 2.0, 4.0, 8.0),
        "alpha_factor":          5.0,
        "min_component_ratio":   0.02,
        "min_component_faces":   50,
        "hole_fill_factor":      15.0,
        "remove_self_intersections":   False,
        "self_intersection_max_faces": 100_000,
        "validate_tau_factor":   3.0,
    },
    "high": {
        "sor_neighbors":         30,
        "sor_std_ratio":         1.5,
        "ror_multiplier":        2.5,
        "ror_min_neighbors":     6,
        "ror_max_removal":       0.12,
        "extent_max_removal":    0.05,
        "extent_min_gap":        1.5,
        "extent_min_shrink":     2.0,
        "normal_radius_factor":  2.5,
        "normal_max_nn":         40,
        "target_points":         500_000,
        "poisson_depth":         10,
        "poisson_max_auto_depth": 12,
        "poisson_cell_factor":   1.0,
        "poisson_linear_fit":    True,
        "density_threshold":     0.02,
        "trim_distance_factor":  2.5,
        "bbox_margin_factor":    2.0,
        "bpa_radius_factors":    (0.75, 1.5, 3.0, 6.0),
        "alpha_factor":          4.0,
        "min_component_ratio":   0.01,
        "min_component_faces":   50,
        "hole_fill_factor":      20.0,
        "remove_self_intersections":   False,
        "self_intersection_max_faces": 150_000,
        "validate_tau_factor":   3.0,
    },
    "ultra": {
        "sor_neighbors":         40,
        "sor_std_ratio":         1.2,
        "ror_multiplier":        2.0,
        "ror_min_neighbors":     8,
        "ror_max_removal":       0.10,
        "extent_max_removal":    0.05,
        "extent_min_gap":        1.5,
        "extent_min_shrink":     2.0,
        "normal_radius_factor":  2.0,
        "normal_max_nn":         60,
        "target_points":         None,   # no downsampling
        "poisson_depth":         11,
        "poisson_max_auto_depth": 13,
        "poisson_cell_factor":   0.8,
        "poisson_linear_fit":    True,
        "density_threshold":     0.02,
        "trim_distance_factor":  2.0,
        "bbox_margin_factor":    1.5,
        "bpa_radius_factors":    (0.5, 1.0, 2.0, 4.0),
        "alpha_factor":          3.0,
        "min_component_ratio":   0.005,
        "min_component_faces":   50,
        "hole_fill_factor":      25.0,
        "remove_self_intersections":   False,
        "self_intersection_max_faces": 200_000,
        "validate_tau_factor":   3.0,
    },
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class MeshResult:
    """Return value from MeshPipeline.run()."""

    success: bool
    output_path: Optional[Path] = None
    face_count: int = 0
    vertex_count: int = 0
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    cloud_path: Optional[Path] = None
    report_path: Optional[Path] = None
    verdict: Optional[str] = None


# ── Pipeline ───────────────────────────────────────────────────────────────────

class MeshPipeline:
    """
    Orchestrates the full mesh reconstruction pipeline.

    Parameters
    ----------
    args : argparse.Namespace  CLI arguments containing the --mesh-* flags.
    """

    def __init__(self, args) -> None:
        self.args = args
        quality = getattr(args, "mesh_quality", "medium")
        if quality not in QUALITY_PRESETS:
            logger.warning("[MESH] Unknown quality preset %r — falling back to 'medium'", quality)
            quality = "medium"
        self.quality = quality
        self.preset = dict(QUALITY_PRESETS[quality])

        # CLI overrides of preset values
        target = getattr(args, "mesh_target_points", None)
        if target is not None:
            self.preset["target_points"] = None if int(target) <= 0 else int(target)

        # Self-intersection handling is opt-in: the test is expensive and a
        # Poisson iso-surface cannot self-intersect by construction, so paying
        # for it on every run buys nothing on the default path.
        self.check_self_intersections = bool(
            getattr(args, "mesh_self_intersections", False)
        )
        self.preset["remove_self_intersections"] = self.check_self_intersections

    def run(
        self,
        pointcloud_path: "str | Path",
        output_path: "str | Path",
        camera_centers: Optional[np.ndarray] = None,
    ) -> MeshResult:
        """
        Run all mesh reconstruction stages.

        Parameters
        ----------
        pointcloud_path : Path to the input PLY point cloud produced by the SfM
                          pipeline (dense when available, otherwise sparse).
        output_path     : Desired output mesh path; the extension determines the
                          format (.obj, .ply, .glb, .stl).
        camera_centers  : (C, 3) world-space camera centres.  Used to orient the
                          cloud normals — strongly recommended, since it replaces
                          a slow and failure-prone heuristic with the ground truth
                          the SfM stage already computed.

        Returns
        -------
        MeshResult with the success flag, resolved paths, face/vertex counts, the
        quality verdict, and a stats dict suitable for
        ``SfMVisualizer.on_mesh_complete``.
        """
        try:
            import open3d as o3d
        except ImportError:
            msg = (
                "\n[MESH] open3d is not installed. Run:\n"
                "         pip install open3d\n"
                "       (or: pip install -e '.[mesh]')\n"
                "       Skipping mesh reconstruction — the point cloud is unaffected."
            )
            print(msg)
            return MeshResult(success=False, error="open3d not installed")

        from .cleaning import clean_mesh
        from .export import export_mesh
        from .pointcloud_prep import prepare_point_cloud
        from .postprocess import postprocess_mesh
        from .reconstruction import reconstruct_surface
        from .validation import format_report, validate_mesh, write_report

        pointcloud_path = Path(pointcloud_path)
        output_path = Path(output_path)
        args = self.args
        method = getattr(args, "mesh_method", "poisson")

        stats: Dict[str, Any] = {"method": method, "quality": self.quality}
        timings: Dict[str, float] = {}
        t_total = time.time()

        logger.info("")
        logger.info("[MESH]  Starting mesh reconstruction")
        logger.info("[MESH]  Input  : %s", pointcloud_path)
        logger.info("[MESH]  Output : %s", output_path)
        logger.info("[MESH]  Method : %s   Quality: %s", method, self.quality)

        # ── Stage 1: Point-cloud preparation ──────────────────────────────
        t = time.time()
        try:
            prepared = prepare_point_cloud(
                pointcloud_path, self.preset, self.quality, camera_centers=camera_centers
            )
        except Exception as exc:
            return MeshResult(success=False, error=f"Point cloud preparation failed: {exc}")
        timings["prepare"] = time.time() - t

        if prepared is None or prepared.n_points == 0:
            return MeshResult(success=False, error="Prepared point cloud is empty")

        pcd = prepared.pcd
        spacing = prepared.spacing
        stats["prep"] = prepared.stats
        stats["spacing"] = spacing

        # The prepared cloud is written before anything can go wrong later: it is
        # the actual input to surface reconstruction and is worth keeping even if
        # the mesh itself fails.
        cloud_path: Optional[Path] = None
        if getattr(args, "mesh_keep_pointcloud", True):
            cloud_path = output_path.parent / f"{output_path.stem}_prepared_cloud.ply"
            try:
                cloud_path.parent.mkdir(parents=True, exist_ok=True)
                if o3d.io.write_point_cloud(str(cloud_path), pcd, compressed=True):
                    logger.info(
                        "[MESH]  Prepared cloud saved -> %s (%d points, with normals)",
                        cloud_path, len(pcd.points),
                    )
                else:
                    logger.warning("[MESH]  Could not write prepared cloud to %s", cloud_path)
                    cloud_path = None
            except Exception as exc:
                logger.warning("[MESH]  Could not write prepared cloud: %s", exc)
                cloud_path = None

        # ── Stage 2: Surface reconstruction ───────────────────────────────
        t = time.time()
        try:
            mesh, recon_stats = reconstruct_surface(
                pcd,
                spacing,
                method,
                self.preset,
                depth_override=getattr(args, "mesh_depth", None),
                trim_override=getattr(args, "mesh_trim", None),
            )
        except Exception as exc:
            return MeshResult(
                success=False,
                error=f"Surface reconstruction failed: {exc}",
                cloud_path=cloud_path,
            )
        timings["reconstruct"] = time.time() - t

        if mesh is None or len(mesh.triangles) == 0:
            return MeshResult(
                success=False,
                error="Reconstruction produced an empty mesh",
                cloud_path=cloud_path,
            )

        stats["recon"] = recon_stats
        stats["faces_raw"] = len(mesh.triangles)
        stats["verts_raw"] = len(mesh.vertices)

        # ── Stage 3: Cleaning ──────────────────────────────────────────────
        t = time.time()
        clean_stats: Dict[str, Any] = {}
        if not getattr(args, "mesh_no_clean", False):
            try:
                mesh, clean_stats = clean_mesh(
                    mesh,
                    self.preset,
                    spacing=spacing,
                    fill_holes=getattr(args, "mesh_fill_holes", True),
                    hole_size=getattr(args, "mesh_hole_size", None),
                    keep_largest_only=getattr(args, "mesh_keep_largest", False),
                )
            except Exception as exc:
                logger.warning(
                    "[MESH] Cleaning failed (%s) — continuing with the uncleaned mesh", exc
                )
        else:
            logger.info("[MESH]  Cleaning disabled (--mesh-no-clean)")
        stats["clean"] = clean_stats
        timings["clean"] = time.time() - t

        if len(mesh.triangles) == 0:
            return MeshResult(
                success=False,
                error="Cleaning removed every triangle",
                cloud_path=cloud_path,
                stats=stats,
            )

        # ── Stage 4: Post-processing ───────────────────────────────────────
        t = time.time()
        try:
            mesh, post_stats = postprocess_mesh(
                mesh,
                pcd,
                smooth=getattr(args, "mesh_smooth", False),
                smooth_iterations=getattr(args, "mesh_smooth_iterations", 5),
                decimate=getattr(args, "mesh_decimate", False),
                decimate_target=getattr(args, "mesh_decimate_target", 100_000),
                texture=not getattr(args, "mesh_no_texture", False),
            )
            stats["post"] = post_stats
        except Exception as exc:
            logger.warning(
                "[MESH] Post-processing failed (%s) — continuing with the current mesh", exc
            )
        timings["postprocess"] = time.time() - t

        # ── Stage 5: Export ────────────────────────────────────────────────
        t = time.time()
        try:
            stats["export"] = export_mesh(mesh, output_path, verbose=True)
        except Exception as exc:
            return MeshResult(
                success=False, error=f"Export failed: {exc}", cloud_path=cloud_path, stats=stats
            )
        timings["export"] = time.time() - t

        # ── Stage 6: Validation ────────────────────────────────────────────
        report_path: Optional[Path] = None
        verdict: Optional[str] = None
        if not getattr(args, "mesh_no_validate", False):
            t = time.time()
            try:
                report = validate_mesh(
                    mesh,
                    pcd,
                    spacing=spacing,
                    tau_factor=float(self.preset.get("validate_tau_factor", 3.0)),
                    trim_factor=stats.get("recon", {}).get("trim_factor"),
                    check_self_intersections=self.check_self_intersections,
                    self_intersection_max_faces=int(
                        self.preset.get("self_intersection_max_faces", 100_000)
                    ),
                )
                stats["quality"] = report
                verdict = report.get("verdict")
                logger.info("\n%s\n", format_report(report))
                report_path = write_report(
                    report, output_path.parent / f"{output_path.stem}_quality.json"
                )
                if report_path:
                    logger.info("[MESH]  Quality report -> %s", report_path)
            except Exception as exc:
                logger.warning("[MESH] Validation failed: %s", exc)
            timings["validate"] = time.time() - t

        # ── Stage 7: Interactive preview ───────────────────────────────────
        if getattr(args, "mesh_preview", False):
            try:
                logger.info("[MESH]  Opening interactive mesh preview...")
                print("\n[MESH] Viewer controls:")
                print("       Left drag=rotate | Right drag=pan | Scroll=zoom | Q=quit\n")
                o3d.visualization.draw_geometries(
                    [mesh], window_name="Mesh Preview", width=1280, height=800
                )
            except Exception as exc:
                logger.warning("[MESH] Preview failed: %s", exc)

        n_faces = len(mesh.triangles)
        n_verts = len(mesh.vertices)
        timings["total"] = time.time() - t_total
        stats["timings"] = timings
        stats["faces_final"] = n_faces
        stats["verts_final"] = n_verts
        stats["watertight"] = bool(mesh.is_watertight())

        logger.info(
            "[MESH]  Complete: %d faces, %d vertices in %.1fs "
            "(prep %.1fs | recon %.1fs | clean %.1fs | post %.1fs | export %.1fs%s)",
            n_faces, n_verts, timings["total"],
            timings.get("prepare", 0.0), timings.get("reconstruct", 0.0),
            timings.get("clean", 0.0), timings.get("postprocess", 0.0),
            timings.get("export", 0.0),
            f" | validate {timings['validate']:.1f}s" if "validate" in timings else "",
        )

        return MeshResult(
            success=True,
            output_path=output_path,
            face_count=n_faces,
            vertex_count=n_verts,
            stats=stats,
            cloud_path=cloud_path,
            report_path=report_path,
            verdict=verdict,
        )
