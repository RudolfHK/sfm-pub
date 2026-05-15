"""
sfm/mesh/pipeline.py — Mesh reconstruction pipeline orchestrator.

Coordinates: point-cloud preparation → surface reconstruction →
cleaning → post-processing → export.

All quality-preset parameters live in QUALITY_PRESETS so they are
never scattered across sub-modules.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Quality presets ────────────────────────────────────────────────────────────

QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "sor_neighbors":       10,
        "sor_std_ratio":       3.0,
        "ror_multiplier":      5.0,
        "normal_max_nn":       20,
        "target_points":       50_000,
        "poisson_depth":       8,
        "density_threshold":   0.05,
        "min_component_ratio": 0.001,
    },
    "medium": {
        "sor_neighbors":       20,
        "sor_std_ratio":       2.0,
        "ror_multiplier":      3.0,
        "normal_max_nn":       30,
        "target_points":       200_000,
        "poisson_depth":       9,
        "density_threshold":   0.10,
        "min_component_ratio": 0.005,
    },
    "high": {
        "sor_neighbors":       30,
        "sor_std_ratio":       1.5,
        "ror_multiplier":      2.0,
        "normal_max_nn":       40,
        "target_points":       500_000,
        "poisson_depth":       10,
        "density_threshold":   0.15,
        "min_component_ratio": 0.010,
    },
    "ultra": {
        "sor_neighbors":       40,
        "sor_std_ratio":       1.2,
        "ror_multiplier":      1.5,
        "normal_max_nn":       60,
        "target_points":       None,   # no downsampling
        "poisson_depth":       11,
        "density_threshold":   0.20,
        "min_component_ratio": 0.010,
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


# ── Pipeline ───────────────────────────────────────────────────────────────────

class MeshPipeline:
    """
    Orchestrates the full mesh reconstruction pipeline.

    Parameters
    ----------
    args : argparse.Namespace  CLI arguments containing --mesh-* flags.
    """

    def __init__(self, args) -> None:
        self.args = args
        quality = getattr(args, "mesh_quality", "medium")
        if quality not in QUALITY_PRESETS:
            logger.warning("[MESH] Unknown quality preset %r — falling back to 'medium'", quality)
            quality = "medium"
        self.quality = quality
        self.preset = QUALITY_PRESETS[quality]

    def run(
        self,
        pointcloud_path: "str | Path",
        output_path: "str | Path",
    ) -> MeshResult:
        """
        Run all mesh reconstruction stages.

        Parameters
        ----------
        pointcloud_path : Path to input PLY point cloud produced by the SfM pipeline.
        output_path     : Desired output mesh path; extension determines format
                          (.obj, .ply, .glb, .stl).

        Returns
        -------
        MeshResult  with success flag, resolved output path, face/vertex counts,
                    and a stats dict suitable for passing to SfMVisualizer.on_mesh_complete.
        """
        try:
            import open3d as o3d  # noqa: F401
        except ImportError:
            msg = (
                "\n[MESH] open3d not installed. Run:\n"
                "         pip install open3d\n"
                "       Skipping mesh reconstruction."
            )
            print(msg)
            return MeshResult(success=False, error="open3d not installed")

        from .pointcloud_prep import prepare_point_cloud
        from .reconstruction import reconstruct_surface
        from .cleaning import clean_mesh
        from .postprocess import postprocess_mesh
        from .export import export_mesh

        pointcloud_path = Path(pointcloud_path)
        output_path = Path(output_path)
        args = self.args
        method = getattr(args, "mesh_method", "poisson")
        stats: Dict[str, Any] = {"method": method, "quality": self.quality}

        logger.info("\n[MESH]  Starting mesh reconstruction…")
        logger.info("[MESH]  Input  : %s", pointcloud_path)
        logger.info("[MESH]  Output : %s", output_path)
        logger.info("[MESH]  Method : %s   Quality: %s", method, self.quality)

        # ── Stage 1: Point-cloud preparation ──────────────────────────────
        try:
            pcd, prep_stats = prepare_point_cloud(
                pointcloud_path, self.preset, self.quality
            )
        except Exception as exc:
            return MeshResult(
                success=False,
                error=f"Point cloud preparation failed: {exc}",
            )

        if pcd is None or len(pcd.points) == 0:
            return MeshResult(success=False, error="Prepared point cloud is empty")

        stats["prep"] = prep_stats

        # ── Stage 2: Surface reconstruction ───────────────────────────────
        mesh_depth = getattr(args, "mesh_depth", None)
        try:
            mesh, recon_stats = reconstruct_surface(
                pcd, method, self.preset, mesh_depth
            )
        except Exception as exc:
            return MeshResult(
                success=False,
                error=f"Surface reconstruction failed: {exc}",
            )

        if mesh is None or len(mesh.triangles) == 0:
            return MeshResult(success=False, error="Reconstruction produced an empty mesh")

        stats["recon"] = recon_stats
        stats["faces_raw"] = len(mesh.triangles)
        stats["verts_raw"] = len(mesh.vertices)

        # ── Stage 3: Mesh cleaning ─────────────────────────────────────────
        do_clean = not getattr(args, "mesh_no_clean", False)
        fill_holes = getattr(args, "mesh_fill_holes", True)
        clean_stats: Dict[str, Any] = {}
        if do_clean:
            try:
                mesh, clean_stats = clean_mesh(mesh, self.preset, fill_holes=fill_holes)
            except Exception as exc:
                logger.warning("[MESH] Cleaning failed (%s) — continuing with uncleaned mesh", exc)
        else:
            logger.info("[MESH]  Cleaning disabled (--mesh-no-clean)")

        stats["clean"] = clean_stats

        # ── Stage 4: Post-processing ───────────────────────────────────────
        do_smooth = getattr(args, "mesh_smooth", False)
        smooth_iters = getattr(args, "mesh_smooth_iterations", 5)
        do_decimate = getattr(args, "mesh_decimate", False)
        decimate_target = getattr(args, "mesh_decimate_target", 100_000)
        do_texture = not getattr(args, "mesh_no_texture", False)

        try:
            mesh = postprocess_mesh(
                mesh,
                pcd,
                smooth=do_smooth,
                smooth_iterations=smooth_iters,
                decimate=do_decimate,
                decimate_target=decimate_target,
                texture=do_texture,
            )
        except Exception as exc:
            logger.warning("[MESH] Post-processing failed (%s) — continuing with current mesh", exc)

        # ── Stage 5: Export ────────────────────────────────────────────────
        keep_cloud = getattr(args, "mesh_keep_pointcloud", False)
        try:
            export_mesh(mesh, output_path, verbose=True)

            if keep_cloud:
                import open3d as o3d
                cloud_out = (
                    output_path.parent / f"{output_path.stem}_prepared_cloud.ply"
                )
                o3d.io.write_point_cloud(str(cloud_out), pcd)
                logger.info("[MESH]  Prepared cloud saved → %s", cloud_out)

        except Exception as exc:
            return MeshResult(success=False, error=f"Export failed: {exc}")

        # ── Stage 6: Interactive preview ───────────────────────────────────
        if getattr(args, "mesh_preview", False):
            try:
                import open3d as o3d
                logger.info("[MESH]  Opening interactive mesh preview…")
                print("\n[MESH] Interactive viewer controls:")
                print("       Left drag=rotate  |  Right drag=pan  |  Scroll=zoom  |  Q=quit\n")
                o3d.visualization.draw_geometries(
                    [mesh],
                    window_name="Mesh Preview",
                    width=1280,
                    height=800,
                )
            except Exception as exc:
                logger.warning("[MESH] Preview failed: %s", exc)

        n_faces = len(mesh.triangles)
        n_verts = len(mesh.vertices)
        stats["faces_final"] = n_faces
        stats["verts_final"] = n_verts
        stats["watertight"] = mesh.is_watertight()

        logger.info("[MESH]  Complete: %d faces, %d vertices", n_faces, n_verts)

        return MeshResult(
            success=True,
            output_path=output_path,
            face_count=n_faces,
            vertex_count=n_verts,
            stats=stats,
        )
