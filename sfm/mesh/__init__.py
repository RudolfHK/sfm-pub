"""
sfm.mesh — surface reconstruction from SfM/MVS point clouds.

Stages: point-cloud preparation -> surface reconstruction -> cleaning ->
post-processing -> export -> validation.  See MESHING_PROMPT.md for the design
brief and PIPELINE_GUIDE.md for usage.

Programmatic entry point:

    from sfm.mesh import MeshPipeline
    result = MeshPipeline(args).run("dense.ply", "mesh.ply", camera_centers=C)

Command line:

    python -m sfm.mesh dense.ply -o mesh.ply --quality high
"""

from .pipeline import QUALITY_PRESETS, MeshPipeline, MeshResult

__all__ = ["MeshPipeline", "MeshResult", "QUALITY_PRESETS"]
