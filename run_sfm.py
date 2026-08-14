#!/usr/bin/env python3
"""
run_sfm.py — Structure from Motion pipeline entry point.

Usage
-----
    # Pure-Python backend (default)
    python run_sfm.py --image_dir ./images --output output.ply

    # COLMAP sparse reconstruction
    python run_sfm.py --image_dir ./images --output output.ply --backend colmap

    # COLMAP sparse + dense (MVS) reconstruction
    python run_sfm.py --image_dir ./images --output output.ply --backend colmap-mvs

    # Example usage for specific dataset:
    python run_sfm.py --image_dir "C:/Users/baldo/Downloads/dataset_buddha-master/dataset_buddha-master/buddha_imgs" --output buddha_python_dense.ply --dense --dense_output  buddha_python_dense2.ply --n_features 12000 --ratio 0.7 --min_inliers 25 --max_reproj_error 3.0 --verbose --visualize
    python run_sfm.py --image_dir "C:/Users/Rudolf/Downloads/dataset_buddha-master/dataset_buddha-master/imgs_only" --output buddha_python_dense.ply --dense --dense_output  buddha_python_dense3.ply --n_features 12000 --ratio 0.7 --min_inliers 25 --max_reproj_error 3.0 --verbose --visualize

Run `python run_sfm.py --help` for all options.
"""

import argparse
import hashlib
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structure from Motion pipeline — pure-Python or COLMAP backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing input JPG/PNG images.",
    )
    p.add_argument(
        "--output",
        default="output.ply",
        help="Output PLY file path (sparse cloud, or dense when --dense is set).",
    )
    # Backend selection
    p.add_argument(
        "--backend",
        choices=["python", "colmap", "colmap-mvs"],
        default="python",
        help=(
            "Reconstruction backend.  "
            "'python' = built-in incremental SfM (default);  "
            "'colmap' = COLMAP sparse reconstruction;  "
            "'colmap-mvs' = COLMAP sparse + dense (patch-match MVS, requires CUDA)."
        ),
    )
    # Feature extraction
    p.add_argument(
        "--n_features",
        type=int,
        default=8_000,
        help="Max features extracted per image (SIFT or SuperPoint).",
    )
    p.add_argument(
        "--feature-backend",
        choices=["sift", "superpoint", "disk"],
        default="sift",
        help=(
            "Feature extraction backend.  'sift' (default) uses OpenCV SIFT with "
            "optional GPU acceleration via kornia or CUDA SURF.  'superpoint' uses "
            "kornia's SuperPoint neural detector+descriptor (requires torch + "
            "kornia>=0.7); automatically activates LightGlue matching.  'disk' uses "
            "kornia's DISK detector with 128-D descriptors; compatible with LightGlue "
            "and standard FLANN matching."
        ),
    )
    p.add_argument(
        "--sift-contrast-threshold",
        type=float,
        default=0.02,
        help=(
            "SIFT contrast threshold. Lower values detect more low-contrast keypoints. "
            "0.02 recovers 30-60%% more features than OpenCV's default 0.04 on "
            "high-resolution images; COLMAP uses 0.02 by default."
        ),
    )
    p.add_argument(
        "--sift-edge-threshold",
        type=float,
        default=10.0,
        help=(
            "SIFT edge threshold. Higher values keep more edge-like keypoints "
            "(often more features, but can include unstable points)."
        ),
    )
    p.add_argument(
        "--sift-n-octave-layers",
        type=int,
        default=3,
        help="Number of SIFT octave layers. Higher values can produce more features.",
    )
    p.add_argument(
        "--sift-sigma",
        type=float,
        default=1.6,
        help="Sigma of the Gaussian applied at octave 0 for SIFT.",
    )
    # Matching strategy
    p.add_argument(
        "--match_strategy",
        choices=["exhaustive", "sequential", "vocab_tree", "loftr"],
        default="exhaustive",
        help=(
            "Pairwise matching strategy.  "
            "'exhaustive' = O(N²) all pairs; "
            "'sequential' = sliding window; "
            "'vocab_tree' = bag-of-words retrieval."
            "'Note: vocab_tree requires a pre-built COLMAP vocabulary tree file when using the COLMAP backend (see --colmap-vocab-tree)."
            "'Recommended' settings:  "
            "Use 'exhaustive' for small datasets (<50 images) with no temporal ordering.  "
        ),
    )
    p.add_argument(
        "--match-workers",
        type=int,
        default=0,
        help=(
            "Threads for the exhaustive matching pair loop.  0 (default) uses "
            "the CPU count; 1 forces the serial loop.  Matching was 91%% of a "
            "67-image run and 4.9× slower than COLMAP on the same CPU purely "
            "because this loop was serial.  Results are order-identical to the "
            "serial path regardless of thread count."
        ),
    )
    p.add_argument(
        "--sequential_window",
        type=int,
        default=5,
        help="Window size for sequential matching (images i vs i+1 … i+W). Example: W=5 means image_000.jpg will be matched against image_001.jpg through image_005.jpg (inclusive). Maximum W is N-1, which degrades to exhaustive matching (default: 5).",
    )
    p.add_argument(
        "--vocab_words",
        type=int,
        default=4096,
        help=(
            "Vocabulary size for vocab_tree matching. "
            "Larger = better image retrieval quality but more memory and build time. "
            "4096 is the practical minimum for small-to-medium datasets; "
            "256 (old default) is too coarse for meaningful retrieval."
        ),
    )
    p.add_argument(
        "--vocab_top_k",
        type=int,
        default=10,
        help="Number of nearest-neighbour images retrieved per query (vocab_tree).",
    )
    p.add_argument(
        "--retrieval",
        choices=["none", "dinov2"],
        default="none",
        help=(
            "Image-retrieval backend to restrict matching to visually similar pairs. "
            "'dinov2' uses DINOv2 CLS-token embeddings (requires torch; FAISS "
            "accelerates ANN search when installed). Overrides --match_strategy "
            "when set to anything other than 'none'."
        ),
    )
    p.add_argument(
        "--retrieval_top_k",
        type=int,
        default=10,
        help="Number of nearest-neighbour images per query for --retrieval dinov2.",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio-test threshold (lower = stricter).",
    )
    p.add_argument(
        "--min_matches",
        type=int,
        default=15,
        help="Min raw matches required to keep a pair.",
    )
    # Geometric verification
    p.add_argument(
        "--min_inliers",
        type=int,
        default=15,
        help="Min RANSAC inliers required to accept a pair.",
    )
    p.add_argument(
        "--ransac_thr",
        type=float,
        default=1.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    # Reconstruction
    p.add_argument(
        "--max_reproj_error",
        type=float,
        default=4.0,
        help="Max reprojection error (px) for triangulation / PnP.",
    )
    p.add_argument(
        "--ba_interval",
        type=int,
        default=5,
        help="Run bundle adjustment every N newly registered cameras.",
    )
    p.add_argument(
        "--no_refine_intrinsics",
        action="store_true",
        help=(
            "Disable joint focal-length and radial-distortion (k1, k2) "
            "refinement in bundle adjustment.  Use when the camera is "
            "pre-calibrated or for speed."
        ),
    )
    p.add_argument(
        "--ba-fix-principal-point",
        action="store_true",
        help=(
            "Keep cx/cy fixed during bundle adjustment (default: optimize cx/cy). "
            "Use on small datasets or when BA convergence is poor."
        ),
    )
    p.add_argument(
        "--loop-closure",
        action="store_true",
        help=(
            "When the scene graph has multiple disconnected components, "
            "attempt to bridge them by running DINOv2 or VocabTree retrieval "
            "between images in different components, re-verifying any found "
            "pairs, and adding them to the scene graph before reconstruction.  "
            "Requires either --retrieval dinov2 or --match_strategy vocab_tree."
        ),
    )
    p.add_argument(
        "--local-ba-window",
        type=int,
        default=0,
        metavar="W",
        help=(
            "Enable local bundle adjustment: after each camera registration, "
            "optimize the last W cameras and their visible 3-D points.  "
            "W=10 gives 10-50× speedup over global BA on large scenes (>50 images).  "
            "0 disables local BA (default).  Combined with --ba_interval for global BA."
        ),
    )
    p.add_argument(
        "--track-merge",
        action="store_true",
        help=(
            "Enable union-find track merging before incremental SfM.  "
            "Removes contradictory multi-assignment matches so each keypoint "
            "participates in at most one 3-D track, reducing duplicate 3-D points."
        ),
    )
    p.add_argument(
        "--per-camera-intrinsics",
        action="store_true",
        help=(
            "Give each camera its own fx, fy, cx, cy, k1, k2 initialised from "
            "EXIF FocalLengthIn35mmFilm when available (falls back to shared "
            "estimate).  Enables T2-01 per-camera intrinsics in reconstruction.  "
            "Required for --ba-backend pyceres per-camera optimisation."
        ),
    )
    p.add_argument(
        "--pnp-backend",
        choices=["cv2", "poselib"],
        default="cv2",
        help=(
            "'cv2' (default) uses cv2.solvePnPRansac with EPnP+LM refinement.  "
            "'poselib' uses the PoseLib p3p minimal solver (3–5× faster on CPU, "
            "better numerical stability on near-planar configurations).  "
            "Requires: pip install poselib"
        ),
    )
    p.add_argument(
        "--ba-separate-focal",
        action="store_true",
        help=(
            "Optimize separate fx and fy focal lengths in bundle adjustment "
            "instead of a single shared scalar f.  Useful for anamorphic lenses "
            "or drone cameras with non-square pixels.  Adds one extra shared "
            "parameter to the BA problem (negligible cost)."
        ),
    )
    p.add_argument(
        "--ba-backend",
        choices=["scipy", "pyceres"],
        default="scipy",
        help=(
            "'scipy' (default) uses the TRF least-squares solver with a "
            "manually constructed Jacobian sparsity pattern.  'pyceres' uses "
            "pyceres (Ceres Solver Python bindings) with SPARSE_SCHUR for "
            "O(C³+P) scaling; falls back to scipy if pyceres is not installed."
        ),
    )
    p.add_argument(
        "--no-track-completion",
        action="store_true",
        help=(
            "Disable track completion and merging after each bundle adjustment. "
            "Completion extends a track into cameras that already observe it; "
            "merging reconciles two 3-D points that a verified match proves "
            "identical.  Without them mean track length was measured at 2.70 "
            "against COLMAP's 4.69 on the same images, with 8%% exact duplicates."
        ),
    )
    p.add_argument(
        "--focal",
        type=float,
        default=None,
        metavar="PX",
        help=(
            "Known focal length in pixels, overriding EXIF and the max(W, H) "
            "fallback.  Supply this whenever the camera is calibrated: the "
            "fallback guess was measured 47%% wrong on an uncalibrated dataset, "
            "which is the single largest source of geometric error."
        ),
    )
    p.add_argument(
        "--intrinsics",
        default=None,
        metavar="fx,fy,cx,cy",
        help=(
            "Full known calibration as four comma-separated pixel values.  "
            "Takes precedence over --focal, EXIF and the size-based fallback."
        ),
    )
    p.add_argument(
        "--focal-search",
        action="store_true",
        help=(
            "Estimate the focal length before reconstruction by sweeping "
            "candidates and keeping the one that yields the most accepted "
            "two-view correspondences.  Use when the images carry no EXIF and "
            "no calibration is known.  Ignored if --focal or --intrinsics is given."
        ),
    )
    p.add_argument(
        "--focal-search-pairs",
        type=int,
        default=12,
        help="Number of strongest image pairs sampled per candidate by --focal-search.",
    )
    p.add_argument(
        "--ba-ftol",
        type=float,
        default=1e-6,
        help="BA cost-change convergence tolerance (scipy least_squares ftol).",
    )
    p.add_argument(
        "--ba-xtol",
        type=float,
        default=1e-6,
        help=(
            "BA step-size convergence tolerance (scipy least_squares xtol).  "
            "The former default of 1e-4 stopped the solver after ~0.1s with the "
            "focal length still 47%% wrong, reported as convergence."
        ),
    )
    p.add_argument(
        "--ba-gtol",
        type=float,
        default=1e-6,
        help="BA gradient convergence tolerance (scipy least_squares gtol).",
    )
    p.add_argument(
        "--ba-max-nfev",
        type=int,
        default=200,
        help=(
            "BA function-evaluation budget, multiplied by the parameter count.  "
            "Lower it to bound the runtime cost of the tighter tolerances."
        ),
    )
    p.add_argument(
        "--ba-no-param-scaling",
        action="store_true",
        help=(
            "Disable analytic parameter scaling in bundle adjustment, restoring "
            "the pre-fix behaviour where focal, translations and point "
            "coordinates share one scale.  Diagnostic use only."
        ),
    )
    # MVS densification
    p.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Run MVS densification (StereoSGBM) after sparse reconstruction "
            "and save the dense + sparse point cloud."
        ),
    )
    p.add_argument(
        "--dense_output",
        default=None,
        help=(
            "Output PLY path for the dense cloud.  "
            "Defaults to <output_stem>_dense.ply."
        ),
    )
    p.add_argument(
        "--mvs-fusion",
        action="store_true",
        help=(
            "Enable multi-view depth consistency filtering after SGBM densification.  "
            "Each world-space point must be visible from at least --mvs-fusion-min-views "
            "cameras (positive depth) to be kept.  Requires --dense."
        ),
    )
    p.add_argument(
        "--mvs-fusion-min-views",
        type=int,
        default=2,
        help="Minimum cameras with positive depth for a point to survive --mvs-fusion.",
    )
    p.add_argument(
        "--max-dense-points",
        type=int,
        default=500_000,
        help=(
            "Cap on the dense cloud size; points above it are randomly "
            "subsampled.  When the cap binds, the output size is set by this "
            "number rather than by the scene — a warning now says so."
        ),
    )
    # Export
    p.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip statistical outlier filtering of the final sparse point cloud.",
    )
    # Misc
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for OpenCV, NumPy and the stdlib RNG.  OpenCV's RANSAC "
            "variants (USAC_MAGSAC, solvePnPRansac) draw from a process-global "
            "generator; without seeding it, identical invocations vary by up to "
            "57%% in registered cameras.  Use --seed -1 to keep the previous "
            "nondeterministic behaviour."
        ),
    )
    p.add_argument(
        "--export-cameras",
        default=None,
        metavar="PATH",
        help=(
            "Write the registered camera poses (R, t, K) and per-run statistics "
            "to a JSON file.  Enables external evaluation against ground-truth "
            "poses.  Ignored by the COLMAP backend."
        ),
    )

    # ── Visualization (completely optional) ──────────────────────────────
    viz = p.add_argument_group("visualization (all ignored unless --visualize is set)")
    viz.add_argument(
        "--visualize",
        action="store_true",
        help="Enable the full visualization suite.  Zero overhead when omitted.",
    )
    viz.add_argument(
        "--viz-samples",
        type=int,
        default=3,
        metavar="N",
        help="Number of images / pairs to sample for feature/match visualizations.",
    )
    viz.add_argument(
        "--viz-output",
        default="sfm_visualization",
        metavar="DIR",
        help="Directory to save all visualization outputs.",
    )
    viz.add_argument(
        "--viz-format",
        default="png",
        choices=["png", "jpg", "pdf"],
        metavar="FMT",
        help="Output image format for saved figures.",
    )
    viz.add_argument(
        "--viz-interactive",
        action="store_true",
        help="Open an interactive open3d point-cloud viewer at end of pipeline.",
    )
    viz.add_argument(
        "--viz-save-video",
        action="store_true",
        help="Export reconstruction growth GIF and point-cloud turntable GIF.",
    )
    viz.add_argument(
        "--viz-dpi",
        type=int,
        default=150,
        metavar="N",
        help="DPI for saved figures.",
    )
    viz.add_argument(
        "--viz-seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducible image/pair sampling.",
    )

    # ── COLMAP options (ignored unless --backend colmap / colmap-mvs) ─────
    col = p.add_argument_group(
        "COLMAP options (ignored unless --backend colmap or colmap-mvs)"
    )
    col.add_argument(
        "--colmap-bin",
        default="colmap",
        metavar="PATH",
        help="Path to the COLMAP executable (default: 'colmap' on PATH).",
    )
    col.add_argument(
        "--colmap-workspace",
        default=None,
        metavar="DIR",
        help=(
            "Directory for COLMAP's internal database and sparse model.  "
            "Defaults to <output_dir>/colmap_workspace/."
        ),
    )
    col.add_argument(
        "--colmap-keep-workspace",
        action="store_true",
        help="Keep the COLMAP workspace after a successful run (useful for debugging).",
    )
    col.add_argument(
        "--colmap-vocab-tree",
        default=None,
        metavar="PATH",
        help=(
            "Path to a pre-built COLMAP vocabulary tree file (.bin).  "
            "Required when --match_strategy vocab_tree is used with the COLMAP backend.  "
            "Download from: https://demuc.de/colmap/#download"
        ),
    )

    # ── Mesh reconstruction (completely optional) ─────────────────────────
    msh = p.add_argument_group(
        "mesh reconstruction (all ignored unless --mesh is set)"
    )
    msh.add_argument(
        "--mesh",
        action="store_true",
        help=(
            "Enable mesh reconstruction from the output point cloud.  "
            "Requires: pip install open3d"
        ),
    )
    msh.add_argument(
        "--mesh-output",
        default=None,
        metavar="PATH",
        help=(
            "Output path for the mesh file.  "
            "Defaults to <output_stem>_mesh.obj.  "
            "Supported formats: .obj  .ply  .glb  .stl"
        ),
    )
    msh.add_argument(
        "--mesh-method",
        choices=["poisson", "bpa", "alpha"],
        default="poisson",
        help=(
            "Surface reconstruction algorithm.  "
            "'poisson' = Screened Poisson (best for smooth objects, default);  "
            "'bpa' = Ball-Pivoting (better for thin/open surfaces);  "
            "'alpha' = Alpha shapes (fast, good for convex/simple objects)."
        ),
    )
    msh.add_argument(
        "--mesh-quality",
        choices=["low", "medium", "high", "ultra"],
        default="medium",
        help=(
            "Quality preset controlling all key reconstruction parameters.  "
            "'low'=fast/coarse  'medium'=balanced (default)  "
            "'high'=detailed  'ultra'=maximum quality (slow)."
        ),
    )
    msh.add_argument(
        "--mesh-depth",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Poisson octree depth override (ignores quality preset).  "
            "Higher = more detail.  Typical range: 8–12."
        ),
    )
    msh.add_argument(
        "--mesh-trim",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Phantom-surface trim distance, in multiples of the cloud's point spacing "
            "(ignores quality preset).  Poisson vertices further than this from any "
            "input point were invented by the solver and are removed.  "
            "Lower = more aggressive.  Typical range: 2–5."
        ),
    )
    msh.add_argument(
        "--mesh-target-points",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Downsample the cloud to roughly N points before meshing "
            "(ignores quality preset).  0 = keep every point."
        ),
    )
    msh.add_argument(
        "--mesh-no-clean",
        action="store_true",
        help="Disable mesh cleaning (raw reconstruction output, not recommended).",
    )
    msh.add_argument(
        "--mesh-smooth",
        action="store_true",
        help=(
            "Apply Taubin smoothing after cleaning.  "
            "Smoothing is lossy — fine surface detail will be reduced."
        ),
    )
    msh.add_argument(
        "--mesh-smooth-iterations",
        type=int,
        default=5,
        metavar="N",
        help="Number of Taubin smoothing iterations (default: 5).",
    )
    msh.add_argument(
        "--mesh-no-texture",
        action="store_true",
        help="Skip RGB color projection from point cloud onto mesh vertices.",
    )
    msh.add_argument(
        "--mesh-decimate",
        action="store_true",
        help="Reduce polygon count via quadric decimation after reconstruction.",
    )
    msh.add_argument(
        "--mesh-decimate-target",
        type=int,
        default=100_000,
        metavar="N",
        help="Target face count after decimation (default: 100 000).",
    )
    msh.add_argument(
        "--mesh-no-fill-holes",
        action="store_false",
        dest="mesh_fill_holes",
        default=True,
        help="Disable hole filling in the mesh surface (hole filling is ON by default).",
    )
    msh.add_argument(
        "--mesh-fill-holes",
        action="store_true",
        dest="mesh_fill_holes_legacy",
        default=False,
        help=argparse.SUPPRESS,
    )
    msh.add_argument(
        "--mesh-hole-size",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Largest hole to close, in scene units.  "
            "Default: derived from the point spacing via the quality preset."
        ),
    )
    msh.add_argument(
        "--mesh-keep-largest",
        action="store_true",
        help=(
            "Keep only the largest connected component of the mesh, discarding all "
            "other surfaces (useful for single-object scans)."
        ),
    )
    msh.add_argument(
        "--mesh-keep-pointcloud",
        action="store_true",
        default=True,
        help=(
            "Save the cleaned/prepared point cloud (with normals) as a separate PLY "
            "alongside the mesh.  ON by default — this is the exact input surface "
            "reconstruction consumed, so it is needed to reproduce the mesh."
        ),
    )
    msh.add_argument(
        "--mesh-no-keep-pointcloud",
        action="store_false",
        dest="mesh_keep_pointcloud",
        help="Do not write the prepared point cloud PLY alongside the mesh.",
    )
    msh.add_argument(
        "--mesh-self-intersections",
        action="store_true",
        help=(
            "Detect and remove self-intersecting triangles, and report them in the "
            "quality check.  Off by default: the test costs seconds to minutes "
            "depending on mesh size, and a Poisson iso-surface cannot self-intersect "
            "by construction — enable it for --mesh-method bpa/alpha."
        ),
    )
    msh.add_argument(
        "--mesh-no-validate",
        action="store_true",
        help=(
            "Skip the mesh quality report (accuracy, completeness, topology).  "
            "Validation is ON by default."
        ),
    )
    msh.add_argument(
        "--mesh-preview",
        action="store_true",
        help="Open the mesh in the open3d interactive viewer immediately after generation.",
    )

    # ── Checkpointing ────────────────────────────────────────────────────
    ckpt = p.add_argument_group("checkpointing")
    ckpt.add_argument(
        "--checkpoint-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to store / load stage checkpoints (features and matches).  "
            "Defaults to <output_dir>/.sfm_checkpoints/.  "
            "Checkpoints are automatically invalidated when the image set changes."
        ),
    )
    ckpt.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the most recent valid checkpoint, skipping already-completed "
            "stages (feature extraction, matching).  Has no effect if no valid "
            "checkpoint exists for the current image set."
        ),
    )
    return p


_SUPPORTED_EXT = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


def _validate_inputs(args) -> Optional[str]:
    """
    Validate CLI arguments before any expensive work begins.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    str  — human-readable error message if validation fails
    None — all checks passed
    """
    import os
    import shutil

    # 1 — Image directory
    if not os.path.isdir(args.image_dir):
        return f"Image directory not found: {args.image_dir!r}"

    images = [
        f
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f.lower())[1] in _SUPPORTED_EXT
    ]
    if len(images) < 2:
        exts = ", ".join(sorted(_SUPPORTED_EXT))
        return (
            f"Need ≥ 2 images in {args.image_dir!r}, "
            f"found {len(images)} supported file(s) (supported extensions: {exts})"
        )

    # 2 — Output parent directory (create if missing, check write access)
    out_parent = os.path.dirname(os.path.abspath(args.output)) or "."
    try:
        os.makedirs(out_parent, exist_ok=True)
    except OSError as exc:
        return f"Cannot create output directory {out_parent!r}: {exc}"
    if not os.access(out_parent, os.W_OK):
        return f"Output directory {out_parent!r} is not writable"

    # 3 — Numerical parameter bounds
    checks = [
        (0.0 < args.ratio < 1.0, f"--ratio must be in (0, 1), got {args.ratio}"),
        (
            args.min_inliers >= 8,
            f"--min_inliers must be ≥ 8 (5-pt algorithm minimum), got {args.min_inliers}",
        ),
        (args.ransac_thr > 0, f"--ransac_thr must be > 0, got {args.ransac_thr}"),
        (
            args.max_reproj_error > 0,
            f"--max_reproj_error must be > 0, got {args.max_reproj_error}",
        ),
        (args.n_features >= 100, f"--n_features must be ≥ 100, got {args.n_features}"),
        (
            args.sift_contrast_threshold > 0,
            f"--sift-contrast-threshold must be > 0, got {args.sift_contrast_threshold}",
        ),
        (
            args.sift_edge_threshold > 0,
            f"--sift-edge-threshold must be > 0, got {args.sift_edge_threshold}",
        ),
        (
            args.sift_n_octave_layers >= 1,
            f"--sift-n-octave-layers must be ≥ 1, got {args.sift_n_octave_layers}",
        ),
        (args.sift_sigma > 0, f"--sift-sigma must be > 0, got {args.sift_sigma}"),
        (
            args.sequential_window >= 1,
            f"--sequential_window must be ≥ 1, got {args.sequential_window}",
        ),
        (args.vocab_words >= 4, f"--vocab_words must be ≥ 4, got {args.vocab_words}"),
        (args.vocab_top_k >= 1, f"--vocab_top_k must be ≥ 1, got {args.vocab_top_k}"),
        (args.ba_interval >= 1, f"--ba_interval must be ≥ 1, got {args.ba_interval}"),
    ]
    for ok, msg in checks:
        if not ok:
            return msg

    # 4 — COLMAP-specific checks
    if args.backend in ("colmap", "colmap-mvs"):
        if shutil.which(args.colmap_bin) is None and not os.path.isfile(
            args.colmap_bin
        ):
            return (
                f"COLMAP executable not found: {args.colmap_bin!r}. "
                "Install COLMAP and ensure it is on PATH, "
                "or specify the full path with --colmap-bin."
            )
        if args.match_strategy == "vocab_tree" and args.colmap_vocab_tree is None:
            return (
                "--match_strategy vocab_tree with the COLMAP backend requires "
                "--colmap-vocab-tree PATH (download from demuc.de/colmap/#download)."
            )

    # 5 — Optional deep-learning backends: fail here with an actionable message
    # rather than deep inside the pipeline with a raw ModuleNotFoundError, and
    # before any expensive stage has been paid for.  Same contract as the
    # COLMAP pre-flight above.
    err = _check_optional_backends(args)
    if err is not None:
        return err

    return None


# Optional backend → (selecting flag, required importable modules, install hint).
_OPTIONAL_BACKENDS = {
    "superpoint": (
        "--feature-backend superpoint",
        ("torch", "kornia"),
        "pip install torch kornia",
    ),
    "disk": (
        "--feature-backend disk",
        ("torch", "kornia"),
        "pip install torch kornia",
    ),
    "loftr": (
        "--match_strategy loftr",
        ("torch", "kornia"),
        "pip install torch kornia",
    ),
    "dinov2": (
        "--retrieval dinov2",
        ("torch",),
        "pip install torch torchvision",
    ),
}


def _check_optional_backends(args) -> Optional[str]:
    """Return an actionable message when a selected optional backend cannot import."""
    import importlib.util

    selected = []
    if getattr(args, "feature_backend", "sift") in ("superpoint", "disk"):
        selected.append(args.feature_backend)
    if getattr(args, "match_strategy", "") == "loftr":
        selected.append("loftr")
    if getattr(args, "retrieval", "none") == "dinov2":
        selected.append("dinov2")

    for name in selected:
        flag, modules, hint = _OPTIONAL_BACKENDS[name]
        missing = [
            m for m in modules if importlib.util.find_spec(m) is None
        ]
        if missing:
            return (
                f"{flag} requires {', '.join(modules)}, but "
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                f"not installed. Install with: {hint}"
            )
    return None


def _image_set_hash(image_paths: list) -> str:
    """
    Hash the image set to decide whether a checkpoint may be reused.

    Digests file *content*, not just name and size.  Hashing name+size alone
    was demonstrably unsafe: `eval/hash_collision_test.py` builds two entirely
    different 4-image scenes with identical filenames and identical byte sizes
    (equal-dimension uncompressed BMPs), and the resumed run reported a cache
    hit and reconstructed scene B from scene A's features, with no warning.
    Any workflow that edits images in place — re-exporting, colour-correcting,
    undistorting — hit the same failure.

    Content hashing costs one sequential read of the image set (a few hundred
    ms for 67 PNGs) against a feature-extraction stage measured in tens of
    seconds, so the check is paid for many times over the first time it
    correctly invalidates.
    """
    h = hashlib.md5()
    for p in sorted(str(p) for p in image_paths):
        h.update(p.encode())
        try:
            h.update(str(os.path.getsize(p)).encode())
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            # Unreadable here is not fatal: extraction reports and skips it.
            pass
    return h.hexdigest()[:16]


def _ckpt_dir(args) -> Path:
    """Resolve the checkpoint directory."""
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return Path(args.output).parent / ".sfm_checkpoints"


def _save_checkpoint(ckpt_dir: Path, name: str, data, img_hash: str) -> None:
    """
    Pickle `data` to ckpt_dir/<name>.pkl alongside a manifest with img_hash.

    Parameters
    ----------
    ckpt_dir : directory to write into (created if missing)
    name     : stage name, e.g. 'features' or 'matches'
    data     : picklable object to save
    img_hash : image-set hash for invalidation on load
    """
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {"img_hash": img_hash, "data": data}
        with open(ckpt_dir / f"{name}.pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("  Checkpoint saved: %s/%s.pkl", ckpt_dir, name)
    except Exception as exc:
        logger.warning("  Could not save checkpoint %s: %s", name, exc)


def _load_checkpoint(ckpt_dir: Path, name: str, img_hash: str):
    """
    Load a checkpoint if it exists and its image-set hash matches.

    Returns the stored data object, or None if unavailable / stale.
    """
    path = ckpt_dir / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("img_hash") != img_hash:
            logger.info(
                "  Checkpoint %s is stale (image set changed) — will recompute.",
                name,
            )
            return None
        logger.info("  Loaded checkpoint: %s/%s.pkl", ckpt_dir, name)
        return payload["data"]
    except Exception as exc:
        logger.warning("  Could not load checkpoint %s: %s", name, exc)
        return None


def _export_cameras(
    path: str,
    cameras: dict,
    K,
    dist_coeffs,
    features: dict,
    points_3d,
    observations: list,
    stage_times: dict,
    args,
    focal_curve: Optional[list] = None,
) -> None:
    """
    Dump registered camera poses and run statistics to JSON.

    The schema is deliberately flat and dependency-free so that external
    evaluation tools (see eval/) can compare the poses against ground truth
    without importing the pipeline.

    Poses use the OpenCV world-to-camera convention: x_cam = R @ x_world + t.
    The camera centre in world coordinates is C = -R.T @ t.
    """
    import json

    import numpy as np

    from sfm.utils import reprojection_error

    errs = [
        reprojection_error(
            points_3d[pt_idx],
            np.array([x, y]),
            cameras[img_idx].get("K", K),
            cameras[img_idx]["R"],
            cameras[img_idx]["t"],
        )
        for img_idx, pt_idx, x, y in observations
        if img_idx in cameras and pt_idx < len(points_3d)
    ]
    errs = np.array([e for e in errs if np.isfinite(e)], dtype=np.float64)

    track_lengths: dict = {}
    for img_idx, pt_idx, _, _ in observations:
        track_lengths[pt_idx] = track_lengths.get(pt_idx, 0) + 1

    payload = {
        "convention": "world_to_camera (x_cam = R @ x_world + t); C = -R.T @ t",
        "n_images": len(features),
        "n_cameras_registered": len(cameras),
        "n_points": int(len(points_3d)),
        "n_observations": len(observations),
        "mean_track_length": (
            float(np.mean(list(track_lengths.values()))) if track_lengths else 0.0
        ),
        "reprojection": {
            "rmse_px": float(np.sqrt(np.mean(errs**2))) if errs.size else None,
            "mean_px": float(np.mean(errs)) if errs.size else None,
            "median_px": float(np.median(errs)) if errs.size else None,
            "p95_px": float(np.percentile(errs, 95)) if errs.size else None,
            "max_px": float(np.max(errs)) if errs.size else None,
            "frac_above_threshold": (
                float(np.mean(errs > args.max_reproj_error)) if errs.size else None
            ),
            "threshold_px": args.max_reproj_error,
            "n_residuals": int(errs.size),
        },
        "focal_search": focal_curve,
        "shared_K": np.asarray(K, dtype=float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).ravel().tolist(),
        "stage_times_s": stage_times,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cameras": [],
    }

    for img_idx in sorted(cameras):
        cam = cameras[img_idx]
        R = np.asarray(cam["R"], dtype=float)
        t = np.asarray(cam["t"], dtype=float).reshape(3)
        payload["cameras"].append(
            {
                "image_index": int(img_idx),
                "image_name": Path(features[img_idx]["image_path"]).name,
                "R": R.tolist(),
                "t": t.tolist(),
                "center": (-R.T @ t).tolist(),
                "K": np.asarray(cam.get("K", K), dtype=float).tolist(),
            }
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("       Camera poses exported → %s", path)


def _seed_rngs(seed: int) -> None:
    """
    Seed every random source the pipeline consumes.

    The NumPy generators used inside the pipeline are already constructed with
    a fixed seed, but OpenCV keeps a *process-global* RNG that USAC_MAGSAC and
    solvePnPRansac draw from, and it is seeded from system state.  Leaving it
    unseeded is what makes byte-identical invocations register different
    numbers of cameras.  Seeding it here is the only place that covers every
    downstream call.

    `seed < 0` restores the previous nondeterministic behaviour.
    """
    if seed < 0:
        logger.info("Seeding disabled (--seed %d) — results will not be reproducible.", seed)
        return

    import random

    import cv2
    import numpy as np

    cv2.setRNGSeed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info("RNG seed: %d (OpenCV, NumPy, random)", seed)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    from sfm.utils import setup_logging

    setup_logging(args.verbose)

    if getattr(args, "mesh_fill_holes_legacy", False):
        logger.warning(
            "[DEPRECATED] --mesh-fill-holes is deprecated and has inverted logic. "
            "Hole filling is ON by default. Use --mesh-no-fill-holes to disable it."
        )

    # ── Input validation (fast checks before any heavy imports) ──────────
    err = _validate_inputs(args)
    if err is not None:
        logger.error("Input validation failed: %s", err)
        return 1

    logger.info("=" * 62)
    logger.info("  Structure from Motion Pipeline  [backend: %s]", args.backend)
    logger.info("=" * 62)

    _seed_rngs(args.seed)

    t_total = time.time()
    stage_times: dict = {}

    # ── COLMAP backend — short-circuit the Python pipeline ────────────────
    if args.backend in ("colmap", "colmap-mvs"):
        from sfm.colmap_backend import ColmapRunner

        runner = ColmapRunner(
            args,
            colmap_bin=args.colmap_bin,
            workspace=args.colmap_workspace,
            keep_workspace=args.colmap_keep_workspace,
        )
        try:
            runner.run()
        except Exception as exc:
            logger.error("COLMAP pipeline failed: %s", exc)
            import traceback

            traceback.print_exc()
            return 1
        elapsed = time.time() - t_total
        logger.info("\n" + "=" * 62)
        logger.info("  COLMAP pipeline complete in %.1fs", elapsed)
        logger.info("  Sparse PLY : %s", Path(args.output).resolve())
        if args.backend == "colmap-mvs" or args.dense:
            dense_out = args.dense_output or str(
                Path(args.output).parent / f"{Path(args.output).stem}_dense.ply"
            )
            logger.info("  Dense  PLY : %s", dense_out)
        logger.info("=" * 62)
        return 0

    # ── Imports (deferred so --help is instant) ───────────────────────────
    from sfm.feature_extraction import FeatureExtractor, SuperPointExtractor, DISKExtractor
    from sfm.feature_matching import (
        FeatureMatcher, SequentialMatcher, VocabTreeMatcher, DINOv2Matcher,
        LightGlueMatcher, LoFTRMatcher,
    )
    from sfm.geometric_verification import GeometricVerifier
    from sfm.reconstruction import IncrementalSfM
    from sfm.point_cloud import PointCloudExporter
    from sfm.utils import list_images, load_image, estimate_intrinsics, check_scene_graph_connectivity
    import numpy as np

    # ── Visualizer (zero cost when --visualize is not set) ────────────────
    if args.visualize:
        from sfm.visualizer import SfMVisualizer

        viz = SfMVisualizer(
            enabled=True,
            output_dir=args.viz_output,
            n_samples=args.viz_samples,
            fmt=args.viz_format,
            interactive=args.viz_interactive,
            save_video=args.viz_save_video,
            dpi=args.viz_dpi,
            seed=args.viz_seed,
        )
    else:
        viz = None  # guaranteed no-op — never imported when disabled

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1 — Discover images & estimate intrinsics
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[1/6]  Loading images…")
    try:
        image_paths = list_images(args.image_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    if len(image_paths) < 2:
        logger.error("Need at least 2 images for reconstruction.")
        return 1

    ckpt_dir = _ckpt_dir(args)
    img_hash = _image_set_hash(image_paths)

    # Intrinsics come from the first *readable* image: a corrupt leading file
    # must not decide the run before extraction has had a chance to skip it.
    sample = None
    sample_path = None
    for _p in image_paths:
        try:
            sample = load_image(_p)
            sample_path = _p
            break
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", Path(_p).name, exc)
    if sample is None:
        logger.error("No readable images in %s", args.image_dir)
        return 1

    K = estimate_intrinsics(sample.shape, image_path=sample_path)
    dist_coeffs = np.zeros(4, dtype=np.float64)  # refined later by BA if enabled

    # A supplied calibration always wins over EXIF and the size-based guess.
    if args.intrinsics:
        try:
            fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        except ValueError:
            logger.error(
                "--intrinsics expects four comma-separated numbers "
                "'fx,fy,cx,cy'; got %r", args.intrinsics,
            )
            return 1
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        logger.info("Intrinsics supplied: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", fx, fy, cx, cy)
    elif args.focal is not None:
        if args.focal <= 0:
            logger.error("--focal must be positive; got %s", args.focal)
            return 1
        logger.info(
            "Focal supplied: %.1f px (overriding estimate of %.1f px)",
            args.focal, K[0, 0],
        )
        K[0, 0] = K[1, 1] = float(args.focal)

    if viz is not None:
        try:
            viz.on_pipeline_start(image_paths, K, args)
        except Exception as _e:
            logger.warning(f"[VIZ] on_pipeline_start: {_e}")

    logger.info(
        f"Camera K (initial):\n"
        f"  [{K[0,0]:.1f}   0   {K[0,2]:.1f}]\n"
        f"  [  0   {K[1,1]:.1f}  {K[1,2]:.1f}]\n"
        f"  [  0     0    1   ]"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2 — Feature extraction
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[2/6]  Feature extraction…")
    features = None
    if args.resume:
        features = _load_checkpoint(ckpt_dir, "features", img_hash)
    if features is None:
        t = time.time()
        _feat_backend = getattr(args, "feature_backend", "sift")
        if _feat_backend == "superpoint":
            extractor = SuperPointExtractor(n_features=args.n_features)
        elif _feat_backend == "disk":
            extractor = DISKExtractor(n_features=args.n_features)
        else:
            extractor = FeatureExtractor(  # type: ignore[call-arg]
                n_features=args.n_features,
                sift_contrast_threshold=args.sift_contrast_threshold,
                sift_edge_threshold=args.sift_edge_threshold,
                sift_n_octave_layers=args.sift_n_octave_layers,
                sift_sigma=args.sift_sigma,
            )
        features = extractor.extract_all(image_paths)
        stage_times["features"] = time.time() - t
        logger.info(f"       Done in {time.time()-t:.1f}s")
        _save_checkpoint(ckpt_dir, "features", features, img_hash)
    else:
        stage_times["features"] = 0.0  # loaded from checkpoint
        logger.info("       Skipped (loaded from checkpoint)")

    total_kps = sum(len(f["keypoints"]) for f in features.values())
    logger.info(f"       {total_kps:,} keypoints total")

    if viz is not None:
        try:
            viz.on_all_features_done(features)
        except Exception as _e:
            logger.warning(f"[VIZ] on_all_features_done: {_e}")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 3 — Feature matching
    # ─────────────────────────────────────────────────────────────────────
    _feat_backend_active = getattr(args, "feature_backend", "sift")
    _match_desc = (
        f"lightglue" if _feat_backend_active == "superpoint"
        else (args.retrieval if args.retrieval != "none" else args.match_strategy)
    )
    logger.info(f"\n[3/6]  Feature matching  [{_match_desc}]…")
    # Include strategy + key params in the checkpoint key so changing
    # --match_strategy, --retrieval, --feature-backend, or --ratio triggers a re-match.
    _key_parts = [f"matches_{_match_desc}_r{args.ratio:.3f}"]
    if _feat_backend_active == "superpoint":
        pass   # no extra params needed for lightglue key
    elif args.retrieval == "dinov2":
        _key_parts.append(f"tk{args.retrieval_top_k}")
    elif args.match_strategy == "sequential":
        _key_parts.append(f"w{args.sequential_window}")
    elif args.match_strategy == "vocab_tree":
        _key_parts.append(f"vw{args.vocab_words}_tk{args.vocab_top_k}")
    match_ckpt_key = "_".join(_key_parts)
    all_matches = None
    if args.resume:
        all_matches = _load_checkpoint(ckpt_dir, match_ckpt_key, img_hash)
    if all_matches is None:
        t = time.time()
        common_kw = dict(
            ratio_threshold=args.ratio,
            cross_check=True,
            min_matches=args.min_matches,
        )
        if _feat_backend_active == "superpoint":
            all_matches = LightGlueMatcher(
                min_matches=args.min_matches,
            ).match_all(features)
        elif args.retrieval == "dinov2":
            all_matches = DINOv2Matcher(
                top_k=args.retrieval_top_k,
                **common_kw,
            ).match_all(features)
        elif args.match_strategy == "sequential":
            all_matches = SequentialMatcher(
                window=args.sequential_window,
                **common_kw,
            ).match_all(features)
        elif args.match_strategy == "vocab_tree":
            all_matches = VocabTreeMatcher(
                n_words=args.vocab_words,
                top_k=args.vocab_top_k,
                **common_kw,
            ).match_all(features)
        elif args.match_strategy == "loftr":
            all_matches = LoFTRMatcher(
                min_matches=args.min_matches,
            ).match_all(features)
        else:
            all_matches = FeatureMatcher(
                workers=args.match_workers, seed=args.seed, **common_kw
            ).match_all(features)
        stage_times["matching"] = time.time() - t
        logger.info(
            f"       Done in {time.time()-t:.1f}s — {len(all_matches)} pairs retained"
        )
        _save_checkpoint(ckpt_dir, match_ckpt_key, all_matches, img_hash)
    else:
        stage_times["matching"] = 0.0  # loaded from checkpoint
        logger.info(
            f"       Skipped (loaded from checkpoint — {len(all_matches)} pairs)"
        )

    if viz is not None:
        try:
            viz.on_all_matching_done(all_matches, features)
        except Exception as _e:
            logger.warning(f"[VIZ] on_all_matching_done: {_e}")

    if not all_matches:
        logger.error("No pairs with sufficient matches — aborting.")
        return 1

    # ── Focal-length search (before verification: E depends on K) ─────────
    focal_curve = None
    if args.focal_search and args.focal is None and not args.intrinsics:
        from sfm.focal_search import search_focal

        t = time.time()
        best_focal, focal_curve = search_focal(
            K, features, all_matches,
            n_pairs=args.focal_search_pairs,
            ransac_threshold=args.ransac_thr,
            min_inliers=args.min_inliers,
            max_reproj_error=args.max_reproj_error,
        )
        K[0, 0] = K[1, 1] = best_focal
        stage_times["focal_search"] = time.time() - t
    elif args.focal_search:
        logger.info(
            "Focal search skipped — an explicit calibration was supplied."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 4 — Geometric verification
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[4/6]  Geometric verification…")
    t = time.time()
    verifier = GeometricVerifier(
        ransac_threshold=args.ransac_thr,
        min_inliers=args.min_inliers,
    )
    verified = verifier.verify_all(features, all_matches, K, dist_coeffs=dist_coeffs)
    stage_times["verification"] = time.time() - t
    logger.info(f"       Done in {time.time()-t:.1f}s — {len(verified)} verified pairs")

    if viz is not None:
        try:
            viz.on_geometric_verification_done(all_matches, verified, features)
        except Exception as _e:
            logger.warning(f"[VIZ] on_geometric_verification_done: {_e}")

    if not verified:
        logger.error("No pairs passed geometric verification — aborting.")
        return 1

    # ── Scene graph connectivity check ────────────────────────────────────
    all_img_indices = list(features.keys())
    components = check_scene_graph_connectivity(
        verified, all_img_indices, min_inliers=args.min_inliers
    )
    if len(components) > 1:
        logger.warning(
            "[SCENE GRAPH] %d disconnected components found:", len(components)
        )
        for ci, comp in enumerate(components):
            comp_names = [
                features[idx]["image_path"].name
                for idx in sorted(comp)
                if idx in features
            ]
            logger.warning(
                "  Component %d: %d images%s",
                ci + 1,
                len(comp),
                "  → " + ", ".join(comp_names) if len(comp) <= 10 else "",
            )
        logger.warning(
            "[SCENE GRAPH] Only the largest component (%d images) will be reconstructed. "
            "Ensure images have sufficient overlap.",
            len(components[0]),
        )

        # Loop-closure: attempt to bridge disconnected components via retrieval
        if getattr(args, "loop_closure", False):
            _lc_backend = getattr(args, "feature_backend", "sift")
            _lc_retrieval = args.retrieval
            _lc_strategy  = args.match_strategy
            if _lc_retrieval == "dinov2" or _lc_strategy == "vocab_tree":
                logger.info(
                    "[LOOP CLOSURE] Attempting to bridge %d disconnected components…",
                    len(components),
                )
                # Build cross-component candidate pairs using the retrieval back-end
                _bridge_pairs: set = set()
                _comp_lists = [sorted(c) for c in components]
                for _ci in range(len(_comp_lists)):
                    for _cj in range(_ci + 1, len(_comp_lists)):
                        for _a in _comp_lists[_ci]:
                            for _b in _comp_lists[_cj]:
                                _bridge_pairs.add(
                                    (min(_a, _b), max(_a, _b))
                                )
                _bridge_pairs -= set(all_matches.keys())  # skip already-matched
                if _bridge_pairs:
                    logger.info(
                        "[LOOP CLOSURE] Matching %d cross-component candidate pairs…",
                        len(_bridge_pairs),
                    )
                    _lc_matcher = _make_flann() if False else None  # import below
                    import cv2 as _cv2
                    from sfm.feature_matching import _match_pair_cpu, _make_flann as _mk_flann
                    _lc_flann = _mk_flann()
                    _lc_new_matches: dict = {}
                    for _a, _b in _bridge_pairs:
                        _m = _match_pair_cpu(
                            _lc_flann,
                            features[_a]["descriptors"],
                            features[_b]["descriptors"],
                            args.ratio,
                            cross_check=True,
                        )
                        if len(_m) >= args.min_matches:
                            _lc_new_matches[(_a, _b)] = _m
                    logger.info(
                        "[LOOP CLOSURE] %d bridge pairs with ≥%d matches — re-verifying…",
                        len(_lc_new_matches), args.min_matches,
                    )
                    _lc_verified = verifier.verify_all(
                        features, _lc_new_matches, K, dist_coeffs=dist_coeffs
                    )
                    if _lc_verified:
                        logger.info(
                            "[LOOP CLOSURE] %d bridge pairs verified — adding to scene graph",
                            len(_lc_verified),
                        )
                        verified.update(_lc_verified)
                        # Recompute connectivity
                        components = check_scene_graph_connectivity(
                            verified, all_img_indices, min_inliers=args.min_inliers
                        )
                        logger.info(
                            "[LOOP CLOSURE] After bridging: %d component(s)",
                            len(components),
                        )
                    else:
                        logger.warning("[LOOP CLOSURE] No bridge pairs survived verification.")
            else:
                logger.warning(
                    "[LOOP CLOSURE] --loop-closure requires --retrieval dinov2 or "
                    "--match_strategy vocab_tree to generate cross-component candidates."
                )

        # Filter verified pairs to the largest component only
        largest_set = components[0]
        verified = {
            k: v for k, v in verified.items()
            if k[0] in largest_set and k[1] in largest_set
        }
    else:
        logger.info(
            "[SCENE GRAPH] Fully connected: %d images in 1 component", len(all_img_indices)
        )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5 — Incremental SfM reconstruction
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[5/6]  Incremental reconstruction…")
    t = time.time()
    refine_intrinsics = not args.no_refine_intrinsics

    per_cam_intr = None
    if getattr(args, "per_camera_intrinsics", False):
        from sfm.intrinsics import estimate_per_image
        per_cam_intr = estimate_per_image(features, K, dist_coeffs)
        logger.info(
            f"Per-camera intrinsics: initialised {len(per_cam_intr)} cameras from EXIF"
        )

    sfm = IncrementalSfM(
        features=features,
        verified_pairs=verified,
        K=K,
        max_reproj_error=args.max_reproj_error,
        ba_interval=args.ba_interval,
        dist_coeffs=dist_coeffs,
        refine_intrinsics=refine_intrinsics,
        fix_principal_point=getattr(args, "ba_fix_principal_point", False),
        visualizer=viz,
        merge_tracks=getattr(args, "track_merge", False),
        per_camera_intrinsics=per_cam_intr,
        ba_backend=getattr(args, "ba_backend", "scipy"),
        local_ba_window=getattr(args, "local_ba_window", 0),
        ba_separate_focal=getattr(args, "ba_separate_focal", False),
        pnp_backend=getattr(args, "pnp_backend", "cv2"),
        track_completion=not args.no_track_completion,
        ba_options={
            "ftol": args.ba_ftol,
            "xtol": args.ba_xtol,
            "gtol": args.ba_gtol,
            "max_nfev": args.ba_max_nfev,
            "param_scaling": not args.ba_no_param_scaling,
        },
    )
    try:
        cameras, points_3d, observations, kp_to_3d = sfm.reconstruct()
    except Exception as exc:
        logger.error(f"Reconstruction failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    stage_times["reconstruction"] = time.time() - t

    # Read back refined intrinsics from the SfM object
    K = sfm.K
    dist_coeffs = sfm.dist_coeffs
    if refine_intrinsics:
        logger.info(
            f"       Refined K: f={K[0,0]:.1f}  "
            f"k1={dist_coeffs[0]:.5f}  k2={dist_coeffs[1]:.5f}"
        )

    logger.info(
        f"       Done in {time.time()-t:.1f}s\n"
        f"       Cameras   : {len(cameras)}/{len(features)} registered\n"
        f"       3-D points: {len(points_3d):,}"
    )

    if len(points_3d) == 0:
        logger.error("No 3-D points reconstructed — aborting.")
        return 1

    if viz is not None:
        try:
            viz.on_reconstruction_complete(
                cameras, points_3d, observations, features, K
            )
        except Exception as _e:
            logger.warning(f"[VIZ] on_reconstruction_complete: {_e}")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6a — Export sparse point cloud
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[6/6]  Exporting sparse point cloud…")
    t = time.time()
    exporter = PointCloudExporter(max_reproj_error=args.max_reproj_error)

    if not args.no_filter:
        points_3d, observations, _ = exporter.filter_outliers(
            points_3d, observations, cameras, K
        )

    colors = exporter.colorize(points_3d, observations, features, cameras, K)

    try:
        exporter.save_ply(args.output, points_3d, colors)
    except Exception as exc:
        logger.error(f"Failed to write PLY: {exc}")
        return 1

    stage_times["export"] = time.time() - t
    logger.info(f"       Done in {time.time()-t:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6b — MVS densification (optional)
    # ─────────────────────────────────────────────────────────────────────
    _dense_out_path = None   # set below if dense succeeds; used by mesh stage
    if args.dense:
        from sfm.mvs import MVSDensifier

        logger.info("\n[6b]   MVS densification (StereoSGBM)…")
        t = time.time()

        # Build covisibility counts from observations: for each pair (i,j) of cameras
        # count the number of 3-D points observed by both.
        _pt_obs: dict = {}
        for _img_idx, _pt_idx, _, _ in observations:
            _pt_obs.setdefault(_pt_idx, set()).add(_img_idx)
        _covis_counts: dict = {}
        for _obs_set in _pt_obs.values():
            _obs_list = sorted(_obs_set)
            for _a in range(len(_obs_list)):
                for _b in range(_a + 1, len(_obs_list)):
                    _key = (_obs_list[_a], _obs_list[_b])
                    _covis_counts[_key] = _covis_counts.get(_key, 0) + 1

        image_paths_map = {idx: features[idx]["image_path"] for idx in features}
        densifier = MVSDensifier(
            mvs_fusion=getattr(args, "mvs_fusion", False),
            fusion_min_views=getattr(args, "mvs_fusion_min_views", 2),
            max_dense_pts=args.max_dense_points,
        )
        dense_pts, dense_colors = densifier.densify(
            cameras=cameras,
            K=K,
            dist_coeffs=dist_coeffs,
            image_paths=image_paths_map,
            max_reproj_error=args.max_reproj_error,
            covisibility_counts=_covis_counts,
        )

        if len(dense_pts) > 0:
            if args.dense_output is None:
                out_stem = Path(args.output).stem
                _dense_out_path = str(
                    Path(args.output).parent / f"{out_stem}_dense.ply"
                )
            else:
                _dense_out_path = args.dense_output

            try:
                exporter.save_ply(_dense_out_path, dense_pts, dense_colors)
                logger.info(
                    f"       Dense PLY saved → {_dense_out_path}  "
                    f"({len(dense_pts):,} points)"
                )
            except Exception as exc:
                logger.error(f"Failed to write dense PLY: {exc}")
                _dense_out_path = None   # export failed; don't pass bad path to mesh
        else:
            logger.warning("       MVS produced no dense points.")

        stage_times["dense"] = time.time() - t
        stage_times["dense_points"] = int(len(dense_pts))
        logger.info(f"       Done in {time.time()-t:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6c — Mesh reconstruction (optional)
    # CRITICAL: mesh stage NEVER crashes the main pipeline — the PLY is
    # already saved.  All mesh errors are logged and swallowed.
    # ─────────────────────────────────────────────────────────────────────
    if args.mesh:
        from sfm.mesh.pipeline import MeshPipeline

        _t_mesh = time.time()

        # Prefer the denser cloud if available, otherwise use sparse output
        _mesh_input = _dense_out_path if _dense_out_path is not None else args.output
        if args.mesh_output is None:
            _mesh_out = str(
                Path(args.output).parent / f"{Path(args.output).stem}_mesh.obj"
            )
        else:
            _mesh_out = args.mesh_output

        # Camera centres let the mesh stage orient point normals towards the
        # views that actually observed each surface point, which is both faster
        # and far more reliable than the tangent-plane MST fallback — and this
        # pipeline has already solved for them.
        from sfm.utils import camera_center as _camera_center

        _cam_centers = np.array(
            [_camera_center(cameras[c]["R"], cameras[c]["t"]) for c in sorted(cameras)],
            dtype=np.float64,
        ) if cameras else None

        try:
            mesh_pipeline = MeshPipeline(args)
            mesh_result = mesh_pipeline.run(
                pointcloud_path=_mesh_input,
                output_path=_mesh_out,
                camera_centers=_cam_centers,
            )

            if mesh_result.success:
                logger.info(f"[MESH] Mesh saved  : {mesh_result.output_path}")
                logger.info(f"[MESH] Faces       : {mesh_result.face_count:,}")
                logger.info(f"[MESH] Vertices    : {mesh_result.vertex_count:,}")
                if mesh_result.cloud_path:
                    logger.info(f"[MESH] Cloud saved : {mesh_result.cloud_path}")
                if mesh_result.report_path:
                    logger.info(f"[MESH] Report      : {mesh_result.report_path}")
                if mesh_result.verdict:
                    logger.info(f"[MESH] Verdict     : {mesh_result.verdict}")
                if viz is not None:
                    try:
                        viz.on_mesh_complete(mesh_result.output_path, mesh_result.stats)
                    except Exception as _e:
                        logger.warning(f"[VIZ] on_mesh_complete: {_e}")
            else:
                logger.warning(f"[MESH] Mesh reconstruction failed: {mesh_result.error}")

        except Exception as exc:
            logger.warning(f"[MESH] Mesh stage failed unexpectedly: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()

        stage_times["mesh"] = time.time() - _t_mesh

    # ── Camera pose export (opt-in, for external evaluation) ──────────────
    if args.export_cameras:
        stage_times["total"] = time.time() - t_total
        try:
            _export_cameras(
                args.export_cameras,
                cameras,
                K,
                dist_coeffs,
                features,
                points_3d,
                observations,
                stage_times,
                args,
                focal_curve=focal_curve,
            )
        except Exception as exc:
            logger.warning("Camera export failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────────
    # Visualization — final outputs
    # ─────────────────────────────────────────────────────────────────────
    if viz is not None:
        from sfm.utils import reprojection_error as _reproj_err

        _errs = [
            _reproj_err(
                points_3d[pt_idx],
                np.array([x, y]),
                K,
                cameras[img_idx]["R"],
                cameras[img_idx]["t"],
            )
            for img_idx, pt_idx, x, y in observations
            if img_idx in cameras and pt_idx < len(points_3d)
        ]
        _rmse = float(np.sqrt(np.mean(np.array(_errs) ** 2))) if _errs else 0.0

        _h, _w = load_image(image_paths[0]).shape[:2]
        try:
            viz.on_pipeline_complete(
                cameras=cameras,
                points_3d=points_3d,
                colors=colors,
                stats={
                    "n_images": len(image_paths),
                    "n_cameras": len(cameras),
                    "n_points": len(points_3d),
                    "resolution": f"{_w}×{_h}",
                    "rmse": _rmse,
                    "elapsed": time.time() - t_total,
                },
            )
        except Exception as _e:
            logger.warning(f"[VIZ] on_pipeline_complete: {_e}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 62)
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Output  : {Path(args.output).resolve()}")
    logger.info(f"  Cameras : {len(cameras)} registered")
    logger.info(f"  Points  : {len(points_3d):,}")
    logger.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
