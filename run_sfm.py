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
        help="Max SIFT features extracted per image. Examples typically have 2k-10k features per image; set higher for large scenes with lots of texture, lower for speed on small/simple scenes.",
    )
    p.add_argument(
        "--sift-contrast-threshold",
        type=float,
        default=0.04,
        help=(
            "SIFT contrast threshold. Lower values detect more low-contrast keypoints "
            "(often more features, but noisier)."
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
        choices=["exhaustive", "sequential", "vocab_tree"],
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
        "--sequential_window",
        type=int,
        default=5,
        help="Window size for sequential matching (images i vs i+1 … i+W). Example: W=5 means image_000.jpg will be matched against image_001.jpg through image_005.jpg (inclusive). Maximum W is N-1, which degrades to exhaustive matching (default: 5).",
    )
    p.add_argument(
        "--vocab_words",
        type=int,
        default=256,
        help="Vocabulary size for vocab_tree matching.",
    )
    p.add_argument(
        "--vocab_top_k",
        type=int,
        default=10,
        help="Number of nearest-neighbour images retrieved per query (vocab_tree).",
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
        "--mesh-fill-holes",
        action="store_true",
        default=True,
        help="Attempt to fill holes in the mesh surface (default: enabled).",
    )
    msh.add_argument(
        "--mesh-keep-pointcloud",
        action="store_true",
        help=(
            "Save the cleaned/prepared point cloud as a separate PLY file "
            "alongside the mesh output."
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

    return None


def _image_set_hash(image_paths: list) -> str:
    """
    Compute a short hash over the sorted image filenames and their sizes.
    Used to invalidate checkpoints when the image set changes.
    """
    h = hashlib.md5()
    for p in sorted(str(p) for p in image_paths):
        h.update(p.encode())
        try:
            h.update(str(os.path.getsize(p)).encode())
        except OSError:
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    from sfm.utils import setup_logging

    setup_logging(args.verbose)

    # ── Input validation (fast checks before any heavy imports) ──────────
    err = _validate_inputs(args)
    if err is not None:
        logger.error("Input validation failed: %s", err)
        return 1

    logger.info("=" * 62)
    logger.info("  Structure from Motion Pipeline  [backend: %s]", args.backend)
    logger.info("=" * 62)

    t_total = time.time()

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
    from sfm.feature_extraction import FeatureExtractor
    from sfm.feature_matching import FeatureMatcher, SequentialMatcher, VocabTreeMatcher
    from sfm.geometric_verification import GeometricVerifier
    from sfm.reconstruction import IncrementalSfM
    from sfm.point_cloud import PointCloudExporter
    from sfm.utils import list_images, load_image, estimate_intrinsics
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

    sample = load_image(image_paths[0])
    K = estimate_intrinsics(sample.shape, image_path=image_paths[0])
    dist_coeffs = np.zeros(4, dtype=np.float64)  # refined later by BA if enabled

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
        extractor = FeatureExtractor(  # type: ignore[call-arg]
            n_features=args.n_features,
            sift_contrast_threshold=args.sift_contrast_threshold,
            sift_edge_threshold=args.sift_edge_threshold,
            sift_n_octave_layers=args.sift_n_octave_layers,
            sift_sigma=args.sift_sigma,
        )
        features = extractor.extract_all(image_paths)
        logger.info(f"       Done in {time.time()-t:.1f}s")
        _save_checkpoint(ckpt_dir, "features", features, img_hash)
    else:
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
    logger.info(f"\n[3/6]  Feature matching  [{args.match_strategy}]…")
    # Include strategy + key params in the checkpoint key so changing
    # --match_strategy or --ratio correctly triggers a re-match.
    match_ckpt_key = f"matches_{args.match_strategy}_r{args.ratio:.3f}"
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
        if args.match_strategy == "sequential":
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
        else:
            all_matches = FeatureMatcher(**common_kw).match_all(features)
        logger.info(
            f"       Done in {time.time()-t:.1f}s — {len(all_matches)} pairs retained"
        )
        _save_checkpoint(ckpt_dir, match_ckpt_key, all_matches, img_hash)
    else:
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
    logger.info(f"       Done in {time.time()-t:.1f}s — {len(verified)} verified pairs")

    if viz is not None:
        try:
            viz.on_geometric_verification_done(all_matches, verified, features)
        except Exception as _e:
            logger.warning(f"[VIZ] on_geometric_verification_done: {_e}")

    if not verified:
        logger.error("No pairs passed geometric verification — aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5 — Incremental SfM reconstruction
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[5/6]  Incremental reconstruction…")
    t = time.time()
    refine_intrinsics = not args.no_refine_intrinsics
    sfm = IncrementalSfM(
        features=features,
        verified_pairs=verified,
        K=K,
        max_reproj_error=args.max_reproj_error,
        ba_interval=args.ba_interval,
        dist_coeffs=dist_coeffs,
        refine_intrinsics=refine_intrinsics,
        visualizer=viz,
    )
    try:
        cameras, points_3d, observations, kp_to_3d = sfm.reconstruct()
    except Exception as exc:
        logger.error(f"Reconstruction failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

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

    logger.info(f"       Done in {time.time()-t:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6b — MVS densification (optional)
    # ─────────────────────────────────────────────────────────────────────
    _dense_out_path = None   # set below if dense succeeds; used by mesh stage
    if args.dense:
        from sfm.mvs import MVSDensifier

        logger.info("\n[6b]   MVS densification (StereoSGBM)…")
        t = time.time()

        image_paths_map = {idx: features[idx]["image_path"] for idx in features}
        densifier = MVSDensifier()
        dense_pts, dense_colors = densifier.densify(
            cameras=cameras,
            K=K,
            dist_coeffs=dist_coeffs,
            image_paths=image_paths_map,
            max_reproj_error=args.max_reproj_error,
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

        logger.info(f"       Done in {time.time()-t:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6c — Mesh reconstruction (optional)
    # CRITICAL: mesh stage NEVER crashes the main pipeline — the PLY is
    # already saved.  All mesh errors are logged and swallowed.
    # ─────────────────────────────────────────────────────────────────────
    if args.mesh:
        from sfm.mesh.pipeline import MeshPipeline

        # Prefer the denser cloud if available, otherwise use sparse output
        _mesh_input = _dense_out_path if _dense_out_path is not None else args.output
        if args.mesh_output is None:
            _mesh_out = str(
                Path(args.output).parent / f"{Path(args.output).stem}_mesh.obj"
            )
        else:
            _mesh_out = args.mesh_output

        try:
            mesh_pipeline = MeshPipeline(args)
            mesh_result = mesh_pipeline.run(
                pointcloud_path=_mesh_input,
                output_path=_mesh_out,
            )

            if mesh_result.success:
                logger.info(f"[MESH] Mesh saved  : {mesh_result.output_path}")
                logger.info(f"[MESH] Faces       : {mesh_result.face_count:,}")
                logger.info(f"[MESH] Vertices    : {mesh_result.vertex_count:,}")
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
