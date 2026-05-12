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

Run `python run_sfm.py --help` for all options.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structure from Motion pipeline — pure-Python or COLMAP backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image_dir", required=True,
        help="Directory containing input JPG/PNG images.",
    )
    p.add_argument(
        "--output", default="output.ply",
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
        "--n_features", type=int, default=8_000,
        help="Max SIFT features extracted per image.",
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
        ),
    )
    p.add_argument(
        "--sequential_window", type=int, default=5,
        help="Window size for sequential matching (images i vs i+1 … i+W).",
    )
    p.add_argument(
        "--vocab_words", type=int, default=256,
        help="Vocabulary size for vocab_tree matching.",
    )
    p.add_argument(
        "--vocab_top_k", type=int, default=10,
        help="Number of nearest-neighbour images retrieved per query (vocab_tree).",
    )
    p.add_argument(
        "--ratio", type=float, default=0.75,
        help="Lowe's ratio-test threshold (lower = stricter).",
    )
    p.add_argument(
        "--min_matches", type=int, default=15,
        help="Min raw matches required to keep a pair.",
    )
    # Geometric verification
    p.add_argument(
        "--min_inliers", type=int, default=15,
        help="Min RANSAC inliers required to accept a pair.",
    )
    p.add_argument(
        "--ransac_thr", type=float, default=1.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    # Reconstruction
    p.add_argument(
        "--max_reproj_error", type=float, default=4.0,
        help="Max reprojection error (px) for triangulation / PnP.",
    )
    p.add_argument(
        "--ba_interval", type=int, default=5,
        help="Run bundle adjustment every N newly registered cameras.",
    )
    p.add_argument(
        "--no_refine_intrinsics", action="store_true",
        help=(
            "Disable joint focal-length and radial-distortion (k1, k2) "
            "refinement in bundle adjustment.  Use when the camera is "
            "pre-calibrated or for speed."
        ),
    )
    # MVS densification
    p.add_argument(
        "--dense", action="store_true",
        help=(
            "Run MVS densification (StereoSGBM) after sparse reconstruction "
            "and save the dense + sparse point cloud."
        ),
    )
    p.add_argument(
        "--dense_output", default=None,
        help=(
            "Output PLY path for the dense cloud.  "
            "Defaults to <output_stem>_dense.ply."
        ),
    )
    # Export
    p.add_argument(
        "--no_filter", action="store_true",
        help="Skip statistical outlier filtering of the final sparse point cloud.",
    )
    # Misc
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ── Visualization (completely optional) ──────────────────────────────
    viz = p.add_argument_group("visualization (all ignored unless --visualize is set)")
    viz.add_argument(
        "--visualize", action="store_true",
        help="Enable the full visualization suite.  Zero overhead when omitted.",
    )
    viz.add_argument(
        "--viz-samples", type=int, default=3, metavar="N",
        help="Number of images / pairs to sample for feature/match visualizations.",
    )
    viz.add_argument(
        "--viz-output", default="sfm_visualization", metavar="DIR",
        help="Directory to save all visualization outputs.",
    )
    viz.add_argument(
        "--viz-format", default="png", choices=["png", "jpg", "pdf"], metavar="FMT",
        help="Output image format for saved figures.",
    )
    viz.add_argument(
        "--viz-interactive", action="store_true",
        help="Open an interactive open3d point-cloud viewer at end of pipeline.",
    )
    viz.add_argument(
        "--viz-save-video", action="store_true",
        help="Export reconstruction growth GIF and point-cloud turntable GIF.",
    )
    viz.add_argument(
        "--viz-dpi", type=int, default=150, metavar="N",
        help="DPI for saved figures.",
    )
    viz.add_argument(
        "--viz-seed", type=int, default=42, metavar="N",
        help="Random seed for reproducible image/pair sampling.",
    )

    # ── COLMAP options (ignored unless --backend colmap / colmap-mvs) ─────
    col = p.add_argument_group(
        "COLMAP options (ignored unless --backend colmap or colmap-mvs)"
    )
    col.add_argument(
        "--colmap-bin", default="colmap", metavar="PATH",
        help="Path to the COLMAP executable (default: 'colmap' on PATH).",
    )
    col.add_argument(
        "--colmap-workspace", default=None, metavar="DIR",
        help=(
            "Directory for COLMAP's internal database and sparse model.  "
            "Defaults to <output_dir>/colmap_workspace/."
        ),
    )
    col.add_argument(
        "--colmap-keep-workspace", action="store_true",
        help="Keep the COLMAP workspace after a successful run (useful for debugging).",
    )
    col.add_argument(
        "--colmap-vocab-tree", default=None, metavar="PATH",
        help=(
            "Path to a pre-built COLMAP vocabulary tree file (.bin).  "
            "Required when --match_strategy vocab_tree is used with the COLMAP backend.  "
            "Download from: https://demuc.de/colmap/#download"
        ),
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    from sfm.utils import setup_logging
    setup_logging(args.verbose)

    logger.info("=" * 62)
    logger.info("  Structure from Motion Pipeline  [backend: %s]", args.backend)
    logger.info("=" * 62)

    t_total = time.time()

    # ── COLMAP backend — short-circuit the Python pipeline ────────────────
    if args.backend in ("colmap", "colmap-mvs"):
        from sfm.colmap_backend import ColmapRunner
        runner = ColmapRunner(
            args,
            colmap_bin      = args.colmap_bin,
            workspace       = args.colmap_workspace,
            keep_workspace  = args.colmap_keep_workspace,
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
    from sfm.feature_extraction    import FeatureExtractor
    from sfm.feature_matching      import FeatureMatcher, SequentialMatcher, VocabTreeMatcher
    from sfm.geometric_verification import GeometricVerifier
    from sfm.reconstruction        import IncrementalSfM
    from sfm.point_cloud           import PointCloudExporter
    from sfm.utils                 import list_images, load_image, estimate_intrinsics
    import numpy as np

    # ── Visualizer (zero cost when --visualize is not set) ────────────────
    if args.visualize:
        from sfm.visualizer import SfMVisualizer
        viz = SfMVisualizer(
            enabled     = True,
            output_dir  = args.viz_output,
            n_samples   = args.viz_samples,
            fmt         = args.viz_format,
            interactive = args.viz_interactive,
            save_video  = args.viz_save_video,
            dpi         = args.viz_dpi,
            seed        = args.viz_seed,
        )
    else:
        viz = None      # guaranteed no-op — never imported when disabled

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
    t = time.time()
    extractor = FeatureExtractor(n_features=args.n_features)
    features  = extractor.extract_all(image_paths)
    logger.info(f"       Done in {time.time()-t:.1f}s")

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
    t = time.time()

    common_kw = dict(
        ratio_threshold=args.ratio,
        cross_check=True,
        min_matches=args.min_matches,
    )
    if args.match_strategy == "sequential":
        matcher = SequentialMatcher(window=args.sequential_window, **common_kw)
    elif args.match_strategy == "vocab_tree":
        matcher = VocabTreeMatcher(
            n_words=args.vocab_words,
            top_k=args.vocab_top_k,
            **common_kw,
        )
    else:
        matcher = FeatureMatcher(**common_kw)

    all_matches = matcher.match_all(features)
    logger.info(f"       Done in {time.time()-t:.1f}s — {len(all_matches)} pairs retained")

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
    K           = sfm.K
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
            viz.on_reconstruction_complete(cameras, points_3d, observations, features, K)
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
                out_stem       = Path(args.output).stem
                dense_out_path = str(Path(args.output).parent / f"{out_stem}_dense.ply")
            else:
                dense_out_path = args.dense_output

            try:
                exporter.save_ply(dense_out_path, dense_pts, dense_colors)
                logger.info(
                    f"       Dense PLY saved → {dense_out_path}  "
                    f"({len(dense_pts):,} points)"
                )
            except Exception as exc:
                logger.error(f"Failed to write dense PLY: {exc}")
        else:
            logger.warning("       MVS produced no dense points.")

        logger.info(f"       Done in {time.time()-t:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Visualization — final outputs
    # ─────────────────────────────────────────────────────────────────────
    if viz is not None:
        from sfm.utils import reprojection_error as _reproj_err
        _errs = [
            _reproj_err(points_3d[pt_idx], np.array([x, y]), K,
                        cameras[img_idx]["R"], cameras[img_idx]["t"])
            for img_idx, pt_idx, x, y in observations
            if img_idx in cameras and pt_idx < len(points_3d)
        ]
        _rmse = float(np.sqrt(np.mean(np.array(_errs) ** 2))) if _errs else 0.0

        _h, _w = load_image(image_paths[0]).shape[:2]
        try:
            viz.on_pipeline_complete(
                cameras   = cameras,
                points_3d = points_3d,
                colors    = colors,
                stats     = {
                    "n_images":   len(image_paths),
                    "n_cameras":  len(cameras),
                    "n_points":   len(points_3d),
                    "resolution": f"{_w}×{_h}",
                    "rmse":       _rmse,
                    "elapsed":    time.time() - t_total,
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
