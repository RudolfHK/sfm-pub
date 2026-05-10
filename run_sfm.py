#!/usr/bin/env python3
"""
run_sfm.py — Structure from Motion pipeline entry point.

Usage
-----
    python run_sfm.py --image_dir ./images --output output.ply

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
        description="Pure-Python incremental Structure from Motion pipeline.",
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    from sfm.utils import setup_logging
    setup_logging(args.verbose)

    logger.info("=" * 62)
    logger.info("  Structure from Motion Pipeline")
    logger.info("=" * 62)

    t_total = time.time()

    # ── Imports (deferred so --help is instant) ───────────────────────────
    from sfm.feature_extraction    import FeatureExtractor
    from sfm.feature_matching      import FeatureMatcher, SequentialMatcher, VocabTreeMatcher
    from sfm.geometric_verification import GeometricVerifier
    from sfm.reconstruction        import IncrementalSfM
    from sfm.point_cloud           import PointCloudExporter
    from sfm.utils                 import list_images, load_image, estimate_intrinsics
    import numpy as np

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
