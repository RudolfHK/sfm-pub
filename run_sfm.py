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
        help="Output PLY file path.",
    )
    # Feature extraction
    p.add_argument(
        "--n_features", type=int, default=8_000,
        help="Max SIFT features extracted per image.",
    )
    # Matching
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
    # Export
    p.add_argument(
        "--no_filter", action="store_true",
        help="Skip statistical outlier filtering of the final point cloud.",
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
    from sfm.feature_extraction   import FeatureExtractor
    from sfm.feature_matching     import FeatureMatcher
    from sfm.geometric_verification import GeometricVerifier
    from sfm.reconstruction       import IncrementalSfM
    from sfm.point_cloud          import PointCloudExporter
    from sfm.utils                import list_images, load_image, estimate_intrinsics

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
    K = estimate_intrinsics(sample.shape)
    logger.info(
        f"Camera K (estimated):\n"
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
    logger.info("\n[3/6]  Feature matching…")
    t = time.time()
    matcher     = FeatureMatcher(
        ratio_threshold=args.ratio,
        cross_check=True,
        min_matches=args.min_matches,
    )
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
    verified = verifier.verify_all(features, all_matches, K)
    logger.info(f"       Done in {time.time()-t:.1f}s — {len(verified)} verified pairs")

    if not verified:
        logger.error("No pairs passed geometric verification — aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5 — Incremental SfM reconstruction
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[5/6]  Incremental reconstruction…")
    t = time.time()
    sfm = IncrementalSfM(
        features=features,
        verified_pairs=verified,
        K=K,
        max_reproj_error=args.max_reproj_error,
        ba_interval=args.ba_interval,
    )
    try:
        cameras, points_3d, observations, kp_to_3d = sfm.reconstruct()
    except Exception as exc:
        logger.error(f"Reconstruction failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info(
        f"       Done in {time.time()-t:.1f}s\n"
        f"       Cameras   : {len(cameras)}/{len(features)} registered\n"
        f"       3-D points: {len(points_3d):,}"
    )

    if len(points_3d) == 0:
        logger.error("No 3-D points reconstructed — aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6 — Export point cloud
    # ─────────────────────────────────────────────────────────────────────
    logger.info("\n[6/6]  Exporting point cloud…")
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
