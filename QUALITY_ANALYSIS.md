# QUALITY_ANALYSIS.md — SfM Pipeline: Technical Depth Assessment

**Prepared by:** Senior CV / Photogrammetry Review  
**Codebase commit:** `claude/sfm-pipeline-SLpu2`  
**Estimated read time:** ~20 minutes

---

## Changelog

### Updated: 2026-05-12

- ✅ **GPU feature detection now functional** — `_extract_kornia()` in `feature_extraction.py` correctly uses GPU-detected LAF keypoints instead of discarding them (previously critical bug).
- ✅ **`SequentialMatcher` implemented** — `feature_matching.py` has a sliding-window sequential matcher, closing GAP-4 for ordered datasets.
- ✅ **`VocabTreeMatcher` implemented** — bag-of-words retrieval with k-means vocabulary, TF-IDF weighting, and GPU-accelerated cosine similarity, closing GAP-4 for large unordered datasets.
- ✅ **GPU descriptor matching implemented** — `_match_pair_gpu()` using `torch.cdist` for full L2 distance matrix, dramatically accelerating brute-force matching when a CUDA GPU is available.
- ✅ **MVS densification implemented** — `sfm/mvs.py` provides a StereoSGBM-based dense reconstruction pipeline, closing GAP-2.
- ✅ **EXIF focal length reading implemented** — `utils.read_exif_focal_px()` reads `FocalLengthIn35mmFilm` tag and converts to pixel units, closing GAP-3 (init side).
- ✅ **Radial distortion (k1, k2) in bundle adjustment** — BA now jointly refines focal length + k1/k2 when `refine_intrinsics=True`, partially closing GAP-1 and GAP-3.
- ✅ **Distortion-aware geometric verification** — `undistort_points()` applied before RANSAC in `geometric_verification.py`, improving epipolar constraint accuracy.
- ✅ **Central device manager** — `sfm/device.py` with `get_device()` / `has_gpu()` utilities.
- ✅ **Visualization suite** — `sfm/visualizer.py`: comprehensive event-driven visualization with zero cost when `--visualize` not passed.
- ✅ **COLMAP backend** — `sfm/colmap_backend.py`: `--backend colmap / colmap-mvs` routes through COLMAP CLI with PLY output compatible with existing tooling.
- 🟡 **Quality tier updated: 🟠 Research Prototype → 🟡 Approaching Solid Open Source** — four of the five originally identified blocking gaps have been addressed or substantially narrowed. Remaining critical gaps: principal point not in BA, no covisibility graph / local BA, no Hartley normalization, no LO-RANSAC on E matrix.
- 🆕 **New gap identified:** Principal point (cx, cy) not refined in BA — only focal length f is optimized.
- 🆕 **New gap identified:** E-matrix estimation uses `cv2.RANSAC` instead of USAC_MAGSAC — inconsistent with F-matrix path.
- 🆕 **New gap identified:** No Hartley normalization (coordinate centering/scaling) before F/E RANSAC — numerical stability issue for wide-resolution images.
- 🆕 **New gap identified:** No SO(3) re-orthogonalization after BA — Rodrigues vectors decode directly to R without SVD projection back to SO(3).
- 🆕 **New gap identified:** No pipeline checkpointing — crash mid-run requires restarting from scratch.
- 🆕 **New gap identified:** No input validation at pipeline entry.

---

## Table of Contents

1. [Full Codebase Audit](#1-full-codebase-audit)
2. [State-of-the-Art Baseline](#2-state-of-the-art-baseline)
3. [Dimension-by-Dimension Comparison](#3-dimension-by-dimension-comparison)
4. [Gap Analysis — Ranked by Impact](#4-gap-analysis--ranked-by-impact)
5. [Overall Quality Rating](#5-overall-quality-rating)
6. [Prioritized Improvement Roadmap](#6-prioritized-improvement-roadmap)
7. [Summary Comparison Table](#7-summary-comparison-table)

---

## 1. Full Codebase Audit

### 1.1 Feature Extraction — `sfm/feature_extraction.py`

**Algorithm:** OpenCV SIFT (Lowe, 2004) with default parameters.

```python
cv2.SIFT_create(nfeatures=8000, nOctaveLayers=3,
                contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
```

**What is implemented well:**
- The parameters match standard practice: `sigma=1.6` and `nOctaveLayers=3` are exactly Lowe's recommended settings.
- A three-tier GPU fallback hierarchy (kornia → CUDA SURF → CPU SIFT) is architecturally sound.
- Output contract (L2-normalized `float32` descriptors) is consistent throughout the pipeline.

**What is simplified or naive:**
- ~~The kornia backend path detects keypoints on GPU but then calls `_extract_sift_cpu(gray)` for descriptors, discarding the kornia keypoints entirely.~~ **✅ FIXED:** `_extract_kornia()` now correctly uses GPU-detected LAF centres and scales to build `cv2.KeyPoint` objects, then calls `sift.compute()` at those positions. The GPU path provides genuine acceleration on CUDA hardware.
- `contrastThreshold=0.04` is conservative; lowering it to `0.02` would increase feature density by 40–60% on real imagery at minimal cost to precision.
- No affine adaptation. SIFT is invariant to similarity transforms (scale + rotation) but not to affine deformations introduced by oblique viewpoints. ASIFT (Morel & Yu, 2009) or MSER covers this case.
- No multi-model fallback when SIFT finds < N features on an image (e.g., smooth surfaces).

**Entirely missing vs. production systems:**
- Learned detectors and descriptors (SuperPoint, DISK, ALIKED, KeyNetAffNet+HardNet8).
- Dense descriptor extraction for MVS (used by COLMAP and Meshroom for patch matching).
- Prefiltering of near-duplicate or near-black images before extraction.

---

### 1.2 Feature Matching — `sfm/feature_matching.py`

**Algorithm:** FLANN KDTree (5 trees, 50 checks), Lowe's ratio test at 0.75, optional cross-check for CPU path; `torch.cdist` L2 brute-force on GPU when available. Three matcher classes: `FeatureMatcher` (exhaustive), `SequentialMatcher` (sliding window), `VocabTreeMatcher` (bag-of-words retrieval).

**What is implemented well:**
- Ratio = 0.75 is the canonical Lowe threshold; cross-check eliminates a significant fraction of incorrect ratio-test survivors.
- FLANN KDTree is the correct choice for SIFT's 128-D `float32` descriptors (preferred over LSH, which suits binary descriptors).
- The `(i, j)` canonical pair ordering avoids duplicate work.
- **✅ NEW:** `SequentialMatcher` pairs only images within a configurable sliding window (`--sequential_window`, default 5), reducing O(N²) to O(N × W) for ordered sequences.
- **✅ NEW:** `VocabTreeMatcher` builds a k-means vocabulary, encodes images as TF-IDF weighted histograms, and retrieves the `top_k` most similar candidates per image via GPU cosine similarity — reducing matching work from O(N²) to O(N × top_k).
- **✅ NEW:** GPU matching path (`_match_pair_gpu`) uses `torch.cdist` for the full L2 distance matrix, with vectorized ratio-test and optional cross-check, dramatically accelerating large batches on CUDA hardware.

**What is simplified or naive:**
- Ratio threshold is fixed at 0.75 regardless of descriptor type, scene complexity, or viewpoint change. Production systems adapt this per-session.
- No guided matching: after a camera is registered, its pose could be used to compute epipolar lines and only search for matches within a narrow band, dramatically improving precision and recall.
- VocabTreeMatcher uses k-means on the local dataset rather than a pre-built large-scale vocabulary (DBoW2/NetVLAD); retrieval quality depends on dataset diversity.

**Entirely missing vs. production systems:**
- Learned matchers: SuperGlue (Sarlin et al., 2020) and LightGlue (Lindenberger et al., 2023) replace ratio + cross-check with an attention graph over all keypoints, achieving far higher correct-match density on low-overlap and textureless scenes.
- Covisibility-based expansion matching: after initial reconstruction, match only image pairs sharing reconstructed 3-D points.

---

### 1.3 Geometric Verification — `sfm/geometric_verification.py`

**Algorithm:** USAC_MAGSAC for F, RANSAC for E, `cv2.recoverPose` for pose.

**What is implemented well:**
- Using USAC_MAGSAC (Barath et al., 2020) is genuinely state-of-the-art for F estimation — it outperforms LO-RANSAC on datasets with moderate noise and is exactly the estimator used by the current OpenCV recommendation. This is the pipeline's strongest algorithmic choice.
- Pre-shuffling matches before RANSAC avoids a documented ordering-dependent performance degradation (and a crash in OpenCV 4.10+).
- The two-stage verification (F → inlier subset → E) is the correct approach: F estimation uses all matches; E estimation re-verifies only the F-inliers, giving a tighter inlier set for `recoverPose`.
- Fallback `E = K^T F K` is mathematically correct when `findEssentialMat` degenerates.
- Near-zero baseline rejection (`‖t‖ < 1e-4`) correctly discards near-pure-rotation pairs that would yield degenerate 3-D structure.

**What is simplified or naive:**
- E estimation uses plain `cv2.RANSAC` (basic 5-point algorithm + random sampling). MAGSAC or LO-RANSAC would be more appropriate here too; the E-inlier set quality directly determines the pose accuracy that seeds the reconstruction. **🆕 NEW GAP:** This inconsistency with the F-matrix path (which uses USAC_MAGSAC) should be fixed.
- `ransac_threshold = 1.0` pixel is a reasonable *starting* value but is applied uniformly across all image pairs regardless of image resolution, estimated baseline, or scene depth. Production systems use scale-adaptive thresholds.
- **🆕 NEW GAP:** No Hartley normalization — pixel coordinates are passed directly to `findFundamentalMat` and `findEssentialMat` without centering and scaling. This is a known numerical stability issue: condition numbers of the DLT system are orders of magnitude better with normalized coordinates (Hartley, 1997).

**Partially addressed:**
- ~~The pipeline operates on raw pixel coordinates throughout.~~ **✅ PARTIAL FIX:** `undistort_points()` is now called before RANSAC in `geometric_verification.py`, correcting for k1/k2 radial distortion when available. However, when the BA-refined k1/k2 are zero (first run), raw coordinates are still used.

**Still entirely missing:**
- Homography + fundamental matrix discrimination: a verified pair that is best explained by a homography (planar scene, pure rotation) should be flagged and excluded from the reconstruction seed, not just from the lowest-baseline filter.
- LO-RANSAC (Local Optimization) for the E-matrix step: after RANSAC converges, re-estimating E from the full inlier set and repeating dramatically improves accuracy at negligible cost.

---

### 1.4 Camera Model — `sfm/utils.py`

**Model:** Pinhole with optional radial distortion (k1, k2), EXIF-initialized focal length, single shared K.

```python
# utils.py estimate_intrinsics(): tries EXIF first, falls back to heuristic
focal = read_exif_focal_px(image_path, H, W)   # ✅ NEW: reads FocalLengthIn35mmFilm
if focal is None:
    focal = float(max(H, W))   # heuristic fallback: ~53° diagonal FoV
K = [[focal, 0, W/2],
     [0, focal, H/2],
     [0, 0,     1  ]]
```

**Current assessment:**

**✅ EXIF focal length reading** — `read_exif_focal_px()` now reads the `FocalLengthIn35mmFilm` EXIF tag (0xA405) and converts it to pixel units using the sensor diagonal formula. This eliminates the systematic 15–25% focal length error on cameras with EXIF data. The heuristic `f = max(H, W)` is used only as a fallback.

**✅ Radial distortion (k1, k2) in BA** — The bundle adjustment now jointly optimizes focal length f and radial coefficients k1, k2 alongside camera extrinsics and 3-D points when `refine_intrinsics=True`. This substantially reduces systematic reprojection error on real cameras.

**🆕 REMAINING GAP — Principal point (cx, cy) not optimized** — The BA parameter vector includes only `[rvec(3), tvec(3), f, k1, k2]` = 9 DOF per camera. cx and cy are fixed at `(W/2, H/2)`. For cameras whose optical center deviates more than ~1% of image dimensions from the image center (common on phone cameras), this introduces a residual systematic bias.

**Remaining limitation** — The shared K assumption means that images taken with different camera models cannot be represented. COLMAP assigns one K per unique camera model, identified via EXIF.

---

### 1.5 Incremental SfM — `sfm/reconstruction.py`

**Algorithm:** Classic incremental SfM (Snavely et al., 2006 / Schönberger & Frahm, 2016).

**What is implemented well:**
- The seed pair selection (max inliers) is a defensible heuristic; the true optimal seed maximizes the *geometric quality* (baseline-to-depth ratio) of initial triangulation, but max-inliers is a reasonable proxy.
- EPNP (`cv2.SOLVEPNP_EPNP`) is the correct algorithm for n > 6 correspondences: it gives a closed-form solution followed by optional Gauss-Newton refinement with O(n) complexity.
- The three-way point acceptance criterion (positive depth in both cameras + bearing angle ≥ 1° + reprojection error ≤ threshold) is sound. The 1° bearing angle filter correctly discards nearly-degenerate triangulations that produce very distant or numerically unstable 3-D points.
- Per-observation storage in `observations` list with correct handling during index remapping is implemented correctly.
- The `_remove_outlier_points` step with median + 3σ threshold mirrors what production pipelines call a "filter track" operation.

**What is simplified or naive:**
- `_get_corr` and `_count_corr` scan all verified pairs to find 2D-3D correspondences for a given image. This is O(N_pairs × N_matches_per_pair) per new image, i.e., O(N²) total as the reconstruction grows. The correct data structure is a *covisibility graph* (COLMAP terminology: camera pairs sharing ≥ k triangulated points are covisible neighbors), which reduces this to O(degree × N_matches) where degree ≈ 10–20.
- Seed pair selection ignores triangulation angle. A pair of images with maximal inliers but minimal baseline (near-identical poses) produces very noisy initial triangulation that contaminates the entire reconstruction. COLMAP selects the seed pair maximizing both inlier count and median triangulation angle.
- No *track building*. Correct SfM maintains a union-find structure over keypoint observations to merge multi-view observations of the same 3-D point into a single *track*. Without this, the same physical world point can be triangulated independently from different pairs, creating duplicate 3-D points with conflicting indices — this is not a correctness bug in this implementation (kp_to_3d prevents re-triangulation of already-assigned keypoints), but the absence means the BA observation set may under-constrain some points.
- Zero distortion coefficients passed to `cv2.solvePnPRansac` (`np.zeros(4)`) means PnP operates on distorted pixel coordinates as if they were undistorted, accumulating systematic errors in the recovered pose.

**Entirely missing:**
- Local bundle adjustment (BA over the covisibility neighborhood of the newly registered camera). COLMAP runs local BA after every new camera, keeping global BA infrequent. Without local BA, drift accumulates between global BA passes.
- Loop closure detection and correction.
- Merge of duplicate tracks (a union-find merge step after each triangulation pass).
- Complete / hierarchical SfM option for large datasets.
- Guided matching expansion post-registration.

---

### 1.6 Bundle Adjustment — `sfm/bundle_adjustment.py`

**Solver:** `scipy.optimize.least_squares`, method `'trf'` (Trust-Region Reflective), Huber loss at f_scale=2.0 px, Jacobian sparsity via `scipy.sparse`.

**What is implemented well:**
- The vectorized `_rodrigues_rotate_batch` implementation avoids looping over observations in Python — the most critical performance decision in the BA code. Verified via the integration test to match `cv2.Rodrigues` output to < 1e-10 error.
- Explicit Jacobian sparsity pattern construction via `lil_matrix → CSR` is the correct approach. Without this, `scipy.optimize.least_squares` would use finite-differences over a dense Jacobian, making it unusably slow for any non-trivial scene.
- Huber loss with f_scale=2.0 px appropriately down-weights observations with reprojection error > 2 px while still penalizing gross outliers.
- The divergence guard (`rmse_final > 3 × rmse_init → revert`) is a sound engineering safety net.
- The consecutive-index remapping for cameras ensures the BA parameter vector is dense (no gaps from unregistered cameras).

**What is simplified or naive:**
- ~~**K is not optimized.**~~ **✅ PARTIAL FIX:** Focal length `f` is now optimized jointly in BA. However, **the principal point (cx, cy) remains fixed** at image center. This is a new gap (GAP-NEW-1) — cx/cy error of > 1% of image dimensions introduces a residual systematic pose bias.
- ~~**No distortion parameters.**~~ **✅ IMPLEMENTED:** k1 and k2 (Brown–Conrady 2-parameter radial model) are now in the BA parameter vector. The distortion Jacobian ∂(u,v)/∂(k1,k2) is computed analytically. **Still missing:** tangential distortion p1, p2.
- **🆕 NEW GAP:** No SO(3) re-orthogonalization — Rodrigues vectors are decoded to R matrices directly without SVD-based projection back onto SO(3). After BA update steps, the recovered R may have determinant slightly ≠ 1 and is not guaranteed to be a proper rotation. This rarely causes visible artifacts but violates the mathematical constraint.
- `scipy.optimize.least_squares` (TRF) uses a Gauss-Newton approximation and is exact in the mathematical formulation, but is substantially slower than Ceres Solver or g2o for large problems. Ceres uses the Levenberg-Marquardt algorithm with analytical Jacobians and a highly optimized sparse linear algebra backend (CHOLMOD / Eigen), while scipy uses finite-differences or user-provided forward-mode differentiation.
- `max_nfev = 200` evaluations is the *per-evaluation* multiplier: the actual limit is `200 × (6C + 3P)`. For C=10, P=1000, this is 3.18 million evaluations — far more than needed. The effective bottleneck is the TRF iteration count, not the evaluation limit.
- Convergence tolerances `ftol = gtol = xtol = 1e-4` are moderately loose. COLMAP uses `1e-6` for final BA.
- **No local BA.** BA runs only every `ba_interval` cameras globally. This is the largest architectural gap in the SfM algorithm itself: without local BA, errors from PnP registration propagate into subsequent triangulation and are not corrected until the next global BA pass.

---

### 1.7 Point Cloud & Export — `sfm/point_cloud.py`

**What is implemented correctly:**
- Bilinear color sampling is sub-pixel accurate and consistent with how the original features were detected.
- The binary little-endian PLY format (float32 XYZ + uint8 RGB) is the most widely supported interchange format (MeshLab, CloudCompare, Open3D all read it natively).
- The `median + 3σ` outlier filter is a standard, defensible approach for the final point cloud.
- Lazy image caching prevents redundant disk reads.

**✅ NEW — MVS Densification implemented (`sfm/mvs.py`):**
The pipeline now includes a StereoSGBM-based dense reconstruction stage (`--dense` flag). For each registered camera pair exceeding a minimum baseline threshold, the pipeline: rectifies the stereo pair, computes Semi-Global Block Matching disparity, back-projects to 3-D, and filters by depth validity. This provides dense coverage orders of magnitude beyond the sparse SIFT keypoints.

**Remaining gaps:**
- GPU PatchMatch stereo (COLMAP's `patch_match_stereo`) is far more accurate and complete than StereoSGBM — accessible via `--backend colmap-mvs`.
- Mesh reconstruction (Poisson surface, Marching Cubes, Delaunay).
- Normal estimation for mesh quality.
- Confidence / weight per point.
- Per-vertex scale from triangulation (useful for noise assessment).

---

## 2. State-of-the-Art Baseline

### 2.1 COLMAP (Schönberger & Frahm, 2016)

**Key algorithmic advantages:**
- SIFT with vocabulary tree retrieval (DBoW2) reduces pair matching from O(N²) to O(N log N).
- Two-view geometry verification with LO-RANSAC, homography/F discrimination, and cheirality check.
- Global SfM via GLOMAP (2024) in addition to incremental.
- Full distortion model (radial k1–k3, tangential p1–p2, thin prism) in BA.
- **Ceres Solver** for BA: sparse LM with analytical Jacobians, orders of magnitude faster than scipy TRF for 1000+ points.
- Local + Global BA cascade: local BA after each new camera, then global BA every N cameras.
- PatchMatch-based MVS dense reconstruction, followed by depth map fusion into a dense point cloud.
- Poisson surface reconstruction pipeline.
- C++ core; GPU-accelerated SIFT and PatchMatch.

### 2.2 OpenSfM (Moulon et al. / Mapillary)

**Key advantages:**
- Global SfM via rotation averaging (Shonan; Rosen et al., 2020) + translation averaging — more robust than incremental for large, loop-closing datasets.
- Supports multiple feature types: SIFT, HAHOG, AKAZE, DSPSift.
- GPS + IMU fusion for aerial/street-level capture.
- Full OpenCV camera models including fisheye.
- Multi-camera rig support.
- Python-native with C++ extensions; relatively accessible codebase.

### 2.3 Meshroom / AliceVision

**Key advantages:**
- Full photogrammetry pipeline in a node-graph UI.
- VLFeat SIFT + AKAZE feature extraction; ANN approximate matching.
- Semi-Global Matching (SGM) dense reconstruction (Hirschmüller, 2007) with depth map fusion.
- Poisson mesh reconstruction + texturing.
- Structured SfM with robust filtering of degenerate configurations.
- Supports custom camera models via an internal camera database.

### 2.4 RealityCapture (Capturing Reality / Epic)

**Key advantages:**
- Proprietary algorithm stack, reportedly the fastest and highest-quality commercial photogrammetry pipeline.
- Handles datasets of 50,000+ images routinely.
- Sub-millimeter accuracy with GCPs.
- Fully GPU-accelerated matching, BA, and dense reconstruction.
- Direct integration of laser scans, LiDAR, and photogrammetric imagery.
- Automatic LOD mesh generation and 8K texture atlasing.

### 2.5 Pix4D

**Key advantages:**
- GPS-anchored bundle adjustment with ground control points (GCPs).
- Processing optimized for regular aerial grids (forward/side overlap).
- DEM, orthophoto, and volumetric measurement outputs.
- Multi-spectral band support (NDVI etc.).
- Calibration database for drone-specific cameras.

### 2.6 GLOMAP (Pan et al., 2024)

**Key advantages:**
- Global SfM reformulation that parametrizes reconstruction in terms of point tracks rather than camera parameters.
- 10–100× faster than incremental COLMAP on large datasets at comparable quality.
- Robust to initialization order (no incremental drift accumulation).
- Directly integrated with COLMAP's feature extraction and matching pipeline.

### 2.7 hloc / Hierarchical Localization (Sarlin et al., 2019)

**Key advantages:**
- **SuperPoint** (DeTone et al., 2018): learned keypoints trained on homographic adaptation; dramatically better repeatability on textureless surfaces, night scenes, and under large illumination changes.
- **SuperGlue** (Sarlin et al., 2020) / **LightGlue** (Lindenberger et al., 2023): attention-based feature matching with context awareness; far outperforms ratio-test + cross-check on low-overlap and wide-baseline image pairs.
- **NetVLAD** image retrieval for efficient pair selection.
- Currently defines the quality ceiling on standard visual localization benchmarks (InLoc, Aachen Day-Night, RobotCar).

---

## 3. Dimension-by-Dimension Comparison

Scoring key: **Poor** / **Basic** / **Good** / **Excellent**

---

### 3.1 Feature Detection & Description Quality

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Viewpoint repeatability | Basic | Excellent |
| Illumination invariance | Basic | Excellent |
| Descriptor distinctiveness | Basic | Excellent |
| Feature density on textureless surfaces | Poor | Good |
| Subpixel localization accuracy | Basic | Good |

**Repo score: Basic**

SIFT (1999/2004) is the foundation on which all modern methods were built and remains serviceable for well-textured scenes under moderate viewpoint change (< 30° between consecutive frames). The default `contrastThreshold=0.04` filters out a meaningful fraction of weakly textured keypoints that learned detectors successfully retain.

For a direct comparison: on the HPatches benchmark (Balntas et al., 2017), standard SIFT achieves ~55% mean matching accuracy at threshold 3px under viewpoint change; SuperPoint achieves ~68%; DISK (Tyszkiewicz et al., 2020) ~73%; ALIKED (Zhao et al., 2023) ~78%. This 23-percentage-point gap in matching precision translates directly to fewer verified pairs and less accurate pose estimates.

**✅ FIXED:** The GPU path in `feature_extraction.py` is now functional. `_extract_kornia()` uses GPU-detected LAF centres and scales to build `cv2.KeyPoint` objects, then calls `sift.compute()` at those positions for descriptors. The GPU path provides genuine acceleration on CUDA hardware.

---

### 3.2 Matching Quality & Robustness

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Correct-match precision | Basic | Excellent |
| Wide-baseline coverage | Poor | Good |
| Repetitive texture handling | Poor | Good |
| Computational scalability (pair selection) | Basic (VocabTree/Sequential avail.) | Good |
| Guided / epipolar-constrained matching | None | Good |

**Repo score: Basic** *(upgraded from Poor–Basic: retrieval strategies now available)*

The Lowe ratio test at 0.75 + cross-check is the industry baseline for descriptor matching and works well when descriptors are genuinely distinctive.

- **✅ UPDATED — Retrieval strategies now available.** `SequentialMatcher` reduces O(N²) to O(N×W) for ordered datasets; `VocabTreeMatcher` implements bag-of-words retrieval (k-means vocabulary + TF-IDF + GPU cosine similarity) reducing pair candidates from O(N²) to O(N×top_k). Default `FeatureMatcher` is still exhaustive — users must opt into the efficient strategies via `--match_strategy`.
- **No guided matching.** After a camera is registered, its pose and the existing 3-D point cloud can be used to predict where each 3-D point should project in the new image (within a few pixels, accounting for uncertainty). Searching only within those predicted regions dramatically increases match precision and recall. This technique is used by COLMAP's `IncrementalMapper::EstimateAndFindNextBestView`.
- Learned matchers (SuperGlue, LightGlue) replace the ratio test entirely with a graph neural network over all keypoint pairs simultaneously. On the ETH3D benchmark, LightGlue recovers 3× more inlier matches than SIFT+ratio on challenging image pairs (large viewpoint change, low texture).

---

### 3.3 Camera Pose Accuracy

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Rotation accuracy (deg) | Basic (1–3°) | Excellent (< 0.1°) |
| Translation accuracy | Basic | Excellent |
| Drift in long sequences | Poor | Good |
| Self-calibration / focal refinement | None | Good–Excellent |
| Distortion correction | None | Excellent |

**Repo score: Basic** *(upgraded from Poor–Basic: distortion model and focal refinement now present)*

The dominant accuracy limiters are:

1. ~~**No distortion model.**~~ **✅ PARTIALLY FIXED:** k1, k2 radial coefficients are now in the BA parameter vector and `undistort_points()` is applied before RANSAC. The principal point (cx, cy) remains fixed at image center (see GAP-NEW-1).

2. ~~**No focal length refinement in BA.**~~ **✅ FIXED:** Focal length f is now jointly optimized in BA alongside extrinsics and distortion coefficients. EXIF-based initialization further reduces the starting bias.

3. **No local BA.** Without local BA, each camera's pose at registration time is based on noisy initial triangulation. Global BA every 5 cameras allows these errors to accumulate, and the 5-camera lag means 4 cameras have sub-optimal poses driving their triangulation steps.

---

### 3.4 Bundle Adjustment Convergence & Accuracy

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Final reprojection RMSE (well-cond. input) | ~1.0–2.0 px | < 0.3 px |
| Optimization scope | Basic (f + k1/k2; cx/cy fixed) | Excellent (K + dist) |
| Robustness to initialization | Good | Excellent |
| Performance (time/memory) | Poor (Python scipy) | Excellent (Ceres C++) |
| Local + global BA cascade | None | Excellent |

**Repo score: Basic**

The BA implementation is mathematically sound. The vectorized Rodrigues rotation and explicit Jacobian sparsity mean it converges to the correct local minimum for the *problem as formulated*. The Huber loss correctly handles outlier observations.

**✅ UPDATED:** Focal length f and radial coefficients k1, k2 are now in the optimization vector, substantially narrowing the gap with COLMAP. The remaining intrinsic gaps are the principal point (cx, cy) and tangential distortion (p1, p2). COLMAP routinely achieves < 0.3 px RMSE because BA jointly optimizes all camera intrinsics, extrinsics, and 3-D points; this repo should now approach 0.5–1.0 px on well-photographed scenes.

The scipy TRF solver is correct but slow. For a problem with C=50 cameras and P=5,000 points (300,000 parameters), Ceres LM completes global BA in ~2–5 seconds on a modern CPU. scipy TRF on the same problem would take 30–120 seconds due to Python dispatch overhead on each residual evaluation, despite the vectorized residual implementation.

---

### 3.5 Reconstruction Completeness

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Spatial coverage of scene | Basic (keypoints only) | Excellent |
| Textureless region coverage | None | Basic–Good |
| Thin structure / fine detail | None | Basic |
| Surface vs. point representation | Sparse only | Dense + Mesh |

**Repo score: Poor (for final output density)**

The output is a *sparse* point cloud of triangulated SIFT keypoints — the same intermediate representation that COLMAP uses before running MVS. For a 100-image scan of a small object, this typically yields 3,000–30,000 points, covering perhaps 2–5% of visible surface area. The same scan via COLMAP+PMVS or Meshroom SGM would yield 500,000–5,000,000 points covering 70–95% of visible surfaces.

For most practical applications (3-D printing, dimensional measurement, mesh generation, texture atlasing), the sparse output is insufficient as a final product. It is only suitable as camera calibration data or as input to a separate MVS step run externally.

---

### 3.6 Point Cloud Density & Noise

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Points per cm² (10 cm object) | ~0.1–2 | 100–1,000 (MVS) |
| Outlier density | Low (filtered) | Very low |
| Surface normal accuracy | N/A | Good–Excellent |
| Confidence / scale metadata | None | Good |

**Repo score: Poor**

The final PLY output contains only triangulated SIFT keypoints. The `filter_outliers` step (median + 3σ on reprojection error) is appropriate for removing geometric outliers from the sparse cloud, and the bilinear color sampling is sub-pixel accurate. However, the cloud density is 2–3 orders of magnitude below what photogrammetry users expect from a dense reconstruction tool.

---

### 3.7 Scalability

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Practical image count limit | ~80–150 | 50,000+ (RC), 5,000+ (COLMAP) |
| Matching time complexity | O(N²) | O(N log N) with retrieval |
| BA time for 100 cams, 10K pts | ~60–300 s | ~2–10 s (Ceres) |
| Memory for 200 images | Moderate (< 8 GB) | Optimized |
| GPU acceleration (actual) | Basic (feature detect + matching) | Full (COLMAP, RC) |

**Repo score: Poor** *(unchanged — VocabTree/Sequential help for matching but BA and _get_corr scalability are still O(N²))*

The pipeline becomes impractical above approximately 100–150 images. With `--match_strategy vocab_tree`, the matching stage now scales sub-quadratically. However: (a) the default exhaustive matcher remains O(N²), (b) scipy BA overhead grows as O((6C + 3P)²) per iteration, and (c) the unoptimized `_get_corr` scan is O(N²) per registration step. COLMAP has demonstrated successful reconstructions of 100,000-image datasets using vocabulary tree retrieval and distributed BA; this pipeline would likely OOM or timeout before reaching 300 images on the Python backend. The COLMAP backend (`--backend colmap`) handles large datasets.

---

### 3.8 Robustness to Difficult Inputs

| Aspect | This Repo | SotA |
|--------|-----------|------|
| Low-overlap images | Poor | Good |
| Reflective / transparent surfaces | Poor | Poor–Basic |
| Low-texture scenes | Poor | Basic–Good |
| Inconsistent lighting | Basic | Good |
| Wide-angle / fisheye lenses | Poor | Good–Excellent |
| Mixed camera models | None | Good |

**Repo score: Poor**

USAC_MAGSAC for F estimation is the strongest robustness feature in the pipeline. Beyond that, the pipeline has no adaptive strategies: thresholds are fixed, there is no scene-aware parameter tuning, and failure on a difficult pair is silent (pair is dropped). Production systems log detailed diagnostics, offer multiple feature types, and have scene-specific parameter profiles.

---

## 4. Gap Analysis — Ranked by Impact

### 🔴 CRITICAL IMPACT

#### ~~GAP-1: No Camera Distortion Model~~ — ✅ SUBSTANTIALLY ADDRESSED

**Status:** `sfm/bundle_adjustment.py` now jointly optimizes focal length + k1 + k2 per camera. `geometric_verification.py` applies `undistort_points()` before RANSAC. `sfm/utils.py` reads EXIF focal length.

**Remaining sub-gap (GAP-NEW-1):** Principal point (cx, cy) not in BA parameter vector. Only 9 DOF per camera [rvec(3), tvec(3), f, k1, k2] rather than the full 11 DOF [rvec(3), tvec(3), fx, fy, cx, cy, k1, k2].

**Quality impact of remaining gap:** Medium — cx/cy offset is typically small (< 1–2% of image dimensions) on modern cameras, but fixing it would further reduce systematic residuals.

**Implementation path:**
```python
# In bundle_adjustment.py: extend cam_params from 9 to 11 DOF
# Add cx, cy to param_vec; extend Jacobian with ∂(u,v)/∂cx and ∂(u,v)/∂cy
# (Both are trivially ±1 in image coordinates)
```

---

#### ~~GAP-2: No MVS / Dense Reconstruction~~ — ✅ ADDRESSED (StereoSGBM)

**Status:** `sfm/mvs.py` implements a StereoSGBM dense reconstruction pipeline. Use `--dense` for the Python backend. For higher-quality GPU PatchMatch stereo, use `--backend colmap-mvs` which invokes COLMAP's `patch_match_stereo` + `stereo_fusion`.

**Remaining gap:** StereoSGBM produces depth maps for image *pairs* only; it does not perform multi-view depth fusion. COLMAP's PatchMatch fuses constraints from multiple views, producing a substantially denser and more accurate cloud. For production use, `--backend colmap-mvs` is recommended when a CUDA GPU is available.

---

### 🟠 HIGH IMPACT

#### ~~GAP-3: No EXIF-Based Focal Length + No K Refinement in BA~~ — ✅ ADDRESSED

**Status:** `utils.read_exif_focal_px()` reads `FocalLengthIn35mmFilm` EXIF tag and converts to pixel units. `bundle_adjustment.py` now jointly optimizes focal length f alongside poses and distortion coefficients. Principal point (cx, cy) is still fixed — see GAP-NEW-1.

---

#### ~~GAP-4: Exhaustive Pairwise Matching — No Retrieval~~ — ✅ ADDRESSED

**Status:** `SequentialMatcher` (sliding window) and `VocabTreeMatcher` (bag-of-words retrieval with GPU cosine similarity) are both implemented in `feature_matching.py`. Select via `--match_strategy sequential` or `--match_strategy vocab_tree`.

---

#### GAP-5: No Local Bundle Adjustment

**Description:** BA runs as a full global operation every `ba_interval` cameras. Without local BA, pose errors from PnP registration propagate uncorrected into downstream triangulation steps.

**Quality impact:** High — drift accumulates in medium/long sequences; final accuracy degrades with sequence length.

**SotA solution:** COLMAP runs local BA over the covisibility neighborhood (cameras sharing ≥ k triangulated points with the new camera) immediately after registration. This is cheap (O(neighborhood_size)) and keeps registration errors bounded.

**Implementation complexity:** Medium — requires a covisibility graph data structure.

**Code-level suggestion:**
```python
# In reconstruction.py: maintain covisibility_graph: Dict[int, Set[int]]
# After _register_image(), call _local_ba(img_idx, covisibility_graph)
# _local_ba extracts the subproblem (covisible cameras + shared points)
# and runs BA only on that subproblem
def _local_ba(self, img_idx, max_neighborhood=10):
    neighbors = self._get_covisible_neighbors(img_idx, max_neighborhood)
    local_cameras = {img_idx: self.cameras[img_idx]}
    local_cameras.update({n: self.cameras[n] for n in neighbors})
    ...
```

---

#### GAP-6: SIFT-Only Feature Extraction — No Learned Features

**Description:** SIFT fails on textureless surfaces, under repeated illumination changes, and on wide-baseline (> 45°) image pairs that are common in practical scans.

**Quality impact:** High on difficult datasets; Low on well-photographed well-textured scenes.

**SotA solution:** SuperPoint (DeTone et al., 2018) — pre-trained on homographic warps of COCO; state-of-the-art on HPatches. DISK (Tyszkiewicz et al., 2020). ALIKED (Zhao et al., 2023).

**Implementation complexity:** Medium — SuperPoint weights are open-source; inference in PyTorch adds ~80 ms/image on CPU, ~5 ms/image on GPU.

**Code-level suggestion:**
```python
# In feature_extraction.py: add _extract_superpoint backend
# Use kornia.feature.SuperPoint if kornia >= 0.7 (already in requirements)
class FeatureExtractor:
    def _try_init_gpu(self):
        ...
        try:
            import kornia.feature as KF
            self._sp = KF.SuperPoint(pretrained=True)
            return "superpoint"
        except: pass
```

---

### 🟡 MEDIUM IMPACT

#### GAP-7: PnP Algorithm — No P3P, No Iterative Refinement

**Description:** `cv2.SOLVEPNP_EPNP` is used without refinement. For correspondences with reprojection noise, EPNP without refinement leaves residuals on the table.

**Quality impact:** Medium — affects registration accuracy in early-stage reconstruction when few correspondences exist.

**SotA solution:** COLMAP uses P3P (Haralick et al., 1991; Gao et al., 2003) inside RANSAC then refines with `SOLVEPNP_ITERATIVE` (Levenberg-Marquardt) on all inliers.

**Implementation complexity:** Easy — one-line change.

**Code-level suggestion:**
```python
# In reconstruction.py _register_image():
ok, rvec, tvec, inliers = cv2.solvePnPRansac(..., flags=cv2.SOLVEPNP_P3P)
if ok and inliers is not None and len(inliers) >= 6:
    rvec, tvec = cv2.solvePnP(
        pts3d[inliers.flatten()], pts2d[inliers.flatten()],
        self.K, np.zeros(4), rvec, tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )[1:3]
```

---

#### GAP-8: No Track Building (Union-Find)

**Description:** The same physical world point triangulated from different pairs appears as separate 3-D points until BA merges them (which it cannot, as point identity is not tracked across observations). The `kp_to_3d` lookup partially mitigates this but does not handle multi-hop correspondence chains.

**Quality impact:** Medium — inflates point count with duplicates; under-constrains BA for points with few registered observations.

**SotA solution:** Union-Find (Disjoint Set) over `(image_idx, keypoint_idx)` nodes. Match chaining merges all keypoints that are matched transitively into one track. COLMAP's `TrackElement` and `TrackManager` classes implement this.

**Implementation complexity:** Medium — classical algorithm, ~100 lines.

---

#### GAP-9: Seed Pair Selection Does Not Maximize Triangulation Angle

**Description:** The seed pair is the pair with the most inliers. For nearby cameras with large overlap, this can be a nearly-pure translation pair with small triangulation angles, producing shallow, noisy initial triangulation.

**Quality impact:** Medium — errors in initial triangulation contaminate all subsequent registration.

**SotA solution:** COLMAP maximizes `score = n_inliers × median_triangulation_angle` for seed selection. Pairs with angle < 5° are penalized heavily.

**Implementation complexity:** Easy — modify `_select_seed_pair` to score pairs by inlier count weighted by median bearing angle.

---

#### GAP-10: No Loop Closure Detection

**Description:** In sequences where the camera returns to a previously photographed region, the reconstruction should detect the loop and add additional constraints to correct accumulated drift. Without this, long sequences and 360° walk-arounds exhibit visible drift.

**Quality impact:** Medium–High for sequences with ground loops; Low for short unclosed sequences.

**SotA solution:** DBoW2 (Gálvez-López & Tardós, 2012) vocabulary tree with temporal/geometric consistency check. COLMAP's vocabulary tree retrieval inherently handles this.

**Implementation complexity:** Hard — requires vocabulary tree construction and a place recognition module.

---

#### GAP-11: No Covisibility Graph

**Description:** O(N²) scanning in `_get_corr` and `_count_corr` becomes the dominant runtime cost once the reconstruction exceeds ~50 cameras. A covisibility graph (`{img_idx: set_of_covisible_img_idxs}`, updated incrementally) reduces this to O(degree).

**Quality impact:** Medium for correctness; High for performance.

**Implementation complexity:** Easy — add a `defaultdict(set)` maintained during `_add_point`/`_link_kp`.

---

### 🟢 LOW IMPACT

#### GAP-12: BA Convergence Tolerances Loose

**Description:** `ftol = gtol = xtol = 1e-4`. Final BA at `1e-6` would reduce residual by ~0.1–0.3 px.

**Implementation complexity:** Easy (one-line change). Only beneficial after distortion and K are added to BA.

#### GAP-13: Feature Count Cap May Be Too Low

**Description:** `nfeatures = 8000` on a 24 MP image may leave useful features on the table. COLMAP uses 8192 by default but on full-resolution images.

**Implementation complexity:** Easy. Set `nfeatures = 0` (unlimited) and let `contrastThreshold` and `edgeThreshold` govern density.

#### GAP-14: No LORANSAC on E Estimation

**Description:** `findEssentialMat` is called with `cv2.RANSAC` (basic). LO-RANSAC would improve E-inlier quality at negligible cost.

**Implementation complexity:** Easy — `cv2.USAC_MAGSAC` also applies to `findEssentialMat`.

---

### 🆕 NEW GAPS (identified in 2026-05-12 re-audit)

#### GAP-NEW-1: Principal Point (cx, cy) Not Optimized in BA

**Description:** The BA parameter vector is `[rvec(3), tvec(3), f, k1, k2]` = 9 DOF per camera. The principal point (cx, cy) is fixed at `(W/2, H/2)` and never refined. For cameras whose optical center deviates more than ~1% of image dimensions from the image center (common on phone cameras with asymmetric lens assemblies), this introduces a residual systematic pose bias.

**Quality impact:** Low–Medium — cx/cy offset is typically < 1–2% on modern cameras, but fixing it would reduce systematic residuals on wide-angle and telephoto lenses.

**Implementation path:**
```python
# In bundle_adjustment.py: extend cam_params from 9 to 11 DOF
# Add cx, cy to param_vec; extend Jacobian with ∂(u,v)/∂cx and ∂(u,v)/∂cy
# (Both are trivially ±1 in image coordinates before focal scaling)
```

---

#### GAP-NEW-2: No Hartley Normalization Before F/E RANSAC

**Description:** Pixel coordinates are passed directly to `findFundamentalMat` and `findEssentialMat` without centering and scaling. Hartley (1997) proved that the condition number of the DLT system is orders of magnitude better with normalized coordinates (centroid → origin, mean distance → √2). This affects numerical accuracy on wide-resolution images (e.g., 4K or higher) where pixel coordinates span thousands of units.

**Quality impact:** Medium — manifests as increased RANSAC iterations required and slightly less accurate F/E estimates on high-resolution inputs. On 1080p inputs the effect is modest; on 4K+ it can be significant.

**Reference:** Hartley, R. (1997). In defense of the eight-point algorithm. *IEEE TPAMI*, 19(6), 580–593.

**Implementation path:**
```python
# In geometric_verification.py: add normalize_points() and denormalize_F()
def _normalize_points(pts):
    """Hartley normalization: center + scale to mean dist √2."""
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    scale = np.sqrt(2.0) / np.maximum(np.linalg.norm(pts_c, axis=1).mean(), 1e-9)
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1.0]])
    return (pts_c * scale), T
# F_denorm = T2.T @ F_norm @ T1
```

---

#### GAP-NEW-3: E-Matrix Estimation Uses cv2.RANSAC Instead of USAC_MAGSAC

**Description:** `findFundamentalMat` uses `cv2.USAC_MAGSAC` (state-of-the-art), but `findEssentialMat` uses `cv2.RANSAC`. This inconsistency means the pose-critical E estimation step uses an inferior estimator compared to F. The E-inlier set directly seeds `cv2.recoverPose` and is the tightest quality gate before camera registration.

**Quality impact:** Medium — USAC_MAGSAC has better inlier recovery on noisy correspondences, so E-inliers are currently slightly under-estimated.

**Implementation path:**
```python
# In geometric_verification.py _estimate_essential():
method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
E, mask_E = cv2.findEssentialMat(pts1_u, pts2_u, K, method=method,
                                  prob=0.9999, threshold=1.0)
```

---

#### GAP-NEW-4: No SO(3) Re-Orthogonalization After BA

**Description:** After BA update steps, Rodrigues vectors are decoded to rotation matrices with `cv2.Rodrigues`. Small floating-point errors in the Rodrigues optimization path can leave R with `det(R) ≈ 1 ± ε` and non-unit row/column norms. SVD-based projection back onto SO(3) (`R = U @ Vt` from `U, S, Vt = np.linalg.svd(R_approx)`) is a one-liner that guarantees exact orthogonality.

**Quality impact:** Low — numerical drift is typically < 1e-6 and rarely causes visible artifacts. However, it is a mathematical correctness issue that compounds over many BA iterations in long sequences.

**Reference:** Grassia, F.S. (1998). Practical parameterization of rotations using the exponential map. *J. Graphics Tools*, 3(3), 29–48.

**Implementation path:**
```python
# In bundle_adjustment.py _unpack_params() or after BA solve:
U, S, Vt = np.linalg.svd(R)
R_ortho = U @ Vt
if np.linalg.det(R_ortho) < 0:
    R_ortho = U @ np.diag([1, 1, -1]) @ Vt
```

---

#### GAP-NEW-5: No Pipeline Checkpointing / --resume

**Description:** If the pipeline crashes or is interrupted mid-run (e.g., after feature extraction, during reconstruction), there is no way to resume — the entire pipeline must restart from scratch including expensive feature extraction and matching steps.

**Quality impact:** Low for correctness, High for usability on large datasets where extraction + matching can take hours.

**Implementation path:** Serialize extracted features, verified pairs, and reconstruction state to disk after each major stage. Add `--resume` flag that loads from checkpoint if available.

---

#### GAP-NEW-6: No Input Validation at Pipeline Entry

**Description:** The pipeline accepts `--image_dir` without verifying that the directory exists, contains supported image formats, or has the minimum number of images required for reconstruction (≥ 2). Invalid inputs produce cryptic errors deep in the pipeline rather than actionable error messages at startup.

**Quality impact:** Low for technical correctness, High for usability.

**Implementation path:**
```python
# In run_sfm.py, before pipeline start:
def validate_inputs(image_dir, output_path):
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    if len(images) < 2:
        raise ValueError(f"Need ≥ 2 images, found {len(images)} in {image_dir}")
```

---

## 5. Overall Quality Rating

### 🟡 Approaching Solid Open Source *(upgraded from 🟠 Research Prototype)*

> The mathematics are correct, the architecture mirrors the classic incremental SfM algorithm, and the implementation is clean and readable. Four of the five originally identified blocking gaps have been addressed: radial distortion + focal length are now in BA, EXIF initialization is implemented, MVS densification is available, and both sequential and vocabulary-tree matching strategies are provided. On well-photographed, well-lit, richly textured scenes, the pipeline now produces geometrically accurate sparse reconstructions and can optionally produce dense point clouds.

**Evidence for upgraded rating:**
- ✅ All fundamental algorithmic steps are present and mathematically correct.
- ✅ USAC_MAGSAC for F estimation is genuinely state-of-the-art.
- ✅ Vectorized BA with Jacobian sparsity is correctly implemented.
- ✅ The integration test demonstrates successful end-to-end reconstruction.
- ✅ Radial distortion (k1, k2) in BA — systematic corner distortion error substantially reduced.
- ✅ EXIF focal length initialization — eliminates 15–25% systematic depth error on cameras with EXIF data.
- ✅ MVS densification available — StereoSGBM Python backend + COLMAP GPU PatchMatch via `--backend colmap-mvs`.
- ✅ Sequential + VocabTree matching — enables large ordered and unordered datasets.
- ✅ GPU feature detection and matching — functional on CUDA hardware.
- ⚠️ Principal point (cx, cy) still fixed — residual systematic bias for off-center lenses.
- ⚠️ No local BA — drift accumulates between global BA passes.
- ⚠️ No Hartley normalization — numerical stability degraded on high-resolution inputs.
- ❌ No loop closure — drift in long sequences is uncorrected.
- ❌ O(N²) Python BA limits scalability to < 150 cameras on pure-Python backend.

**Comparison with Bundler (Snavely et al., 2006) + PMVS (Furukawa & Ponce, 2010):** This repo is now broadly comparable to the Bundler era (2006–2010) in algorithmic scope. The key remaining gaps versus Bundler are local BA and union-find track building.

### What is needed to reach 🟢 Solid Open Source:

1. Add local BA after each camera registration (highest-priority remaining gap).
2. Add Hartley normalization + USAC_MAGSAC for E-matrix estimation.
3. Add principal point (cx, cy) to BA parameter vector.
4. Add covisibility graph for O(degree) correspondence lookup.
5. Add SO(3) re-orthogonalization after BA.
6. These changes together would bring quality and robustness close to early COLMAP (2016) on moderate-scale datasets.

---

## 6. Prioritized Improvement Roadmap

### Phase 1 — Quick Wins (< 1 week, high quality impact)

| Item | What to implement | Reference | Expected improvement |
|------|------------------|-----------|---------------------|
| **P1-1** | Read EXIF focal length via `piexif` or `Pillow`; use as K init | COLMAP EXIF parsing | -15–25% depth error on typical cameras |
| **P1-2** | Add P3P + iterative refinement after PnP RANSAC | `cv2.SOLVEPNP_P3P` + `SOLVEPNP_ITERATIVE` | +0.2–0.5 px pose accuracy |
| **P1-3** | Apply `cv2.USAC_MAGSAC` to E estimation (not just F) | `findEssentialMat(..., cv2.USAC_MAGSAC, ...)` | +10–20% E-inlier quality |
| **P1-4** | Lower `contrastThreshold` to 0.02; raise `nfeatures` to 16000 | Lowe (2004) | +30–60% keypoint density on semi-textured scenes |
| **P1-5** | Add `SequentialMatcher`: match only within a sliding window of K images | — | Enables video/sequential-image datasets |
| **P1-6** | Score seed pair by inliers × median triangulation angle | Schönberger & Frahm (2016) §3.1 | Better initial triangulation quality |
| **P1-7** | Tighten BA tolerances from `1e-4` to `1e-6` | — | -0.05–0.15 px final RMSE (synergistic with P2 items) |

---

### Phase 2 — Core Quality Upgrades (2–6 weeks, critical impact)

| Item | What to implement | Reference | Expected improvement |
|------|------------------|-----------|---------------------|
| **P2-1** | Radial distortion (k1, k2) in BA parameter vector + `cv2.undistortPoints` pre-processing | COLMAP `CameraModelId::RADIAL` | -1–5 px systematic RMSE on real imagery |
| **P2-2** | Optimize fx, fy, cx, cy jointly with poses in BA | Standard SfM self-calibration | Enables accurate reconstruction without EXIF |
| **P2-3** | Covisibility graph (`defaultdict(set)`) maintained during `_add_point` | COLMAP `SceneGraph` | O(N²) → O(degree·N) for `_get_corr` |
| **P2-4** | Local BA over covisibility neighborhood after each camera registration | Schönberger & Frahm (2016) §3.3 | -0.3–1.0 px per-camera pose error |
| **P2-5** | Vocabulary tree pair selection using FAISS + compressed SIFT aggregates | COLMAP VocabTree; Philbin et al. (2007) | N²→N log N matching; enables 1000+ image datasets |
| **P2-6** | Union-Find track builder to merge multi-hop correspondences | Snavely et al. (2006) §3.2 | Better BA constraint coverage; fewer duplicate points |
| **P2-7** | Export COLMAP-compatible `cameras.bin`, `images.bin`, `points3D.bin` | COLMAP data format | Enables use of COLMAP MVS / OpenMVS as next step |

---

### Phase 3 — Advanced Features (1–3 months, high completeness impact)

| Item | What to implement | Reference | Expected improvement |
|------|------------------|-----------|---------------------|
| **P3-1** | Dense reconstruction via OpenMVS subprocess or OpenCV `StereoBM`/`StereoSGBM` on rectified pairs | OpenMVS; Hirschmüller (2007) | Dense point cloud; 100–1000× point density |
| **P3-2** | SuperPoint backend (open-source weights via kornia) | DeTone et al., CVPRW 2018 | +20–30% matches on textureless scenes |
| **P3-3** | Poisson surface reconstruction from dense cloud | Kazhdan et al. (2006) via Open3D | Mesh output suitable for 3-D printing/CAD |
| **P3-4** | Loop closure: DBoW2-style vocabulary tree + geometric re-verification | Gálvez-López & Tardós, TRO 2012 | Correct drift in 360° walk-arounds |
| **P3-5** | Fisheye / equirectangular camera model | OpenCV fisheye model | Enables GoPro, action cameras, 360° rigs |

---

### Phase 4 — SotA Features (research-level, optional)

| Item | What to implement | Reference | Expected improvement |
|------|------------------|-----------|---------------------|
| **P4-1** | LightGlue / SuperGlue matcher | Lindenberger et al., ICCV 2023 | Near-perfect matching on textureless scenes |
| **P4-2** | Global SfM via rotation + translation averaging (GLOMAP) | Pan et al., 2024 | 10–100× faster on large datasets; no drift |
| **P4-3** | Ceres Solver BA via PyCeres or g2o Python bindings | Agarwal et al. (2012) | 10–50× BA speedup on large problems |
| **P4-4** | NeRF-initialized depth for MVS seed | NeRF; Instant-NGP | Improved MVS on low-texture scenes |
| **P4-5** | Multi-camera rig support (shared extrinsics) | OpenSfM multi-camera | Enables stereo rigs and 360° camera arrays |

---

## 7. Summary Comparison Table

Scoring: ❌ None · ⚠️ Poor · 🔵 Basic · 🟡 Good · ✅ Excellent

| Dimension | **This Repo** | COLMAP | OpenSfM | Meshroom | RealityCapture | hloc |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Feature quality | 🔵 SIFT | 🟡 SIFT+GPU | 🟡 Multi | 🟡 Multi | ✅ Prop. | ✅ SuperPoint |
| Matching strategy | 🔵 VocabTree avail. | 🟡 VocabTree | 🟡 Graph | 🟡 ANN | ✅ Prop. | ✅ SuperGlue |
| Camera model | 🔵 Pinhole+k1k2 | ✅ Full dist. | ✅ Fisheye | ✅ Multi | ✅ Full | ✅ Full |
| Geometric verification | 🟡 MAGSAC | ✅ LO-RANSAC | 🟡 RANSAC | 🟡 RANSAC | ✅ Prop. | ✅ MAGSAC+ |
| Bundle adjustment | 🔵 scipy TRF | ✅ Ceres LM | ✅ Ceres | 🟡 Custom | ✅ Prop. | ✅ Ceres |
| K + distortion in BA | 🔵 f+k1k2 only | ✅ | ✅ | ✅ | ✅ | ✅ |
| Local BA | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Loop closure | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dense MVS output | 🔵 StereoSGBM | ✅ | ⚠️ | ✅ | ✅ | ❌ |
| Mesh generation | ❌ | 🔵 | ❌ | ✅ | ✅ | ❌ |
| Scalability (N images) | ⚠️ < 150 | ✅ > 50,000 | ✅ > 10,000 | 🟡 > 1,000 | ✅ > 50,000 | 🟡 > 1,000 |
| GPU acceleration | 🔵 Partial | ✅ | 🔵 | 🔵 | ✅ | ✅ |
| **Overall** | 🟡 Approaching Solid | ✅ SotA | ✅ SotA | 🟡 Solid | ✅ SotA | ✅ SotA |

---

### Key Takeaway

This codebase is a **complete, educationally valuable, and mathematically correct implementation** of incremental SfM. Every stage is present, the code is clean, and the architectural choices (USAC_MAGSAC, Huber loss, Jacobian sparsity, EPNP) reflect genuine knowledge of the literature. Following recent improvements, it now handles lens distortion (k1, k2), EXIF focal initialization, sequential/vocabulary-tree matching, and MVS densification.

It will reconstruct well-photographed, richly textured scenes with moderate overlap and up to ~150 images on the Python backend (unlimited via `--backend colmap`). For practical photogrammetry applications, the remaining critical gaps are: **(1) local BA after each camera registration**, **(2) Hartley normalization + USAC_MAGSAC for E-matrix**, and **(3) covisibility graph for scalable correspondence lookup**. Addressing these three items, estimated at 2–3 weeks of focused engineering, would elevate the pipeline to 🟢 **Solid Open Source** quality — comparable to early COLMAP circa 2016.

---

*References: Lowe (2004) IJCV; Snavely et al. (2006) SIGGRAPH; Hirschmüller (2007) TPAMI; Agarwal et al. (2012) ECCV; Schönberger & Frahm (2016) CVPR; DeTone et al. (2018) CVPRW; Sarlin et al. (2020) CVPR; Barath et al. (2020) CVPR; Lindenberger et al. (2023) ICCV; Pan et al. (2024) ECCV.*
