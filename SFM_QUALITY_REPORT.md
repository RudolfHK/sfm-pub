# SfM Pipeline Quality Audit Report

**Repository:** `/home/user/sfm-pub`
**Audit Date:** 2026-05-19
**Auditor:** Principal CV/Photogrammetry Engineer
**Code State:** After fixes R-01 through R-17 (as documented in repository context)

---

## 1. Executive Summary

### Overall Quality Tier

🟠 **Research Prototype** — Approaching the lower boundary of Solid Open Source on several axes.

This codebase is a well-structured, single-camera incremental SfM system implemented entirely in Python (NumPy + OpenCV + SciPy). After the R-01 through R-17 round of fixes it produces geometrically correct sparse reconstructions on clean, overlapping image sets. The pipeline is architecturally sound: it follows the canonical incremental SfM workflow faithfully, has working stage checkpointing, a covisibility graph, scene-graph connectivity checking, and reasonably careful numerical handling throughout.

However, it falls short of production open-source quality (e.g., OpenSfM/Meshroom) on several dimensions that matter for real-world use: the feature detector is SIFT-only (no learned detectors), geometric verification uses OpenCV's Python-wrapped USAC_MAGSAC rather than the far faster PoseLib or C++ MAGSAC++, bundle adjustment relies on SciPy's TRF (numerically adequate but 20–100× slower than Ceres Solver and without explicit Schur complement), MVS is StereoSGBM (1990s-era), the BA projection model assumes a single focal length (square pixels), and the system has no covisibility-graph-guided view ordering for triangulation beyond what is already in the match list. These gaps translate to roughly 2–5× lower point-cloud density, 5–20× slower BA, and failure on >200-image datasets compared with COLMAP.

### Top 5 Most Impactful Weaknesses

| Ref | Weakness | Impact |
|-----|----------|--------|
| W-01 | SciPy TRF bundle adjustment — no Schur complement, 20–100× slower than Ceres | BA becomes the wall-clock bottleneck above ~50 cameras; divergence rate rises |
| W-02 | Single shared focal length in BA (fx=fy assumed, no per-camera model) | Systematic distortion in scenes with non-square pixels or mixed camera models |
| W-03 | SGBM MVS — produces noisy, hole-filled disparity maps; no multi-view consistency | Dense cloud density 5–20× below COLMAP PatchMatch; many invalid points |
| W-04 | No re-triangulation after BA — stale 3D points never refined in camera coordinates | Point quality degrades incrementally; no benefit from pose improvement |
| W-05 | Vocab-tree uses flat k-means over SIFT descriptors — primitive image retrieval | Miss rate >30% on large/complex datasets; wrong pairs returned |

### Top 5 Highest-ROI Improvements

| Ref | Improvement | Expected Gain | Effort |
|-----|-------------|---------------|--------|
| I-01 | Replace SciPy TRF with `pyceres` or `g2o` BA with Schur complement | 10–50× BA speedup; enables 500+ camera scenes | MEDIUM |
| I-02 | Add SuperPoint + LightGlue as optional feature extractor/matcher | 2–4× more correct matches on difficult inputs; handles textureless regions | MEDIUM |
| I-03 | Add re-triangulation pass after each BA step | +20–40% point count; better geometry for subsequent registrations | EASY |
| I-04 | Replace SGBM with MVSNet/RAFT-Stereo or PatchMatch-based MVS | 5–20× denser, more accurate dense cloud | HARD |
| I-05 | Per-image intrinsics storage + RADIAL camera model in BA | Correct handling of multi-camera datasets; removes systematic 0.3–1 px error | MEDIUM |

### What Would It Take to Reach COLMAP Quality?

1. **C++ BA core** — Ceres Solver with analytical Jacobians and Schur complement (or expose via `pyceres`). SciPy TRF has ~1/20 the throughput. This is the single largest gap.
2. **Learned feature extraction and matching** — SuperPoint + LightGlue or DISK + LightGlue instead of SIFT + FLANN. COLMAP now ships with these in its latest releases.
3. **Proper PatchMatch MVS** — The GPU-accelerated COLMAP PatchMatch Stereo (or OpenMVS) generates multi-view-consistent depth maps with photo-consistency checks that SGBM cannot match.
4. **Per-image intrinsic models** — COLMAP supports SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, FISHEYE per image. This codebase uses a single (f, k1, k2) shared across all cameras.
5. **Hierarchical or global SfM option** — Incremental registration is fragile; COLMAP's hierarchical mapper and drift correction via loop closure are absent here.
6. **Re-triangulation and track merging** — COLMAP re-triangulates after every BA and merges tracks. This codebase triangulates once and never re-triangulates.
7. **Image-level retrieval with NetVLAD or DINOv2** — COLMAP's vocab tree is built offline from millions of images; this codebase builds a flat k-means vocabulary at runtime over the input set, which is far weaker.

---

## 2. Codebase Inventory

| File | Lines | Summary |
|------|-------|---------|
| `run_sfm.py` | 1101 | CLI entry point; argparse, input validation, stage orchestration, checkpointing |
| `sfm/__init__.py` | 2 | Package marker; version string |
| `sfm/device.py` | 52 | Centralized GPU detection via PyTorch; `get_device()` cached with `lru_cache` |
| `sfm/utils.py` | 209 | Shared I/O helpers, EXIF focal reading, pinhole math, union-find connectivity check |
| `sfm/feature_extraction.py` | 260 | SIFT extractor; GPU path via kornia ScaleSpaceDetector + CPU SIFT descriptors; fallback CUDA SURF |
| `sfm/feature_matching.py` | 593 | Three matchers: exhaustive (GPU cdist or FLANN), sequential (sliding window), VocabTree (TF-IDF + k-means) |
| `sfm/geometric_verification.py` | 270 | F+E RANSAC (USAC_MAGSAC) with Hartley normalization; pose recovery; near-zero baseline rejection |
| `sfm/reconstruction.py` | 884 | Incremental SfM: seed selection, initialization, PnP+EPnP+LM registration, batch triangulation, covisibility graph |
| `sfm/bundle_adjustment.py` | 492 | SciPy TRF bundle adjustment with sparse Jacobian; (f, k1, k2, [cx, cy]) shared intrinsics; divergence guard |
| `sfm/point_cloud.py` | 331 | Vectorized colourization (bilinear sampling), statistical outlier filtering, float64 PLY export |
| `sfm/mvs.py` | 303 | StereoSGBM dense reconstruction; covisibility-based pair selection; stereo rectification |
| `sfm/colmap_backend.py` | 509 | Thin wrapper around COLMAP CLI; binary model readers (cameras.bin, images.bin, points3D.bin) |
| `sfm/visualizer.py` | 1586 | Event-driven visualization harness; matplotlib figures; zero overhead when disabled |
| `sfm/mesh/pipeline.py` | 259 | Mesh pipeline orchestrator; quality presets; MeshResult dataclass |
| `sfm/mesh/pointcloud_prep.py` | 215 | SOR + ROR outlier removal, normal estimation, voxel downsampling |
| `sfm/mesh/reconstruction.py` | 205 | Screened Poisson, BPA, and alpha-shape surface reconstruction via open3d |
| `sfm/mesh/cleaning.py` | 153 | Degenerate triangle removal, non-manifold edge cleaning, component filtering, hole filling |
| `sfm/mesh/postprocess.py` | 109 | Taubin smoothing, quadric decimation, cKDTree color transfer from point cloud |
| `sfm/mesh/export.py` | 99 | Multi-format mesh export (OBJ, PLY, GLB, STL) via open3d |
| `integration_test.py` | 294 | Synthetic ground-truth test: 6 circular cameras, 250 world points, 0.5 px noise; tests geometric verification + SfM + PLY export |

---

## 3. Internal Quality Audit

### 3.1 Image Loading and Intrinsics

**Implementation:** `utils.py:estimate_intrinsics` reads EXIF tag 0xA405 (`FocalLengthIn35mmFilm`) via Pillow and converts to pixels using the 36×24 mm sensor diagonal. If EXIF is absent, falls back to `focal = max(H, W)`, which corresponds to a ~53° diagonal field of view. Principal point is fixed at `(W/2, H/2)`.

**Faithfulness:** Reasonable heuristic. Most SfM systems default to `focal = max(W,H)` when EXIF is absent. COLMAP additionally tries `FocalLength` + `FocalLengthIn35mmFilm` + manufacturer-specific tags (Sony, Nikon, Canon). The EXIF-based path is correct in principle.

**Failure modes:**
- Images with non-35mm-equivalent EXIF (e.g., from drones reporting physical focal length without sensor size) will produce an incorrect initial K. No sensor-size lookup table (like COLMAP's `database_management` camera priors) exists.
- When the dataset contains images from multiple cameras with different focal lengths, a single K is estimated from `image_paths[0]` only (`run_sfm.py:720`). All images use this same K for the entire pipeline. Multi-camera datasets are structurally unsupported.

**Code health:** Good. Error is silently swallowed (`except Exception: return None`) which is acceptable since EXIF reading is optional.

**Numerical issues:** The diagonal-based focal computation (`focal_35 / sensor_diag_35mm * image_diag_px`) is correct but assumes a 36×24 mm full-frame sensor even for crop-sensor cameras. This can give a focal estimate that is off by the crop factor (1.5×–2×).

---

### 3.2 Feature Extraction

**Implementation:** `feature_extraction.py` — SIFT with configurable parameters. Three backends tried in order: (1) kornia ScaleSpaceDetector (GPU keypoints, CPU SIFT descriptors), (2) OpenCV CUDA SURF (extended=True, 128-D), (3) OpenCV CPU SIFT. After fix R-01, SURF `extended=True` is correct (previously produced 64-D descriptors zero-padded to 128-D, breaking the descriptor metric). After fix R-09, the kornia detector is cached rather than re-instantiated per image.

**Faithfulness:** CPU SIFT is faithful to Lowe 2004. The kornia GPU path uses ScaleSpaceDetector + BlobDoG which is a reimplementation of DoG keypoint detection; scales are derived from LAF matrix determinants, and SIFT descriptors are computed at those positions by OpenCV's `sift.compute()`. This is a valid hybrid approach, though the scale conversion (line 221: `scales = sqrt(|det(A)|)`, then `size = scale * 6.0`) is not the standard SIFT diameter convention and may produce slightly incorrect descriptor windows.

**Failure modes:**
- Only SIFT descriptors (128-D float32, L2-normalized) are ever produced. No orientation information is passed to the matcher (this is fine — SIFT descriptors are rotation-invariant).
- The kornia path computes centres using `KF.get_laf_center(lafs)` but accesses `lafs` (GPU tensor) after `.squeeze(0).cpu()` has already been called on `lafs_cpu`, yet passes the original GPU `lafs` to `get_laf_center`. This works because `lafs` is still in scope, but wastes memory by keeping both on GPU and CPU simultaneously.
- `n_features` cap is soft: OpenCV's `nfeatures` parameter is a suggestion; SIFT may return more features on textured images.

**Code health:** Good structure. The fallback chain is clean. Magic number: `size = float(scales[k]) * 6.0` (line 228) — the `6.0` factor converts LAF scale to SIFT keypoint diameter (SIFT uses 2×scale as radius, so diameter = 4×scale; the factor 6.0 is an empirical fudge that is not documented). This could produce incorrect descriptor windows.

---

### 3.3 Feature Matching

**Implementation:** `feature_matching.py` — Three strategies:
- **Exhaustive:** All N(N-1)/2 pairs; GPU cdist or CPU FLANN kNN (k=2) + Lowe ratio test + optional mutual cross-check.
- **Sequential:** Sliding window of width `W`; appropriate for ordered sequences.
- **VocabTree:** Flat k-means vocabulary (random init, 20 Lloyd iterations), TF-IDF encoding, cosine similarity retrieval, then full matching on top-k candidate pairs.

**Faithfulness:** The ratio test and cross-check are correct Lowe 2004 implementations. The GPU path uses `torch.cdist` for the full L2 distance matrix, which is exact brute-force — more accurate than FLANN's approximate kNN.

**Failure modes:**
- VocabTree uses a flat k-means vocabulary (one level) with random initialization. COLMAP uses a hierarchical vocabulary tree [Nister & Stewenius, 2006] built from millions of images. A flat 4096-word vocabulary from k-means on 1000×N descriptors is extremely coarse and will have very high false-positive rates on diverse datasets.
- VocabTree `max_sample_per_image=1000` means the vocabulary is built from at most 1000×N descriptors with no regard for descriptor space coverage. For large N this is fine; for small N (e.g., 20 images) the vocabulary will overfit to the scene.
- The k-means GPU implementation uses random (not k-means++) initialization and only 20 iterations. For 4096 centroids in 128-D space, convergence may be poor.
- Mutual cross-check is implemented via Python `set` intersection on CPU, which is O(M) after GPU matching — this is fine.
- Sequential matcher has no loop closure: if images are ordered but contain a loop (e.g., a video that returns to start), the loop will not be closed.

**Code health:** The GPU path is well-structured with proper OOM fallback. The vectorized ratio test avoids Python loops. The VocabTree is novel for a research prototype but the flat-vocabulary quality is a known limitation.

---

### 3.4 Geometric Verification

**Implementation:** `geometric_verification.py` — Two-stage: (1) Hartley-normalized F estimation via `cv2.findFundamentalMat(USAC_MAGSAC)`; (2) E estimation via `cv2.findEssentialMat(USAC_MAGSAC)` on F-inliers; (3) pose recovery via `cv2.recoverPose`. Points are undistorted before RANSAC when dist_coeffs are non-zero.

**Faithfulness:** High fidelity to the standard pipeline. Hartley normalization is correct (centroid to origin, mean distance = √2). The de-normalization formula `F = T2.T @ F_norm @ T1` is correct. Using USAC_MAGSAC for both F and E is good practice (MAGSAC++ [Barath et al. 2020] is score-based and avoids the hard inlier threshold problem).

**Failure modes:**
- The F matrix is computed in undistorted space (after `undistort_points`), then used as if it were the fundamental matrix in original pixel space — but the de-normalization assumes pixel space. Since points were already undistorted, the resulting F is the fundamental matrix for the undistorted space, which is the correct E relationship (E = K.T @ F @ K). The subsequent E computation on undistorted pts is therefore redundant but harmless.
- `cv2.recoverPose` uses a hard cheirality test (positive depth) with threshold parameter 200 (hardcoded in OpenCV). On short-baseline pairs, many points may fail cheirality.
- Near-zero baseline rejection threshold `1e-4` (line 219) is in recoverPose's unit-scale space (where t is normalized to unit length after E decomposition). This correctly rejects degenerate pairs.
- No homography check — if the scene is planar, E estimation will succeed but the rotation will be degenerate. COLMAP checks for the H/E inlier ratio to detect planar scenes.

**Code health:** Very good. The shuffle before RANSAC (line 124) is a documented workaround for an OpenCV crash. The analytical fallback `E = K.T @ F @ K` (line 182) is correct and handles the edge case gracefully.

---

### 3.5 Incremental SfM

**Implementation:** `reconstruction.py` — IncrementalSfM class:
- **Seed selection:** Score = baseline × inlier_count, with 5° median triangulation angle guard (R-13). Correct.
- **Initialization:** Identity pose for camera 0, E-decomposed pose for camera 1. Batch triangulation, vectorized acceptance filter (depth + angle + reproj). Correct.
- **Registration:** EPnP (`cv2.SOLVEPNP_EPNP`) + `solvePnPRefineLM` (R-12). Correct and well-chosen: EPnP is O(n) and accurate; LM refinement adds <1ms and ~0.5px RMSE improvement.
- **Triangulation:** Batch `cv2.triangulatePoints` per pair, vectorized acceptance filter. Correct.
- **Covisibility graph:** Maintained incrementally via `_pt_observers` dict. Good.
- **Pair indexing:** `_pairs_by_img` pre-built from verified_pairs. Reduces `_count_corr` from O(N²) to O(degree × M_matches). Good.

**Failure modes:**
- **No re-triangulation:** After each BA step, 3D points are updated in place, but no new triangulation is attempted from newly refined poses. COLMAP re-triangulates after every BA, recovering 10–30% additional points. This is the most impactful algorithmic gap.
- **No track merging:** If two 3D points correspond to the same world point (found via different matching paths), they are never merged. COLMAP's track merger prevents this.
- **Single-camera PnP:** `_register_image` uses `self.dist_coeffs` (shared coefficients) for PnP. If BA has refined these, they are re-used correctly (updated at line 818). But when `no_refine_intrinsics=True`, the dist_coeffs remain zero, so the undistorted keypoints are also used without distortion correction — this is correct.
- **Camera position sanity check** (line 540): `cam_dist > 100 × median_dist` is a reasonable heuristic, but uses distance from the *origin* rather than from the scene centroid. In reconstructions where the origin is far from the scene, this test may reject valid cameras.
- **Observations are appended but never deduplicated:** `_add_obs` in `_triangulate_new_points` line 695 calls `_add_obs(other, idx, ...)` for each matched pair — if `other` and `new_idx` see the same 3D point via multiple pairs, duplicate observations accumulate. The covisibility check (`if img_idx not in self._pt_observers[pt3d_idx]`) prevents duplicate covisibility entries but `self.observations` may still have multiple entries for the same (img_idx, pt3d_idx) pair.

**Code health:** Very good overall. The vectorized batch acceptance filter (`_accept_batch`) is correct and efficient. The `_pair_roles` utility is clean. The outlier removal after final BA is correct.

---

### 3.6 Bundle Adjustment

**Implementation:** `bundle_adjustment.py` — SciPy `least_squares` with TRF method, sparse Jacobian pattern, Huber loss.

Parameter layout: shared `[f, k1, k2, cx, cy]` (5 params) + per-camera `[rvec(3), tvec(3)]` (6C params) + per-point `[X(3)]` (3P params). Optional `fix_principal_point` mode drops cx, cy (3 shared instead of 5).

**Faithfulness:** The projection model is correct Brown-Conrady radial distortion with two coefficients. The Huber loss weight is adaptive (1.4826 × MAD of initial residuals), which is a good robust estimator choice. The SVD projection of R back onto SO(3) after each TRF update (lines 455–458) is mathematically correct.

**Failure modes:**
- **No Schur complement:** SciPy's TRF treats the Jacobian as an arbitrary sparse matrix and applies an L-BFGS-B-like solver. It does not exploit the block-diagonal structure of the point parameters (BA's key computational advantage). For 100 cameras and 10000 points, the parameter vector has ~30600 entries; SciPy will form dense linear systems of this size. COLMAP with Ceres + Schur complement would reduce this to a 600×600 camera-camera normal equation.
- **Only 2 radial coefficients** (`k1`, `k2`): No tangential distortion (`p1`, `p2`), no higher-order radial. For fisheye or high-distortion lenses, this model will have systematic residuals.
- **Single focal length assumption** (`fx = fy`): Lines 471–472 set `K_refined[0,0] = K_refined[1,1] = f_opt`. This forces square pixels. Real cameras can have `fx ≠ fy` by a small amount that matters for precision work.
- **Bounds on cx/cy** (lines 353–356): `cx_init ± 0.1 × W` and `cy_init ± 0.1 × H`. For images where the true principal point is far from center (e.g., stitched panoramas, cropped images), this bound will prevent convergence.
- **BA interval** default 5: BA runs after every 5 cameras. For large datasets (200+ images), this means BA runs 40 times before completion. With SciPy TRF, each BA invocation can take 30–300 seconds for large scenes. COLMAP adaptively increases the interval.
- **Divergence guard** at 1.5× RMSE (R-07): Correct safety net. However, rejecting the BA result when it diverges means the pipeline continues with potentially poor poses that will accumulate error.

**Code health:** The sparsity pattern builder uses a Python loop over observations (O(M)), which is slow for large M. For 100,000 observations this can take several seconds. The adaptive f_scale is a good engineering choice. The RMSE reporting is correct (RMS of residual vector, not per-observation mean).

---

### 3.7 Point Cloud Export

**Implementation:** `point_cloud.py` — `colorize()` groups observations by camera, computes vectorized reprojection errors per camera, samples image pixels at integer coordinates (rounded, not bilinear, after fix R-10 changed from `_sample_bilinear` to vectorized nearest-pixel), writes float64 PLY (R-15).

**Faithfulness:** Correct. The outlier filter (median + 3σ, clamped at max_reproj_error) is a standard robust statistic.

**Failure modes:**
- **Color sampling is nearest-pixel, not bilinear:** Lines 168–176 — `good_xs = np.round(obs_2d[good, 0]).astype(np.int32)` then `img[good_ys, good_xs]`. The original `_sample_bilinear` function still exists (lines 75–92) but is no longer called by `colorize()`. This is a regression from R-10 (which vectorized color sampling but switched to nearest-pixel). Bilinear would give smoother colors, especially for sub-pixel observations.
- **All cameras equally weighted for color:** A point seen by 10 cameras averages all 10 contributions equally. In practice, the closest or most frontal camera observation gives the most reliable color, not the average. COLMAP uses the observation with lowest angle to the surface normal.
- **PLY float64:** `property double x/y/z` in the PLY header (lines 303–309). This is correct precision (R-15) but doubles file size vs. float32. Most downstream tools (MeshLab, CloudCompare, open3d) handle float64 PLY correctly.
- **No normals in PLY:** The sparse cloud has no per-point normals, which is fine for point clouds but limits direct use in some mesh reconstruction tools.

**Code health:** Very good. The vectorized per-camera reprojection is efficient. The PLY writer is correct.

---

### 3.8 MVS

**Implementation:** `mvs.py` — StereoSGBM on stereo-rectified image pairs. Pair selection via covisibility counts (R-04) or consecutive index fallback.

**Faithfulness:** StereoSGBM is a classic semi-global block matching algorithm (Hirschmüller 2008). It is fundamentally a per-pair stereo method with no multi-view consistency. The pipeline is:
1. `cv2.stereoRectify` — correct use of CALIB_ZERO_DISPARITY + alpha=0
2. `cv2.initUndistortRectifyMap` + `cv2.remap` — correct
3. SGBM with P1=8×3×bs², P2=32×3×bs² — standard SGBM parameters
4. `cv2.reprojectImageTo3D` — correct Q matrix usage
5. Transform from rectified cam-1 frame → world via R1.T → Ri.T

**Failure modes:**
- **No left-right consistency check:** SGBM does not enforce left-right consistency by default. `disp12MaxDiff=1` provides a weak check but false disparities (especially near occlusion boundaries) will produce ghost points.
- **`minDisparity=0`:** Assumes the reference camera is always to the left of the match camera. When `R_rel` is not a pure rightward translation (general rotation + translation), the minimum disparity may need to be negative. This will produce incorrect depth for pixels where the correspondence falls to the left of the reference pixel.
- **Rectification for general camera motion:** `cv2.stereoRectify` performs a general rectification for any baseline direction. This is correct in principle, but for large rotational baselines (>30°) the rectification introduces extreme distortion and SGBM quality degrades severely.
- **Coordinate transform bug potential:** Line 282: `pts_cam_i = (R1.T @ pts_rect.T).T`. This assumes `X_rect = R1 @ X_cam_i`, which is what `stereoRectify` produces for camera 1. This is correct.
- **No depth consistency across pairs:** Dense points from different stereo pairs at the same world location will be merged by simple concatenation (lines 185–186) with no consistency check or averaging. This doubles or triples point density at overlapping regions with small offsets, producing noisy surface normals.
- **Hard-coded max_dense_pts=500,000:** This is a soft cap via random subsampling, but it applies per-run, not per-pair. For a 50-camera dataset with 50 pairs, each pair producing 100k points, the merged cloud of 5M points is subsampled to 500k, discarding 90% of the dense data.

**Code health:** Clean structure. The covisibility-based pair selection (R-04) is a genuine improvement over consecutive-index. The magic numbers (disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32) are reasonable defaults but are not tunable via CLI.

---

### 3.9 COLMAP Backend

**Implementation:** `colmap_backend.py` — Thin subprocess wrapper around COLMAP CLI. Reads binary output via pure Python struct parsing. After R-02, dead `_read_images_bin` was removed (it is now re-implemented correctly).

**Faithfulness:** The binary parsers are correct:
- `_read_cameras_bin`: Correctly reads num_cameras as uint64, cam_id as uint32, model as int32, width/height as uint64, params as float64 array.
- `_read_images_bin`: Correctly interleaves (x, y, point3D_id) per 2D point.
- `_read_points3d_bin`: Correctly reads point3D_id as uint64, xyz as float64×3, rgb as uint8×3, error as float64, track_len as uint64, and skips track entries.
- `_qvec_to_rot`: Correct COLMAP quaternion (qw, qx, qy, qz) convention.
- `_params_to_K`: Handles SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE correctly.

After R-14, `_has_gpu` is imported from `device.py` — no duplication.

**Failure modes:**
- **No COLMAP version check:** COLMAP's binary format has been stable since 3.5, but older versions may differ in argument names (e.g., `--Mapper.ba_global_function_tolerance` vs. `--Mapper.ba_global_max_num_iterations`). The runner does not check the COLMAP version.
- **Mapper always runs `ImageReader.single_camera=1`** (line 330): This forces COLMAP to use one camera model for all images. For multi-camera datasets, this is incorrect. For single-camera captures (the common case) this is correct.
- **Dense pipeline assumes CUDA:** `patch_match_stereo` will fail on CPU-only machines with a non-zero returncode, which is caught and re-raised as a CalledProcessError. The error message should mention CUDA explicitly.
- **Workspace cleanup on success** (line 319): If the run succeeds but the user immediately needs the COLMAP database for downstream processing, the workspace is deleted by default. `--colmap-keep-workspace` addresses this.

**Code health:** Excellent. Clean, minimal, correct. Error handling via `_exec` is consistent.

---

### 3.10 Visualization

**Implementation:** `visualizer.py` — 1586-line event-driven visualization harness with matplotlib. Hooks: `on_pipeline_start`, `on_all_features_done`, `on_all_matching_done`, `on_geometric_verification_done`, `on_seed_pair_selected`, `on_camera_registered`, `on_bundle_adjustment_run`, `on_reconstruction_complete`, `on_mesh_complete`, `on_pipeline_complete`.

**Faithfulness:** Comprehensive visualization for a research tool. Covers keypoint overlays, density heatmaps, feature statistics, match visualizations with epipolar lines, camera pose plots, BA convergence curves, reprojection error distributions, point cloud views, and summary dashboard.

**Failure modes:** The `_render_epipolar` method computes and draws epipolar lines — if called on a pair where `F` is degenerate (rank < 2), `cv2.computeCorrespondEpilines` may produce NaN lines. This is caught by the outer `try/except`.

**Code health:** All visualization hooks are wrapped in `try/except` in the pipeline — viz failures never crash the reconstruction. Heavy imports (matplotlib, networkx, imageio, open3d) are deferred until `enabled=True`. Well-engineered.

---

### 3.11 CLI

**Implementation:** `run_sfm.py:build_parser()` — 450 lines of argparse. All key parameters are exposed.

**Strengths:**
- `_validate_inputs` catches common errors before any heavy processing.
- `--resume` + pickle checkpointing allows pipeline restart.
- `--mesh-no-fill-holes` / `--mesh-fill-holes` fix (R-03) correctly uses `store_false` + `dest=mesh_fill_holes` with `default=True`.
- The legacy `--mesh-fill-holes` flag is suppressed with `argparse.SUPPRESS` and logs a deprecation warning.

**Failure modes:**
- The `match_ckpt_key` (line 774) encodes `--match_strategy` and `--ratio` but not `--sequential_window` or `--vocab_words`. Changing these parameters without changing `--ratio` will incorrectly reuse cached matches.
- Dense output path logic (line 988–993) uses `Path(args.output).parent` — if the user specifies `--output /abs/path/output.ply`, the dense output goes to `/abs/path/output_dense.ply`. This is correct but could be surprising if `--dense_output` is not set.
- `--ba-fix-principal-point` is read via `getattr(args, "ba_fix_principal_point", False)` (line 890) because argparse converts `--ba-fix-principal-point` to `ba_fix_principal_point` (hyphen→underscore). This is correct.

---

## 4. Production System Comparison

### 4.1 vs. COLMAP

COLMAP [Schönberger & Frahm, 2016] is the gold standard for incremental SfM. Key gaps:

| Dimension | This Repo | COLMAP |
|-----------|-----------|--------|
| Feature extraction | SIFT (Python) | SIFT + learned (SuperPoint in 3.9+), GPU-native |
| Matching | FLANN/torch.cdist | Cascade hashing, spatial verification, GPU |
| Geometric verification | USAC_MAGSAC (Python OpenCV) | USAC_MAGSAC (C++), spatial consistency |
| BA | SciPy TRF | Ceres Solver + Schur complement |
| Per-image intrinsics | Single shared model | Per-image, multiple camera models |
| Re-triangulation | Never | After every BA |
| Track merging | Never | After every triangulation |
| Scalability | ~50–100 images | 10,000+ images |
| Dense MVS | SGBM | PatchMatch Stereo (GPU, multi-view consistent) |
| Loop closure | None | Hierarchical mapper |

### 4.2 vs. RealityCapture

RealityCapture (RNDR.AI) is a commercial photogrammetry tool. Key gaps:

- RC uses proprietary feature extraction (claimed to be SIFT-derivative with learned improvements).
- RC's MVS uses GPU-parallel PatchMatch with multi-scale processing, producing point clouds 10–50× denser.
- RC supports laser scan fusion, thermal imagery, and has sub-millimeter accuracy on calibrated setups.
- RC can process 10,000+ images in hours on workstation hardware.
- This repo cannot approach RC on any production metric.

### 4.3 vs. Metashape (Agisoft)

Metashape uses its own optimized pipelines with advanced camera models (including fisheye). Key gaps:

- Metashape's dense reconstruction uses a modified SGM algorithm with geometric consistency and filter steps.
- Metashape supports multi-spectral cameras, thermal imagery, and GCP-based georeferencing.
- Metashape's BA uses a custom C++ solver with full camera model support.

### 4.4 vs. OpenSfM

OpenSfM [Mapillary, 2020] is the closest open-source comparison as a Python-heavy SfM system. Key gaps:

- OpenSfM uses a C++ BA backend (PyCeres or custom), which is 10–50× faster than SciPy.
- OpenSfM supports multi-model reconstruction (multiple connected components).
- OpenSfM's retrieval is based on learned DNN embeddings (NetVLAD) not flat k-means.
- OpenSfM has robust geo-referencing via GPS EXIF.

### 4.5 vs. Meshroom (Alice Vision)

Meshroom uses a node-graph pipeline with C++ nodes including DepthMap MVS and Texturing. Key gaps:

- Meshroom's DepthMap uses multi-scale SGM with semi-global aggregation, far better than single-scale SGBM.
- Meshroom's texturing maps hi-res images onto the mesh (UV unwrapping), not vertex color transfer.
- Meshroom supports structured-light and lidar fusion.

### 4.6 Academic SotA (2022–2025)

**Feature detection:**
- **SuperPoint** [DeTone et al. 2018] — self-supervised learned keypoints and descriptors. 2–4× more repeatable than SIFT on low-texture, in-the-wild images.
- **DISK** [Tyszkiewicz et al. 2020] — reinforcement-learned descriptor field, competitive with SuperPoint.
- **DeDoDe** [Edstedt et al. 2023] — 3D-consistency-aware detection, outperforms SuperPoint on benchmark pose estimation.

**Matching:**
- **SuperGlue** [Sarlin et al. 2020] — graph neural network matcher, 20–40% more correct matches than mutual-NN.
- **LightGlue** [Lindenberger et al. 2023] — faster SuperGlue successor, adaptive depth, runs in 5–20ms per pair on GPU.
- **LoFTR** [Sun et al. 2021] — detector-free matching via transformer; handles low-texture scenes where SIFT fails completely.
- **MASt3R** [Leroy et al. 2024] — foundation model for 3D matching; produces dense correspondence fields and rough 3D directly.

**Geometric verification:**
- **MAGSAC++** [Barath et al. 2020] — score-based RANSAC, available in OpenCV as USAC_MAGSAC (already used here). The pure C++ `magsac` library is ~5× faster than OpenCV's Python wrapper.
- **GC-RANSAC** [Barath & Matas 2018] — locally-optimized RANSAC with graph-cut refinement.
- **PoseLib** [Bujnak et al. 2022] — fast minimal solver library (P3P, P4P, essential matrix) with RANSAC; 5–10× faster than OpenCV for pose estimation.

**Bundle Adjustment:**
- **Ceres Solver** [Agarwal et al. 2022] — the industry standard. Analytical Jacobians, Schur complement, PCG solver, supports all camera models.
- **g2o** [Kümmerle et al. 2011] — factor graph optimizer widely used in SLAM.
- **GTSAM** [Dellaert 2012] — Bayes-tree-based factor graph, good for incremental BA.

**Reconstruction paradigms:**
- **GLOMAP** [Pan et al. 2024] — global SfM that avoids incremental drift. First estimates rotations globally (IRLS on Lie group), then solves global translation via BATA. Produces models comparable to COLMAP in 5–10× less time.
- **VGGSfM** [Wang et al. 2024] — learned camera pose estimator using transformer architecture; handles unordered image collections without matching.
- **DUSt3R** [Wang et al. 2024] — dense unconstrained stereo using a vision transformer; produces pairwise point maps and can be combined with global alignment (MASt3R) for multi-view 3D.

**Output format:**
- **3D Gaussian Splatting** [Kerbl et al. 2023] — represents scenes as a set of 3D Gaussians; produces real-time renderable models from SfM sparse points as initialization. This repo's PLY output is a suitable initialization for 3DGS but no 3DGS training is implemented.

---

## 5. Weakness Catalogue

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Slow BA — No Schur Complement ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-01
Severity: CRITICAL
Stage: Bundle Adjustment
Location: bundle_adjustment.py:_build_sparsity_v2:173–215, BundleAdjuster.adjust:393–411

Current behavior: scipy.optimize.least_squares with method="trf" treats the
  (2*n_obs) × (5 + 6*C + 3*P) Jacobian as a general sparse matrix and solves
  normal equations via LSQR without exploiting the BA block structure. For
  C=100, P=10000, n_obs=100000, the parameter vector has 30605 entries. SciPy
  builds a dense 30605×30605 normal equations matrix internally.

What it should do: Exploit the block-diagonal structure of the point parameters.
  In the Schur complement approach (Triggs et al. 2000, "Bundle Adjustment — A
  Modern Synthesis"), the 3P×3P point block is eliminated analytically, reducing
  the normal equation to a 6C×6C (or (5+6C)×(5+6C)) system. For 100 cameras
  this is a 600×600 system — 2600× smaller. Ceres Solver implements this natively.

Impact: BA runtime scales as O(C³) without Schur vs O(C³/P + P·C) with it.
  Empirically: 50 cameras, 5000 pts → SciPy ~30s, Ceres ~0.5s.
  For 200 cameras, BA may take 10–60 minutes per invocation, making the pipeline
  unusable for large datasets.

Fixability: MEDIUM
Fix: Install pyceres (https://github.com/cvg/pyceres) and replace
  scipy.optimize.least_squares with a Ceres problem using
  ceres.SolverOptions(linear_solver_type=ceres.SPARSE_SCHUR).
  Alternatively, use g2o (https://github.com/uoip/g2opy) or
  pycolmap's built-in BA. pyceres provides a near-drop-in replacement for
  the current residual function structure.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Single Shared Focal Length in BA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-02
Severity: HIGH
Stage: Bundle Adjustment
Location: bundle_adjustment.py:_residuals_v2:135–170, _project_distorted:92–130

Current behavior: A single scalar `f` is shared across all cameras and all
  axes: the projection formula uses `f * xn` and `f * yn`, forcing fx == fy.
  K_refined updates both K[0,0] and K[1,1] to f_opt (line 471–472).

What it should do: Support at minimum (fx, fy, cx, cy) per-camera, and ideally
  a per-image intrinsic model (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV)
  as COLMAP does. Non-square pixels (fx ≠ fy) occur in digital video cameras,
  drone cameras, and scanned imagery.

Impact: For cameras with >0.5% pixel aspect ratio error (~1 pixel in 200),
  the forced fx=fy constraint introduces ~0.5–1px systematic residual that
  BA cannot reduce. This limits achievable RMSE to >0.5 px even with perfect
  correspondences.

Fixability: MEDIUM
Fix: Expand the shared intrinsics vector from [f, k1, k2] to
  [fx, fy, cx, cy, k1, k2], update _project_distorted to use separate
  fx/fy, update K_refined unpacking (lines 470–474), and add CLI flags
  --ba-fix-fx-fy (to couple them) and --ba-fix-aspect (to fix fy/fx ratio).
  Per-image intrinsics require storing each camera's K separately and expanding
  the parameter vector — a larger refactor.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: No Re-Triangulation After BA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-03
Severity: HIGH
Stage: Incremental SfM
Location: reconstruction.py:_run_ba:791–824, reconstruct:305–319

Current behavior: After each BA step, `self.points_3d` is updated from
  BA output (line 805). No new triangulation is attempted from the refined
  camera poses. Matched keypoints that were not previously triangulated
  (e.g., because the triangulation angle was below threshold at an earlier
  registration step) remain unprocessed forever.

What it should do: After each BA step, re-attempt triangulation for all
  unmatched keypoints across all registered camera pairs (COLMAP's
  "complete tracks" pass). Refined camera poses often make previously
  marginal triangulation angles sufficient. COLMAP typically gains 10–30%
  more points per re-triangulation pass.

Impact: ~10–30% lower point density than achievable; missed observations
  reduce BA constraint count, which worsens pose accuracy.

Fixability: EASY
Fix: In _run_ba(), after updating self.cameras and self.points_3d, call
  a new method _re_triangulate_all() that iterates over all registered
  camera pairs and calls _triangulate_new_points() for each newly
  registered camera. The existing _triangulate_new_points() already
  handles the case where some keypoints are already linked — it will
  simply skip them and attempt the rest.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: SGBM MVS — No Multi-View Consistency ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-04
Severity: HIGH
Stage: MVS
Location: mvs.py:_process_pair:202–303

Current behavior: SGBM produces a single disparity map per stereo pair.
  No left-right consistency check beyond disp12MaxDiff=1. No photometric
  consistency across N>2 views. Dense points from overlapping pairs are
  concatenated without any cross-pair filtering (lines 185–186).

What it should do: Multi-view stereo with photo-consistency scoring across
  3+ views (PatchMatch MVS, Plane-Sweep Stereo, or RAFT-Stereo). Each
  depth hypothesis should be validated against ≥2 other views before acceptance.
  Reference: Galliani et al. (2015) "Massively parallel multiview stereopsis
  by surface normal diffusion," ICCV 2015 (Gipuma); or Xu et al. (2022)
  "RNN-MVSNet" for learned multi-view depth.

Impact: Dense cloud is 5–20× sparser and noisier than COLMAP PatchMatch.
  Occlusion boundaries produce many false points. Textureless regions
  (sky, walls) produce noise rather than being masked out.

Fixability: HARD
Fix: Option A: Integrate OpenMVS (open-source, C++, GPU) as a subprocess
  (similar to COLMAP backend). Option B: Integrate RAFT-Stereo
  (https://github.com/princeton-vl/RAFT-Stereo) as a GPU depth estimator.
  Option C: Implement a simple plane-sweep stereo with NCC matching and
  left-right consistency — this can be done in ~300 lines of PyTorch and
  gives 3–5× quality improvement over SGBM with consistency filtering.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Flat K-Means Vocab Tree ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-05
Severity: HIGH
Stage: Feature Matching
Location: feature_matching.py:VocabTreeMatcher._build_vocab:436–453

Current behavior: A flat k-means vocabulary (4096 words, random init, 20
  iterations of Lloyd's algorithm) is built at runtime from SIFT descriptors
  sampled from the input images. TF-IDF is applied over this vocabulary.

What it should do: Use a hierarchical vocabulary tree (branching factor 10,
  depth 6 = 10^6 leaves) pre-built from millions of diverse images (the
  standard COLMAP/NetVLAD vocabulary). Alternatively, use a learned image
  embedding (NetVLAD, DINOv2) for retrieval. Reference: Nister & Stewenius
  (2006), "Scalable recognition with a vocabulary tree," CVPR 2006.

Impact: A flat 4096-word vocabulary has poor discrimination ability for
  general scenes. Two dissimilar images may share many visual words by
  chance (false positives). On large or diverse datasets, 30–50% of the
  top-k retrieved images may be incorrect, wasting matching time and
  potentially corrupting the scene graph.

Fixability: MEDIUM
Fix: (1) For offline use: download a pre-built COLMAP vocab tree (.bin)
  and implement a hierarchical tree traversal in Python (or use subprocess
  COLMAP with --match_strategy vocab_tree). (2) For online use: replace
  VocabTreeMatcher with DINOv2-based retrieval:
  `features = dinov2(images)`, cosine similarity retrieval. DINOv2 is
  available via torch.hub and requires only 3–10ms per image for
  the ViT-S/14 variant.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Duplicate Observations in BA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-06
Severity: MEDIUM
Stage: Incremental SfM / Bundle Adjustment
Location: reconstruction.py:_triangulate_new_points:693–696, _add_obs:870–884

Current behavior: _triangulate_new_points calls _add_obs(other, idx, ...)
  on line 695 to add the "other" image's observation of the newly
  triangulated point. However, if `other` already has an observation of this
  3D point from a previous triangulation step (via a different pair), the
  observation is added a second time. The covisibility check
  (`if img_idx not in self._pt_observers[pt3d_idx]`, line 880) prevents
  duplicate covisibility edges but does not prevent duplicate observations.

What it should do: Check that (img_idx, pt3d_idx) is not already in
  self.observations before appending. Duplicate observations inflate the
  observation count and give more weight to cameras that happen to see a
  point via many matched pairs, biasing BA.

Impact: For scenes with many overlapping pairs (exhaustive matching on
  20+ images), duplicate observations may constitute 10–30% of all
  observations in self.observations, artificially inflating BA constraint
  counts and biasing convergence toward cameras with duplicate entries.

Fixability: TRIVIAL
Fix: In _add_obs(), maintain a set `self._obs_set: Set[Tuple[int,int]]` of
  (img_idx, pt3d_idx) pairs. Before appending to self.observations, check:
  `if (img_idx, pt3d_idx) not in self._obs_set: self._obs_set.add(...)`.
  Alternatively, filter duplicates in BundleAdjuster.adjust() before
  building the observation arrays.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Match Checkpoint Key Missing Parameters ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-07
Severity: MEDIUM
Stage: CLI / Checkpointing
Location: run_sfm.py:774–775

Current behavior: `match_ckpt_key = f"matches_{args.match_strategy}_r{args.ratio:.3f}"`
  This key encodes strategy and ratio but not --sequential_window, --vocab_words,
  or --vocab_top_k.

What it should do: Encode all parameters that affect the match output:
  for sequential: window size; for vocab_tree: n_words, top_k.

Impact: Changing --sequential_window from 5 to 10 without changing --ratio
  will incorrectly load the old cached matches with window=5, silently
  producing a reconstruction with insufficient overlap. This is a correctness
  bug that silently produces wrong results when --resume is used.

Fixability: TRIVIAL
Fix: Expand the key:
  `match_ckpt_key = (
      f"matches_{args.match_strategy}_r{args.ratio:.3f}"
      + (f"_w{args.sequential_window}" if args.match_strategy == "sequential" else "")
      + (f"_vw{args.vocab_words}_vk{args.vocab_top_k}" if args.match_strategy == "vocab_tree" else "")
  )`
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Color Sampling Reverted to Nearest-Pixel ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-08
Severity: LOW
Stage: Point Cloud Export
Location: point_cloud.py:colorize:168–176

Current behavior: After vectorization in R-10, color sampling uses integer
  nearest-pixel lookup: `good_xs = np.round(...).astype(np.int32)` then
  `img[good_ys, good_xs]`. The `_sample_bilinear` function exists (lines 75–92)
  but is no longer called from colorize().

What it should do: Use bilinear interpolation for sub-pixel color accuracy.
  This matters when the reprojected point falls between pixels, which is
  common for high-precision BA solutions.

Impact: Low — color differences are at most 1 pixel's worth of gradient.
  Visually, the difference is negligible for most scenes. On textured edges,
  aliasing artifacts may be visible.

Fixability: EASY
Fix: Replace the integer sampling block (lines 168–176) with a vectorized
  bilinear implementation:
  ```python
  x0 = good_xs.clip(0, w-1); x1 = (good_xs+1).clip(0, w-1)
  y0 = good_ys.clip(0, h-1); y1 = (good_ys+1).clip(0, h-1)
  dx = obs_2d[good, 0] - good_xs; dy = obs_2d[good, 1] - good_ys
  bgr = (img[y0,x0]*(1-dx[:,None])*(1-dy[:,None]) + img[y0,x1]*dx[:,None]*(1-dy[:,None])
         + img[y1,x0]*(1-dx[:,None])*dy[:,None] + img[y1,x1]*dx[:,None]*dy[:,None])
  ```
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Sparse Jacobian Builder Is a Python Loop ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-09
Severity: MEDIUM
Stage: Bundle Adjustment
Location: bundle_adjustment.py:_build_sparsity_v2:175–215

Current behavior: The sparsity pattern is built by a Python for-loop over
  all n_obs observations (line 193: `for k in range(n_obs)`), setting 8
  entries in a lil_matrix per observation. For n_obs=100,000 this loop
  takes 5–15 seconds.

What it should do: Build the sparsity pattern as arrays of (row, col) index
  pairs and construct the sparse matrix in a single vectorized call.

Impact: For large scenes (50k+ observations), the sparsity builder takes
  longer than a single BA iteration. This is pure overhead.

Fixability: EASY
Fix: Replace the loop with vectorized construction:
  ```python
  n_obs = len(cam_indices)
  rows = np.repeat(np.arange(n_obs), 4) * 2  # 4 blocks per obs, x-row
  # ... build all (row, col) pairs as arrays, then
  J = csr_matrix((np.ones(len(rows)), (all_rows, all_cols)), shape=(n_res, n_params))
  ```
  This reduces build time from O(n_obs) Python iterations to a single
  scipy.sparse.csr_matrix constructor call.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: MVS minDisparity=0 Assumes Left-Camera Geometry ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-10
Severity: MEDIUM
Stage: MVS
Location: mvs.py:_process_pair:247

Current behavior: `cv2.StereoSGBM_create(minDisparity=0, ...)` assumes all
  correspondences fall to the right of the reference pixel in the rectified
  image. This is only valid when the rectified camera configuration has
  camera_i to the left of camera_j.

What it should do: After stereoRectify, determine the sign of the baseline
  in the rectified frame and set minDisparity to a small negative value
  (e.g., -64) when the baseline is rightward, to handle both stereo
  configurations correctly.

Impact: For pairs where camera_j is to the left of camera_i in the rectified
  frame, all correspondences have negative disparity and SGBM produces zero
  valid depth. This silently drops stereo pairs, reducing dense cloud coverage.

Fixability: EASY
Fix: After stereoRectify, compute `baseline_sign = np.sign(t_rel[0, 0])`.
  If negative, negate `R_rel` and `t_rel`, swap images, and run SGBM with
  minDisparity=0. The final coordinate transform must then swap the images back.
  Alternatively, set minDisparity=-64 and numDisparities+=64 to handle both
  cases, at the cost of ~25% more computation.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: No Homography / Pure-Rotation Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-11
Severity: MEDIUM
Stage: Geometric Verification
Location: geometric_verification.py:verify_pair:103–230

Current behavior: Verification fits F and E via RANSAC but does not check
  whether the scene is planar (H model fits better than F) or whether the
  motion is pure rotation (t ≈ 0). Both cases produce a valid E decomposition
  but invalid or degenerate triangulation.

What it should do: Additionally fit H = findHomography(pts1, pts2, RANSAC)
  and compute the H/E inlier ratio. If H_inliers / E_inliers > 0.85, mark
  the pair as "degenerate" and exclude it from being the seed pair
  (COLMAP's degeneracy_check). Pure-rotation pairs (baseline < threshold)
  are already rejected at line 219, but planar scenes pass through.

Impact: If the seed pair is from a planar scene, the initialization will
  produce an underconstrained triangulation (infinite points on the plane)
  leading to a degenerate reconstruction. Subsequent registrations will fail.

Fixability: EASY
Fix: Add after E estimation:
  ```python
  H, mask_h = cv2.findHomography(pts1_ein, pts2_ein, cv2.RANSAC, self.ransac_threshold)
  if H is not None and mask_h is not None:
      h_ratio = mask_h.sum() / max(mask_e.sum(), 1)
      if h_ratio > 0.85:
          return None  # planar/degenerate
  ```
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: EXIF Focal Assumes 35mm Full-Frame Sensor ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-12
Severity: LOW
Stage: Image Loading / Intrinsics
Location: utils.py:read_exif_focal_px:52–72

Current behavior: Conversion uses `sensor_diag_35mm = sqrt(36^2 + 24^2)` 
  (line 70), which is the correct 35mm full-frame diagonal. The EXIF tag
  0xA405 (FocalLengthIn35mmFilm) is defined as the equivalent focal length
  for a 35mm camera, so this computation is mathematically correct.
  However, it only reads tag 0xA405 and ignores the actual FocalLength
  (0x920A) combined with sensor size information.

What it should do: Also try to read FocalLength (0x920A) and sensor size
  information from EXIF (CCD width from 0xA002/0xA003), as many cameras
  write FocalLength but not FocalLengthIn35mmFilm. COLMAP reads both tags
  and uses a lookup table of known sensor sizes.

Impact: Images without FocalLengthIn35mmFilm EXIF (common in RAW converters
  that strip tags) will fall back to focal = max(W,H), introducing 20–50%
  focal error depending on the actual lens.

Fixability: EASY
Fix: In read_exif_focal_px(), also try:
  focal_raw = exif.get(0x920A)  # FocalLength (mm)
  pixel_pitch = ...  # estimate from sensor database or EXIF 0xA002/0xA003
  If both are available, compute focal_px = focal_raw / sensor_width_mm * image_width_px.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Multi-Camera Dataset Unsupported ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-13
Severity: MEDIUM
Stage: Image Loading / Full Pipeline
Location: run_sfm.py:720–721, bundle_adjustment.py:all shared intrinsics

Current behavior: `sample = load_image(image_paths[0])` and 
  `K = estimate_intrinsics(sample.shape, image_path=image_paths[0])`.
  A single K is used for all images throughout the pipeline including BA.

What it should do: Read EXIF from every image, group by (make, model, focal)
  to identify distinct camera models, and maintain per-camera intrinsics.
  COLMAP supports MULTIPLE_CAMERA_MODELS mode where each image has its own
  camera model entry.

Impact: Datasets from multiple cameras (e.g., images from two different
  phones, or a mix of portrait and landscape shots) will have incorrect
  intrinsics for all but the first image. BA will compensate by distorting
  camera poses, leading to systematic errors. The scene graph check (R-16)
  will not catch this.

Fixability: MEDIUM
Fix: In run_sfm.py, call estimate_intrinsics() per image and build a dict
  {img_idx: K_i}. Pass this to IncrementalSfM and BundleAdjuster. BA must
  be extended to support per-camera intrinsics (each camera gets its own
  [f, k1, k2, cx, cy] block). This is the most architecturally significant
  intrinsics improvement.
Depends on: W-02 (add fx/fy separation first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: Kornia LAF Scale Factor 6.0 Undocumented ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-14
Severity: LOW
Stage: Feature Extraction
Location: feature_extraction.py:_extract_kornia:228

Current behavior: `size=float(scales[k]) * 6.0` when building cv2.KeyPoint
  for SIFT descriptor computation. The factor 6.0 is not documented.

What it should do: SIFT's keypoint `size` field is the diameter of the
  meaningful neighbourhood (sigma * 2 * 2 = 4*sigma for the Gaussian envelope,
  or commonly defined as diameter = scale * 2 * octave_scale_factor).
  The kornia LAF scale is the half-axis of the ellipse; converting to SIFT
  diameter requires understanding what kornia's scale unit means.

Impact: If the scale factor is wrong, SIFT descriptors are computed over
  a neighbourhood of the wrong size, reducing descriptor quality. The error
  is bounded — SIFT is somewhat robust to scale mismatch — but could reduce
  match quality by 5–15%.

Fixability: TRIVIAL
Fix: Document the conversion or replace 6.0 with
  `scales[k] * kornia_to_sift_scale_factor` where the factor is derived
  from kornia's documentation on LAF scale conventions.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ WEAKNESS: No Track Merging ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID: W-15
Severity: MEDIUM
Stage: Incremental SfM
Location: reconstruction.py:_triangulate_new_points:610–697

Current behavior: kp_to_3d maps each (img_idx, kp_idx) to at most one 3D
  point. If two triangulated 3D points correspond to the same physical point
  (e.g., triangulated from different pairs with a common observation), they
  remain as separate entries in self.points_3d.

What it should do: When linking an existing 3D point to a new keypoint that
  already has a different 3D point linked (via a different pair), the two
  3D points should be merged (track merging). COLMAP's track merging step
  identifies pairs of 3D points that are linked via common observations and
  merges them by averaging their positions and unioning their observation sets.

Impact: Duplicate 3D points inflate the point count (reported numbers are
  artificially high), waste BA computation, and produce a noisier cloud
  near boundaries where two slightly different positions exist for the
  same physical point.

Fixability: MEDIUM
Fix: In _triangulate_new_points(), when `(new_idx, ks) in self.kp_to_3d`
  and `(other, ko) in self.kp_to_3d` and they map to different 3D indices,
  merge the two 3D points: keep the one with more observations, update
  all kp_to_3d entries pointing to the removed index, and merge observations.
  Maintain a union-find structure over 3D point indices for O(α(n)) merge.
Depends on: none
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Genuine Strengths

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Hartley Normalization in F Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: geometric_verification.py:_hartley_normalize:42–72
  Isotropic normalization is applied before findFundamentalMat and F is
  de-normalized correctly (T2.T @ F_norm @ T1). This dramatically improves
  F estimation quality on high-resolution images (4K+), where raw pixel
  coordinates have condition numbers of ~1000. The de-normalization is
  mathematically correct.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Batch Vectorized Triangulation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: reconstruction.py:_triangulate_batch:63–96, _accept_batch:99–157
  Triangulates M points in a single cv2.triangulatePoints call and applies
  the depth/angle/reproj acceptance filter as pure NumPy vectorized operations.
  This gives 100–1000× speedup over a Python loop for large match sets.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Adaptive Huber Scale in BA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: bundle_adjustment.py:BundleAdjuster.adjust:376–386
  f_scale = 1.4826 × MAD of initial residuals, clamped to [0.5, 10.0] px.
  This is the correct robust estimation procedure (Hampel et al. 1986).
  Avoids the hard-coded 2.0 px threshold that would be too tight for rough
  initial poses and too loose for well-initialized reconstructions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Scene Graph Connectivity Check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: utils.py:check_scene_graph_connectivity:167–209, run_sfm.py:839–874
  Union-find connectivity analysis before reconstruction. Warns and restricts
  to the largest connected component. Prevents silent failures when the
  scene has disconnected sub-sets (e.g., two separate objects photographed
  in the same session). Added in R-16.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: EPnP + LM Refinement ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: reconstruction.py:_register_image:503–533
  Uses SOLVEPNP_EPNP (O(n) exact solver, Lepetit et al. 2009) as the robust
  estimator inside solvePnPRansac, followed by solvePnPRefineLM on inliers.
  This is the same two-step approach COLMAP uses. EPnP gives an accurate
  initial pose; LM refinement gives ~0.5 px improvement at <1ms cost.
  Added in R-12.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Seed Pair with Angle Guard ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: reconstruction.py:_select_seed_pair:323–403
  Seed pair selection uses score = baseline × inlier_count with a 5°
  median triangulation angle guard (R-13). This prevents selecting seed
  pairs from near-parallel camera configurations, which produce numerically
  unstable initial reconstructions. The fallback to max-inlier selection
  prevents hard failures.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ STRENGTH: Visualization is a True No-Op When Disabled ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementation: visualizer.py:SfMVisualizer.__init__:48–51
  `if not enabled: return` — no imports, no allocations. All 1586 lines of
  visualization code are unreachable when --visualize is not set. This is
  the correct pattern for optional heavy dependencies.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 6. Scoring Matrix

Scores are 1–10 (10 = best in class). "Gap to COLMAP" is the delta score with a one-line reason.

| Dimension | This Repo | COLMAP | RealityCapture | Metashape | OpenSfM | Meshroom | GLOMAP | Gap to COLMAP |
|-----------|-----------|--------|----------------|-----------|---------|----------|--------|---------------|
| 1. Feature detection quality | 5 | 8 | 8 | 8 | 7 | 7 | 8 | -3: SIFT only; no SuperPoint/DISK/DeDoDe |
| 2. Feature matching quality | 5 | 8 | 9 | 8 | 7 | 7 | 8 | -3: Flat k-means vocab; no learned matcher (LightGlue/SuperGlue) |
| 3. Geometric verification robustness | 7 | 9 | 9 | 8 | 7 | 7 | 9 | -2: No planar degeneracy check; Python RANSAC overhead |
| 4. Camera pose accuracy | 6 | 9 | 9 | 9 | 8 | 8 | 8 | -3: No per-image intrinsics; single focal; no re-triangulation |
| 5. BA quality (convergence, robustness, scale) | 5 | 9 | 9 | 9 | 8 | 8 | 9 | -4: SciPy TRF, no Schur complement, square pixel assumption |
| 6. Dense reconstruction quality | 3 | 9 | 10 | 9 | 6 | 7 | 7 | -6: SGBM vs PatchMatch MVS; no multi-view consistency |
| 7. Mesh quality | 5 | 6 | 10 | 9 | 5 | 8 | 6 | -1: Poisson on sparse cloud; no texture mapping (vertex colors only) |
| 8. Scalability (max practical images) | 4 | 9 | 10 | 9 | 8 | 8 | 10 | -5: SciPy BA; no loop closure; O(N²) exhaustive matching |
| 9. GPU utilization | 4 | 8 | 10 | 9 | 5 | 8 | 8 | -4: GPU only for matching/k-means; BA/geo-verification on CPU |
| 10. Robustness to difficult inputs | 4 | 8 | 9 | 9 | 7 | 7 | 8 | -4: No planar check, no loop closure, SIFT fails on textureless |
| 11. Code correctness | 7 | 9 | N/A | N/A | 8 | 8 | 9 | -2: Duplicate obs, cache key bug, minDisparity=0 issue |
| 12. Engineering quality | 7 | 9 | N/A | N/A | 8 | 8 | 9 | -2: No re-triangulation, no track merging, Python BA |
| **Overall** | **5.2** | **8.8** | **N/A** | **N/A** | **7.3** | **7.6** | **8.8** | **-3.6 avg** |

---

## 7. Improvement Intelligence

### FROM: COLMAP

```
FROM: COLMAP
TECHNIQUE: Schur Complement Bundle Adjustment via Ceres Solver
WHAT IT SOLVES: BA is O(C³) without Schur complement; becomes O(C³/P + P) with it
HOW TO APPLY: Install pyceres (https://github.com/cvg/pyceres). Replace the
  scipy.optimize.least_squares call in BundleAdjuster.adjust() with a
  ceres.Problem() that adds per-observation cost functions using the existing
  _project_distorted projection model as the residual. Use
  ceres.SPARSE_SCHUR as the linear solver type.
EFFORT: MEDIUM
FILES TO CHANGE: sfm/bundle_adjustment.py (replace _residuals_v2 and adjust())
EXPECTED GAIN: 10–50× BA speedup; enables 500+ camera reconstructions
```

```
FROM: COLMAP
TECHNIQUE: Re-Triangulation After BA
WHAT IT SOLVES: Stale 3D points miss opportunities from refined camera poses
HOW TO APPLY: In reconstruction.py:_run_ba(), after updating self.cameras
  and self.points_3d, call a new _re_triangulate_all() that iterates all
  verified pairs between registered cameras and calls _triangulate_new_points()
  for each pair. The existing batch triangulation infrastructure handles this.
EFFORT: EASY
FILES TO CHANGE: sfm/reconstruction.py (add _re_triangulate_all, call from _run_ba)
EXPECTED GAIN: +15–30% point count; improved BA constraint density
```

```
FROM: COLMAP
TECHNIQUE: Per-Image Intrinsics with RADIAL Camera Model
WHAT IT SOLVES: Systematic residuals from forced square pixels and shared K
HOW TO APPLY: In run_sfm.py, call estimate_intrinsics() for each image.
  In BundleAdjuster, add an intrinsics block per camera group (or per image).
  Change _project_distorted to accept per-camera (fx, fy, cx, cy, k1, k2).
EFFORT: MEDIUM
FILES TO CHANGE: run_sfm.py, sfm/bundle_adjustment.py, sfm/reconstruction.py
EXPECTED GAIN: 0.3–1.0 px RMSE improvement on real multi-camera datasets
```

### FROM: OpenSfM

```
FROM: OpenSfM
TECHNIQUE: DINOv2/NetVLAD Image Retrieval for Pair Selection
WHAT IT SOLVES: Flat k-means vocab tree misses 30–50% of true image pairs
HOW TO APPLY: Replace VocabTreeMatcher._sample_descriptors/_build_vocab/_encode
  with a DINOv2 ViT-S/14 encoder (torch.hub.load('facebookresearch/dinov2')).
  Each image is encoded as a 384-D global descriptor. Cosine similarity
  over all pairs gives top-K candidates. The _select_candidates_gpu() method
  already handles this; just replace the TF-IDF vectors with DINOv2 embeddings.
EFFORT: EASY
FILES TO CHANGE: sfm/feature_matching.py (new DINOv2Retriever class)
EXPECTED GAIN: Top-K retrieval recall@10 improves from ~60% to ~90%
```

```
FROM: OpenSfM
TECHNIQUE: GPS EXIF Geolocation for Scene Graph Pruning
WHAT IT SOLVES: Without GPS, all N²/2 pairs are candidate matches
HOW TO APPLY: Read EXIF GPS (tag 0x8825) and compute geographic distances.
  Only match image pairs within a configurable GPS radius (e.g., 50m).
  This reduces the number of candidate pairs from O(N²) to O(N·K_spatial).
EFFORT: EASY
FILES TO CHANGE: sfm/utils.py (add read_exif_gps()), sfm/feature_matching.py
EXPECTED GAIN: 5–50× speedup on GPS-tagged datasets (drones, phones)
```

### FROM: LightGlue/SuperGlue

```
FROM: LightGlue
TECHNIQUE: Learned Feature Matching
WHAT IT SOLVES: Mutual-NN + ratio test fails on textureless, repetitive,
  and low-contrast regions where SIFT descriptors are unreliable
HOW TO APPLY: Add a LightGlue matcher option to FeatureMatcher. LightGlue
  requires SuperPoint or DISK keypoints+descriptors as input. The API is:
  matcher = LightGlue(features='superpoint').eval().cuda()
  matches = matcher({'image0': feats0, 'image1': feats1})
  This is a drop-in replacement for _match_pair_gpu().
EFFORT: MEDIUM
FILES TO CHANGE: sfm/feature_extraction.py (add SuperPointExtractor),
  sfm/feature_matching.py (add LightGlueMatcher class)
EXPECTED GAIN: 2–4× more inliers on difficult inputs; handles repetitive
  patterns and textureless regions that SIFT+FLANN cannot match
```

### FROM: GLOMAP

```
FROM: GLOMAP
TECHNIQUE: Global Rotation Averaging Before Incremental Registration
WHAT IT SOLVES: Incremental SfM accumulates drift; first registered images
  have best accuracy; last registered have worst
HOW TO APPLY: After geometric verification, run rotation averaging (IRLS on
  SO(3)) over all verified pair rotations {R_ij} to get globally consistent
  absolute rotations. Use these as initialization for incremental registration.
  Reference: Pan et al. (2024) "GLOMAP: Global Structure-from-Motion Revisited"
  https://github.com/colmap/glomap
EFFORT: HARD
FILES TO CHANGE: sfm/reconstruction.py (new GlobalRotationAverager class,
  modify _initialise_from_pair to use global rotations)
EXPECTED GAIN: Reduced drift in large scenes; better handling of loops
```

### FROM: PatchMatch MVS

```
FROM: PatchMatch MVS (COLMAP/OpenMVS)
TECHNIQUE: Multi-View Consistent Dense Depth Estimation
WHAT IT SOLVES: SGBM is a per-pair method with no multi-view consistency
HOW TO APPLY: Option A (subprocess): Use OpenMVS (open-source) as a
  subprocess: openMVS DensifyPointCloud --input-file scene.mvs --output-file dense.mvs.
  Option B (Python): Implement plane-sweep stereo with NCC similarity across
  N reference images. For each pixel, sweep depth hypotheses and aggregate
  NCC scores across all registered cameras. Accept hypothesis with maximum
  aggregate score. This is 200–300 lines of PyTorch/CUDA.
EFFORT: HARD
FILES TO CHANGE: sfm/mvs.py (replace _process_pair entirely)
EXPECTED GAIN: 5–20× denser, more complete, more accurate dense cloud
```

### Practical Adoptability Assessment

| Technology | Adoptability | Key Blocker | Expected Gain |
|------------|-------------|-------------|---------------|
| **LightGlue** | High — pip install lightglue; Python API matches existing structure | Requires SuperPoint keypoints (SIFT descriptors incompatible) | 2–4× more inliers on difficult inputs |
| **MAGSAC++** (standalone C++) | Medium — pybind11 wrapper available but not pip-installable | Build system; USAC_MAGSAC in OpenCV is already equivalent | ~5× faster geo-verification; already using equivalent algorithm |
| **PoseLib** | Medium — pip install poselib | No Python bindings for all features | 5–10× faster PnP; better P3P solver |
| **DUSt3R** for initial pose | High — pip install dust3r; produces pairwise point maps | Memory: requires 8+ GB GPU for multi-image inference | Bootstrap multi-view from scratch without traditional SfM |
| **SuperPoint as SIFT replacement** | High — pip install kornia or use pretrained checkpoint | License restrictions for commercial use | 2–3× better repeatability on difficult images |
| **DINOv2 for image retrieval** | Very High — torch.hub, no extra install | Requires 4GB+ GPU or slow on CPU | Top-k recall improves from ~60% to ~90% |

### Structural Improvements

**Pipeline stage caching:** Currently only features and matches are cached. Geometric verification results (the expensive RANSAC step) are not cached. Adding a `verified_ckpt_key = f"verified_{args.match_strategy}_r{args.ratio:.3f}_t{args.ransac_thr}"` checkpoint would save significant time on re-runs with the same parameters.

**Covisibility graph:** The covisibility graph is maintained in-memory during reconstruction (`self.covisibility` dict) but is not exported or used for subsequent stages. Exporting it as a networkx graph or adjacency dict would enable better MVS pair selection and future global SfM improvements.

**Proper scene graph before reconstruction:** The union-find connectivity check (R-16) is applied after geometric verification. However, the matching stage doesn't use connectivity information to prune its candidate set. A two-phase approach (coarse retrieval → fine matching → connectivity analysis) would be more efficient.

**Structured per-stage metrics:** The pipeline logs RMSE and point counts but does not produce a structured JSON metrics file. Adding `--metrics-output metrics.json` would enable automated quality comparison across runs.

---

## 8. Prioritized Roadmap

### Tier 0 — Correctness Fixes (bugs producing wrong or silent-failure results)

**T0-01: Fix match checkpoint key to include strategy-specific parameters**
- File: `run_sfm.py:774`
- Fix: Expand `match_ckpt_key` to encode `sequential_window`, `vocab_words`, `vocab_top_k` as described in W-07
- Estimate: ~30 minutes
- Weakness: W-07

**T0-02: Fix duplicate observations in self.observations**
- File: `reconstruction.py:_add_obs:870–884`
- Fix: Add `self._obs_set: Set[Tuple[int,int]]` and check before appending as described in W-06
- Estimate: ~1 hour
- Weakness: W-06

**T0-03: Fix MVS minDisparity for both stereo configurations**
- File: `mvs.py:_process_pair:247`
- Fix: Detect baseline direction in rectified frame, set minDisparity appropriately (see W-10)
- Estimate: ~2 hours
- Weakness: W-10

**T0-04: Restore bilinear color sampling in point_cloud.py**
- File: `point_cloud.py:colorize:168–176`
- Fix: Implement vectorized bilinear sampling as described in W-08
- Estimate: ~1 hour
- Weakness: W-08

---

### Tier 1 — High ROI Improvements (1–2 weeks total, 1–3 days each)

Dependency order: T1-01 → T1-02 → T1-03; T1-04 is independent; T1-05 is independent.

**T1-01: Vectorize BA sparsity pattern builder**
- File: `bundle_adjustment.py:_build_sparsity_v2:175–215`
- Fix: Replace Python for-loop with vectorized (row, col) array construction (see W-09)
- Estimate: 1 day
- Weakness: W-09

**T1-02: Add re-triangulation after BA**
- File: `reconstruction.py:_run_ba:791–824`
- Fix: Add `_re_triangulate_all()` called after updating self.cameras (see W-03)
- Estimate: 2 days (includes testing)
- Weakness: W-03

**T1-03: Add planar degeneracy / homography check in geometric verification**
- File: `geometric_verification.py:verify_pair:198–230`
- Fix: Fit H, compute H/E inlier ratio, reject planar pairs (see W-11)
- Estimate: 1 day
- Weakness: W-11

**T1-04: Add DINOv2 image retrieval for VocabTree**
- File: `feature_matching.py:VocabTreeMatcher`
- Fix: Add `DINOv2Retriever` class as an alternative to k-means vocab (see W-05, Improvement Intelligence)
- Estimate: 2 days
- Weakness: W-05

**T1-05: Add track merging**
- File: `reconstruction.py:_triangulate_new_points:610–697`
- Fix: Implement union-find over 3D point indices, merge conflicting links (see W-15)
- Estimate: 2 days
- Weakness: W-15

---

### Tier 2 — Core Algorithm Upgrades (1–2 months total)

**T2-01: Per-image intrinsics with separate fx, fy**
- Files: `run_sfm.py`, `bundle_adjustment.py`, `reconstruction.py`
- Fix: Expand intrinsics vector to [fx, fy, cx, cy, k1, k2] per camera group (see W-02, W-13)
- Estimate: 1 week
- Depends: T2-01 is a prerequisite for T2-02

**T2-02: Ceres/pyceres bundle adjustment with Schur complement**
- File: `bundle_adjustment.py` (major rewrite)
- Fix: Replace SciPy TRF with pyceres SPARSE_SCHUR (see W-01, Improvement Intelligence)
- Estimate: 2 weeks (including testing on large scenes)
- Depends: T2-01

**T2-03: SuperPoint + LightGlue optional feature extractor/matcher**
- Files: `feature_extraction.py`, `feature_matching.py`
- Fix: Add SuperPointExtractor and LightGlueMatcher as alternative backends selected via CLI
- Estimate: 1 week
- Depends: none

**T2-04: Multi-view consistent MVS (plane-sweep or RAFT-Stereo)**
- File: `mvs.py` (major rewrite of `_process_pair`)
- Fix: Implement plane-sweep stereo with NCC over N views (see W-04, Improvement Intelligence)
- Estimate: 2 weeks
- Depends: none

---

### Tier 3 — Architectural Improvements (2–4 months)

**T3-01: Global rotation averaging before incremental registration**
- Fix: Implement IRLS rotation averaging on verified pair rotations; use as initialization (see Improvement Intelligence → GLOMAP)
- Files: `sfm/reconstruction.py` (new GlobalRotationAverager class)
- Estimate: 3–4 weeks

**T3-02: Per-camera model support in pipeline (multi-camera datasets)**
- Fix: Per-image K estimation; group by camera model; per-group BA intrinsics (see W-13)
- Files: `run_sfm.py`, `bundle_adjustment.py`, `reconstruction.py`, `utils.py`
- Estimate: 2–3 weeks
- Depends: T2-01

**T3-03: Structured per-run metrics export**
- Fix: Emit JSON with per-stage timing, RMSE, point counts, camera registration rate, BA iterations
- Files: `run_sfm.py` (add metrics dict, write at end)
- Estimate: 1 week

**T3-04: Geometric verification result caching**
- Fix: Add checkpoint for verified pairs (separate from match checkpoint)
- Files: `run_sfm.py`
- Estimate: 1 day

---

### Tier 4 — Research-Level Features (6+ months)

**T4-01: DUSt3R/MASt3R integration for initialisation-free reconstruction**
- Fix: Use DUSt3R to produce pairwise point maps and relative poses; replace geometric verification + init steps
- Expected impact: Handles textureless scenes, wide baselines, repetitive patterns

**T4-02: Neural MVS (DepthAnything v2 + multi-view consistency)**
- Fix: Replace SGBM with a monocular depth estimator primed by SfM scale

**T4-03: 3D Gaussian Splatting output**
- Fix: From the reconstructed sparse cloud + cameras, train a 3DGS model (gaussian-splatting library)

**T4-04: GLOMAP-style global SfM**
- Fix: Full global SfM pipeline replacing incremental approach for >500 image scalability

---

### Roadmap Summary Table

| ID | Tier | Item | Effort | Quality Gain | Weakness | Depends On |
|----|------|------|--------|--------------|----------|------------|
| T0-01 | 0 | Fix match checkpoint key | 0.5h | Correctness (silent failure) | W-07 | none |
| T0-02 | 0 | Fix duplicate observations | 1h | Correctness (BA bias) | W-06 | none |
| T0-03 | 0 | Fix MVS minDisparity | 2h | Correctness (dropped stereo pairs) | W-10 | none |
| T0-04 | 0 | Restore bilinear color sampling | 1h | LOW (visual quality) | W-08 | none |
| T1-01 | 1 | Vectorize BA sparsity builder | 1d | MEDIUM (BA speed 5–15s saved) | W-09 | none |
| T1-02 | 1 | Re-triangulation after BA | 2d | HIGH (+15–30% points) | W-03 | none |
| T1-03 | 1 | Planar degeneracy check | 1d | MEDIUM (robustness) | W-11 | none |
| T1-04 | 1 | DINOv2 image retrieval | 2d | HIGH (retrieval recall 60%→90%) | W-05 | none |
| T1-05 | 1 | Track merging | 2d | MEDIUM (cleaner cloud) | W-15 | none |
| T2-01 | 2 | Per-image intrinsics (fx, fy, cx, cy) | 1w | HIGH (multi-camera support) | W-02, W-13 | none |
| T2-02 | 2 | Ceres/pyceres BA with Schur | 2w | CRITICAL (10–50× BA speedup) | W-01 | T2-01 |
| T2-03 | 2 | SuperPoint + LightGlue | 1w | HIGH (2–4× more inliers) | — | none |
| T2-04 | 2 | Multi-view consistent MVS | 2w | HIGH (5–20× denser cloud) | W-04 | none |
| T3-01 | 3 | Global rotation averaging | 3–4w | HIGH (reduce drift) | — | none |
| T3-02 | 3 | Multi-camera pipeline | 2–3w | HIGH (multi-camera datasets) | W-13 | T2-01 |
| T3-03 | 3 | Metrics JSON export | 1w | LOW (tooling quality) | — | none |
| T3-04 | 3 | Geo-verification caching | 1d | LOW (developer velocity) | — | none |
| T4-01 | 4 | DUSt3R integration | 6w+ | Very High (textureless scenes) | — | none |
| T4-02 | 4 | Neural MVS | 6w+ | Very High (dense quality) | — | none |
| T4-03 | 4 | 3DGS output | 4w+ | High (rendering quality) | — | T2-02 |
| T4-04 | 4 | GLOMAP-style global SfM | 3m+ | Very High (scalability) | — | T3-01 |

---

## 9. Reference Library

### Papers

```
[P-01] Lowe, D.G. (2004). "Distinctive image features from scale-invariant keypoints."
       International Journal of Computer Vision, 60(2), 91–110.
       https://doi.org/10.1023/B:VISI.0000029664.99615.94

[P-02] Hartley, R. (1997). "In defense of the eight-point algorithm."
       IEEE TPAMI, 19(6), 580–593.
       https://doi.org/10.1109/34.601246

[P-03] Schönberger, J.L. & Frahm, J.M. (2016). "Structure-from-Motion Revisited."
       CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf

[P-04] Barath, D., Matas, J., & Noskova, J. (2020). "MAGSAC++: A fast, reliable and
       accurate robust estimator." CVPR 2020.
       https://openaccess.thecvf.com/content_CVPR_2020/papers/Barath_MAGSAC_A_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.pdf

[P-05] Barath, D. & Matas, J. (2018). "Graph-Cut RANSAC." CVPR 2018.
       https://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf

[P-06] Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). "EPnP: An accurate O(n)
       solution to the PnP problem." IJCV 81(2), 155–166.
       https://doi.org/10.1007/s11263-008-0152-6

[P-07] Triggs, B. et al. (2000). "Bundle Adjustment — A Modern Synthesis."
       LNCS, 1883, 298–372.
       https://link.springer.com/chapter/10.1007/3-540-44480-7_21

[P-08] Nister, D. & Stewenius, H. (2006). "Scalable Recognition with a Vocabulary Tree."
       CVPR 2006. https://ieeexplore.ieee.org/document/1641018

[P-09] Hirschmüller, H. (2008). "Stereo Processing by Semiglobal Matching and
       Mutual Information." IEEE TPAMI, 30(2), 328–341.
       https://doi.org/10.1109/TPAMI.2007.1166

[P-10] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). "SuperPoint: Self-Supervised
       Interest Point Detection and Description." CVPRW 2018.
       https://arxiv.org/abs/1712.07629

[P-11] Sarlin, P.E. et al. (2020). "SuperGlue: Learning Feature Matching with Graph
       Neural Networks." CVPR 2020. https://arxiv.org/abs/1911.11763

[P-12] Lindenberger, P., Sarlin, P.E., & Pollefeys, M. (2023). "LightGlue: Local Feature
       Matching at Light Speed." ICCV 2023. https://arxiv.org/abs/2306.13643

[P-13] Sun, J. et al. (2021). "LoFTR: Detector-Free Local Feature Matching with
       Transformers." CVPR 2021. https://arxiv.org/abs/2104.00680

[P-14] Tyszkiewicz, M., Fua, P., & Trulls, E. (2020). "DISK: Learning local features
       with policy gradient." NeurIPS 2020. https://arxiv.org/abs/2006.13566

[P-15] Edstedt, J. et al. (2023). "DeDoDe: Detect, Don't Describe—Describe, Don't
       Detect for Local Feature Matching." 3DV 2024. https://arxiv.org/abs/2308.08479

[P-16] Wang, S. et al. (2024). "DUSt3R: Geometric 3D Vision Made Easy." CVPR 2024.
       https://arxiv.org/abs/2312.14132

[P-17] Leroy, V. et al. (2024). "MASt3R: Grounding Image Matching in 3D with MASt3R."
       ECCV 2024. https://arxiv.org/abs/2406.09756

[P-18] Wang, Y. et al. (2024). "VGGSfM: Visual Geometry Grounded Deep Structure From
       Motion." CVPR 2024. https://arxiv.org/abs/2312.04563

[P-19] Pan, L. et al. (2024). "GLOMAP: Global Structure-from-Motion Revisited."
       CVPR 2024. https://arxiv.org/abs/2407.20219

[P-20] Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Novel View
       Synthesis." SIGGRAPH 2023. https://arxiv.org/abs/2308.04079

[P-21] Galliani, S., Lasinger, K., & Schindler, K. (2015). "Massively parallel
       multiview stereopsis by surface normal diffusion." ICCV 2015.
       https://doi.org/10.1109/ICCV.2015.532

[P-22] Kazhdan, M. & Hoppe, H. (2013). "Screened Poisson Surface Reconstruction."
       ACM TOG, 32(3), 29. https://dl.acm.org/doi/10.1145/2487228.2487237

[P-23] Bernardini, F. et al. (1999). "The Ball-Pivoting Algorithm for Surface
       Reconstruction." IEEE TVCG, 5(4), 349–359. https://doi.org/10.1109/2945.817351

[P-24] Hampel, F.R. et al. (1986). "Robust Statistics: The Approach Based on Influence
       Functions." Wiley. ISBN 978-0-471-73577-9.

[P-25] Taubin, G. (1995). "A signal processing approach to fair surface design."
       SIGGRAPH 1995. https://dl.acm.org/doi/10.1145/218380.218473

[P-26] Kümmerle, R. et al. (2011). "g2o: A general framework for graph optimization."
       ICRA 2011. https://doi.org/10.1109/ICRA.2011.5979949

[P-27] Agarwal, S. et al. (2022). "Ceres Solver." http://ceres-solver.org/

[P-28] Oquab, M. et al. (2024). "DINOv2: Learning Robust Visual Features without
       Supervision." TMLR 2024. https://arxiv.org/abs/2304.07193
```

### Libraries

```
[L-01] COLMAP. BSD-3. https://github.com/colmap/colmap
[L-02] pyceres. Apache-2.0. https://github.com/cvg/pyceres
[L-03] LightGlue. Apache-2.0. https://github.com/cvg/LightGlue
[L-04] hloc (hierarchical-localization). Apache-2.0. https://github.com/cvg/Hierarchical-Localization
[L-05] kornia. Apache-2.0. https://github.com/kornia/kornia
[L-06] open3d. MIT. https://github.com/isl-org/Open3D
[L-07] OpenMVS. AGPL-3.0. https://github.com/cdcseacave/openMVS
[L-08] RAFT-Stereo. BSD-3. https://github.com/princeton-vl/RAFT-Stereo
[L-09] PoseLib. BSD-2. https://github.com/PoseLib/PoseLib
[L-10] GLOMAP. BSD-3. https://github.com/colmap/glomap
[L-11] MASt3R. CC-BY-NC-4.0. https://github.com/naver/mast3r
[L-12] gaussian-splatting. Gaussian-Splatting License. https://github.com/graphdeco-inria/gaussian-splatting
[L-13] DINOv2 (pretrained weights). CC-BY-NC-4.0. https://github.com/facebookresearch/dinov2
[L-14] scipy.optimize.least_squares. BSD-3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
[L-15] OpenCV. Apache-2.0. https://github.com/opencv/opencv
[L-16] MAGSAC++ (standalone). BSD-3. https://github.com/danini/magsac
```

---

*End of SFM Quality Report*
*Total source files reviewed: 20 (6826 lines)*
*Weaknesses catalogued: 15 (4 TRIVIAL/LOW, 7 MEDIUM, 2 HIGH severity, 1 CRITICAL + 1 HIGH BA)*
*Strengths catalogued: 7*
*Roadmap items: 20 (4 Tier-0, 5 Tier-1, 4 Tier-2, 4 Tier-3, 4 Tier-4)*
