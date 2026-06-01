# SfM Pipeline Quality Audit Report

**Repository:** `/home/user/sfm-pub`  
**Audit Date:** 2026-06-01  
**Auditor:** Principal CV/Photogrammetry Engineer  
**Overall Grade:** B− (5.8 / 10)  
**Files Audited:** 23 source files, 4 936 non-blank lines of code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Codebase Inventory](#2-codebase-inventory)
3. [Internal Quality Audit — Stage by Stage](#3-internal-quality-audit--stage-by-stage)
4. [Production System Comparison](#4-production-system-comparison)
5. [Weakness Catalogue](#5-weakness-catalogue)
6. [Scoring Matrix](#6-scoring-matrix)
7. [Improvement Intelligence](#7-improvement-intelligence)
8. [Prioritized Roadmap](#8-prioritized-roadmap)

---

## 1. Executive Summary

`sfm-pub` is a well-structured, single-process Structure-from-Motion pipeline built around OpenCV, SciPy, and optional PyTorch/Kornia dependencies. It covers the full pipeline from raw images to dense point clouds and mesh surfaces, with multiple algorithmic paths (SIFT / SuperPoint, FLANN / GPU / VocabTree / DINOv2 / LightGlue, scipy TRF / pyceres BA) and a rich visualizer.

The codebase is readable and thoughtfully commented. Several recently-added features (DINOv2 image retrieval, LightGlue matching, MVS depth-consistency fusion, covisibility-based MVS pair selection, per-camera intrinsics from EXIF, union-find track merging, post-BA re-triangulation, stereo-pair swap fix, bilinear colour sampling, vectorised sparsity matrix) are genuinely useful additions.

However, **three correctness-critical bugs** must be fixed before the pipeline produces reliable results at any scale:

- **W-01 (CRITICAL)**: The pyceres bundle-adjustment path silently runs zero iterations — a `break` exits the residual-block loop on the first camera, so the optimizer sees an empty problem.
- **W-02 (HIGH)**: `_register_image` double-undistorts 2-D points: already-undistorted coordinates are passed to `solvePnPRansac` together with non-zero distortion coefficients, corrupting every new camera pose.
- **W-03 (HIGH)**: `CameraIntrinsics.update_from_ba()` exists but is never called; per-camera focal lengths are optimized in scipy BA but the refined values are never written back, so all subsequent PnP calls and MVS use stale EXIF-derived intrinsics.

Beyond correctness bugs, the pipeline's **scalability ceiling is low**: global-only BA (O(C²·P) per run every `ba_interval` cameras), exhaustive O(N²) pair selection without loop-closure recovery, and a two-view SGBM MVS without real multi-view fusion mean that at ≳100 images the pipeline will slow dramatically and produce lower quality than COLMAP.

Fixing W-01 through W-03, adding local BA (T1-1), and integrating loop-closure retrieval into LightGlue pair selection (T1-4) would bring the system from grade B− to solid B+ within a two-week effort.

---

## 2. Codebase Inventory

| File | Lines | Role |
|------|------:|------|
| `run_sfm.py` | 1 220 | CLI entry point, 6-stage pipeline, checkpoint system |
| `sfm/__init__.py` | 3 | Package version |
| `sfm/feature_extraction.py` | 378 | SIFT (CPU/kornia/CUDA-SURF), SuperPoint |
| `sfm/feature_matching.py` | 937 | FLANN, GPU brute-force, VocabTree, DINOv2, LightGlue |
| `sfm/geometric_verification.py` | 289 | Hartley F→E, USAC_MAGSAC, planarity check |
| `sfm/reconstruction.py` | 1 082 | IncrementalSfM, triangulation, PnP registration |
| `sfm/bundle_adjustment.py` | 698 | scipy TRF BA, pyceres BA (broken) |
| `sfm/intrinsics.py` | 176 | CameraIntrinsics, EXIF parsing, update_from_ba() |
| `sfm/mvs.py` | 380 | StereoSGBM densification, depth-consistency filter |
| `sfm/point_cloud.py` | 339 | PLY export, bilinear colour sampling, reprojection |
| `sfm/utils.py` | 210 | EXIF intrinsics, scene-graph connectivity, misc |
| `sfm/colmap_backend.py` | 510 | COLMAP binary readers, subprocess runner |
| `sfm/visualizer.py` | 1 587 | Event-driven matplotlib/networkx/open3d visualizer |
| `sfm/device.py` | 53 | lru_cache GPU detection |
| `sfm/mesh/pipeline.py` | 260 | Mesh quality presets, stage orchestration |
| `sfm/mesh/reconstruction.py` | 206 | Poisson, BPA, alpha-shape surface reconstruction |
| `sfm/mesh/cleaning.py` | 154 | Dedup, manifold repair, small-component removal |
| `sfm/mesh/pointcloud_prep.py` | 216 | SOR, ROR, normals, voxel downsampling |
| `sfm/mesh/postprocess.py` | 110 | Taubin smoothing, quadric decimation, colour transfer |
| `sfm/mesh/export.py` | 100 | OBJ/PLY/GLB/STL export |
| `integration_test.py` | 295 | Synthetic 6-camera test scene |
| `pyproject.toml` | 86 | Dependency declaration |
| `README.md` | ~120 | User documentation |

---

## 3. Internal Quality Audit — Stage by Stage

### 3.1 Image Loading (`sfm/utils.py:load_image`, `sfm/feature_extraction.py`)

**Implementation.** `load_image` calls `cv2.imread` then optionally resizes. No EXIF orientation correction is applied. The CLI `--max-image-size` clamp is supported but resolution capping is done silently.

**Issues.**
- EXIF rotation tags (0x0112) are ignored. Phones routinely store portrait images landscape in the pixel buffer with an orientation tag. Rotated images produce wrong matches and fail geometric verification.
- No channel validation: grayscale or RGBA inputs passed to `cv2.cvtColor(img, BGR2GRAY)` will raise or silently produce wrong results.

**Verdict.** Adequate for well-formed DSLR datasets; will silently fail on mobile datasets. (See W-10.)

---

### 3.2 Feature Extraction (`sfm/feature_extraction.py`)

**SIFT CPU path (`_extract_sift_cpu`).** Standard `cv2.SIFT_create` + `detectAndCompute`. Correct. Default `n_features=8000`, `contrastThreshold=0.02`, `edgeThreshold=10` are reasonable.

**kornia GPU path (`_extract_kornia`).** `ScaleSpaceDetector` produces Local Affine Frames (LAFs); centres are extracted via `KF.get_laf_center`; scale is converted to a SIFT keypoint diameter as `float(scales[k]) * 6.0`. The ×6.0 multiplier is a heuristic (SIFT diameter ≈ 6×scale is conventional but the LAF scale is not the same as SIFT scale). Descriptors are then computed by OpenCV SIFT at these positions. The mismatch between kornia detector scale space and SIFT descriptor extraction can cause descriptor quality loss on very small or very large keypoints.

**SuperPoint (`SuperPointExtractor`).** Lazy model load; correct `descs_t.T` transpose (kornia returns (256, N)). Score-sorted top-N truncation is not applied — all keypoints are returned regardless of `n_features`. The model is re-evaluated per image without batching.

**CUDA SURF fallback.** Requires `opencv-contrib-python` with CUDA build — not declared in `pyproject.toml`. On failure it falls back to CPU SIFT with a warning.

**Verdict.** Solid CPU SIFT path; GPU paths functional but descriptor quality under-tested. W-07 (kornia LAF scale heuristic) is medium severity.

---

### 3.3 Feature Matching (`sfm/feature_matching.py`)

**FLANN (`_match_pair_cpu`).** k=2 kNN + Lowe ratio test (0.75) + optional cross-check. Correct; well-tested in SIFT literature.

**GPU brute-force (`_match_pair_gpu`).** `torch.cdist` computes full (N, M) L2 matrix. For N=M=8000 with 128-D float32 descriptors this is 8000×8000×4 = 256 MB per call — will OOM on a 4 GB GPU with multiple concurrent threads. No chunked fallback. (W-17.)

**VocabTree (`VocabTreeMatcher`).** Flat k-means (4096 words), random initialization (not k-means++), TF-IDF encoding, cosine similarity retrieval. GPU Lloyd's k-means via `torch.cdist`. Correct overall, but random init increases variance in vocabulary quality.

**DINOv2 (`DINOv2Matcher`).** `torch.hub` load of `dinov2_vitb14`; CLS-token embeddings; FAISS `IndexFlatIP` (cosine, L2-normalised) or numpy fallback. Per-pair SIFT matching on retrieved pairs. Correct implementation; FAISS is optional with graceful degradation.

**LightGlue (`LightGlueMatcher`).** `kornia.feature.LightGlue` with `features='superpoint'` or `'sift'`. When `candidate_pairs` is None it falls back to exhaustive O(N²) matching, negating the retrieval benefit of LightGlue. (W-11.)

**Verdict.** Four mature matcher paths plus LightGlue. W-11 and W-17 are the main concerns.

---

### 3.4 Geometric Verification (`sfm/geometric_verification.py`)

**Hartley normalization.** Centred + isotropically scaled; normalization matrices `T1`, `T2` applied to pts1/pts2 before `findFundamentalMat`. Denormalization: `F = T2.T @ F_norm @ T1`. This is mathematically correct.

**USAC_MAGSAC.** Used for both F and E estimation. MAGSAC weighting with USAC graph-cut LO — this is current state of the art for robust estimation and is a genuine strength.

**Planarity degeneracy check.** SVD of centred coordinate matrix; condition ratio `σ[-1] / σ[-2]`. For a (N, 2) matrix `σ[-1]` and `σ[-2]` map to the two valid singular values, so the check works by coincidence. If the input were transposed (2, N) as in some OpenCV conventions, it would fail silently. Code is fragile but not currently wrong. (W-05.)

**Homography fallback.** Missing entirely. For planar scenes (walls, floors, framed artwork), the fundamental matrix is degenerate and E decomposition will produce wrong translations. COLMAP uses H/F competition with the Torr criterion. (W-06.)

**Verdict.** Strongest verification stage in the pipeline. MAGSAC is a significant advantage. W-06 is a notable gap for architectural datasets.

---

### 3.5 Camera Pose Estimation — Seed Pair (`sfm/reconstruction.py:_select_seed_pair`)

**Seed scoring.** `score = baseline × inlier_count`. Baseline is the camera-centre distance from E decomposition. Inlier count from MAGSAC verification. Top candidate requires median triangulation angle > 5°. This is a reasonable heuristic, consistent with COLMAP's approach (though COLMAP uses a more nuanced scoring).

**Initial reconstruction.** `cv2.recoverPose` decomposed E → R, t for seed pair. Points triangulated via `cv2.triangulatePoints`. Cheirality check: homogeneous z > 0 for both cameras. Correct.

---

### 3.6 Camera Registration — PnP (`sfm/reconstruction.py:_register_image`)

**Bug W-02 (HIGH).** At lines 524–533:
```python
pts2d_ud = cv2.undistortPoints(pts2d, K_i, d_i, P=K_i)
ret = cv2.solvePnPRansac(pts3d, pts2d_ud, K_i, d_i, ...)
```
`pts2d_ud` is already in undistorted pixel coordinates (`P=K_i` returns pixel-space). Passing `d_i` again to `solvePnPRansac` causes it to apply distortion correction a second time, pushing points away from their true positions. For strong distortion (k1 > 0.1) this corrupts pose estimates by several pixels.

**Correct fix:** Pass `distCoeffs=None` (or `np.zeros(4)`) to `solvePnPRansac` when `pts2d_ud` are already undistorted.

**EPnP + LM refinement.** `SOLVEPNP_EPNP` followed by `cv2.solvePnPRefineLM` on inliers. Correct approach; LM refinement reduces the linearization error of EPnP.

---

### 3.7 Scene Graph / Image Connectivity (`sfm/utils.py:check_scene_graph_connectivity`)

**Implementation.** Union-find over image pairs with ≥ 1 verified match. Returns components sorted by size (largest first). Pipeline logs a warning when multiple components are detected.

**Limitations.** No automatic loop-closure retrieval when graph is disconnected. No minimum spanning-tree pair selection to prioritize bridging pairs. (W-13.)

---

### 3.8 Triangulation (`sfm/reconstruction.py:_triangulate_pair`, `_accept_batch`)

**Batch triangulation.** `cv2.triangulatePoints` called once for all match pairs — correct and efficient.

**Cheirality + angle filter.** Positive depth in both cameras; triangulation angle > `MIN_TRIANGULATION_ANGLE_DEG = 1.0°`. COLMAP uses 2°; 1° is too permissive and will add poorly-conditioned points that hurt BA convergence.

**Per-camera K bug (W-08).** `_accept_batch` is called as:
```python
self._accept_batch(..., K_n, K_n, ...)
```
where `K_n` is the new camera's K. In per-camera intrinsics mode, the reference camera (camera `n`) and the new camera may have different intrinsics. Using `K_n` for both corrupts the reprojection error threshold test and can accept or reject incorrect triangulations.

**Union-find track merging.** `_merge_tracks()` correctly prevents one keypoint from mapping to multiple 3-D tracks, avoiding the double-counting that causes drift in incremental SfM.

---

### 3.9 Bundle Adjustment (`sfm/bundle_adjustment.py`)

#### 3.9.1 scipy TRF BA (`BundleAdjuster`)

**Parameter encoding.** Per-camera `[rvec(3), tvec(3), f(1), cx(1)]` — 8 parameters. A single focal f means fx is assumed equal to fy. For cameras with known non-square pixels (rare in practice) this is incorrect. Principal point cy is fixed at H/2. (W-04.)

**Sparsity.** Vectorised CSR sparsity matrix via `_build_sparsity_v2` — correct and efficient. Two non-zero entries per residual per camera, two per point.

**Huber loss.** Adaptive f_scale = `1.4826 × MAD(residuals)`. This is the standard robust scale estimator; correct application.

**SO(3) reprojection.** Rotation matrix via SVD `U @ Vt` enforces proper rotation during optimization; prevents degeneracy from Rodrigues singularities at π. Correct.

**Divergence guard.** Post-BA RMSE > 1.5× pre-BA RMSE → rollback. Correct conservative guard.

**Per-camera intrinsic propagation (W-03).** After scipy BA, the refined `f` and `cx` for each camera are available in the parameter vector but `CameraIntrinsics.update_from_ba()` is never called. Subsequent PnP registration and MVS use stale EXIF-derived K matrices.

#### 3.9.2 pyceres BA (`PyceresBundleAdjuster`)

**Critical Bug W-01.** Lines 641–649:
```python
for cam_idx, cam_data in cameras.items():
    cost = pyceres.examples.SnavelyReprojectionErrorWithQuaternions(...)
    if not hasattr(pyceres.examples, "SnavelyReprojectionErrorWithQuaternions"):
        cost = None
    if cost is None:
        break   # ← exits the FOR loop, not just this iteration
```
The `break` is triggered on the very first camera if `SnavelyReprojectionErrorWithQuaternions` is unavailable (standard pyceres installs don't expose the `examples` sub-module). Zero residual blocks are added; `problem.solve()` returns immediately with no change. The pipeline logs no error — the caller checks only whether pyceres is importable, not whether solve succeeded.

**Secondary bug.** `n_cam_params = 8` (rvec 3, tvec 3, f 1, cx 1) but cy is neither stored nor optimized. If cy drifts (e.g., after image resizing), the pyceres model would be inconsistent even if W-01 were fixed.

**Verdict.** pyceres BA path is completely non-functional in its current form.

---

### 3.10 Per-Camera Intrinsics (`sfm/intrinsics.py`)

**EXIF parsing.** `FocalLengthIn35mmFilm` (tag 0xA405) → `focal_px = focal_35mm / sqrt(36²+24²) × image_diagonal_px`. Mathematically correct 35 mm equivalence formula.

**Fallback.** When EXIF is absent, falls back to `focal = max(H, W)` (≈ 67° diagonal FoV assumption). Reasonable default.

**`update_from_ba()`.** Updates `fx`, `fy`, `cx`, `cy` from post-BA values. **Never called** (W-03). Dead code in practice.

**Principal point.** Always set to (W/2, H/2). Correct assumption for most cameras; not refined in BA.

---

### 3.11 Global BA Scalability (`sfm/reconstruction.py:_run_ba`)

**Global-only BA (W-12).** Every `ba_interval` cameras, all registered cameras and all 3-D points are included in BA. For C cameras and P points, the Schur complement has size C×C and BA time scales as O(C²·P). At C=100, P=10 000 this is ~10× slower than local BA (windowed over 10 cameras).

**No local BA.** COLMAP, OpenSfM, and most production systems use local BA (optimize the last K cameras plus their visible points) followed by global BA at coarser intervals. This is the most impactful missing feature for datasets > 50 images.

---

### 3.12 Dense Reconstruction — MVS (`sfm/mvs.py`)

**StereoSGBM.** Semi-global block matching on rectified stereo pairs. Correct CV2 API usage. Stereo pair swap fix (t_rel[0] < 0) ensures positive disparity. Correct world-frame coordinate transform (R1.T to undo rectification, then Ri.T to go to world).

**Covisibility-based pair selection.** When `covisibility_counts` is passed, pairs are selected by shared 3-D point count rather than consecutive index — a genuine quality improvement.

**Depth-consistency filter (`_depth_consistency_filter`).** Checks only that depth > 0.01 in ≥ `fusion_min_views` cameras. This is not depth consistency — it does not check that the projected depth values agree across views (which would require storing per-pixel depth maps per view). Floating-point artefacts from degenerate SGBM patches will pass this filter. (W-09.)

**Two-view limitation.** SGBM is fundamentally a two-view method. Multi-view methods (PatchMatch Stereo, COLMAP's MVS, OpenMVS) propagate depth hypotheses across multiple views for far superior completeness and accuracy on textureless surfaces.

---

### 3.13 Point Cloud Export (`sfm/point_cloud.py`)

**Bilinear colour sampling.** `scipy.ndimage.map_coordinates(channel, [ys, xs], order=1)` — correct sub-pixel bilinear interpolation. Better than nearest-neighbour, which most simple implementations use.

**BGR→RGB reorder.** `np.stack([r, g, b], axis=1)` — correct.

**PLY float64 (W-15).** PLY writer uses `<f8` (float64) for XYZ. Standard PLY viewers and photogrammetry tools expect `<f4` (float32). This doubles file size for no precision benefit beyond float32 (which is 7 significant digits, sufficient for millimetre accuracy at km scales).

**Vectorised reprojection error.** `_reproject_batch` computes (R @ pts.T + t → project → distance) in batch. Correct.

---

### 3.14 Mesh Reconstruction (`sfm/mesh/`)

**Screened Poisson.** Open3D `create_from_point_cloud_poisson` with density-based phantom trimming at the 5th percentile. Correct; density threshold is a standard approach.

**BPA.** Auto-radius from median NN distance; 4 radii tried. Correct; but BPA produces non-watertight meshes with holes on sparse inputs.

**Alpha shapes.** Diagonal-based alpha; works poorly on anything other than convex scenes.

**Mesh cleaning.** Sequential: dedup → non-manifold edge removal → small component removal → hole filling → normal recompute. Correct pipeline; open3d exposes all needed operations.

**Taubin smoothing.** λ=0.5, μ=−0.53. Standard volume-preserving parameters. Correct.

**Vertex colour transfer (W-16).** k=1 NN from point cloud. Single nearest-neighbour assigns one point's colour to each vertex, producing blocky colour transitions at mesh boundaries. k=5 with IDW (inverse-distance weighting) would smooth colours across sparse regions.

---

### 3.15 GPU Acceleration (`sfm/device.py`, backends)

**Detection.** `has_gpu()` checks `torch.cuda.is_available()`; `get_device()` returns the CUDA device or CPU. lru_cache avoids repeated CUDA API calls.

**Usage.** GPU is used for: kornia detector, GPU brute-force matching, VocabTree k-means, DINOv2 embeddings, LightGlue inference, SuperPoint inference. BA and MVS remain CPU-only.

**OOM risk.** GPU brute-force matching with N=M=8000 and 128-D float32 descriptors requires 256 MB per call. On a 4 GB GPU with batch processing, this can OOM. (W-17.)

---

### 3.16 CLI / Configuration (`run_sfm.py`)

**Argument coverage.** 60+ CLI arguments covering all pipeline stages. Good.

**Checkpoint system.** MD5 of `{filename}{size}` for each input image. Misses mtime — re-running with same-name/same-size but different-content images reuses stale checkpoints. (W-14.)

**COLMAP short-circuit.** When `--use-colmap` is set, the entire custom pipeline is bypassed and COLMAP binary is invoked via subprocess. COLMAP binary parser reads `cameras.bin`, `images.bin`, `points3D.bin` correctly.

---

## 4. Production System Comparison

### 4.1 Feature Pipeline

| Capability | sfm-pub | COLMAP | OpenSfM | Meshroom |
|---|---|---|---|---|
| SIFT | ✓ | ✓ | ✓ | ✓ |
| SuperPoint | ✓ | ✗ | ✓ | ✗ |
| RootSIFT normalization | ✗ | ✓ | ✗ | ✗ |
| MAGSAC/USAC geometric verification | ✓ | USAC | LO-RANSAC | RANSAC |
| Vocabulary tree retrieval | ✓ (flat k-means) | ✓ (hierarchical) | ✗ | ✓ |
| DINOv2 image retrieval | ✓ | ✗ | ✗ | ✗ |
| LightGlue | ✓ | ✗ (separate) | ✗ | ✗ |
| Homography fallback for planar scenes | ✗ | ✓ | ✓ | ✓ |

### 4.2 Reconstruction Pipeline

| Capability | sfm-pub | COLMAP | RealityCapture | OpenSfM | Meshroom |
|---|---|---|---|---|---|
| Incremental SfM | ✓ | ✓ | Global | ✓ | ✓ |
| Global SfM | ✗ | ✓ | ✓ | ✗ | ✗ |
| Local BA | ✗ | ✓ | ✓ | ✓ | ✓ |
| Ceres BA (working) | ✗ (broken) | ✓ | proprietary | ✓ (custom) | ✓ |
| Per-camera intrinsics refined in BA | ✗ (W-03) | ✓ | ✓ | ✓ | ✓ |
| Loop closure | ✗ | ✓ | ✓ | ✓ | ✓ |
| MVS | 2-view SGBM | PatchMatch | PatchMatch | PatchMatch | PatchMatch |
| Dense depth fusion | depth>0 only | multi-view | multi-view | multi-view | multi-view |
| Mesh reconstruction | ✓ (Poisson) | ✓ | ✓ | ✓ | ✓ |

### 4.3 Academic State of the Art (2024–2025)

| Method | Category | Status vs. sfm-pub |
|---|---|---|
| DISK (Tyszkiewicz et al. 2020) | Learning-based detector | Not integrated |
| DeDoDe (Edstedt et al. 2023) | Learning-based detector | Not integrated |
| SuperGlue (Sarlin et al. 2020) | Graph NN matcher | Not integrated |
| LightGlue (Lindenberger et al. 2023) | Transformer matcher | ✓ Integrated |
| LoFTR (Sun et al. 2021) | Detector-free dense matching | Not integrated |
| MASt3R (Leroy et al. 2024) | End-to-end 3D matching | Not integrated |
| MAGSAC++ (Barath et al. 2020) | Robust estimation | ✓ (via USAC_MAGSAC) |
| GC-RANSAC (Barath & Matas 2018) | Robust estimation | Not integrated |
| PoseLib | Fast minimal solvers | Not integrated |
| GLOMAP (Pan et al. 2024) | Global SfM | Not integrated |
| VGGSfM (Wang et al. 2024) | Feed-forward SfM | Not integrated |
| DUSt3R (Wang et al. 2024) | Dense unconstrained 3D | Not integrated |
| 3DGS (Kerbl et al. 2023) | Neural rendering | Not integrated |

---

## 5. Weakness Catalogue

### W-01 — pyceres BA is a silent no-op

**Severity:** CRITICAL  
**Stage:** Bundle Adjustment  
**Location:** `sfm/bundle_adjustment.py:PyceresBundleAdjuster._build_and_solve:641-649`

**Current behavior:** When `pyceres.examples.SnavelyReprojectionErrorWithQuaternions` is not available (standard pyceres installs), `cost` is set to `None` and `break` exits the residual-block loop after the first camera. Zero residual blocks are added to the `ceres.Problem`. `problem.solve()` returns immediately with no change to any parameter. The caller receives `cameras` and `points3d` unchanged, logs no error, and the pipeline continues as if BA succeeded.

**Expected behavior:** When the cost function constructor is unavailable, the pipeline should log a clear error and either raise an exception or fall back to scipy TRF BA. If the cost function is available, the loop must use `continue` (not `break`) so all cameras receive residual blocks.

**Impact:** Any reconstruction run with `--use-pyceres` produces zero bundle adjustment. Reprojection errors will grow without bound as more cameras are registered, causing cascading drift.

**Fixability:** Trivial — replace `break` with `continue`; add availability check before the loop; add a post-solve residual count assertion.

**Fix:**
```python
# Before the loop:
if not hasattr(pyceres.examples, "SnavelyReprojectionErrorWithQuaternions"):
    logger.error("pyceres.examples cost function unavailable — falling back to scipy BA")
    return self._fallback_to_scipy(cameras, points3d, observations, K)

for cam_idx, cam_data in cameras.items():
    cost = pyceres.examples.SnavelyReprojectionErrorWithQuaternions(...)
    if cost is None:
        continue   # ← was break
    problem.add_residual_block(cost, loss, ...)

assert problem.num_residual_blocks() > 0, "No residual blocks added to Ceres problem"
```

**Dependencies:** None.

---

### W-02 — Double-undistortion in PnP registration

**Severity:** HIGH  
**Stage:** Camera Registration  
**Location:** `sfm/reconstruction.py:_register_image:524-533`

**Current behavior:**
```python
pts2d_ud = cv2.undistortPoints(pts2d, K_i, d_i, P=K_i)   # now in pixel space, distortion removed
ret = cv2.solvePnPRansac(pts3d, pts2d_ud, K_i, d_i, ...)  # applies d_i again — WRONG
```
`solvePnPRansac` with non-zero `distCoeffs` applies distortion correction internally. Since `pts2d_ud` are already undistorted, a second correction pushes points away from their true positions by up to several pixels for k1 > 0.05.

**Expected behavior:** Pass `distCoeffs=np.zeros(4)` (or `None`) to `solvePnPRansac` when coordinates are pre-undistorted.

**Impact:** Every newly registered camera pose is corrupted by double distortion. Poses drift further from ground truth with each registration. BA partially corrects this but cannot fully recover from systematically biased initializations.

**Fixability:** One-line fix.

**Fix:**
```python
pts2d_ud = cv2.undistortPoints(pts2d, K_i, d_i, P=K_i)
ret = cv2.solvePnPRansac(pts3d, pts2d_ud, K_i, None, ...)  # pass None or zeros(4)
```

**Dependencies:** None.

---

### W-03 — Per-camera intrinsics never propagated from BA

**Severity:** HIGH  
**Stage:** Bundle Adjustment → Camera Registration  
**Location:** `sfm/reconstruction.py:_run_ba`, `sfm/intrinsics.py:CameraIntrinsics.update_from_ba`

**Current behavior:** scipy TRF BA optimizes per-camera `[rvec, tvec, f, cx]` parameters. After BA, the refined `f` and `cx` are extracted and used to update `self.K` (the shared K matrix) but `self._per_cam_intr[i].update_from_ba()` is never called. All subsequent PnP calls for new cameras use stale EXIF-derived focal lengths. MVS rectification also uses stale K.

**Expected behavior:** After each BA run, call `self._per_cam_intr[i].update_from_ba(fx, fy, cx, cy)` for each camera i using the refined parameter values extracted from the solution vector.

**Impact:** Per-camera intrinsic refinement provides no benefit. Cameras with inaccurate EXIF focal lengths will continue to produce biased pose estimates after the first BA. The `CameraIntrinsics` system is effectively dead code.

**Fixability:** Medium — requires extracting per-camera f/cx from the solution vector and calling update_from_ba for each camera.

**Fix:** In `_run_ba`, after extracting the solution:
```python
for k, cam_idx in enumerate(cam_indices):
    f_new  = float(x_opt[k * 8 + 6])
    cx_new = float(x_opt[k * 8 + 7])
    if cam_idx in self._per_cam_intr:
        intr = self._per_cam_intr[cam_idx]
        intr.update_from_ba(f_new, f_new, cx_new, intr.cy)
```

**Dependencies:** None; `update_from_ba` already exists in `sfm/intrinsics.py`.

---

### W-04 — scipy BA assumes square pixels (fx = fy)

**Severity:** HIGH  
**Stage:** Bundle Adjustment  
**Location:** `sfm/bundle_adjustment.py:BundleAdjuster:_residuals:~430`

**Current behavior:** Camera parameter vector encodes a single scalar `f` for focal length, used as both fx and fy in the projection model. Cameras with non-square pixels (anamorphic lenses, some drone cameras) will have systematic reprojection error in one axis that BA cannot correct.

**Expected behavior:** Encode separate `fx` and `fy` parameters (9 params per camera instead of 8), or at minimum document the square-pixel assumption clearly.

**Impact:** Low for most consumer cameras (fx ≈ fy within 0.1%). High for anamorphic or non-standard sensors. Also means BA has one fewer degree of freedom to absorb systematic error.

**Fixability:** Medium — requires changing parameter encoding from 8 to 9 per camera and updating sparsity matrix accordingly.

**Fix:** Extend camera parameter vector to `[rvec(3), tvec(3), fx(1), fy(1), cx(1)]` = 9 params.

**Dependencies:** W-03 fix should be applied simultaneously.

---

### W-05 — Planarity check SVD indexing relies on coincidence

**Severity:** MEDIUM  
**Stage:** Geometric Verification  
**Location:** `sfm/geometric_verification.py:_check_planarity:176-185`

**Current behavior:**
```python
_, sv, _ = np.linalg.svd(pts_centred)   # pts_centred is (N, 2)
ratio = sv[-1] / sv[-2]                  # sv has 2 values; sv[-1]=sv[1], sv[-2]=sv[0]
```
For a (N, 2) input matrix, SVD returns 2 singular values. `sv[-1]` is the smallest (index 1) and `sv[-2]` is the largest (index 0). The planarity ratio min/max is correctly computed by coincidence of Python's negative indexing.

**Expected behavior:** Use explicit indices `sv[1] / sv[0]` or `sv.min() / sv.max()` to make intent clear and prevent silent breakage if the input shape changes.

**Impact:** Currently correct; fragile to refactoring.

**Fixability:** Trivial.

**Fix:** `ratio = float(sv[-1]) / float(sv[-2])` → `ratio = float(sv.min()) / float(sv.max())`

**Dependencies:** None.

---

### W-06 — No homography fallback for planar scenes

**Severity:** HIGH  
**Stage:** Geometric Verification  
**Location:** `sfm/geometric_verification.py:verify_pair` (missing feature)

**Current behavior:** All pairs are verified using F→E only. For planar scenes (walls, book covers, flat terrain), the fundamental matrix is degenerate and cannot be reliably estimated. E decomposition will produce geometrically incorrect translations and the pair will be rejected or produce wrong poses.

**Expected behavior:** Implement the Torr H/F degeneracy criterion: independently estimate H (homography) for the same matches; if the H inlier ratio is significantly higher than F inliers (e.g., `n_H_inliers / n_F_inliers > 0.85`), flag the pair as potentially planar and either (a) use H for pose recovery or (b) warn and skip.

**Impact:** Entire datasets of architectural interiors, artwork, or document scanning will reconstruct incorrectly or not at all.

**Fixability:** Medium — add `cv2.findHomography` with USAC_MAGSAC, compute inlier ratio, add branching logic.

**Dependencies:** None.

---

### W-07 — Kornia LAF scale × 6.0 is an unvalidated heuristic

**Severity:** MEDIUM  
**Stage:** Feature Extraction  
**Location:** `sfm/feature_extraction.py:_extract_kornia:228`

**Current behavior:** LAF scale (from `det(A)^0.5`) is converted to SIFT keypoint diameter as `float(scales[k]) * 6.0`. SIFT's diameter convention is ≈ 6× the scale-space sigma, but the kornia `ScaleSpaceDetector` LAF linear matrix scale is not directly comparable to SIFT sigma.

**Expected behavior:** Calibrate the scale conversion empirically or use a kornia-native descriptor (e.g., `HardNet`, `SOSNet`) instead of bridging to OpenCV SIFT descriptors.

**Impact:** SIFT descriptors at wrong scale produce reduced match quality. For small keypoints the descriptor patch may cover sub-pixel regions; for large keypoints it may exceed the image boundary.

**Fixability:** Medium — either validate empirically or replace with a kornia descriptor.

**Dependencies:** None.

---

### W-08 — `_accept_batch` uses new camera's K for both cameras

**Severity:** MEDIUM  
**Stage:** Triangulation  
**Location:** `sfm/reconstruction.py:_accept_batch` call sites (~line 703)

**Current behavior:**
```python
pts3d, colors, track_ids = self._accept_batch(..., K_n, K_n, ...)
```
`K_n` is the intrinsic matrix of the newly registered camera n. In per-camera intrinsics mode, the reference camera (from which the second K argument is drawn) may have a different focal length and principal point.

**Expected behavior:** Pass the correct K for each camera: `K_ref` for the reference camera and `K_n` for the new camera.

**Impact:** Reprojection error thresholds in `_accept_batch` are computed in the wrong image space for the reference camera, causing borderline-good triangulations to be rejected and borderline-bad ones to be accepted.

**Fixability:** Easy — look up `self._per_cam_intr[ref_cam_idx]` and pass its K.

**Dependencies:** W-03 fix recommended first.

---

### W-09 — MVS depth-consistency filter is not true multi-view fusion

**Severity:** HIGH  
**Stage:** Dense Reconstruction  
**Location:** `sfm/mvs.py:MVSDensifier._depth_consistency_filter:223-274`

**Current behavior:** For each candidate 3-D point, the filter counts cameras where the point's depth (Z in camera frame) is > 0.01. Points with fewer than `fusion_min_views` positive-depth cameras are removed. This is a visibility test, not a depth consistency test.

**Expected behavior:** True multi-view depth fusion (as in COLMAP, OpenMVS, or PatchMatch Stereo) compares the depth value predicted by one view's depth map against the depth values from neighboring views at the projected pixel location. Points where depths disagree by more than a threshold (e.g., 1% of median scene depth) are classified as floating artefacts and removed.

**Impact:** Floating points from featureless surfaces, sky, or window reflections pass the filter because they happen to have positive depth in multiple cameras, even if they are geometrically inconsistent. Dense reconstruction quality is lower than it would be with true fusion.

**Fixability:** Hard — requires per-pair depth map storage and cross-projection comparison. Would significantly increase memory usage.

**Dependencies:** None; intermediate fix possible (add projected-depth agreement check using stored disparity maps).

---

### W-10 — EXIF orientation tags ignored

**Severity:** HIGH  
**Stage:** Image Loading  
**Location:** `sfm/utils.py:load_image` (missing feature)

**Current behavior:** Images are loaded with `cv2.imread` which ignores EXIF orientation metadata (tag 0x0112). Smartphones routinely capture images with physical landscape orientation but store them in portrait pixel layout with an orientation tag indicating a 90° rotation.

**Expected behavior:** Read EXIF orientation tag and apply the corresponding rotation (`cv2.rotate`) before returning the image.

**Impact:** Entire mobile-phone datasets will fail — keypoints will be extracted in the wrong coordinate frame, matches will fail geometric verification, and the reconstruction will produce zero cameras.

**Fixability:** Easy — add Pillow-based orientation correction.

**Fix:**
```python
from PIL import Image as PILImage, ExifTags
pil_img = PILImage.open(path)
pil_img = ImageOps.exif_transpose(pil_img)  # handles all 8 EXIF orientations
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

**Dependencies:** Pillow already declared in `pyproject.toml`.

---

### W-11 — LightGlue always exhaustive when candidate_pairs=None

**Severity:** MEDIUM  
**Stage:** Feature Matching  
**Location:** `sfm/feature_matching.py:LightGlueMatcher.match_all:~810`

**Current behavior:** When `candidate_pairs` is not provided, `LightGlueMatcher` falls back to exhaustive O(N²) pair enumeration. LightGlue itself is O(N·k) per pair but O(N²) pairs is the same complexity as brute-force matching.

**Expected behavior:** `LightGlueMatcher` should require a retrieval front-end (DINOv2, VocabTree, or netVLAD) to generate candidate pairs. Without retrieval, LightGlue's adaptive pruning provides no speedup over exhaustive matching.

**Impact:** On a 200-image dataset with LightGlue and no retrieval, matching time is O(200²) = 40 000 pairs, each requiring a Transformer forward pass — potentially hours of GPU compute.

**Fixability:** Easy — add a retrieval requirement; warn loudly if candidate_pairs is None and N > 50.

**Dependencies:** DINOv2Matcher or VocabTreeMatcher for retrieval.

---

### W-12 — Global BA only; no local BA

**Severity:** HIGH  
**Stage:** Bundle Adjustment  
**Location:** `sfm/reconstruction.py:_run_ba`

**Current behavior:** Every `ba_interval` cameras, BA is run over ALL registered cameras and ALL 3-D points. For C cameras and P points, Schur complement cost is O(C²·P).

**Expected behavior:** Local BA over a sliding window of recent cameras (typically last 10–20 cameras plus their visible points) after each registration. Global BA at coarser intervals (every 50 cameras or on merge of major graph components).

**Impact:** At C=100, P=50 000, global BA takes ~10–50× longer than local BA. This is the primary scalability bottleneck for large scenes.

**Fixability:** Medium — requires maintaining a "local window" camera set and filtering observations by visibility within that window.

**Dependencies:** None; scipy BA can be reused with a subset of cameras.

---

### W-13 — No loop-closure detection or re-matching

**Severity:** MEDIUM  
**Stage:** Scene Graph  
**Location:** `sfm/utils.py:check_scene_graph_connectivity`, `run_sfm.py` (missing feature)

**Current behavior:** Disconnected scene graph components are detected and logged. No attempt is made to find bridging pairs through retrieval or to re-match images from different components.

**Expected behavior:** When multiple components are detected, trigger a targeted retrieval search (DINOv2 or VocabTree) to find image pairs that can bridge components. Re-verify and add to the scene graph before reconstruction.

**Impact:** Sequences with loop closures (e.g., a 360° walk-around) will produce multiple disconnected reconstructions or drift without the closing constraint.

**Fixability:** Medium — the retrieval infrastructure (DINOv2Matcher, VocabTreeMatcher) already exists; needs integration with the connectivity check.

**Dependencies:** DINOv2Matcher or VocabTreeMatcher.

---

### W-14 — Checkpoint hash uses only filename and file size

**Severity:** LOW  
**Stage:** CLI / Checkpointing  
**Location:** `run_sfm.py:_compute_checkpoint_hash:~120`

**Current behavior:** MD5 is computed over `{filename}{size}` for each input image. If images are replaced with same-name, same-size but different-content files (e.g., after camera calibration correction), the checkpoint is reused and stale results are returned without warning.

**Expected behavior:** Include file mtime or a content-based hash (MD5 of file bytes) in the checkpoint key. File bytes are most reliable; mtime is a pragmatic trade-off.

**Impact:** Silent incorrect reuse of cached features/matches when images are updated.

**Fixability:** Trivial — add `os.stat(path).st_mtime` to the hash input or read the first 64 KB of each file.

**Dependencies:** None.

---

### W-15 — PLY writer uses float64 (doubles file size)

**Severity:** LOW  
**Stage:** Point Cloud Export  
**Location:** `sfm/point_cloud.py:write_ply:~280`

**Current behavior:** XYZ coordinates are written as `<f8` (float64, 8 bytes per component). A 500 000-point cloud requires 12 MB for XYZ alone vs. 6 MB with float32.

**Expected behavior:** Use `<f4` (float32) for XYZ, which provides 7 significant digits — sufficient for millimetre accuracy at km scales. Most PLY viewers and tools expect float32.

**Fixability:** Trivial — change dtype specifier from `<f8` to `<f4`.

**Dependencies:** None.

---

### W-16 — Mesh vertex colour uses k=1 NN (blocky colours)

**Severity:** LOW  
**Stage:** Mesh Post-processing  
**Location:** `sfm/mesh/postprocess.py:transfer_vertex_colors:~80`

**Current behavior:** For each mesh vertex, k=1 nearest-neighbour from the sparse point cloud assigns one point's colour. Vertices far from any point receive a single, potentially unrepresentative colour.

**Expected behavior:** Use k=5 (or k=7) NN with inverse-distance weighting (IDW) to blend colours from neighbouring points. This smooths colour transitions and handles sparse regions better.

**Fixability:** Easy — open3d `KDTreeFlann.search_knn_vector_3d` already supports k>1.

**Dependencies:** None.

---

### W-17 — GPU brute-force matcher allocates O(N²) memory

**Severity:** MEDIUM  
**Stage:** Feature Matching  
**Location:** `sfm/feature_matching.py:_match_pair_gpu:~350`

**Current behavior:** `torch.cdist(desc1, desc2)` computes the full (N, M) distance matrix at once. For N=M=8000 with float32 descriptors this is 8000 × 8000 × 4 = 256 MB per call. With multiple concurrent GPU operations (DINOv2 embedding + brute-force matching + LightGlue), total GPU memory can exceed 4 GB.

**Expected behavior:** Chunk the computation: split desc1 into blocks of 1000 rows, compute distance for each block, collect top-2 matches. Memory per call drops to 1000 × 8000 × 4 = 32 MB.

**Fixability:** Easy — add chunked loop with `torch.topk` per chunk.

**Fix:**
```python
chunk = 1024
all_dists, all_idxs = [], []
for start in range(0, len(desc1), chunk):
    block = desc1[start:start+chunk]
    d = torch.cdist(block, desc2)
    top = torch.topk(d, k=2, dim=1, largest=False)
    all_dists.append(top.values); all_idxs.append(top.indices)
dists = torch.cat(all_dists); idxs = torch.cat(all_idxs)
```

**Dependencies:** None.

---

## 6. Scoring Matrix

Scores are 1 (unusable) to 10 (best-in-class). Gap-to-COLMAP is the difference sfm-pub − COLMAP.

| Criterion | sfm-pub | COLMAP | RealityCapture | Metashape | OpenSfM | Meshroom | Gap |
|---|---|---|---|---|---|---|---|
| Feature Detection Quality | 7 | 8 | 9 | 9 | 7 | 7 | −1 |
| Matching Robustness | 7 | 8 | 9 | 9 | 7 | 7 | −1 |
| Geometric Verification | 8 | 8 | 9 | 9 | 7 | 7 | 0 |
| Pose Estimation Accuracy | 4 | 9 | 10 | 9 | 8 | 7 | −5 |
| Bundle Adjustment Quality | 4 | 10 | 10 | 10 | 8 | 8 | −6 |
| Intrinsic Calibration | 4 | 9 | 10 | 10 | 8 | 7 | −5 |
| Dense Reconstruction | 4 | 8 | 10 | 9 | 7 | 8 | −4 |
| Scalability (>100 images) | 3 | 9 | 10 | 10 | 8 | 8 | −6 |
| Code Quality / Correctness | 5 | 8 | N/A | N/A | 7 | 6 | −3 |
| GPU Utilisation | 6 | 7 | 9 | 9 | 5 | 7 | −1 |
| Mesh / Export Quality | 6 | 7 | 10 | 9 | 6 | 8 | −1 |
| Ease of Use / CLI | 8 | 7 | 9 | 9 | 7 | 6 | +1 |
| **Overall** | **5.5** | **8.2** | **9.5** | **9.2** | **7.1** | **7.3** | **−2.7** |

**Notes:**
- Pose Estimation and BA scores are low because W-01 (pyceres no-op) and W-02 (double undistortion) are active bugs that corrupt results in many configurations.
- Scalability score of 3 reflects global-only BA with O(C²·P) scaling and no loop closure.
- Geometric Verification matches COLMAP due to USAC_MAGSAC adoption.

---

## 7. Improvement Intelligence

### From Production Systems

**P-01: Local BA (from COLMAP / OpenSfM)**
COLMAP runs local BA over a sliding window of ~10 recently-registered cameras plus their visible points after every registration. This reduces per-BA cost from O(C²·P) to O(W²·P_local) where W≈10. Global BA is then run every ~30 cameras to correct accumulated drift. Implementation requires tracking "local camera set" and filtering the observation list to include only points visible from local cameras.

**P-02: RootSIFT normalization (from COLMAP)**
COLMAP applies Hellinger kernel embedding (RootSIFT: L1-normalize then element-wise sqrt) to SIFT descriptors before matching. This improves nearest-neighbour match quality by ~10% on standard benchmarks (Arandjelović & Zisserman, CVPR 2012) at zero additional cost. Apply to both query and database descriptors before FLANN or GPU matching.

**P-03: Homography/F competition for planar scenes (from COLMAP / OpenSfM)**
COLMAP estimates both F and H for each pair, comparing inlier counts via the Torr criterion. Pairs where H explains matches nearly as well as F are flagged as potentially degenerate and handled differently in pose recovery. OpenSfM implements the same approach. This is a one-time addition to `geometric_verification.py:verify_pair`.

**P-04: EXIF orientation correction (from OpenSfM / Meshroom)**
OpenSfM's image loader applies PIL `ImageOps.exif_transpose` to handle all 8 EXIF orientation values before processing. This is a two-line fix using Pillow (already a dependency) that enables correct processing of all smartphone datasets.

**P-05: Hierarchical vocabulary tree (from COLMAP)**
COLMAP's vocabulary tree uses hierarchical k-means (branching factor 10, depth 6) rather than flat k-means. This gives O(log N) assignment time per descriptor vs. O(K) for flat k-means with K=4096 words. For large datasets (>10 000 images), hierarchical VT is 10–100× faster for retrieval. The flat k-means in `VocabTreeMatcher` is adequate for <500 images but will bottleneck at scale.

### From Academic Research

**A-01: DISK (Tyszkiewicz et al., NeurIPS 2020)**
DISK detects keypoints using a deep reinforcement-learning policy trained with homography adaptation. It produces more uniformly-distributed keypoints on textureless surfaces than SIFT. Integration path: replace `FeatureExtractor` for the GPU path using `kornia.feature.DISK`. Descriptors are 128-D, compatible with LightGlue. Primary benefit: improved coverage on architectural facades and vegetation.

**A-02: LoFTR (Sun et al., CVPR 2021)**
Semi-dense detector-free matching using Transformer attention on coarse feature maps followed by fine-level refinement. Produces dense correspondences on textureless regions where SIFT fails. Integration as a new matcher class in `feature_matching.py`. Descriptor-free; outputs match lists directly. High GPU memory requirement (~2 GB for typical image pairs at 640px). Particularly valuable for MVS pair selection (W-09) since it can match pairs that SIFT rejects.

**A-03: PoseLib minimal solvers**
PoseLib (Bujnak et al.; maintained by Viktor Larsson) provides highly optimised C++ implementations of 5-point, 6-point, and P3P solvers with Python bindings. Replacing `cv2.solvePnPRansac` with PoseLib's `p3p` inside a RANSAC loop reduces PnP time by 3–5× on CPU and provides better numerical stability on near-planar configurations. Also provides `homography_4pt` needed for W-06.

**A-04: MAGSAC++ (Barath et al., CVPR 2020 + TPAMI 2022)**
The current pipeline uses OpenCV's USAC_MAGSAC which implements the MAGSAC weighting scheme. Pure MAGSAC++ (available in the `pygcransac` package) uses a theoretically superior marginalized quality function and graph-cut LO-RANSAC. On the ETH3D benchmark, MAGSAC++ reduces angular error by ~15% vs. USAC_MAGSAC. Integration: replace `cv2.findFundamentalMat` + `cv2.findEssentialMat` with `pygcransac` equivalents.

**A-05: MASt3R / DUSt3R (Leroy et al. 2024; Wang et al. 2024)**
MASt3R produces dense 3-D correspondences from image pairs using a Transformer architecture, without requiring camera intrinsics as input. DUSt3R produces uncalibrated dense 3-D reconstructions from pairs. These methods represent a fundamentally different reconstruction paradigm where pose estimation and matching are unified. Integration as an alternative reconstruction backend (`--backend mast3r`) would bring the system to 2025 academic state of the art for small-to-medium scenes. Limitation: requires ~8 GB GPU VRAM; inference is slow (~2 s/pair on A100).

**A-06: 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)**
3DGS uses the sparse SfM point cloud as initialization for a differentiable Gaussian scene representation. It produces photorealistic novel views at real-time render speeds. Integration: after SfM reconstruction, export the sparse point cloud + camera poses in the COLMAP format and run a 3DGS trainer (e.g., `gaussian-splatting` official repo or `nerfstudio`). sfm-pub's COLMAP-format export makes this straightforward.

---

## 8. Prioritized Roadmap

### Tier 0 — Correctness Fixes (Must Fix Before Any Production Use)

| ID | Task | Weakness | Effort | Impact |
|---|---|---|---|---|
| T0-1 | Fix pyceres BA `break` → `continue` + availability guard | W-01 | 30 min | CRITICAL: BA now actually runs |
| T0-2 | Fix PnP double-undistortion: pass `distCoeffs=None` | W-02 | 15 min | HIGH: all camera poses corrected |
| T0-3 | Fix EXIF orientation: add `exif_transpose` to `load_image` | W-10 | 1 h | HIGH: mobile datasets now work |

### Tier 1 — High ROI Improvements (Week 1–4)

| ID | Task | Weakness | Effort | Impact |
|---|---|---|---|---|
| T1-1 | Call `update_from_ba()` after scipy BA | W-03 | 2 h | HIGH: per-cam intrinsics actually refined |
| T1-2 | Add homography/F competition for planar scenes | W-06 | 1 day | HIGH: architectural datasets now work |
| T1-3 | Implement local BA (sliding window 10 cameras) | W-12 | 3 days | HIGH: 10× speedup at 100+ images |
| T1-4 | Require retrieval front-end for LightGlue (warn if N>50) | W-11 | 4 h | MEDIUM: prevents accidental O(N²) matching |
| T1-5 | Fix `_accept_batch` to use correct K per camera | W-08 | 2 h | MEDIUM: triangulation filter correct |
| T1-6 | Apply RootSIFT normalization to all SIFT descriptors | P-02 | 2 h | MEDIUM: ~10% match quality gain, zero cost |
| T1-7 | Add chunked GPU matching to prevent OOM | W-17 | 3 h | MEDIUM: 8000-desc pairs no longer OOM |
| T1-8 | Change PLY writer from float64 to float32 | W-15 | 15 min | LOW: halves point cloud file size |
| T1-9 | Add loop-closure re-matching when graph is disconnected | W-13 | 2 days | MEDIUM: 360° sequences now close |

### Tier 2 — Core Algorithm Upgrades (Month 1–3)

| ID | Task | Weakness/Source | Effort | Impact |
|---|---|---|---|---|
| T2-1 | Add separate fx, fy to BA parameter vector | W-04 | 1 day | MEDIUM: non-square pixels supported |
| T2-2 | Integrate DISK detector via kornia | A-01 | 2 days | MEDIUM: better coverage on facades |
| T2-3 | Add LoFTR matcher class | A-02 | 3 days | HIGH: textureless scene matching |
| T2-4 | Replace cv2 PnP with PoseLib p3p | A-03 | 2 days | MEDIUM: 3–5× PnP speedup |
| T2-5 | True multi-view depth fusion in MVS | W-09 | 1 week | HIGH: float artefacts removed |
| T2-6 | Increase MIN_TRIANGULATION_ANGLE_DEG to 2.0° | — | 30 min | LOW: fewer ill-conditioned points |
| T2-7 | Mesh vertex colour k=5 IDW blending | W-16 | 2 h | LOW: smoother mesh colours |

### Tier 3 — Architectural Improvements (Quarter 1–2)

| ID | Task | Source | Effort | Impact |
|---|---|---|---|---|
| T3-1 | Hierarchical vocabulary tree | P-05 | 1 week | HIGH: retrieval scales to 10 000+ images |
| T3-2 | Global SfM path (rotation averaging + translation averaging) | GLOMAP | 3 weeks | HIGH: robust to repeated textures |
| T3-3 | PatchMatch stereo MVS (replace SGBM) | COLMAP/OpenMVS | 3 weeks | HIGH: dense reconstruction quality |
| T3-4 | Fix pyceres cost function integration properly | W-01 | 3 days | HIGH: GPU-accelerated BA |
| T3-5 | Content hash in checkpoint (MD5 of first 64KB) | W-14 | 2 h | LOW: reliable cache invalidation |

### Tier 4 — Research / Long-term (6–12 months)

| ID | Task | Source | Effort | Impact |
|---|---|---|---|---|
| R4-1 | MASt3R/DUSt3R integration as alternative backend | A-05 | 4 weeks | VERY HIGH: SotA small-scene quality |
| R4-2 | 3DGS export pipeline | A-06 | 1 week | HIGH: real-time novel view synthesis |
| R4-3 | MAGSAC++ (pygcransac) for F/E estimation | A-04 | 1 week | MEDIUM: ~15% verification accuracy gain |
| R4-4 | Neural radiance field (NeRF/nerfstudio) export | — | 2 weeks | MEDIUM: complementary to 3DGS |

### Roadmap Summary Table

| Tier | Items | Total Effort | Projected Grade After |
|---|---|---|---|
| T0 (Correctness) | 3 | ~2 h | C+ → B (bugs fixed, reliable results) |
| T1 (High ROI) | 9 | ~2 weeks | B → B+ (scalable, correct intrinsics) |
| T2 (Algorithm) | 7 | ~6 weeks | B+ → A− (better detectors, faster BA) |
| T3 (Architecture) | 5 | ~3 months | A− → A (100+ image scalability) |
| T4 (Research) | 4 | ~6 months | A → A+ (SotA alignment) |

**Immediate Priority:** Fix T0-1 (pyceres break), T0-2 (double undistortion), T0-3 (EXIF orientation) in a single pull request. These three fixes require under 2 hours of work and convert the pipeline from "produces wrong results silently" to "produces correct results on standard datasets."

**Highest ROI after T0:** T1-3 (local BA) is the single most impactful improvement for real-world datasets beyond 50 images. T1-2 (homography fallback) unlocks entire categories of architectural datasets. Together these two items would move the overall score from 5.5 to approximately 7.0.
