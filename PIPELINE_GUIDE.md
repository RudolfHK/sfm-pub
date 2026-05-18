# Pipeline Guide

Structure from Motion pipeline — conceptual reference and full CLI documentation.

---

## Chapter 1 — Overview

Structure from Motion (SfM) is the process of recovering a three-dimensional scene and the camera poses that observed it from a set of unordered or ordered 2D photographs. This pipeline implements incremental SfM in pure Python using OpenCV and SciPy, with an optional COLMAP backend for larger or more demanding datasets. Starting from a directory of images, it detects and matches image features, verifies geometric consistency between pairs, grows a sparse reconstruction one camera at a time, and refines everything jointly via bundle adjustment before writing a colored point cloud. Optional stages add MVS densification, mesh reconstruction, and a visualization suite.

### The Six Core Stages

```
  Images on disk
       |
       v
  [1] Intrinsics estimation   -- EXIF focal or heuristic f=max(W,H)
       |
       v
  [2] Feature extraction      -- SIFT keypoints + 128-D descriptors
       |
       v
  [3] Feature matching        -- exhaustive / sequential / vocab-tree
       |
       v
  [4] Geometric verification  -- Hartley-normalized F, E, pose recovery
       |
       v
  [5] Incremental reconstruction -- seed pair, PnP registration, DLT triangulation
       |
       v
  [6] Bundle adjustment + export -- sparse TRF, Huber loss, binary PLY
       |
       v
  output.ply  (sparse colored point cloud — always produced)
```

### Optional Stages

- **MVS densification** (`--dense`): StereoSGBM on covisibility-selected stereo pairs, producing a dense point cloud alongside the sparse one.
- **Mesh reconstruction** (`--mesh`): Screened Poisson, Ball-Pivoting, or Alpha-shapes surface reconstruction via open3d, with cleaning, smoothing, decimation, and texture projection.
- **Visualization** (`--visualize`): matplotlib figures for features, matches, the scene graph, reprojection error, and camera trajectories; optional GIF exports and an interactive open3d viewer.

### Architecture Diagram

```
run_sfm.py (entry point / argument parser)
     |
     +-- sfm/utils.py          list_images, estimate_intrinsics, reprojection_error
     +-- sfm/feature_extraction.py   FeatureExtractor (SIFT / kornia / CUDA SURF)
     +-- sfm/feature_matching.py     FeatureMatcher / SequentialMatcher / VocabTreeMatcher
     +-- sfm/geometric_verification.py  GeometricVerifier (F + E + recoverPose)
     +-- sfm/reconstruction.py      IncrementalSfM (seed, PnP, triangulation)
     |     +-- sfm/bundle_adjustment.py  BundleAdjuster (scipy TRF, Huber, sparse J)
     +-- sfm/point_cloud.py         PointCloudExporter (outlier filter, colorize, PLY)
     +-- sfm/mvs.py                 MVSDensifier (SGBM stereo pairs)
     +-- sfm/mesh/pipeline.py       MeshPipeline
     |     +-- sfm/mesh/pointcloud_prep.py
     |     +-- sfm/mesh/reconstruction.py
     |     +-- sfm/mesh/cleaning.py
     |     +-- sfm/mesh/postprocess.py
     |     +-- sfm/mesh/export.py
     +-- sfm/visualizer.py          SfMVisualizer (matplotlib + open3d)
     +-- sfm/colmap_backend.py      ColmapRunner (wraps COLMAP binary)
```

---

## Chapter 2 — The Pipeline Stages

### Stage 1 — Intrinsics Estimation

Before any feature work begins, the pipeline needs an initial camera intrinsic matrix K. The process is:

1. Attempt to read the `FocalLengthIn35mmFilm` EXIF tag from the first image using Pillow.
2. If the tag is present, convert it to pixels using the ratio of the image diagonal to the 35 mm sensor diagonal (36 × 24 mm).
3. If EXIF data is absent or zero, fall back to the heuristic `f = max(W, H)`, which corresponds to roughly 53° diagonal field of view and is a reasonable prior for consumer cameras and telephoto shots alike.

The initial matrix is:

```
K = [[f,   0,   W/2],
     [0,   f,   H/2],
     [0,   0,   1  ]]
```

The principal point is initialized to the image center. Both focal length and principal point are refined later by bundle adjustment unless `--no_refine_intrinsics` or `--ba-fix-principal-point` are set.

**Key parameters:** none at this stage — K is derived automatically. Supply well-tagged JPEG files for best initialization; raw or stripped files fall back to the heuristic.

---

### Stage 2 — Feature Extraction

The extractor detects keypoints and computes 128-dimensional descriptors for every image. The backend is selected automatically at startup:

1. **kornia** (GPU preferred): uses a PyTorch `ScaleSpaceDetector` with `BlobDoG` response and `ConvQuadInterp3d` NMS to locate keypoints on the GPU, then computes SIFT descriptors at those positions using CPU OpenCV. This separates GPU-accelerated detection from CPU descriptor computation.
2. **OpenCV CUDA SURF**: falls back to `cv2.cuda.SURF_CUDA_create` if kornia is unavailable but an OpenCV CUDA build is present.
3. **OpenCV SIFT CPU**: the default when no GPU path succeeds. Uses `cv2.SIFT_create` with the parameters below.

All backends return the same contract: `keypoints` as (N, 2) float32 pixel coordinates and `descriptors` as (N, 128) float32 L2-normalized SIFT vectors.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--n_features` | 8000 | Maximum keypoints per image |
| `--sift-contrast-threshold` | 0.02 | Lower = more low-contrast keypoints |
| `--sift-edge-threshold` | 10.0 | Higher = keep more edge-like keypoints |
| `--sift-n-octave-layers` | 3 | Layers per octave in scale space |
| `--sift-sigma` | 1.6 | Gaussian sigma at octave 0 |

---

### Stage 3 — Feature Matching

Three matching strategies are available. All use Lowe's ratio test and cross-check filtering before handing pairs to geometric verification.

**Exhaustive** (`--match_strategy exhaustive`): Compares every image against every other image — O(N²) pairs. Appropriate for small datasets (under 50 images) where temporal ordering is unknown.

**Sequential** (`--match_strategy sequential`): Each image is matched only against the next `--sequential_window` images in sorted filename order. This reduces the pair count to O(N × W) and is suitable for video sequences or ordered photo walks.

**Vocabulary tree** (`--match_strategy vocab_tree`): A bag-of-words vocabulary is built from all descriptors using k-means (with `--vocab_words` cluster centers), and each image is assigned a TF-IDF weighted histogram over visual words. The `--vocab_top_k` most similar images per query are retrieved for matching. This gives sublinear pair selection for large unordered datasets.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--match_strategy` | exhaustive | Pair selection strategy |
| `--sequential_window` | 5 | Window size for sequential |
| `--vocab_words` | 4096 | Vocabulary size for vocab_tree |
| `--vocab_top_k` | 10 | Retrieved neighbors per image |
| `--ratio` | 0.75 | Lowe's ratio test threshold |
| `--min_matches` | 15 | Minimum raw matches to keep a pair |

---

### Stage 4 — Geometric Verification

Each matched pair is tested for geometric consistency. The verifier runs three sequential filters:

1. **Hartley normalization**: Raw pixel coordinates are translated to the centroid and scaled so the mean distance from the origin equals √2. This dramatically reduces the condition number of the DLT system inside `findFundamentalMat`, improving numerical stability on high-resolution images where pixel values span thousands of units.

2. **Fundamental matrix**: `cv2.findFundamentalMat` is run on the normalized coordinates using `USAC_MAGSAC` (more robust than `FM_RANSAC` and avoids crashes seen in some OpenCV 4.x builds on specific data distributions). The result is de-normalized back to pixel space via F = T2.T @ F_norm @ T1.

3. **Essential matrix and pose recovery**: `cv2.findEssentialMat` is run on the F-inlier subset using the calibrated camera matrix K. If this call fails, the analytic fallback E = K.T @ F @ K is used. `cv2.recoverPose` then selects the geometrically correct (R, t) from the four E decompositions using the cheirality constraint. Pairs with near-zero baseline are rejected.

After verification, the pipeline checks scene graph connectivity using union-find. If multiple disconnected components are found, only the largest is used for reconstruction.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--min_inliers` | 15 | Minimum RANSAC inliers to accept a pair |
| `--ransac_thr` | 1.0 | Inlier threshold in pixels |

---

### Stage 5 — Incremental Reconstruction

The reconstruction grows one camera at a time:

**Seed pair selection**: The pair maximizing `score = baseline × inlier_count` is chosen, subject to a minimum median triangulation angle of 5°. If no pair clears this angle threshold, the pair with the most inliers is used as a fallback.

**Initialization**: Camera 0 is placed at the world origin (R = I, t = 0). Camera 1 is set from the E decomposition of the seed pair. Inlier matches are triangulated in batch using `cv2.triangulatePoints` (DLT), and accepted points must pass: positive depth in both cameras, triangulation angle ≥ 1°, and reprojection error ≤ `--max_reproj_error` in both cameras.

**Registration loop**: At each step, the unregistered image with the most 2D-3D correspondences to the current reconstruction is selected. It is registered via `cv2.solvePnPRansac` with `SOLVEPNP_EPNP`, followed by Levenberg-Marquardt refinement (`cv2.solvePnPRefineLM`). New 3D points are triangulated with all covisible registered cameras. After every `--ba_interval` new cameras, bundle adjustment is run.

A camera is rejected if its position is more than 50× the median scene scale from the origin (degenerate PnP outlier).

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--max_reproj_error` | 4.0 | Max reprojection error for triangulation and PnP |
| `--ba_interval` | 5 | Run BA every N newly registered cameras |

---

### Stage 6 — Bundle Adjustment

Bundle adjustment jointly refines all camera poses and 3D point positions to minimize reprojection error. The implementation uses `scipy.optimize.least_squares` with the Trust Region Reflective (`trf`) method and a sparse Jacobian pattern.

**Parameter layout** (when `refine_intrinsics=True`, `fix_principal_point=False`):
```
[ f(1)  k1(1)  k2(1)  cx(1)  cy(1) | rvec_0(3) tvec_0(3) | ... | X_0(3) X_1(3) ... ]
```
Total length = 5 + 6C + 3P. Shared intrinsics (f, k1, k2, cx, cy) appear at the start and are touched by every residual, so their Jacobian columns are dense.

**Projection model**: Radial distortion (Brown-Conrady, 2 coefficients):
```
X_cam = R * X_world + t
xn = X_cam[0] / X_cam[2],  yn = X_cam[1] / X_cam[2]
r2 = xn^2 + yn^2
u  = f * xn * (1 + k1*r2 + k2*r2^2) + cx
v  = f * yn * (1 + k1*r2 + k2*r2^2) + cy
```

**Robust loss**: Huber loss with an adaptive scale derived from 1.4826 × MAD of the initial residuals (the MAD consistency factor for Gaussian noise). This adapts to actual reprojection noise rather than using a fixed pixel threshold. The scale is clamped to [0.5, 10.0] px.

**Bounds**: Focal length is bounded to [0.5f, 2f]. Distortion coefficients k1, k2 are bounded to [-2, 2]. Principal point cx, cy are bounded to within ±10% of image dimensions from their initial values.

**SO(3) re-orthogonalization**: After each optimization step, each rotation matrix is projected back onto SO(3) via SVD (U @ Vt, with sign correction for det < 0) to correct floating-point drift.

**Divergence guard**: If the final RMSE exceeds 1.5× the initial RMSE, the BA result is discarded and the previous estimates are kept.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--no_refine_intrinsics` | off | Disables f + k1 + k2 + cx + cy refinement |
| `--ba-fix-principal-point` | off | Keeps cx/cy fixed, optimizes only f + k1 + k2 |

---

### Stage 7 — Point Cloud Export

After the final bundle adjustment, outlier filtering and colorization are applied:

**Outlier filtering** (skipped with `--no_filter`): For each 3D point, the mean reprojection error across all observations is computed using vectorized per-camera batches. The keep threshold is `min(max_reproj_error, median + 3σ)`. Points above the threshold are discarded and the index map is remapped.

**Colorization**: For each 3D point, all observations with reprojection error below `max_reproj_error` contribute a color sample. Colors are looked up by integer pixel index from the source image (clipped to image bounds) and averaged in float64. Unobserved points receive mid-grey (128, 128, 128).

**PLY format**: Binary little-endian with double-precision XYZ and uint8 RGB per vertex. Compatible with MeshLab, CloudCompare, and open3d.

---

### Optional Stage 8 — MVS Densification

Activated with `--dense`. The densifier runs Semi-Global Block Matching (StereoSGBM) on stereo pairs selected by covisibility (shared 3D point count):

1. `cv2.stereoRectify` computes rectification transforms R1/R2/P1/P2/Q for each pair.
2. `cv2.initUndistortRectifyMap` + `cv2.remap` warp both images to a common epipolar plane.
3. `cv2.StereoSGBM_create` (3-way mode) computes dense disparities.
4. `cv2.reprojectImageTo3D` lifts the disparity map to 3D in the rectified camera-1 frame.
5. Points are filtered: disparity > 1.0, finite and positive depth, distance within 10× the median.
6. Points are transformed from rectified camera-1 frame → world frame via the rectification rotation R1 and camera extrinsics.

The output is merged across all stereo pairs and subsampled to at most 500,000 points. The dense PLY is written separately from the sparse PLY.

---

### Optional Stage 9 — Mesh Reconstruction

Activated with `--mesh`. Requires `pip install open3d`. The mesh pipeline has five internal stages:

1. **Point cloud preparation**: SOR (statistical outlier removal) and ROR (radius outlier removal) cleaning, normal estimation, optional downsampling.
2. **Surface reconstruction**: Screened Poisson (smooth closed surfaces), Ball-Pivoting (thin/open surfaces), or Alpha-shapes (fast, convex objects).
3. **Mesh cleaning**: Remove degenerate triangles, small disconnected components, and optionally fill holes.
4. **Post-processing**: Optional Taubin smoothing, quadric decimation, and RGB color projection from the point cloud onto mesh vertices.
5. **Export**: Writes to `.obj`, `.ply`, `.glb`, or `.stl` based on the output path extension.

Quality presets (low/medium/high/ultra) control Poisson depth (8–11), SOR/ROR parameters, normal estimation neighborhood size, and density threshold for artifact removal.

---

## Chapter 3 — Full CLI Reference

The pipeline is invoked as:

```bash
sfm [options]              # after pip install -e .
python run_sfm.py [options] # directly
```

### Core Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_dir DIR` | required | Directory containing input JPG/PNG/TIF/BMP images |
| `--output PATH` | `output.ply` | Output PLY path for the sparse point cloud |
| `--backend {python,colmap,colmap-mvs}` | `python` | Reconstruction backend |
| `--verbose` | off | Enable DEBUG-level logging |

**Backend choices:**
- `python`: built-in incremental SfM (default, no external dependencies)
- `colmap`: delegate to COLMAP binary for sparse reconstruction
- `colmap-mvs`: COLMAP sparse + GPU PatchMatch MVS (requires CUDA)

---

### Feature Extraction

| Argument | Default | Description |
|---|---|---|
| `--n_features N` | 8000 | Maximum SIFT features per image |
| `--sift-contrast-threshold F` | 0.02 | Lower = more low-contrast keypoints (COLMAP default) |
| `--sift-edge-threshold F` | 10.0 | Higher = keep more edge-like keypoints |
| `--sift-n-octave-layers N` | 3 | SIFT octave layers per scale level |
| `--sift-sigma F` | 1.6 | Gaussian sigma at octave 0 |

---

### Matching

| Argument | Default | Description |
|---|---|---|
| `--match_strategy {exhaustive,sequential,vocab_tree}` | `exhaustive` | Pair selection strategy |
| `--sequential_window N` | 5 | Window size for sequential matching |
| `--vocab_words N` | 4096 | Vocabulary size for vocab_tree |
| `--vocab_top_k N` | 10 | Top-K retrieved images per query |
| `--ratio F` | 0.75 | Lowe's ratio test threshold (lower = stricter) |
| `--min_matches N` | 15 | Minimum raw matches to retain a pair |

---

### Geometric Verification

| Argument | Default | Description |
|---|---|---|
| `--min_inliers N` | 15 | Minimum RANSAC inliers to accept a pair |
| `--ransac_thr F` | 1.0 | RANSAC reprojection threshold in pixels |

---

### Reconstruction

| Argument | Default | Description |
|---|---|---|
| `--max_reproj_error F` | 4.0 | Max reprojection error (px) for triangulation and PnP |
| `--ba_interval N` | 5 | Run bundle adjustment every N new cameras |
| `--no_refine_intrinsics` | off | Disable focal length and distortion refinement in BA |
| `--ba-fix-principal-point` | off | Keep cx/cy fixed during BA |
| `--no_filter` | off | Skip statistical outlier filtering of the final cloud |

---

### MVS Densification

| Argument | Default | Description |
|---|---|---|
| `--dense` | off | Run StereoSGBM densification after sparse reconstruction |
| `--dense_output PATH` | `<stem>_dense.ply` | Output PLY path for the dense cloud |

---

### Visualization

All visualization arguments are ignored unless `--visualize` is set.

| Argument | Default | Description |
|---|---|---|
| `--visualize` | off | Enable the full visualization suite |
| `--viz-samples N` | 3 | Images/pairs to sample for feature and match visualizations |
| `--viz-output DIR` | `sfm_visualization` | Directory for all visualization outputs |
| `--viz-format {png,jpg,pdf}` | `png` | Output format for saved figures |
| `--viz-interactive` | off | Open interactive open3d point cloud viewer at end |
| `--viz-save-video` | off | Export reconstruction growth GIF and turntable GIF |
| `--viz-dpi N` | 150 | DPI for saved figures |
| `--viz-seed N` | 42 | Random seed for reproducible image/pair sampling |

---

### COLMAP Options

All COLMAP arguments are ignored unless `--backend colmap` or `--backend colmap-mvs`.

| Argument | Default | Description |
|---|---|---|
| `--colmap-bin PATH` | `colmap` | Path to the COLMAP executable |
| `--colmap-workspace DIR` | auto | Internal COLMAP workspace directory |
| `--colmap-keep-workspace` | off | Keep workspace after run (for debugging) |
| `--colmap-vocab-tree PATH` | none | Path to pre-built COLMAP vocabulary tree (.bin) |

The `--colmap-vocab-tree` argument is required when `--match_strategy vocab_tree` is used with the COLMAP backend. Vocabulary tree files can be downloaded from https://demuc.de/colmap/#download.

---

### Mesh Reconstruction

All mesh arguments are ignored unless `--mesh` is set. Requires `pip install open3d`.

| Argument | Default | Description |
|---|---|---|
| `--mesh` | off | Enable mesh reconstruction |
| `--mesh-output PATH` | `<stem>_mesh.obj` | Output mesh path (.obj .ply .glb .stl) |
| `--mesh-method {poisson,bpa,alpha}` | `poisson` | Surface reconstruction algorithm |
| `--mesh-quality {low,medium,high,ultra}` | `medium` | Quality preset |
| `--mesh-depth N` | from preset | Poisson octree depth override (8–12) |
| `--mesh-no-clean` | off | Skip mesh cleaning |
| `--mesh-smooth` | off | Apply Taubin smoothing after cleaning |
| `--mesh-smooth-iterations N` | 5 | Number of Taubin smoothing iterations |
| `--mesh-no-texture` | off | Skip RGB color projection onto mesh |
| `--mesh-decimate` | off | Reduce polygon count via quadric decimation |
| `--mesh-decimate-target N` | 100000 | Target face count after decimation |
| `--mesh-no-fill-holes` | off | Disable hole filling (enabled by default) |
| `--mesh-keep-pointcloud` | off | Save cleaned point cloud alongside the mesh |
| `--mesh-preview` | off | Open interactive viewer after mesh generation |

**Method guide:**
- `poisson`: Screened Poisson — best for smooth, closed objects. Requires watertight or near-watertight input.
- `bpa`: Ball-Pivoting — better for thin structures and open surfaces.
- `alpha`: Alpha shapes — fast and predictable for convex or simple geometry.

**Quality preset parameters:**

| Preset | Poisson depth | Target points | SOR std ratio |
|---|---|---|---|
| `low` | 8 | 50,000 | 3.0 |
| `medium` | 9 | 200,000 | 2.0 |
| `high` | 10 | 500,000 | 1.5 |
| `ultra` | 11 | none | 1.2 |

---

### Checkpointing

| Argument | Default | Description |
|---|---|---|
| `--checkpoint-dir DIR` | `<output_dir>/.sfm_checkpoints` | Directory for stage checkpoints |
| `--resume` | off | Resume from most recent valid checkpoint |

Checkpoints store feature extraction and matching results as pickle files. They are automatically invalidated when the image set changes (detected via an MD5 hash over sorted filenames and file sizes). The matching checkpoint key includes the strategy and ratio threshold, so changing `--match_strategy` or `--ratio` triggers a re-match even with `--resume`.

---

## Chapter 4 — Usage Examples

### 1. Minimal Run

Smallest possible invocation — pure Python backend, exhaustive matching, all defaults:

```bash
sfm --image_dir ./images --output output.ply
```

### 2. Recommended Run for an Ordered Dataset (20–80 images)

Sequential matching for a video walk or ordered photo set, with visualization:

```bash
sfm \
  --image_dir ./images \
  --output scene.ply \
  --match_strategy sequential \
  --sequential_window 8 \
  --n_features 10000 \
  --visualize \
  --viz-output ./viz \
  --verbose
```

### 3. Large Unordered Dataset with Vocabulary Tree

For 100+ unordered photos where exhaustive matching would be too slow:

```bash
sfm \
  --image_dir ./photos \
  --output monument.ply \
  --match_strategy vocab_tree \
  --vocab_words 8192 \
  --vocab_top_k 20 \
  --n_features 12000 \
  --ratio 0.70 \
  --min_inliers 20 \
  --checkpoint-dir ./checkpoints \
  --resume
```

### 4. COLMAP Backend

Delegate reconstruction to the COLMAP binary:

```bash
sfm \
  --image_dir ./images \
  --output output.ply \
  --backend colmap \
  --colmap-bin /usr/local/bin/colmap \
  --match_strategy sequential \
  --colmap-keep-workspace
```

### 5. Dense Reconstruction

Run sparse reconstruction followed by StereoSGBM densification:

```bash
sfm \
  --image_dir ./images \
  --output sparse.ply \
  --dense \
  --dense_output dense.ply \
  --n_features 12000 \
  --ratio 0.70 \
  --min_inliers 20 \
  --max_reproj_error 3.0
```

### 6. Full Pipeline with Mesh Output

Sparse reconstruction, dense MVS, then mesh reconstruction from the dense cloud:

```bash
sfm \
  --image_dir ./images \
  --output sparse.ply \
  --dense \
  --dense_output dense.ply \
  --mesh \
  --mesh-output model.obj \
  --mesh-method poisson \
  --mesh-quality high \
  --mesh-smooth \
  --mesh-smooth-iterations 3 \
  --visualize \
  --viz-interactive
```

### 7. Resume from Checkpoint After Interruption

If the pipeline was interrupted after feature extraction or matching:

```bash
sfm \
  --image_dir ./images \
  --output output.ply \
  --checkpoint-dir ./checkpoints \
  --resume \
  --verbose
```

The pipeline will load feature and match checkpoints for the current image set and skip straight to geometric verification. If the image set has changed (files added or removed), stale checkpoints are automatically discarded and recomputed.

---

## Chapter 5 — Understanding the Output

### Output Files

| File | Produced when | Description |
|---|---|---|
| `output.ply` | always | Sparse colored point cloud (binary little-endian PLY) |
| `output_dense.ply` | with `--dense` | Dense point cloud from StereoSGBM MVS |
| `output_mesh.obj` | with `--mesh` | Mesh (format determined by extension) |
| `sfm_visualization/` | with `--visualize` | All visualization figures and GIFs |

### Reading a PLY File

The sparse PLY uses double-precision XYZ and uint8 RGB per vertex. It can be opened directly in:
- **MeshLab**: File → Import Mesh
- **CloudCompare**: File → Open
- **open3d** (Python): `o3d.io.read_point_cloud("output.ply")`
- **view_ply.py** (included): `python view_ply.py output.ply`

### Interpreting Reprojection RMSE

The pipeline logs bundle adjustment RMSE after each BA run and at the end of reconstruction. As a rough guide:

| RMSE | Assessment |
|---|---|
| < 1.0 px | Excellent — tight reconstruction, well-calibrated images |
| 1.0–2.0 px | Good — suitable for most photogrammetry purposes |
| 2.0–3.0 px | Acceptable — some drift, minor calibration issues |
| > 3.0 px | Problematic — inspect visualization output; likely cause: wide-baseline pairs with few inliers, rolling shutter, or mixed camera models |

### Visualization Directory

When `--visualize` is set, `sfm_visualization/` contains:
- Feature visualization images (keypoint overlays on sampled images)
- Match visualization images (side-by-side inlier matches for sampled pairs)
- Scene graph plot (image connectivity as a node-edge graph)
- Camera trajectory plot (3D camera positions)
- Reprojection error histogram
- BA convergence plot (RMSE per BA round)
- Optional: `reconstruction_growth.gif` and `turntable.gif` with `--viz-save-video`

---

## Chapter 6 — Tuning for Difficult Datasets

### Low-Overlap Images

Symptoms: few verified pairs, fewer cameras registered than expected.

```bash
sfm \
  --image_dir ./images \
  --output output.ply \
  --n_features 15000 \
  --sift-contrast-threshold 0.015 \
  --ratio 0.70 \
  --min_inliers 10
```

- Raise `--n_features` to detect more candidates per image.
- Lower `--sift-contrast-threshold` to include more low-contrast regions.
- Lower `--ratio` to keep only high-confidence matches.
- Lower `--min_inliers` carefully — too low allows degenerate pairs.

### Textureless Scenes

Symptoms: very few keypoints per image, most pairs fail geometric verification.

- Raise `--n_features` to the maximum the scene can provide.
- Use `--match_strategy vocab_tree` — TF-IDF retrieval is more robust to sparse feature histograms than exhaustive distance comparisons.
- Consider switching to `--backend colmap`, which has a more battle-tested feature detector and can use NetVLAD-based retrieval.

### Large Datasets (100+ Images)

Symptoms: exhaustive matching is too slow, memory pressure.

```bash
sfm \
  --image_dir ./images \
  --output output.ply \
  --match_strategy vocab_tree \
  --vocab_words 8192 \
  --vocab_top_k 20 \
  --checkpoint-dir ./checkpoints \
  --resume
```

- Always use `--match_strategy vocab_tree` for datasets over 100 images.
- Enable `--checkpoint-dir` and `--resume` so matching does not have to restart on interruption.
- Increase `--vocab_words` (8192–16384) for better retrieval quality.
- Raise `--vocab_top_k` if coverage is poor (at the cost of more pairs to verify).

### Slow Bundle Adjustment

Symptoms: each BA round takes minutes.

- Lower `--ba_interval` (e.g., `--ba_interval 3`) to run BA more frequently on smaller problems.
- Use `--no_refine_intrinsics` if the camera is pre-calibrated — removing the 5 shared intrinsic parameters from the parameter vector speeds up each Jacobian evaluation.
- For scenes over 200 cameras, consider `--backend colmap`, which uses a C++ sparse solver that scales substantially better than the Python SciPy implementation.

### Poor Reconstruction (Few Points or Cameras)

Symptoms: only the seed pair registers, or many cameras fail PnP.

1. Run with `--visualize` and inspect the scene graph and match visualizations. A disconnected graph means images share no verified pairs.
2. Check `--min_inliers` — if it is too high, valid pairs are discarded.
3. Try increasing `--min_matches` threshold temporarily lowered to 8 to see how many pairs almost pass.
4. Verify the `--image_dir` contains only images from the same scene — mixed scenes, panoramas, or pure rotation sequences all degrade reconstruction.
5. Check reprojection error in logs; if BA RMSE starts high (> 5 px) the seed pair geometry is poor. The pipeline scores seed pairs by `baseline × inlier_count` — try `--verbose` to see which pair was selected and its baseline and inlier count.
6. Review the IMAGE_INPUT_GUIDE.md for capture guidelines.
