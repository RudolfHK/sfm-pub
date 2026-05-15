# DEEP_ANALYSIS.md — sfm-pub: Research-Grade Codebase Autopsy

**Prepared against:** commit `af37564` (branch `claude/sfm-pipeline-SLpu2`)  
**Reference landscape cutoff:** May 2025  
**Estimated read time:** ~45 minutes

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Complete Codebase Internalization](#2-complete-codebase-internalization)
3. [State-of-the-Art Landscape (2025)](#3-state-of-the-art-landscape-2025)
4. [Systematic Stage-by-Stage Weakness Audit](#4-systematic-stage-by-stage-weakness-audit)
5. [Comparative Scoring Matrix](#5-comparative-scoring-matrix)
6. [Fixability / ROI Analysis](#6-fixability--roi-analysis)
7. [Production System Insights](#7-production-system-insights)
8. [Research Frontier Applicability](#8-research-frontier-applicability)
9. [Prioritized 5-Phase Improvement Roadmap](#9-prioritized-5-phase-improvement-roadmap)

---

## 1. Executive Summary

**Current tier: 🟡 Solid Research Prototype / Approaching Publishable Open Source**

This codebase is a well-engineered, self-contained incremental SfM pipeline that has grown substantially from a single-file proof-of-concept into a multi-module system with GPU support, three matching strategies, optional dense reconstruction, optional mesh reconstruction, checkpointing, input validation, and a comprehensive visualization suite. The recent robustness-improvements branch added Hartley normalization, USAC_MAGSAC for both F and E, SO(3) re-orthogonalization, adaptive Huber scale, and a covisibility graph — all genuine improvements that COLMAP and RC implement.

**What it does well relative to its scope:**
- Clean module boundaries with zero circular imports
- Graceful GPU → CPU fallback hierarchy throughout
- Near-production-quality error handling in all public entry points
- Meaningful logging with correct level discipline
- Non-crashing optional stages (dense, mesh, viz all wrapped in try/except)
- Hartley normalization + USAC_MAGSAC is better than what many academic repos ship

**Where it still falls short of production:**
- Feature detection is classical SIFT; no learned detector/descriptor
- Bundle adjustment uses scipy TRF with a single shared focal length and only k1/k2 distortion — no cx/cy refinement, no tangential distortion, no per-camera model
- MVS is SGBM stereo on consecutive pairs — output is an order of magnitude sparser than COLMAP PatchMatch and completely unusable on unordered collections or wide baselines
- No global SfM mode; incremental drift accumulates with N
- No local BA (windowed) between global BA rounds
- Seed selection is pure inlier-count greedy; no stability check for near-degenerate pairs
- No track merging / scene graph deduplication
- Scaling ceiling ≈ 100–200 images before runtime and quality diverge from COLMAP

**Single most important next step:** Replace SIFT+FLANN with SuperPoint+LightGlue. This alone would close the largest quality gap with the least risk to the rest of the pipeline.

---

## 2. Complete Codebase Internalization

### File inventory and line counts

| File | Lines | Role |
|---|---|---|
| `run_sfm.py` | 1022 | CLI entry point, pipeline orchestration, checkpointing |
| `sfm/feature_extraction.py` | 262 | SIFT/kornia/SURF feature extraction |
| `sfm/feature_matching.py` | 594 | Exhaustive / sequential / vocab-tree matching |
| `sfm/geometric_verification.py` | 271 | F + E + pose estimation, Hartley norm |
| `sfm/reconstruction.py` | 788 | Incremental SfM, PnP, triangulation, covisibility |
| `sfm/bundle_adjustment.py` | 425 | Sparse TRF BA, Huber loss, SO(3) projection |
| `sfm/point_cloud.py` | 324 | Colorization, outlier filter, PLY export |
| `sfm/mvs.py` | 268 | SGBM dense reconstruction |
| `sfm/visualizer.py` | 1587 | Event-driven matplotlib/open3d visualization |
| `sfm/colmap_backend.py` | 573 | COLMAP CLI wrapper + binary format reader |
| `sfm/device.py` | 53 | GPU device singleton |
| `sfm/utils.py` | 165 | Image I/O, EXIF, camera math |
| `sfm/mesh/pipeline.py` | 260 | Mesh orchestrator, QUALITY_PRESETS |
| `sfm/mesh/pointcloud_prep.py` | 216 | SOR, ROR, normals, voxel downsample |
| `sfm/mesh/reconstruction.py` | 206 | Poisson / BPA / Alpha shapes |
| `sfm/mesh/cleaning.py` | 154 | Dedup, manifold, components, hole fill |
| `sfm/mesh/postprocess.py` | 107 | Taubin smooth, decimation, color transfer |
| `sfm/mesh/export.py` | 100 | OBJ / PLY / GLB / STL export |
| **Total** | **~7,375** | |

### Data flow summary

```
images/ ──► FeatureExtractor ──► {keypoints, descriptors}
                                         │
                              FeatureMatcher (exhaustive | sequential | vocab_tree)
                                         │
                              GeometricVerifier (F → E → pose)
                                         │
                              IncrementalSfM (seed → PnP loop → BA every N)
                                         │
                     ┌────────────────────┴────────────────────┐
                PointCloudExporter                    MVSDensifier (optional)
                (colorize + PLY)                   (SGBM per pair → PLY)
                     │                                        │
                     └──────────────────┬────────────────────┘
                                  MeshPipeline (optional)
                          (prep → Poisson/BPA/Alpha → clean → post → export)
                                        │
                               SfMVisualizer (optional, zero-cost when off)
```

### Key algorithm choices (as shipped)

| Stage | Algorithm | Implementation |
|---|---|---|
| Detection | SIFT (Lowe 2004) | `cv2.SIFT_create` |
| GPU detection | kornia ScaleSpaceDetector + SIFT descriptors | `kornia.feature.ScaleSpaceDetector` |
| GPU alt | CUDA SURF (padded to 128D) | `cv2.cuda.SURF_CUDA_create` |
| Matching | FLANN KDTree (5 trees, 50 checks) | `cv2.FlannBasedMatcher` |
| GPU matching | Brute-force L2 via `torch.cdist` | custom |
| Large-scale retrieval | Flat k-means vocab (256 words) + TF-IDF + cosine sim | custom |
| Normalization | Isotropic Hartley (mean dist → √2) | custom, `_hartley_normalize` |
| F estimation | USAC_MAGSAC (fallback: FM_RANSAC) | `cv2.findFundamentalMat` |
| E estimation | USAC_MAGSAC (fallback: RANSAC) | `cv2.findEssentialMat` |
| Pose | Cheirality decomposition | `cv2.recoverPose` |
| Registration | EPnP + RANSAC | `cv2.solvePnPRansac(SOLVEPNP_EPNP)` |
| Triangulation | DLT (batch) | `cv2.triangulatePoints` |
| Bundle Adjustment | Sparse TRF, Huber loss, adaptive f_scale | `scipy.optimize.least_squares` |
| SO(3) projection | SVD (Procrustes) | `np.linalg.svd` |
| Distortion | Brown–Conrady radial k1/k2 | custom in BA |
| Dense | SGBM stereo on consecutive pairs | `cv2.StereoSGBM_create` |
| Surface recon | Screened Poisson / BPA / Alpha shapes | `open3d` |

---

## 3. State-of-the-Art Landscape (2025)

### 3.1 Feature Detection and Description

**Classical (baseline):**
- **SIFT** (Lowe, 2004, IJCV) — scale/rotation invariant, 128-D L2-norm descriptor. Still competitive on textured scenes. Patent expired 2020.
- **ORB** (Rublee et al., 2011, ICCV) — fast binary descriptor, poor for SfM at scale.
- **SURF** (Bay et al., 2006, ECCV) — patented, rarely used in new work.

**Learned detectors (2018–2025):**
- **SuperPoint** (DeTone et al., 2018, CVPR) — homographic adaptation self-supervision; produces 65-D descriptors. Widely used; included in COLMAP 3.x, Meshroom, many academic baselines. *The de facto learned SIFT replacement.*
- **DISK** (Tyszkiewicz et al., 2020, NeurIPS) — reinforcement-learning training; handles non-uniform illumination better than SuperPoint; matches well with LightGlue.
- **DeDoDe** / **DeDoDe v2** (Edstedt et al., 2023 ECCV, 2024) — 3D supervision using known geometry; produces very dense, accurate detections. Currently SotA on MegaDepth-1500.
- **ALIKED** (Zhao et al., 2023, ICCV) — efficient deformable convolutions; fast inference, near-SuperPoint quality, smaller model. Good for edge-device deployment.
- **XFeat** (Potje et al., 2024, CVPR) — ultra-lightweight (< 1 ms per image on RTX 4090); accuracy competitive with SIFT on Megadepth. Best speed/accuracy trade-off for real-time applications.
- **KeyNetAffNet + HardNet8** (Barroso-Laguna et al., 2020, ACCV) — affine-covariant, trained on Hpatches; useful when strong affine deformations are expected.

**Dense descriptor extraction (MVS preprocessing):**
- **NetVLAD** (Arandjelovic et al., 2016, CVPR) — global descriptor for image retrieval at scale.
- **R2D2** (Revaud et al., 2019, NeurIPS) — joint detection and description with repeatability + reliability losses.
- **Patch2Pix** (Zhou et al., 2021, CVPR) — end-to-end patch refinement with dense supervision.

### 3.2 Feature Matching

**Classical:**
- FLANN KDTree + Lowe ratio test — O(N log N) approximate kNN; this repo's baseline.
- Brute-force L2 on GPU via `torch.cdist` — exact, used in this repo.

**Graph-based / attention:**
- **SuperGlue** (Sarlin et al., 2020, CVPR) — GNN with self/cross attention over SuperPoint keypoints. State of practice 2020–2022.
- **LightGlue** (Lindenberger et al., 2023, ICCV) — adaptive depth attention; 2–6× faster than SuperGlue on the same hardware with similar or better accuracy. Runs natively with SuperPoint, DISK, ALIKED, SIFT. *The current default for real-time learned matching.*
- **MatchFormer** (Wang et al., 2022, ACCV) — hierarchical transformer for matching.

**Detector-free (dense) matching:**
- **LoFTR** (Sun et al., 2021, CVPR) — dense coarse-to-fine matching with dual-softmax; no keypoints required. Very robust to low-texture or repetitive scenes but ~5× slower than SuperPoint+LightGlue.
- **DKM** (Edstedt et al., 2023, CVPR) — dense kernelized matching with warp-and-cycle loss.
- **RoMa** (Edstedt et al., 2024, CVPR) — robust dense matching with confidence estimation; SotA on MegaDepth-1500 and HPatches as of 2024.
- **EfficientLoFTR** (Wang et al., 2024) — distilled LoFTR with 4× speed improvement.

**Image retrieval for large-scale matching:**
- **Hierarchical Localization (hloc)** (Sarlin et al., 2019, CVPR) — modular retrieval + matching framework combining NetVLAD/AP-GeM retrieval with SuperGlue/LightGlue.
- **DBoW3** (Gálvez-López & Tardós, 2012, IEEE T-RO) — fast bag-of-binary-words, used in ORB-SLAM; comparable to COLMAP vocab tree.
- **EigenPlaces** / **CosPlace** (Berton et al., 2022, 2023) — aggregated descriptors for place recognition.

### 3.3 Robust Estimation

- **RANSAC** (Fischler & Bolles, 1981, CACM) — baseline. O(k · N²) for F estimation.
- **LORANSAC** (Chum et al., 2003, BMVC) — local optimization step after each inlier set found; standard practice in COLMAP.
- **DEGENSAC** (Chum et al., 2005, CVPR) — degenerate configuration detection (planar scenes).
- **MLESAC** (Torr & Zisserman, 2000, CVIU) — maximum-likelihood flavour.
- **MAGSAC** (Barath & Matas, 2018, CVPR) — sigma-consensus, marginalization over inlier thresholds.
- **MAGSAC++** (Barath et al., 2020, CVPR) — faster with quality-weighted sampling. ✅ *Already in this repo as `USAC_MAGSAC`.*
- **USAC** (Raguram et al., 2013, PAMI) — universal RANSAC framework. OpenCV implements this.
- **GC-RANSAC** (Barath & Matas, 2018) — graph-cut spatial coherence constraint; reduces iterations.

### 3.4 SfM Reconstruction Paradigms

**Incremental SfM (classical):**
- **COLMAP** (Schönberger & Frahm, 2016, CVPR) — gold standard. Guided matching, track merging, local BA + global BA, observation graph.
- **OpenSfM** (Moulon et al., 2016) — Python, multi-model, global or incremental.
- **VisualSFM** (Wu, 2011) — older, GPU-accelerated; largely superseded by COLMAP.

**Global SfM (rotation averaging):**
- **OpenMVG** (Moulon et al., 2017) — implements rotation + translation averaging.
- **GLOMAP** (Pan et al., 2024, CVPR) — constrained 1-D radial cameras for rotation averaging; orders of magnitude faster than COLMAP on very large scenes (>10k images) with near-identical accuracy. Uses a fundamentally different pose graph optimization.
- **Theia** (Sweeney et al., 2015) — academic global SfM library.

**Deep / end-to-end SfM:**
- **VGGSfM** (Wang et al., 2024, CVPR) — fully differentiable SfM pipeline; 2-view seeds → track prediction → bundle adjustment, trained on CO3D. Zero-shot generalisation is impressive but fragile on in-the-wild scenes with > 50 images.
- **DUSt3R** (Wang et al., 2024, CVPR) — Pairwise dense matching + global alignment via optimization over the pointmap graph. No explicit feature matching. Remarkable zero-shot 3D understanding from arbitrary image pairs.
- **MASt3R** (Leroy et al., 2024, ECCV) — extends DUSt3R with better matching supervision; produces dense correspondences + pointmaps simultaneously.
- **Instant3D** / **Zero123++** / **One-2-3-45** — multi-view synthesis from a single image; complementary to SfM for object-centric reconstruction when input images are sparse.

### 3.5 Bundle Adjustment

- **Ceres Solver** (Agarwal et al., 2010) — C++ sparse nonlinear LS; used by COLMAP, GLOMAP, VGGSfM. Orders of magnitude faster than scipy TRF at scale.
- **g2o** (Kümmerle et al., 2011, ICRA) — graph-based; common in SLAM.
- **GTSAM** (Dellaert et al., 2012) — factor-graph formulation; excellent for incremental updates.
- **GBA / SchurBA** — Schur complement-based elimination for block-structured Jacobians; essentially what COLMAP does internally.
- **Theseus** (Pineda et al., 2022, NeurIPS) — differentiable nonlinear LS in PyTorch; enables end-to-end training through BA.
- **ParticleSfM** (Wang et al., 2022, ECCV) — graph-based BA formulation for sequential video.

### 3.6 Dense Reconstruction (MVS)

**Traditional MVS:**
- **COLMAP PatchMatch** (Schönberger et al., 2016, ECCV) — GPU PatchMatch; reference quality.
- **ACMH / ACMM / ACMP** (Ma et al., 2021, 2022) — adaptive checkboard + multi-hypothesis; often better than COLMAP on indoor scenes.
- **OpenMVS** (Cernea, 2015) — open-source, comparable to COLMAP MVS.
- **PMVS** (Furukawa & Ponce, 2010, PAMI) — older but still referenced.

**Learned MVS:**
- **MVSNet** (Yao et al., 2018, ECCV) — recurrent cost volume; seminal learned MVS.
- **PatchMatchNet** (Wang et al., 2021, CVPR) — integrates classical PatchMatch with learned confidence.
- **Uni-MVSNet** (Peng et al., 2022, CVPR) — unified probability volume.
- **GeoMVSNet** (Zhang et al., 2023, CVPR) — geometry-guided coarse-to-fine.
- **IterMVS** (Wang et al., 2022, CVPR) — recurrent updates; good speed/quality trade-off.

### 3.7 Neural Rendering (post-reconstruction)

- **NeRF** (Mildenhall et al., 2020, ECCV) — volume rendering, implicit representation.
- **Mip-NeRF 360** (Barron et al., 2022, CVPR) — unbounded scenes.
- **Instant NGP** (Müller et al., 2022, SIGGRAPH) — hash encoding, real-time training.
- **3D Gaussian Splatting (3DGS)** (Kerbl et al., 2023, SIGGRAPH) — explicit 3D Gaussian primitives; real-time rendering at high quality. Requires a reasonable sparse SfM point cloud as initialization. *The most practically impactful development in 3D rendering since NeRF.*
- **Scaffold-GS** (Lu et al., 2024, CVPR) — structured 3DGS.
- **2D Gaussian Splatting** (Huang et al., 2024, SIGGRAPH) — surface-aligned Gaussians for better geometry.

---

## 4. Systematic Stage-by-Stage Weakness Audit

### Template applied to each stage

For every stage, weaknesses are classified by:
- **Severity:** 🔴 Critical (blocks production use) | 🟠 High (degrades quality significantly) | 🟡 Medium (noticeable but manageable) | 🟢 Low (minor polish)
- **Fixability:** Easy (< 1 week) | Moderate (1–4 weeks) | Hard (1–3 months) | Research (open problem)
- **Reference:** Where applicable

---

### Stage A — Feature Extraction (`sfm/feature_extraction.py`)

**What it does:** OpenCV SIFT (8000 features, σ=1.6, contrastThr=0.04, nOctaveLayers=3) with GPU fallback to kornia ScaleSpaceDetector+SIFT or CUDA SURF.

**Weakness A-1 — Classical descriptors only**  
Severity: 🟠 High | Fixability: Moderate  
SIFT descriptors are 128-D handcrafted and lack the discriminative power of learned descriptors trained on millions of image pairs. On low-texture surfaces (marble, concrete, metallic objects), SIFT fails to produce sufficient repeatable keypoints. SuperPoint trained on synthetic homographies + outdoor scenes recovers features that SIFT simply cannot detect.  
*Reference: DeTone et al. (2018) "SuperPoint: Self-supervised interest point detection and description." CVPR Workshops.*

**Weakness A-2 — No affine covariance**  
Severity: 🟡 Medium | Fixability: Moderate  
SIFT is invariant under similarity transforms but not affine. Oblique viewpoints (wall photography, ground-to-sky, wide-baseline) introduce affine deformations. ASIFT (6 simulated tilt/rotation copies per image) or AffNet+HardNet8 cover this.  
*Reference: Morel & Yu (2009) "ASIFT: A new framework for fully affine invariant image comparison." SIAM J. Imaging Sciences.*

**Weakness A-3 — `contrastThreshold=0.04` too conservative**  
Severity: 🟢 Low | Fixability: Easy (1 line)  
Lowering to 0.02 reliably increases keypoint count by 30–60% on real imagery with negligible precision loss, as the ratio test and cross-check downstream filter poor matches. COLMAP uses 0.02.

**Weakness A-4 — No image pre-screening**  
Severity: 🟡 Medium | Fixability: Easy  
No detection of blurry, overexposed, near-duplicate, or near-black images. These images extract essentially zero useful features but cause the O(N²) matcher to waste cycles on doomed pairs. Production systems (COLMAP via `ImageReader.camera_mask`, Metashape) pre-filter before extraction.

**Weakness A-5 — CUDA SURF pads 64D → 128D with zeros**  
Severity: 🟡 Medium | Fixability: Easy  
`feature_extraction.py:254`: when CUDA SURF produces 64-D descriptors, zeros are appended to reach 128-D so downstream code is uniform. This halves the effective dimensionality and will cause worse ratio-test behavior than true 128-D SIFT descriptors. The CUDA SURF path should either use the 128-D SURF variant (`extended=True`) or keep a separate 64-D code path through matching.

**Weakness A-6 — No laplacian-of-Gaussian scale-space for GPU path**  
Severity: 🟢 Low | Fixability: Easy  
The kornia path builds a `ScaleSpaceDetector` with `BlobDoG()` — correct — but the scale derived from the LAF determinant is scaled by `* 6.0` to convert to SIFT diameter. This constant is empirically derived and may not match the scale convention used by `cv2.SIFT.compute()`, causing slight descriptor inconsistency. No critical bug, but worth a unit test.

---

### Stage B — Feature Matching (`sfm/feature_matching.py`)

**What it does:** Three matchers (exhaustive O(N²), sequential, vocab-tree). CPU via FLANN, GPU via `torch.cdist`. Lowe ratio test, optional cross-check.

**Weakness B-1 — No learned matching**  
Severity: 🟠 High | Fixability: Moderate  
All matching is nearest-descriptor search + ratio test. This is fundamentally a local descriptor comparison with no geometric consistency at matching time. LightGlue introduces global context via transformer attention, yielding dramatically fewer false positives on difficult pairs (repetitive patterns, illumination changes). Quality gap is most visible on pairs with < 30% overlap.  
*Reference: Lindenberger et al. (2023) "LightGlue: Local feature matching at light speed." ICCV.*

**Weakness B-2 — Vocab tree uses flat k-means (256 words)**  
Severity: 🟠 High | Fixability: Moderate  
256 visual words is far too small for a discriminative vocabulary. COLMAP's vocabulary tree uses 256^4 = 4 billion (in practice ~65k–1M) hierarchical nodes. The flat 256-word vocabulary will fail to distinguish images on any non-trivial dataset (> 50 images) and degrade to near-random pair selection. Minimum viable vocabulary: 4096 words with IDF weighting.

**Weakness B-3 — GPU cross-check uses Python set intersection**  
Severity: 🟡 Medium | Fixability: Easy  
`feature_matching.py:162`: After the backward pass, the mutual consistency check does `sorted(set_12 & set_21)`. This is a CPU Python set operation over potentially tens of thousands of pairs. The backward distance matrix is already on GPU; using `torch.nonzero` + tensor comparison would keep the entire mutual-check on GPU.

**Weakness B-4 — No guided matching**  
Severity: 🟠 High | Fixability: Hard  
COLMAP performs guided matching: after a rough set of verified pairs is found, the epipolar geometry is used to constrain the search space in a second matching pass, dramatically increasing inlier count. This is especially important for the seed pair and early registration images. Without guided matching, pairs with < 30% overlap will often have fewer than `min_inliers` and be discarded, causing gaps in the connectivity graph.

**Weakness B-5 — VocabTreeMatcher falls back to exhaustive when N ≤ top_k + 1**  
Severity: 🟢 Low | Fixability: Easy  
The fallback is correct but the threshold `N ≤ top_k + 1` means for `top_k=10` and 12 images, exhaustive is used. Fine. However the log message doesn't explain the fallback reason to the user, which can be confusing.

---

### Stage C — Geometric Verification (`sfm/geometric_verification.py`)

**What it does:** Hartley normalization → USAC_MAGSAC F → de-normalize F → USAC_MAGSAC E on F-inliers → recoverPose with cheirality → near-zero baseline rejection.

**Weakness C-1 — No LORANSAC / local optimization on E**  
Severity: 🟡 Medium | Fixability: Moderate  
USAC_MAGSAC already includes its own local optimization steps internally, which is why it's better than FM_RANSAC. However, the fallback path (`E = K^T F K`) bypasses RANSAC entirely. On datasets where `cv2.findEssentialMat` frequently crashes (certain OpenCV builds), the fallback produces a non-refined E whose inlier mask is trivially the whole F-inlier set, inflating inlier counts.

**Weakness C-2 — Normalization applied only to F, not to E**  
Severity: 🟡 Medium | Fixability: Easy  
`_hartley_normalize` is called on the raw pixel coordinates before F estimation, but E estimation operates on the already-undistorted F-inlier subset `pts1_fin / pts2_fin`. These are in pixel space and benefit from normalization too. The numerical conditioning argument applies equally to `findEssentialMat`.

**Weakness C-3 — No degeneracy detection (planar scenes)**  
Severity: 🟠 High | Fixability: Moderate  
On planar scenes (walls, floors, single flat objects), the F matrix becomes degenerate and the homography model is more appropriate. DEGENSAC tests for planarity during RANSAC and switches models. Without this, planar-dominant scenes produce F estimates with inflated inlier counts that decompose to incorrect poses. The symptom is drift and camera registration failures in the incremental loop.  
*Reference: Chum et al. (2005) "Two-View Geometry Estimation Unaffected by a Dominant Plane." CVPR.*

**Weakness C-4 — `np.random.permutation` shuffle before RANSAC**  
Severity: 🟢 Low | Fixability: Easy  
`geometric_verification.py:125`: matches are shuffled before being passed to RANSAC to avoid a secondary OpenCV crash. This is a workaround for a known OpenCV 4.x bug. The comment explains it, which is good. However this adds a fixed O(M) overhead; an alternative is to set `flags=cv2.USAC_MAGSAC` which internally shuffles. Worth removing the Python shuffle once the minimum supported OpenCV version is known to be unaffected.

**Weakness C-5 — Principal point assumed at image centre**  
Severity: 🟡 Medium | Fixability: Easy  
`utils.estimate_intrinsics` places `cx = w/2`, `cy = h/2`. No EXIF or ICC profile reading for the principal point. For most consumer cameras this is within 10px of the true value, which is acceptable for initialization. However the BA stage never optimizes cx/cy (only f, k1, k2), so systematic principal-point offset accumulates as a constant 2D shift in all reprojections. COLMAP refines cx/cy by default.

---

### Stage D — Incremental SfM (`sfm/reconstruction.py`)

**What it does:** Max-inlier seed selection → DLT triangulation of seed pair → PnP+RANSAC registration loop → batch triangulation → covisibility graph → periodic BA.

**Weakness D-1 — Seed selection is pure inlier-count greedy**  
Severity: 🟠 High | Fixability: Moderate  
`_select_seed_pair` picks the pair with the most RANSAC inliers. This ignores:
- Baseline length (near-zero baseline = poor triangulation even with many inliers)
- Triangulation angle distribution (all inliers at small angle → poor depth)
- Scene coverage (a pair of overlapping telephoto images ≠ a good seed)

COLMAP's seed selection additionally requires a minimum median triangulation angle and a minimum baseline-to-depth ratio. Without this, certain dataset configurations reliably produce degenerate initializations.

**Weakness D-2 — EPnP only; no iterative refinement**  
Severity: 🟡 Medium | Fixability: Easy  
`cv2.solvePnPRansac(flags=cv2.SOLVEPNP_EPNP)` is used. EPnP is an efficient O(1) algorithm but produces a non-locally-optimal solution. COLMAP follows EPnP with an iterative refinement step using the same linear system residuals. `cv2.SOLVEPNP_ITERATIVE` or DLS would improve registration accuracy without changing the interface.

**Weakness D-3 — No local bundle adjustment**  
Severity: 🟠 High | Fixability: Hard  
Between global BA rounds, newly registered cameras accumulate small systematic errors that compound. COLMAP runs a local BA window over the newly registered camera and its `k`-nearest covisible cameras (typically `k=6`) immediately after each registration, before the global BA. This is the single most important architectural difference between this pipeline and COLMAP's. Without local BA, the incremental drift between global BA rounds is O(n_cameras_since_last_BA).

**Weakness D-4 — Single `_MAX_POINT_DISTANCE_RATIO = 50.0` heuristic for bad camera rejection**  
Severity: 🟡 Medium | Fixability: Easy  
The heuristic `cam_dist > 50 × median_scene_dist` rejects wildly displaced cameras but has no sensitivity to rotation outliers or cameras that are correctly placed but looking the wrong direction. A complementary check would verify that the inliers from PnP have a plausible angular distribution (field-of-view consistency).

**Weakness D-5 — `_remove_outlier_points` threshold is a blend of fixed and adaptive**  
Severity: 🟢 Low | Fixability: Easy  
`thr = min(max_reproj_err * 2.0, median + 3σ)`. The `2 × max_reproj_err` ceiling can dominate and may not be tight enough when `max_reproj_err = 4.0` and the scene is clean (median error ~ 0.5 px). Using just `median + 3σ` with a minimum floor of 1.5 px would be more robust.

**Weakness D-6 — No track merging**  
Severity: 🟠 High | Fixability: Hard  
When the same physical point is observed from different paths through the image graph (a camera sees two already-triangulated keypoints that should be the same 3D point), both 3D points persist and are never merged. This creates duplicate 3D points, inflates point counts, and degrades BA performance. COLMAP implements `FilterPoints` + `CompleteAndMergeTracks` as a post-registration step.

**Weakness D-7 — `_quick_rmse` iterates all observations in Python**  
Severity: 🟡 Medium | Fixability: Easy  
`_quick_rmse` runs a Python for-loop over `self.observations` with dict lookups per iteration. At 500k+ observations this adds ~2–5 seconds per BA round. A vectorized version building camera-indexed arrays (as in `_run_ba`) would run in milliseconds.

---

### Stage E — Bundle Adjustment (`sfm/bundle_adjustment.py`)

**What it does:** Sparse TRF least-squares over [f, k1, k2, pose_0...pose_C, pts_0...pts_P]; Huber loss; adaptive MAD-based f_scale; SO(3) re-orthogonalization post-optimization.

**Weakness E-1 — scipy TRF is 10–100× slower than Ceres at scale**  
Severity: 🔴 Critical (for large scenes) | Fixability: Hard  
`scipy.optimize.least_squares` with the TRF method computes explicit Jacobian columns; even with sparsity hints the overhead of Python object allocation per evaluation is prohibitive above ~1000 cameras. COLMAP uses Ceres with Schur complement and analytical Jacobians. For 100–500 cameras this pipeline will take minutes where COLMAP takes seconds.  
*Reference: Agarwal et al. (2010) "Ceres Solver." ceres-solver.org.*

**Weakness E-2 — Single shared intrinsics for all cameras**  
Severity: 🟠 High | Fixability: Moderate  
All cameras share one f, k1, k2. This is correct only if all images come from the same physical camera with fixed zoom. Mixed datasets (phone + DSLR, zoomed shots, multiple cameras) will have systematically incorrect intrinsics for all but the dominant camera. COLMAP supports per-camera or per-image intrinsics.

**Weakness E-3 — cx, cy not refined**  
Severity: 🟡 Medium | Fixability: Easy  
Only f, k1, k2 are in the parameter vector. `cx = K[0,2]` and `cy = K[1,2]` are fixed at image center throughout BA. For cameras with significant principal point offset (wide-angle lenses, stitched panoramas, microscopy), this introduces a persistent 2D bias in all reprojection residuals that the optimizer cannot eliminate. Adding cx/cy to the shared parameter block with bounds [0.4W, 0.6W] × [0.4H, 0.6H] is a low-risk 3-line change.

**Weakness E-4 — No tangential distortion (p1, p2)**  
Severity: 🟢 Low | Fixability: Easy  
Brown–Conrady model (p1, p2) is important for lenses with decentering. The BA cost function already includes `k1, k2`; adding p1, p2 would change `dist_factor` from purely radial to full. For typical DSLR/phone cameras the effect is small (<0.5 px), but for wide-angle action cameras (GoPro) it can be 3–5 px.

**Weakness E-5 — BA divergence guard is too permissive**  
Severity: 🟡 Medium | Fixability: Easy  
`bundle_adjustment.py:372`: if `rmse_final > rmse_init * 3.0`, BA is reverted. A 3× degradation threshold allows BA to significantly worsen the solution before reverting. A tighter bound (1.5×) plus a fallback to halved `f_scale` and retry would be more robust.

**Weakness E-6 — Intrinsics refinement conditional on change magnitude**  
Severity: 🟡 Medium | Fixability: Easy  
`bundle_adjustment.py:412`: `K_refined` is only returned (and hence used downstream) if `abs(f_opt - f_init) / f_init > 0.005 or abs(k1_opt) > 1e-4`. This suppresses small but real corrections. For a 5000-pixel-wide image, a 0.4% focal length change is 20 pixels — significant. The threshold should be removed; the downstream cost of applying a refined K is negligible.

---

### Stage F — Point Cloud Export (`sfm/point_cloud.py`)

**What it does:** Per-camera vectorized reprojection filtering, bilinear BGR sampling, binary PLY write.

**Weakness F-1 — Colorization loop is Python-level per point**  
Severity: 🟡 Medium | Fixability: Moderate  
`point_cloud.py:170`: the inner loop `for k in range(len(good_pts))` calls `_sample_bilinear` per point-observation pair in Python. For 100k points × 5 cameras = 500k iterations, this takes 3–8 seconds on CPU. NumPy vectorization of bilinear interpolation (using `map_coordinates` from `scipy.ndimage` or `cv2.remap`) would reduce this to < 100 ms.

**Weakness F-2 — No per-point observation count threshold**  
Severity: 🟡 Medium | Fixability: Easy  
Points observed by only one camera are added with a single color sample but no geometric validation from multiple views (only triangulation angle from the seed pair checks this). Adding a minimum-track-length filter (e.g., min 2 cameras) would reduce noise in the sparse cloud.

**Weakness F-3 — PLY uses float32 XYZ**  
Severity: 🟢 Low | Fixability: Easy  
`point_cloud.py:308–309`: 32-bit floats are used for XYZ. For large outdoor scenes (>1 km extent) or when the reconstruction origin is far from zero, float32 precision (~7 decimal digits) causes visible quantization. Double precision PLY or centroid-normalization before export would fix this.

---

### Stage G — Multi-View Stereo (`sfm/mvs.py`)

**What it does:** SGBM stereo on consecutive sorted-index camera pairs (up to `max_pairs_per_image=2` neighbors), world-space point lift, distance outlier filter, random subsample to `max_dense_pts=500k`.

**Weakness G-1 — SGBM is 2-camera stereo, not multi-view**  
Severity: 🔴 Critical | Fixability: Hard  
The SGBM-based approach processes pairs in isolation with no cross-view consistency. Each stereo pair produces its own depth map and these are merged by concatenation + random subsampling. This means:
- No visibility reasoning between pairs (a point occluded in pair A is not masked in pair B)
- Duplicate points at boundaries between adjacent pairs
- No photometric consistency constraint across views
- Dense quality is far below COLMAP PatchMatch (which does photometric consistency voting across all covering views simultaneously)

*Reference: Schönberger et al. (2016) "Pixelwise view selection for unstructured multi-view stereo." ECCV.*

**Weakness G-2 — Pair selection by consecutive index, not covisibility**  
Severity: 🔴 Critical | Fixability: Moderate  
Stereo pairs are selected as `cam_list[r+1 : r+1+max_pairs]` — consecutive sorted indices. But image indices are assigned by filename sort order, not by camera spatial proximity. For unordered collections (common in photogrammetry), this produces pairs with zero or near-zero overlap. The stereo pipeline should select pairs from the covisibility graph computed in IncrementalSfM, which already knows which cameras share visible 3D points.

**Weakness G-3 — `max_pairs_per_image=2` produces very sparse coverage**  
Severity: 🟠 High | Fixability: Easy  
2 neighbors per camera is extremely conservative. COLMAP uses up to 10 neighbors per camera in its MVS stage. On a 20-camera scene this means at most 40 stereo pairs, many of which may have insufficient baseline. Increasing to 5–10 with covisibility-based selection would dramatically improve dense coverage.

**Weakness G-4 — No geometric consistency check**  
Severity: 🟠 High | Fixability: Hard  
COLMAP PatchMatch filters dense points by requiring geometric consistency: a point reprojected from depth map A should produce a consistent depth in depth map B. Without this, depth discontinuities and low-texture regions produce large numbers of erroneous dense points. SGBM has a `disp12MaxDiff=1` parameter that provides forward-backward consistency within a pair, but no cross-pair consistency.

**Weakness G-5 — SGBM parameters are hard-coded**  
Severity: 🟡 Medium | Fixability: Easy  
`numDisparities=256`, `blockSize=5`, `P1/P2 = 8/32 × 3 × bs²` are hard-coded. No `--dense-num-disparities`, `--dense-block-size` CLI arguments exist. For scenes with large depth ranges (outdoor landscapes) `numDisparities=256` at full resolution may not cover the full disparity range.

---

### Stage H — Mesh Reconstruction (`sfm/mesh/`)

**What it does:** SOR + ROR + normal estimation + voxel downsample → Poisson/BPA/Alpha shapes → component cleaning + hole filling → Taubin smooth + decimation + color transfer → OBJ/PLY/GLB/STL export.

**Weakness H-1 — Normal orientation uses tangent-plane heuristic, not camera positions**  
Severity: 🟠 High | Fixability: Moderate  
`pointcloud_prep.py:119`: `orient_normals_consistent_tangent_plane(k=15)` is used because camera positions are not passed into the mesh pipeline. However, the pipeline has access to camera centers from IncrementalSfM. Using `orient_normals_towards_camera_location` with the actual camera centers would produce consistently outward-facing normals on objects, dramatically improving Poisson surface quality. The fix requires threading camera center data from the pipeline into MeshPipeline.run().

**Weakness H-2 — Poisson depth range fixed to preset, not adaptive**  
Severity: 🟡 Medium | Fixability: Easy  
`QUALITY_PRESETS["medium"]["poisson_depth"] = 9` is fixed regardless of point cloud density. A small point cloud (< 10k points) doesn't benefit from depth=9; a very dense cloud (> 1M) could use depth=11 or 12. Auto-scaling depth based on point count: `depth = max(8, min(11, round(log2(sqrt(n_points / 1000)) + 8)))` would be more adaptive.

**Weakness H-3 — Alpha shape factor hardcoded to preset depth**  
Severity: 🟡 Medium | Fixability: Easy  
`reconstruction.py:195`: `alpha_factors = {8: 0.05, 9: 0.03, ...}` — alpha is derived from the Poisson preset depth. This is a reasonable heuristic but the alpha shape method is rarely the best choice and the factors need tuning per-dataset. The `--mesh-depth` override currently only applies to Poisson; it should also affect alpha computation.

**Weakness H-4 — No UV texture mapping**  
Severity: 🟡 Medium | Fixability: Hard  
Vertex colors from nearest-neighbor point cloud lookup are used (postprocess.py). This produces visually noisy coloring compared to UV-mapped texture atlases used by Metashape, RC, and Meshroom. Proper texture mapping requires:
1. UV parameterization of the mesh
2. Per-face image selection (best viewing angle)
3. Multi-band blending (Burt & Adelson, 1983)

**Weakness H-5 — Hole filling uses open3d built-in with fixed size**  
Severity: 🟡 Medium | Fixability: Easy  
`cleaning.py:109`: `hole_size = max_extent × 0.1`. For small objects, this may fill legitimate through-holes. For large objects, holes larger than 10% of the bounding box are left open. A user-configurable `--mesh-max-hole-size` (as fraction of bounding box diagonal) would improve control.

---

### Stage I — CLI / Entry Point (`run_sfm.py`)

**What it does:** argparse parser (≈40 arguments), input validation, checkpoint load/save, full Python SfM pipeline orchestration with per-stage timing and logging.

**Weakness I-1 — No progress bars**  
Severity: 🟢 Low | Fixability: Easy  
`extract_all` logs progress at 10% intervals. For 1000-image datasets, users see no feedback for minutes. `tqdm` integration would improve UX significantly.

**Weakness I-2 — Checkpoint hashes filename+size but not content**  
Severity: 🟡 Medium | Fixability: Easy  
`_image_set_hash` hashes `sorted(str(path) for path in image_paths)` + file sizes. File modification time is not included. If an image is replaced with another of the same name and size, the stale checkpoint is loaded. Adding `os.path.getmtime()` to the hash would prevent this.

**Weakness I-3 — Feature checkpoint stores full numpy arrays uncompressed**  
Severity: 🟡 Medium | Fixability: Easy  
Features are pickled at `HIGHEST_PROTOCOL`. For 1000 images × 8000 keypoints × (2+128) float32 values = ~4 GB uncompressed. `numpy.savez_compressed` with a manifest file would reduce checkpoint size by 10–20× and load times proportionally.

**Weakness I-4 — No resume support for geometric verification or reconstruction stages**  
Severity: 🟡 Medium | Fixability: Moderate  
Only features and matches are checkpointed. Geometric verification (fast, ~30s) and reconstruction (can be hours for large datasets) have no checkpoints. Adding checkpoints for `verified` and `(cameras, points_3d, observations, kp_to_3d)` would allow full pipeline resume.

**Weakness I-5 — Argparse `dest` uses hyphens inconsistently**  
Severity: 🟢 Low | Fixability: Easy  
`--sift-contrast-threshold` is accessed as `args.sift_contrast_threshold` (argparse converts hyphens to underscores), which is correct. But `--mesh-fill-holes` with `default=True` and no `store_false` counterpart means it cannot be disabled via CLI — only `--mesh-no-clean` exists for cleaning, but `--mesh-no-fill-holes` does not exist. The argument `--mesh-fill-holes` with `action="store_true"` and `default=True` is a no-op; only `store_false` under a `--mesh-no-fill-holes` name makes semantic sense.

---

### Stage J — Visualizer (`sfm/visualizer.py`)

**What it does:** Event-driven hook system; 1587 lines; renders features/matching/reconstruction/pointcloud/mesh/summary figures; optional video and interactive viewer.

**Weakness J-1 — `_render_reprojection_errors` creates one arrow per observation in Python loop**  
Severity: 🟡 Medium | Fixability: Moderate  
`visualizer.py:1067`: each observation triggers an `ax.annotate(arrowprops=...)` call. For an image with 500 observations this produces 500 matplotlib annotation objects, taking several seconds per image. Replacing with `ax.quiver` for vectorized arrow plotting would be significantly faster.

**Weakness J-2 — No coverage map / heatmap for unregistered cameras**  
Severity: 🟢 Low | Fixability: Easy  
`_render_camera_poses` only shows registered cameras. Unregistered cameras (failed PnP) are not visualized, making it hard to diagnose which images are causing registration failures. Showing failed cameras in red would directly aid debugging.

**Weakness J-3 — `_save_reconstruction_video` reads PNG frames from disk**  
Severity: 🟢 Low | Fixability: Easy  
PNG frames are written during registration steps, then re-read for video assembly. This requires all frames to remain on disk. Accumulating frames in memory (or writing directly to GIF via imageio) would avoid the double I/O.

---

### Stage K — Infrastructure (`sfm/device.py`, `sfm/utils.py`, `sfm/colmap_backend.py`)

**Weakness K-1 — COLMAP binary reader `_read_images_bin` is incorrect (dead code)**  
Severity: 🟠 High | Fixability: Easy  
`colmap_backend.py:88`: `_read_images_bin` allocates `xys` and `point3d_ids` arrays but fills them with zeros/`-1` without actually parsing the interleaved binary layout. It is dead code — the pipeline uses `_read_images_bin_correct` instead. The incorrect function should be removed to avoid confusion or accidental use.

**Weakness K-2 — `estimate_intrinsics` only reads `FocalLengthIn35mmFilm`**  
Severity: 🟡 Medium | Fixability: Moderate  
Tag `0xA405` (`FocalLengthIn35mmFilm`) is often missing on RAW-converted images, phone photos without GPS, and some DSLR files. The actual focal length (`0x920A`, `FocalLength`) combined with sensor size from `0xA002/A003` (image dimensions) and a lookup table of known sensor sizes would be more robust. Tools like ExifTool and piexif provide richer EXIF access.

**Weakness K-3 — `has_gpu()` in `colmap_backend.py` is a local duplicate**  
Severity: 🟢 Low | Fixability: Easy  
`colmap_backend.py:567–572` defines a module-level `_has_gpu()` that duplicates `sfm.device.has_gpu()`. Should import from `sfm.device`.

**Weakness K-4 — No graceful handling of COLMAP workspace permission errors**  
Severity: 🟢 Low | Fixability: Easy  
`ColmapRunner._clean_workspace` silently catches `Exception` on `shutil.rmtree` failure. This can leave stale workspaces that grow over repeated runs. At minimum, the path should be logged at WARNING level.

---

## 5. Comparative Scoring Matrix

Scores are 1–10 (10 = best-in-class). Dimensions are weighted by practical importance.

| Dimension | Weight | **This Repo** | COLMAP | RealityCapture | Metashape | OpenSfM | Meshroom | GLOMAP | VGGSfM |
|---|---|---|---|---|---|---|---|---|---|
| **Feature Detection Quality** | 15% | 5 | 8 | 9 | 9 | 8 | 8 | 8 | 9 |
| **Matching Scalability** | 10% | 5 | 9 | 9 | 9 | 8 | 8 | 9 | 7 |
| **Geometric Verification** | 10% | 7 | 9 | 9 | 9 | 8 | 8 | 9 | 9 |
| **Reconstruction Quality** | 20% | 6 | 9 | 10 | 9 | 7 | 8 | 9 | 9 |
| **Bundle Adjustment** | 15% | 5 | 9 | 10 | 10 | 7 | 8 | 9 | N/A |
| **Dense Reconstruction** | 10% | 3 | 8 | 10 | 9 | 5 | 8 | 6 | N/A |
| **Mesh / Surface Quality** | 5% | 5 | 7 | 10 | 9 | 3 | 7 | 5 | N/A |
| **Large-Scale Robustness** | 5% | 4 | 8 | 9 | 9 | 9 | 8 | 10 | 5 |
| **Code Readability** | 5% | 9 | 7 | N/A | N/A | 8 | 7 | 7 | 7 |
| **Ease of Installation** | 5% | 9 | 5 | 3 | 4 | 7 | 6 | 6 | 7 |
| **Weighted Total** | | **5.4** | **8.5** | **9.4** | **9.2** | **7.3** | **7.9** | **8.6** | **7.8*** |

*VGGSfM weighted total excludes dense/mesh dimensions (marked N/A).

### Interpretation

- **Feature Detection (5/10):** Classical SIFT vs. learned SuperPoint/DISK in competitors. Gap most visible on textureless, repetitive, or low-light imagery.
- **Matching Scalability (5/10):** 256-word vocabulary tree is non-functional above ~30 images. Competitors use 65k–1M word hierarchical trees or attention-based matching.
- **Geometric Verification (7/10):** Hartley + USAC_MAGSAC is genuinely SotA for classical pipelines. The gap comes from missing degeneracy detection and guided matching.
- **Reconstruction Quality (6/10):** Incremental SfM with EPnP + adaptive Huber BA is competitive for small datasets (< 50 images). Gap grows with N due to missing local BA.
- **Bundle Adjustment (5/10):** scipy TRF with a single shared focal + k1/k2 falls behind Ceres (speed) and COLMAP's per-camera model (accuracy). No cx/cy refinement.
- **Dense Reconstruction (3/10):** SGBM on consecutive pairs is a placeholder. No multi-view consistency, no GPU PatchMatch, no neural MVS. This is the largest absolute quality gap.
- **Mesh Quality (5/10):** Open3d Screened Poisson with vertex-color transfer is usable but no UV atlas, no multi-band blending. Comparable to running open3d standalone on a COLMAP cloud.
- **Code Readability (9/10):** Best-in-class for open-source SfM. Clear module separation, docstrings, no circular imports, consistent conventions.

---

## 6. Fixability / ROI Analysis

### Effort × Impact Quadrant

```
            HIGH IMPACT
                 │
    ┌────────────┼────────────────────────────────┐
    │  QUICK WINS│ STRATEGIC INVESTMENTS          │
    │            │                                │
    │ cx/cy in BA│ SuperPoint+LightGlue           │
    │ SIFT 0.02  │ Local BA window                │
    │ Guided     │ Covisibility-based MVS         │
    │  matching  │ per-camera intrinsics          │
    │ Better     │ Ceres BA (or pyceres)          │
    │  vocab tree│                                │
    │            │                                │
    ├────────────┼────────────────────────────────┤
    │  POLISH    │ AVOID NOW                      │
    │            │                                │
    │ GPU cross  │ Full neural rendering pipeline │
    │  check vec.│ (3DGS, NeRF)                  │
    │ Progress   │ Full deep SfM (DUSt3R)        │
    │  bars      │ Distributed reconstruction     │
    │ K-3 dedup  │ Real-time mode                 │
    │ tqdm       │                                │
    └────────────┼────────────────────────────────┘
     LOW EFFORT  │  HIGH EFFORT
                 │
            LOW IMPACT
```

### Top-10 items by salvageability score (impact × (1 / effort))

| Rank | Fix | Impact | Effort | Score | Risk |
|---|---|---|---|---|---|
| 1 | SIFT `contrastThreshold` 0.04 → 0.02 | +30% features | 1 line | ∞ | Zero |
| 2 | Add cx/cy to BA parameter vector | -0.5 px systematic bias | 10 lines | Very High | Low |
| 3 | Covisibility-based MVS pair selection | Dense quality: 3→6 | 30 lines | High | Low |
| 4 | Remove incorrect `_read_images_bin` | Correctness | 5 lines | Very High | Zero |
| 5 | `--mesh-fill-holes` argparse fix | Correctness | 3 lines | Very High | Zero |
| 6 | Vocab tree: 256 → 4096+ words | Retrieval: 5→7 | 5 lines | Very High | Low |
| 7 | Guided matching (2nd pass with F) | Inlier count +50% | 1 week | High | Medium |
| 8 | SuperPoint + LightGlue backend | Quality: 5→8 | 2 weeks | High | Medium |
| 9 | cx/cy joint refinement in BA | Accuracy on real cameras | 1 week | High | Low |
| 10 | Local BA window | Quality: 6→8 at N>50 | 3 weeks | High | Medium |

### Salvageability assessment by component

| Component | Current Quality | Salvageable to | Effort Required |
|---|---|---|---|
| Feature extraction | 5/10 | 8/10 | Replace SIFT → SuperPoint (2 weeks) |
| Feature matching | 5/10 | 9/10 | Replace FLANN → LightGlue (3 weeks) |
| Geometric verification | 7/10 | 8/10 | Add DEGENSAC + guided matching (1 week) |
| Incremental SfM | 6/10 | 8/10 | Local BA + better seed + track merge (6 weeks) |
| Bundle adjustment | 5/10 | 7/10 | Add cx/cy, per-camera models (3 weeks) |
| Dense reconstruction | 3/10 | 7/10 | Full rewrite or OpenMVS binding (4 weeks) |
| Mesh reconstruction | 5/10 | 7/10 | Camera-oriented normals + UV atlas (3 weeks) |

---

## 7. Production System Insights

### 7.1 COLMAP (Schönberger & Frahm, 2016 CVPR; Schönberger et al., 2016 ECCV)

**Why it works at production scale:**

1. **Guided matching is architecturally central.** After a first coarse match pass, COLMAP re-runs matching with the F-matrix constraint, checking only matches near the epipolar line. This roughly doubles effective inlier count at no cost to precision. This single feature accounts for a significant fraction of COLMAP's quality advantage over naive pipelines.

2. **Local BA + global BA interleaving.** COLMAP runs local BA (6-image window) after every camera registration, preventing error accumulation between global BA rounds. This is the reason COLMAP scales to 10k+ cameras while incremental pipelines without local BA degrade badly above ~200.

3. **Observation graph deduplication.** COLMAP maintains a track graph (point3D → set of (image, keypoint) observations) and actively merges tracks that should be the same 3D point. The `CompleteAndMergeTracks` operation reduces point count by ~15–30% and dramatically improves BA conditioning.

4. **Per-image intrinsics model.** The `--ImageReader.single_camera 1` flag forces one camera model, but by default COLMAP assigns per-image models. This matters enormously for datasets with mixed cameras or zoom lenses.

5. **Robust seed selection.** COLMAP's `InitializeReconstruction` checks median triangulation angle, baseline/depth ratio, and homography inlier fraction before accepting a seed pair. Degenerate seeds (frontoparallel, planar scenes) are systematically avoided.

### 7.2 Reality Capture (Capturing Reality / Epic Games)

1. **Graph cut-based scene partitioning.** RC partitions large scenes into overlapping tiles using a graph-cut on the image connectivity graph, processes each tile with full BA, then merges tiles via a global alignment step. This achieves near-linear scaling and routinely processes 50k+ images.

2. **Photogrammetric lens model.** RC supports every known camera model (including fisheye, omnidirectional, frame cameras with full Brown–Conrady) with a rich per-camera calibration. Automatic model selection based on residual statistics.

3. **Hierarchical PatchMatch MVS.** RC's MVS engine operates at multiple resolution levels, using coarse depth estimates to seed fine-level PatchMatch. The coarse-to-fine approach dramatically reduces the search space and avoids local minima.

4. **UV atlas generation with multi-band blending.** RC automatically generates UV-parameterized meshes with seamless texture atlases using Laplacian pyramid blending, producing smooth, photorealistic textures without visible seams.

5. **Laser scan / point cloud alignment.** RC integrates photogrammetry with lidar scans via ICP alignment, enabling hybrid reconstruction that leverages the accuracy of structured light while filling gaps with photogrammetry.

### 7.3 Metashape (Agisoft)

1. **Multi-stage alignment with camera groups.** Metashape supports "camera groups" (fixed relative poses, e.g., stereo rigs) and "ground control points" for absolute orientation. This enables integration of GPS, total-station, and RTK measurements directly into BA.

2. **Adaptive reprojection error thresholding.** Metashape's `gradual selection` workflow uses the per-point reprojection error distribution to progressively remove the worst points and re-run BA, iteratively tightening quality. The pipeline converges to a tighter error distribution than a single-pass outlier removal.

3. **Depth map fusion with confidence weighting.** Metashape's depth map generation assigns per-pixel confidence scores, and the fusion step weights contributions by confidence. Low-confidence regions (texture-poor surfaces, specular highlights) contribute less to the final cloud.

4. **Built-in processing reports.** Metashape generates PDF quality reports with per-camera coverage, overlap maps, reprojection error histograms, and GCP residuals — critical for photogrammetric deliverables in surveying and AEC industries.

5. **Network processing (cluster distribution).** Metashape supports multi-node cluster processing for very large projects, distributing tile processing across a network.

### 7.4 OpenSfM (Mapillary / Meta)

1. **Modular pipeline configuration via YAML.** Every algorithm choice in OpenSfM is a YAML-configurable parameter with sensible defaults. This makes ablation studies and production tuning straightforward without code changes. Lesson: expose more of this repo's parameters as a config file rather than CLI-only.

2. **GPS-constrained bundle adjustment.** OpenSfM integrates GPS priors as soft constraints in BA with a configurable weight. On geotagged street-level imagery this dramatically reduces drift in large scenes where visual connectivity is weak at scale transitions.

3. **Multi-model reconstruction.** OpenSfM's `reconstruction.py` supports simultaneous tracking of multiple disconnected reconstruction components that are later merged, rather than failing when connectivity breaks.

4. **Undistorted image export for downstream processing.** OpenSfM explicitly exports undistorted images at each scale, making it straightforward to feed into external MVS tools (OpenMVS, PMVS) without re-implementing undistortion.

5. **Dense matching using multiple view candidates.** OpenSfM selects depth map candidates based on overlap score and angular diversity, not just sequential order — a direct analogue to the covisibility-based selection gap identified in Stage G.

### 7.5 Meshroom (AliceVision)

1. **Node-based dataflow graph.** Meshroom exposes every stage as a node in a directed acyclic graph (DAG) with typed ports. Users can replace individual stages (e.g., swap feature extractor) without modifying pipeline code. This is architecturally superior to the monolithic `run_sfm.py` orchestration.

2. **Alembic-format scene graph export.** Meshroom uses Alembic (`.abc`) as its intermediate format, enabling direct integration with Blender, Maya, and Houdini pipelines. This is particularly valuable for VFX workflows.

3. **Depth map denoising (bilateral median filter).** Before fusion, Meshroom applies a bilateral median filter to depth maps to remove outlier pixels while preserving depth discontinuities. The denoised depth maps produce cleaner surface normals in the fused cloud.

4. **Separate meshing and texturing nodes.** Meshroom's meshing (Delaunay + visibility carving) and texturing nodes are independent and both expose quality parameters. This allows users to re-texture a mesh at different resolutions without re-running the expensive MVS step.

5. **Multi-view stereo visibility score for view selection.** Meshroom uses a score combining angular diversity, photometric overlap, and estimated depth reliability to select the best 8–12 views per depth map — a key quality factor that this repo's consecutive-pair selection completely lacks.

---

## 8. Research Frontier Applicability

### 8.1 DUSt3R / MASt3R (Wang et al., 2024 CVPR; Leroy et al., 2024 ECCV)

**What it does:** A vision transformer trained to produce pairwise 3D pointmaps directly from image pairs, without any explicit feature matching. Global alignment via optimization over the pointmap graph produces a full 3D reconstruction.

**Applicability to this repo:** DUSt3R could replace **Stages B, C, D** entirely. Given 2–50 images, DUSt3R produces a reconstruction in seconds with no RANSAC, no feature matching, and competitive accuracy on standard benchmarks. The limitation is that DUSt3R's accuracy degrades significantly above ~50 images (the pointmap graph optimization becomes computationally expensive), and it requires ~24 GB VRAM for large batches.

**Integration path:** Implement a `--backend dust3r` option in `run_sfm.py` that routes through DUSt3R → returns (cameras, points_3d) in the same format as IncrementalSfM. The downstream stages (colorization, dense, mesh, viz) would work unchanged.

**Risk:** DUSt3R is non-deterministic, not interpretable, and produces mediocre results on textureless industrial scenes where classical SIFT+COLMAP still excels. It should be an alternative backend, not a replacement.

### 8.2 3D Gaussian Splatting (Kerbl et al., 2023, SIGGRAPH)

**What it does:** Represents a scene as millions of 3D Gaussian primitives with position, covariance, opacity, and spherical harmonic color coefficients. Rasterizes in real-time at high quality.

**Applicability to this repo:** 3DGS requires a sparse SfM point cloud as initialization and camera poses as input — exactly what this pipeline produces. Adding a `--export-3dgs` flag that writes cameras + sparse cloud in the format expected by the canonical 3DGS trainer (gaussian-splatting GitHub) would make this repo a front-end for the entire Gaussian Splatting ecosystem. The implementation effort is ~100 lines (camera format conversion + COLMAP-compatible output).

**Impact:** This would dramatically increase the practical value of the pipeline, since 3DGS outputs are the most-requested deliverable format in the 3D content creation community as of 2024–2025.

### 8.3 GLOMAP (Pan et al., 2024, CVPR)

**What it does:** Global SfM with constrained 1-D radial cameras for rotation averaging; resolves the scale-drift problem of incremental SfM for very large scenes. Claims 10–100× speed over COLMAP with competitive quality.

**Applicability:** GLOMAP requires the same input as this pipeline (feature tracks + pairwise matches). A `--backend glomap` option would be straightforward if GLOMAP is installed, using the existing matching output to seed GLOMAP's mapper. For scenes > 500 images, GLOMAP would produce significantly better results than this pipeline's incremental SfM.

### 8.4 LightGlue (Lindenberger et al., 2023, ICCV)

**Current gap:** This repo uses FLANN + ratio test. LightGlue provides an off-the-shelf drop-in replacement for the matching step.

**Applicability:** LightGlue is released under the Apache 2.0 license, has a Python API, and supports SuperPoint, DISK, ALIKED, and SIFT as input descriptors. The `hloc` library (hierarchical localization) provides a standardized interface. The integration path is:
1. Add `from lightglue import LightGlue, SuperPoint` to `feature_extraction.py` / `feature_matching.py`
2. Implement `LightGlueMatcher` as a fourth matcher class
3. Expose via `--match_strategy lightglue` CLI option

The most important near-term improvement available to this codebase.

### 8.5 VGGSfM (Wang et al., 2024, CVPR)

**What it does:** Differentiable end-to-end SfM in PyTorch. Given a small image set, VGGSfM predicts camera poses, track predictions, and 3D points jointly via a transformer decoder.

**Applicability:** VGGSfM is competitive on datasets it was trained on (CO3D, RealEstate10K) but generalizes poorly to industrial, aerial, or fine-structure scenarios. Best used as a comparison baseline or an alternative backend for object-centric reconstruction (< 20 images, known object category).

### 8.6 Generalized Camera Models

This repo assumes a standard pinhole model. For drone imagery (fisheye lenses), 360° cameras (equirectangular), or underwater cameras (refraction), generalized camera models are necessary. Adding a `--camera-model {pinhole,fisheye,equirectangular}` parameter with the corresponding undistortion in `utils.py` and modified projection in BA would extend the pipeline's applicability significantly.

---

## 9. Prioritized 5-Phase Improvement Roadmap

### Phase 0 — Zero-Risk Immediate Wins (≤ 1 week, zero backward-compat risk)

These changes require no structural modifications and have no regression risk.

| # | Change | File | Lines changed | Expected delta |
|---|---|---|---|---|
| 0-A | `contrastThreshold` 0.04 → 0.02 | `feature_extraction.py:71` | 1 | +30–60% keypoints |
| 0-B | Add cx/cy bounds to BA parameter vector | `bundle_adjustment.py:306–312` | 4 | -0.3–0.8 px systematic error |
| 0-C | Remove dead `_read_images_bin` function | `colmap_backend.py:88–141` | −54 | Correctness |
| 0-D | Fix `--mesh-fill-holes` argparse semantics | `run_sfm.py:389–394` | 3 | Correctness |
| 0-E | Vocab tree words: 256 → 4096 (new default) | `run_sfm.py:122` | 1 | Retrieval on medium datasets |
| 0-F | Add `os.path.getmtime()` to image hash | `run_sfm.py:538–542` | 2 | Stale checkpoint prevention |
| 0-G | Apply Hartley normalization to E-matrix inputs | `geometric_verification.py:170–179` | 5 | Better E conditioning |
| 0-H | Covisibility-based MVS pair selection | `mvs.py:111` | 20 | Dense quality on unordered sets |
| 0-I | Remove `_has_gpu()` duplicate in colmap_backend | `colmap_backend.py:567–572` | −6 | Code hygiene |
| 0-J | Add `os.path.getmtime()` to checkpoint hash | `run_sfm.py:537` | 1 | Stale checkpoint prevention |

**Commit message:** `fix: zero-risk improvements — SIFT threshold, cx/cy BA, vocab tree, dead code removal`

---

### Phase 1 — High-Impact Algorithm Upgrades (2–4 weeks, low risk)

#### 1-A: SuperPoint + LightGlue Matcher Backend

Replace FLANN kNN with the SuperPoint+LightGlue pipeline as a fourth matching strategy.

```python
# New class in feature_extraction.py
class SuperPointExtractor:
    def __init__(self, max_num_keypoints=2048, ...):
        from lightglue import SuperPoint
        self._model = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()
    
    def extract(self, image):
        # Returns (N, 2) keypoints + (N, D) descriptors in the same format

# New class in feature_matching.py
class LightGlueMatcher:
    def match_all(self, features_sp, features_lg):
        from lightglue import LightGlue
        self._matcher = LightGlue(features='superpoint').eval().cuda()
        ...
```

Exposed via `--match_strategy lightglue` in `run_sfm.py`. Falls back gracefully to FLANN if `lightglue` is not installed.

**Expected improvement:** +40–80% inlier count on difficult pairs; reconstruction quality on challenging datasets (low overlap, illumination change) approaches COLMAP.

*Reference: Lindenberger et al. (2023) "LightGlue." ICCV. License: Apache 2.0.*

#### 1-B: Guided Matching (Second Pass with Epipolar Constraint)

After `verify_all`, extract the F matrices for verified pairs and re-run matching on the unverified pairs that share image-level visibility with already-verified neighbors, using the F matrix to constrain the search window.

```python
def guided_match_pass(features, verified, all_matches, F_matrices, K, tol=2.0):
    """For each pair (i,j) with F known, re-match using epipolar search."""
```

This follows COLMAP's approach and can increase inlier count on borderline pairs by 50–100%.

#### 1-C: Per-Camera Intrinsics in BA

Extend the BA parameter vector to support per-camera `(f_i, cx_i, cy_i, k1_i, k2_i)` with a shared-intrinsics flag for single-camera datasets.

```python
# bundle_adjustment.py: add --ba-per-camera-intrinsics flag
# Parameter vector extends from [f, k1, k2, pose_0...] 
# to [f_0, cx_0, cy_0, k1_0, k2_0, ..., f_C, pose_0...] when per-camera=True
```

---

### Phase 2 — Structural Improvements (4–8 weeks, medium risk)

#### 2-A: Local Bundle Adjustment Window

After each camera registration, run a windowed BA over the new camera and its k=6 closest covisible cameras. Uses the existing `BundleAdjuster.adjust()` with a filtered observation list.

```python
def _run_local_ba(self, new_img_idx, k=6):
    neighbors = sorted(self.covisibility[new_img_idx], 
                       key=lambda j: len(self.covisibility[new_img_idx] & self.covisibility[j]),
                       reverse=True)[:k]
    local_cams = {new_img_idx: self.cameras[new_img_idx]}
    local_cams.update({j: self.cameras[j] for j in neighbors if j in self.cameras})
    local_obs = [(i, p, x, y) for (i, p, x, y) in self.observations if i in local_cams]
    ...
```

This is the single most impactful structural change for quality on N > 50 images.

#### 2-B: Covisibility-Based MVS with Multi-View Consistency

Replace consecutive-index pair selection in `mvs.py` with covisibility graph-based selection. Add forward-backward disparity consistency check across pairs.

```python
def _select_stereo_pairs(self, cameras, covisibility, max_pairs_per_image=8):
    """Select pairs from covisibility graph, sorted by baseline/depth ratio."""
```

#### 2-C: Track Merging / Deduplication

Implement a post-reconstruction pass that finds 3D point pairs within a threshold distance that share no common observation and merges them, updating the observation list.

#### 2-D: DEGENSAC for Planar Scene Handling

Add a planarity check in `GeometricVerifier.verify_pair`: if the H inlier ratio exceeds 0.6, use the homography decomposition for pose recovery instead of the F→E pathway.

---

### Phase 3 — Production-Grade Features (2–3 months)

#### 3-A: DUSt3R / MASt3R Backend

```python
# run_sfm.py
if args.backend == "dust3r":
    from sfm.dust3r_backend import DUSt3RRunner
    cameras, points_3d = DUSt3RRunner(args).run(image_paths)
    # ... rest of pipeline (colorization, dense, mesh) unchanged
```

Implement `sfm/dust3r_backend.py` that wraps the DUSt3R inference + global alignment and returns cameras/points in the standard format.

#### 3-B: 3DGS Export

```python
# sfm/exporters/gaussian_splatting.py
def export_for_3dgs(cameras, points_3d, colors, image_paths, output_dir):
    """Write COLMAP-compatible sparse/images.bin + sparse/cameras.bin for 3DGS trainer."""
```

Exposed via `--export-3dgs OUTPUT_DIR` in `run_sfm.py`. ~150 lines of format conversion code.

#### 3-C: OpenMVS Integration for Dense Reconstruction

Replace SGBM-based MVS with an optional `--dense-backend openmvs` that invokes OpenMVS via subprocess (similar to the existing COLMAP wrapper pattern).

#### 3-D: UV Texture Atlas

Use `xatlas` (Python bindings available) for UV parameterization of the reconstructed mesh, then project images onto UV space using the camera poses from reconstruction.

---

### Phase 4 — Research Frontier (3+ months)

#### 4-A: Differentiable SfM Mode

Integrate Theseus for differentiable BA, enabling end-to-end training of feature extraction parameters jointly with geometric optimization. This positions the codebase as a research platform for learning-based SfM.

#### 4-B: Global SfM Mode (GLOMAP Integration)

Implement `--backend glomap` that uses the existing matching output to seed GLOMAP's global rotation averaging pipeline. For scenes > 500 images, this would be the recommended backend.

#### 4-C: Incremental 3DGS Training Pipeline

After Phase 3-B, implement an incremental 3DGS update: as new cameras are registered, re-train only the Gaussians in the visible region. This enables a live reconstruction → neural rendering loop.

#### 4-D: Multi-Category Reconstruction (Object vs. Scene Mode)

Implement automatic detection of whether the input is an object-centric set (< 50 images, circular coverage) or a scene-level set (> 100 images, mostly forward motion), and route to different backends: VGGSfM for object-centric, GLOMAP for scene-level.

---

## Appendix: Key References

### Foundational

- Lowe, D. (2004). "Distinctive image features from scale-invariant keypoints." *IJCV 60(2).*
- Fischler, M. & Bolles, R. (1981). "Random sample consensus." *CACM 24(6).*
- Hartley, R. (1997). "In defense of the eight-point algorithm." *IEEE TPAMI 19(6).*
- Triggs, B. et al. (2000). "Bundle adjustment — a modern synthesis." *Vision Algorithms, LNCS.*

### Feature Detection/Description

- DeTone, D. et al. (2018). "SuperPoint: Self-supervised interest point detection and description." *CVPR Workshops.*
- Tyszkiewicz, M. et al. (2020). "DISK: Learning local features with policy gradient." *NeurIPS.*
- Lindenberger, P. et al. (2023). "LightGlue: Local feature matching at light speed." *ICCV.*
- Sun, J. et al. (2021). "LoFTR: Detector-free local feature matching with transformers." *CVPR.*

### Robust Estimation

- Barath, D. & Matas, J. (2018). "MAGSAC: Marginalizing sample consensus." *CVPR.*
- Barath, D. et al. (2020). "MAGSAC++, a fast, reliable and accurate robust estimator." *CVPR.*
- Chum, O. et al. (2003). "Locally optimized RANSAC." *BMVC.*
- Chum, O. et al. (2005). "Two-view geometry estimation unaffected by a dominant plane." *CVPR.*

### SfM Systems

- Schönberger, J. & Frahm, J. (2016). "Structure-from-motion revisited." *CVPR.*
- Schönberger, J. et al. (2016). "Pixelwise view selection for unstructured multi-view stereo." *ECCV.*
- Pan, H. et al. (2024). "GLOMAP: Global structure-from-motion revisited." *CVPR.*
- Wang, S. et al. (2024). "VGGSfM: Visual geometry grounded deep structure from motion." *CVPR.*
- Wang, S. et al. (2024). "DUSt3R: Geometric 3D vision made easy." *CVPR.*
- Leroy, V. et al. (2024). "Grounding image matching in 3D with MASt3R." *ECCV.*

### Bundle Adjustment

- Agarwal, S. et al. (2010). "Ceres solver." https://ceres-solver.org.
- Grassia, F. (1998). "Practical parameterization of rotations using the exponential map." *J. Graphics Tools 3(3).*
- Hampel, F. et al. (1986). "Robust statistics: The approach based on influence functions." *Wiley.*

### Neural Rendering

- Mildenhall, B. et al. (2020). "NeRF: Representing scenes as neural radiance fields." *ECCV.*
- Kerbl, B. et al. (2023). "3D Gaussian splatting for real-time radiance field rendering." *SIGGRAPH.*
- Müller, T. et al. (2022). "Instant neural graphics primitives." *SIGGRAPH.*

### Dense Reconstruction

- Ma, Z. et al. (2021). "Acmh: Multi-scale geometric consistency guided multi-view stereo." *CVPR.*
- Yao, Y. et al. (2018). "MVSNet: Depth inference for unstructured multi-view stereo." *ECCV.*
- Kazhdan, M. & Hoppe, H. (2013). "Screened Poisson surface reconstruction." *ACM TOG 32(3).*
- Bernardini, F. et al. (1999). "The ball-pivoting algorithm for surface reconstruction." *IEEE TVCG 5(4).*
- Edelsbrunner, H. & Mücke, E.P. (1994). "Three-dimensional alpha shapes." *ACM TOG 13(1).*
- Taubin, G. (1995). "A signal processing approach to fair surface design." *SIGGRAPH.*
