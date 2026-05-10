# sfm-pub — Pure-Python Structure from Motion

A complete, incremental SfM pipeline that reconstructs a coloured 3-D point
cloud from an unordered directory of images.  No COLMAP, no OpenSfM — every
stage is implemented from scratch in Python with NumPy / SciPy / OpenCV.

---

## Quick start

```bash
pip install -r requirements.txt
python run_sfm.py --image_dir ./images --output output.ply
```

Open `output.ply` in **MeshLab** or **CloudCompare** to view the result.

---

## Pipeline stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `feature_extraction.py` | SIFT keypoint + descriptor extraction (CPU or GPU) |
| 2 | `feature_matching.py`   | Exhaustive pairwise matching — Lowe's ratio test + cross-check |
| 3 | `geometric_verification.py` | Fundamental / Essential matrix RANSAC → (R, t) per pair |
| 4 | `reconstruction.py`     | Incremental SfM — seed, PnP registration, triangulation |
| 4b| `bundle_adjustment.py`  | Sparse LM bundle adjustment via `scipy.optimize.least_squares` |
| 5 | `point_cloud.py`        | Outlier filtering, bilinear colour sampling, binary PLY export |

---

## CLI reference

```
python run_sfm.py --help

  --image_dir PATH       Input image directory (JPG / PNG)   [required]
  --output PATH          Output .ply file                    [output.ply]
  --n_features INT       SIFT features per image             [8000]
  --ratio FLOAT          Lowe's ratio threshold              [0.75]
  --min_matches INT      Min raw matches to keep a pair      [15]
  --min_inliers INT      Min RANSAC inliers to accept a pair [15]
  --ransac_thr FLOAT     RANSAC pixel threshold              [1.0]
  --max_reproj_error F   Max reprojection error (px)         [4.0]
  --ba_interval INT      BA every N new cameras              [5]
  --no_filter            Skip point-cloud outlier filter
  --verbose              DEBUG-level logging
```

---

## Project layout

```
sfm_project/
├── run_sfm.py                 CLI entry point
├── requirements.txt
├── README.md
└── sfm/
    ├── __init__.py
    ├── utils.py               Shared helpers (logging, I/O, camera maths)
    ├── feature_extraction.py  SIFT extraction (CPU + optional GPU)
    ├── feature_matching.py    Pairwise FLANN matching + ratio/cross-check
    ├── geometric_verification.py  F → E → (R,t) RANSAC
    ├── reconstruction.py      Incremental SfM core
    ├── bundle_adjustment.py   Sparse BA via scipy LM
    └── point_cloud.py         Colourisation + binary PLY export
```

---

## Mathematics

### Camera model

```
x = K [R | t] X
```

* **K** — 3×3 intrinsic matrix, estimated from image dimensions:
  `f = max(W, H)`,  `cx = W/2`,  `cy = H/2`
* **R, t** — extrinsic rotation and translation

### Essential matrix

```
E = K'^T F K
```

Decomposed via SVD into four (R, t) candidates; the cheirality constraint
(points in front of both cameras) selects the unique valid solution.

### Triangulation

Linear DLT via `cv2.triangulatePoints`.  Each new point is validated by:
1. Positive depth in both cameras
2. Bearing angle ≥ 1°
3. Reprojection error < threshold

### Bundle adjustment

Minimises the robust cost

```
∑_{i,j}  ρ( ||π(K, R_j, t_j, X_i) − x_{ij}||² )
```

where ρ is the Huber loss.  Camera poses are parameterised as Rodrigues
axis-angle (6 DOF each) so the Jacobian sparsity can be fully exploited by
`scipy.optimize.least_squares(method='trf', jac_sparsity=…)`.

---

## GPU acceleration

Set `use_cuda=True` in `FeatureExtractor` (or let it auto-detect).  It tries
in order:

1. **kornia** — GPU keypoint detection (requires PyTorch + kornia)
2. **OpenCV CUDA SURF** — requires `opencv-contrib-python` built with CUDA

If neither is available the pipeline silently falls back to CPU SIFT.

---

## Tips

* **Too few points?** Lower `--ratio` (e.g. 0.70) and `--min_inliers` (e.g. 10).
* **Noisy cloud?** Raise `--ransac_thr` to 2.0 or lower `--max_reproj_error`.
* **Slow on large sets?** Reduce `--n_features` or add an image-retrieval
  pre-filter before matching.
* **Intrinsics known?** Modify `estimate_intrinsics()` in `utils.py` to return
  your calibrated K directly.
