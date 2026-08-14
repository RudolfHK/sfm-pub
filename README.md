[TOC]

# sfm-pub

Incremental Structure from Motion in Python. Give it a folder of photos; get a
colored 3D point cloud (`.ply`) back. Optional dense reconstruction, surface mesh,
and a full visualization suite.

---

## Quick Start

```bash
git clone https://github.com/rudolfhk/sfm-pub.git
cd sfm-pub
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

sfm --image_dir ./images --output output.ply
```

## Guides

| Document | Contents |
|----------|----------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Installation on Windows and Linux, optional extras, COLMAP setup, GPU setup, troubleshooting |
| [IMAGE_INPUT_GUIDE.md](IMAGE_INPUT_GUIDE.md) | Capture guidelines — overlap, lighting, focus, shooting strategies by subject type |
| [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) | How the pipeline works, full CLI reference, usage examples, tuning tips |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | Reading the diagnostic figures, interpreting reprojection error, green/red flag guide |

## Install Options

| Extra | Installs | Enables |
|-------|---------|---------|
| *(none)* | numpy, scipy, opencv, Pillow | Core pipeline |
| `gpu` | torch, kornia, cupy-cuda12x | CUDA feature detection and matching |
| `viz` | matplotlib, networkx, imageio | `--visualize` flag, diagnostic figures |
| `mesh` | open3d | `--mesh` flag, Poisson/BPA surface reconstruction |
| `all` | gpu + viz + mesh | Everything |

```bash
pip install -e ".[gpu,viz,mesh]"   # install specific extras
pip install -e ".[all]"            # install all extras
```

## Basic Usage

```bash
# Minimal
sfm --image_dir ./photos --output model.ply

# With visualization (saves diagnostic figures to sfm_visualization/)
sfm --image_dir ./photos --output model.ply --visualize

# Dense reconstruction (StereoSGBM)
sfm --image_dir ./photos --output model.ply --dense

# COLMAP sparse reconstruction
sfm --image_dir ./photos --output model.ply --backend colmap

# Full pipeline: dense + mesh + visualization
sfm --image_dir ./photos --output model.ply \
    --dense --mesh --mesh-method poisson --visualize

# Mesh an existing point cloud without re-running SfM
python -m sfm.mesh model_dense.ply -o model_mesh.ply --quality high

# Large dataset (sequential matching + checkpointing)
sfm --image_dir ./photos --output model.ply \
    --match_strategy sequential --checkpoint-dir ./ckpt --resume
```

Run `sfm --help` for all options, or see [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).

## Pipeline Stages

```
images/ → Feature extraction (SIFT) → Pairwise matching (FLANN)
       → Geometric verification (USAC_MAGSAC) → Incremental SfM
       → Bundle adjustment (scipy TRF, Huber) → Sparse PLY
       → [optional] MVS densification (StereoSGBM) → Dense PLY
       → [optional] Mesh reconstruction (Screened Poisson) → OBJ/PLY
```

## Requirements

- Python 3.10+
- COLMAP binary (only for `--backend colmap / colmap-mvs`)
- NVIDIA GPU + CUDA 12.x (only for `pip install -e ".[gpu]"`)

## Testing

```bash
python integration_test.py
# Expected: === INTEGRATION TEST PASSED ===
```
