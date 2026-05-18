[TOC]

# Setup Guide

Installation and environment configuration for the SfM pipeline.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 recommended |
| Git | 2.x | Any recent version |
| Visual C++ Build Tools | 2019 or later | Windows only — required to compile opencv-python wheels |
| gcc / g++ | 9+ | Linux only — `build-essential` on Debian/Ubuntu |
| CUDA Toolkit | 12.x | Optional — GPU acceleration for feature extraction |
| COLMAP binary | 3.9+ | Optional — `--backend colmap` and `--backend colmap-mvs` |

The core pipeline (pure-Python backend) requires only Python, pip, and a C compiler to build the opencv-python wheels. GPU and COLMAP features are opt-in.

---

## Quick Install

```bash
git clone https://github.com/rudolfhk/sfm-pub.git
cd sfm-pub
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Verify the installation:

```bash
sfm --help
```

You should see the full argument list printed to the terminal. If the `sfm` command is not found, ensure the virtual environment is activated and that `pip install -e .` completed without errors.

---

## Optional Extras

The pipeline ships several optional dependency groups. Install only what you need.

| Extra | Install command | What it enables |
|---|---|---|
| `gpu` | `pip install -e ".[gpu]"` | CUDA feature extraction via kornia and torch, torch.cdist matching |
| `viz` | `pip install -e ".[viz]"` | `--visualize` flag, matplotlib figures, networkx scene graph, imageio GIFs |
| `mesh` | `pip install -e ".[mesh]"` | `--mesh` flag, Screened Poisson / BPA / Alpha surface reconstruction via open3d |
| `dev` | `pip install -e ".[dev]"` | pytest, ruff (linting), black (formatting) |
| `all` | `pip install -e ".[all]"` | gpu + viz + mesh (excludes dev) |

Examples:

```bash
# Visualization only
pip install -e ".[viz]"

# Full feature set
pip install -e ".[all]"

# Development environment
pip install -e ".[all,dev]"
```

---

## Windows Setup

### Step 1 — Install Python

Download Python 3.11 or 3.12 from https://www.python.org/downloads/. During installation:

- Check **Add python.exe to PATH**.
- Check **Install pip**.

Verify in a new Command Prompt or PowerShell window:

```cmd
python --version
pip --version
```

### Step 2 — Install Git

Download Git for Windows from https://git-scm.com/download/win. Use the default installer options. After installation, verify:

```cmd
git --version
```

### Step 3 — Install Visual C++ Build Tools

opencv-python requires compiled extensions. Install the Microsoft C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/. In the Visual Studio Installer, select:

- **Desktop development with C++** workload
- Make sure **MSVC v143** and **Windows 10/11 SDK** are checked

This is a one-time installation and is not required again for future Python packages that have pre-built wheels (most do).

### Step 4 — Clone and Install

Open a Command Prompt or PowerShell window:

```cmd
git clone https://github.com/rudolfhk/sfm-pub.git
cd sfm-pub
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

To install optional extras:

```cmd
pip install -e ".[viz]"
pip install -e ".[mesh]"
pip install -e ".[all]"
```

### Step 5 — Verify

```cmd
sfm --help
```

If `sfm` is not recognized, confirm the virtual environment is active — the prompt should show `(.venv)` at the start.

### VSCode Integration

1. Install the **Python** extension from the Extensions panel.
2. Open the Command Palette (`Ctrl+Shift+P`) and run **Python: Select Interpreter**.
3. Choose the interpreter at `.venv\Scripts\python.exe` in the repository root.
4. Open a new terminal in VSCode — it will automatically activate the virtual environment.

---

## Linux Setup

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git build-essential
```

If `python3.11` is not available in the default repositories, add the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

Clone and install:

```bash
git clone https://github.com/rudolfhk/sfm-pub.git
cd sfm-pub
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Fedora / RHEL / CentOS Stream

```bash
sudo dnf install python3.11 python3-pip git gcc gcc-c++ make
```

Clone and install:

```bash
git clone https://github.com/rudolfhk/sfm-pub.git
cd sfm-pub
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verify

```bash
sfm --help
```

---

## COLMAP Installation

COLMAP is only needed for `--backend colmap` or `--backend colmap-mvs`. The pure-Python backend works without it.

### Method 1 — Prebuilt Binary (Recommended)

Download a prebuilt binary for your platform from https://colmap.github.io/install.html. Extract the archive and either:

- Add the directory containing the `colmap` binary to your `PATH`, or
- Pass the full path at runtime: `--colmap-bin /path/to/colmap`

### Method 2 — Windows winget

```cmd
winget install COLMAP.COLMAP
```

After installation, `colmap` should be available on the PATH. Verify with:

```cmd
colmap --help
```

### Method 3 — conda

```bash
conda install -c conda-forge colmap
```

This installs COLMAP into the active conda environment. Use this method if you are already managing dependencies with conda.

### Verify COLMAP

```bash
colmap --help
```

You should see the COLMAP command listing. If the command is not found, check that the binary directory is on your PATH or use `--colmap-bin` to specify the full path.

### Vocabulary Tree for Vocab-Tree Matching with COLMAP

If you plan to use `--match_strategy vocab_tree` with the COLMAP backend, you need a pre-built COLMAP vocabulary tree file. Download one from https://demuc.de/colmap/#download. The larger files give better retrieval quality:

- `vocab_tree_flickr100K_words32K.bin` — fast, suitable for small datasets
- `vocab_tree_flickr100K_words256K.bin` — good balance for medium datasets
- `vocab_tree_flickr100K_words1M.bin` — best quality for large datasets

Pass the path at runtime:

```bash
sfm --image_dir ./images --output output.ply \
    --backend colmap \
    --match_strategy vocab_tree \
    --colmap-vocab-tree /path/to/vocab_tree_flickr100K_words256K.bin
```

The vocabulary tree is not needed for the pure-Python `vocab_tree` strategy — that builds its own vocabulary at runtime from the dataset's descriptors.

---

## GPU Setup

GPU acceleration uses PyTorch and kornia for feature detection, with cupy for optional CUDA SURF fallback.

### Requirements

- NVIDIA GPU with CUDA compute capability 6.0 or higher
- CUDA Toolkit 12.x installed (https://developer.nvidia.com/cuda-downloads)
- Matching cuDNN (installed automatically with the PyTorch wheels)

### Install the GPU Extra

```bash
pip install -e ".[gpu]"
```

This installs:
- `torch >= 2.0` (CPU+CUDA wheels; PyTorch auto-selects the CUDA build based on the detected toolkit)
- `kornia >= 0.7` (GPU keypoint detection via ScaleSpaceDetector)
- `cupy-cuda12x >= 12.0` (CUDA SURF fallback)

If your CUDA version is not 12.x, replace `cupy-cuda12x` with the matching package name. For example, for CUDA 11.8: `pip install cupy-cuda11x`.

### Verify GPU Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output: the name of your GPU.

When GPU is available, the pipeline logs the selected backend at startup:

```
Feature extraction backend: kornia (GPU keypoints + CPU SIFT descriptors) — detector initialized once on cuda:0
```

If you see `Feature extraction backend: OpenCV SIFT (CPU)` despite installing the GPU extra, CUDA is not accessible. Check the CUDA Toolkit installation and that `nvidia-smi` reports your GPU.

---

## Running the Tests

The integration test exercises the full pipeline (geometric verification, incremental reconstruction, bundle adjustment, and export) with synthetic ground-truth correspondences, bypassing SIFT to make the test fast and deterministic.

```bash
python integration_test.py
```

Expected output on success:

```
=== INTEGRATION TEST PASSED ===
```

The test uses 6 synthetic cameras looking at 250 random 3D points. It passes if at least 4 cameras are registered, at least 50 3D points are reconstructed, and the mean reprojection error is below 2.0 px.

If you have `pytest` installed (via the `dev` extra), you can also run:

```bash
pytest integration_test.py -v
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `cv2.error: SIFT not available` | `opencv-python` installed instead of `opencv-contrib-python` | `pip uninstall opencv-python && pip install opencv-contrib-python` |
| `ModuleNotFoundError: No module named 'open3d'` | `open3d` not installed | `pip install open3d` or `pip install -e ".[mesh]"` |
| `colmap: command not found` | COLMAP binary not on PATH | Add COLMAP to PATH, or use `--colmap-bin /full/path/to/colmap` |
| `No images found in <dir>` | Wrong directory or unsupported extension | Check that `--image_dir` points to a folder containing `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, or `.bmp` files |
| `CUDA out of memory` | GPU memory exhausted during feature extraction | Reduce `--n_features`, or uninstall the `gpu` extra to use CPU SIFT |
| `Reconstruction failed: 0 cameras registered` | Scene graph is disconnected or images have insufficient overlap | See IMAGE_INPUT_GUIDE.md for capture guidelines; try `--visualize` to inspect the scene graph |
| `sfm: command not found` | Virtual environment not activated, or `pip install -e .` not run | Activate the venv (`source .venv/bin/activate` or `.venv\Scripts\activate`), then re-run `pip install -e .` |
| `Need >= 2 images` | Fewer than 2 supported images found | Check file extensions; rename `.JPG` to `.jpg` on case-sensitive filesystems |
| `--match_strategy vocab_tree with the COLMAP backend requires --colmap-vocab-tree` | Vocab tree path not specified for COLMAP vocab tree matching | Download a vocabulary tree file from demuc.de/colmap and pass the path with `--colmap-vocab-tree` |
| `BA: only N valid observations — skipped` | Too few observations to run bundle adjustment | Increase `--n_features` and lower `--sift-contrast-threshold` to get more matches per image |
