# Windows + VSCode Setup Guide

This guide gets you running the SfM pipeline — including the optional COLMAP
backend — on Windows 10/11 with Visual Studio Code.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Python environment](#2-python-environment)
3. [Install pipeline dependencies](#3-install-pipeline-dependencies)
4. [Install COLMAP (optional)](#4-install-colmap-optional)
5. [VSCode configuration](#5-vscode-configuration)
6. [Quick-start: run the pipeline](#6-quick-start-run-the-pipeline)
7. [GPU acceleration](#7-gpu-acceleration)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Tool | Version | Where to get it |
|------|---------|-----------------|
| Windows | 10 22H2 or 11 | — |
| Python | 3.10 – 3.12 | [python.org/downloads](https://www.python.org/downloads/) |
| Git | any | [git-scm.com](https://git-scm.com) |
| Visual Studio Code | latest | [code.visualstudio.com](https://code.visualstudio.com) |
| COLMAP *(optional)* | 3.8+ | [colmap.github.io](https://colmap.github.io/install.html) |
| CUDA Toolkit *(optional)* | 11.8 or 12.x | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |

> **Python installer tip** — during installation tick both:
> - ✅ Add Python to PATH
> - ✅ Install pip

---

## 2. Python environment

Open a **PowerShell** (or **Command Prompt**) terminal and run:

```powershell
# 1. Clone the repository
git clone https://github.com/yourorg/sfm-pub.git
cd sfm-pub

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1        # PowerShell
# or
.venv\Scripts\activate.bat        # Command Prompt
```

> **Execution-policy error?**  Run this once in an elevated PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## 3. Install pipeline dependencies

```powershell
# Core dependencies (required)
pip install -r requirements.txt

# Visualization dependencies (optional — only needed with --visualize)
pip install -r requirements-viz.txt
```

The `requirements.txt` installs OpenCV, NumPy, SciPy (for bundle adjustment),
and related packages. `requirements-viz.txt` adds matplotlib, networkx,
imageio, and Pillow for the optional visualization layer.

### PyTorch (for GPU feature matching — optional)

If you have an NVIDIA GPU and want GPU-accelerated descriptor matching and
kornia keypoint detection, install PyTorch **before** the requirements above:

```powershell
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify the install:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
# True  ← GPU available
# False ← CPU only (pipeline still works, just slower)
```

---

## 4. Install COLMAP (optional)

Only needed when you use `--backend colmap` or `--backend colmap-mvs`.

### Option A — Pre-built binary (recommended)

1. Go to [github.com/colmap/colmap/releases](https://github.com/colmap/colmap/releases)
2. Download the latest `COLMAP-<version>-windows-cuda.zip` (CUDA GPU)
   or `COLMAP-<version>-windows-no-cuda.zip` (CPU only)
3. Extract to a permanent location, e.g. `C:\Tools\COLMAP`
4. Add the folder to your **PATH**:
   - Win key → "Edit the system environment variables" → Environment Variables
   - Under *User variables*, select **Path** → Edit → New →
     `C:\Tools\COLMAP`
5. Open a new terminal and verify:
   ```powershell
   colmap -h
   # Should print COLMAP version and help text
   ```

### Option B — winget

```powershell
winget install --id COLMAP.COLMAP
```

### Option C — Conda / Miniforge

```powershell
conda install -c conda-forge colmap
```

### Vocabulary tree (vocab_tree matching only)

If you want `--match_strategy vocab_tree` with COLMAP, download a pre-built
vocabulary tree from [demuc.de/colmap/#download](https://demuc.de/colmap/#download)
and point to it with `--colmap-vocab-tree`.

---

## 5. VSCode configuration

### Install extensions

Open VSCode, press `Ctrl+Shift+X`, and install:

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| Python | Microsoft | Linting, IntelliSense, test runner |
| Pylance | Microsoft | Fast type-checking |
| GitLens *(optional)* | GitKraken | Enhanced git history |

### Select the virtual environment interpreter

1. Open the project folder: **File → Open Folder → sfm-pub**
2. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
3. Choose **Enter interpreter path…** → browse to `.venv\Scripts\python.exe`

VSCode will create a `.vscode/settings.json` automatically.

### Recommended `.vscode/settings.json`

Create or update `.vscode/settings.json` with:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.python"
    },
    "terminal.integrated.defaultProfile.windows": "PowerShell"
}
```

### Recommended `.vscode/launch.json` (debug configurations)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "SfM — Python backend",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run_sfm.py",
            "args": [
                "--image_dir", "${workspaceFolder}/images",
                "--output",    "${workspaceFolder}/output.ply",
                "--verbose"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "SfM — COLMAP sparse",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run_sfm.py",
            "args": [
                "--image_dir",  "${workspaceFolder}/images",
                "--output",     "${workspaceFolder}/output.ply",
                "--backend",    "colmap",
                "--verbose"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "SfM — COLMAP + MVS dense",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run_sfm.py",
            "args": [
                "--image_dir",  "${workspaceFolder}/images",
                "--output",     "${workspaceFolder}/output.ply",
                "--backend",    "colmap-mvs",
                "--verbose"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

---

## 6. Quick-start: run the pipeline

All commands assume your virtual environment is active and you are in the
`sfm-pub` directory.

### Pure-Python backend (no COLMAP needed)

```powershell
python run_sfm.py --image_dir .\images --output output.ply --verbose
```

### COLMAP sparse reconstruction

```powershell
python run_sfm.py `
    --image_dir .\images `
    --output    output.ply `
    --backend   colmap `
    --verbose
```

### COLMAP + dense MVS

```powershell
python run_sfm.py `
    --image_dir  .\images `
    --output     output.ply `
    --backend    colmap-mvs `
    --verbose
```

> The dense step (`colmap-mvs`) runs COLMAP's PatchMatch stereo, which
> **requires a CUDA GPU**. Expect several minutes per image pair.

### Sequential matching (large ordered datasets)

```powershell
python run_sfm.py `
    --image_dir      .\images `
    --output         output.ply `
    --match_strategy sequential `
    --backend        colmap
```

### Vocabulary-tree matching with COLMAP

```powershell
python run_sfm.py `
    --image_dir        .\images `
    --output           output.ply `
    --match_strategy   vocab_tree `
    --backend          colmap `
    --colmap-vocab-tree C:\Tools\vocab_tree_flickr100K_words32K.bin
```

### Custom COLMAP binary path

```powershell
python run_sfm.py `
    --image_dir  .\images `
    --output     output.ply `
    --backend    colmap `
    --colmap-bin "C:\Tools\COLMAP\colmap.exe"
```

### Keep the COLMAP workspace for inspection

```powershell
python run_sfm.py `
    --image_dir            .\images `
    --output               output.ply `
    --backend              colmap `
    --colmap-keep-workspace
```

The workspace is written to `<output_dir>/colmap_workspace/` by default, or
override with `--colmap-workspace <dir>`.

---

## 7. GPU acceleration

### Python backend (kornia + torch.cdist matching)

With PyTorch and CUDA installed, the pipeline automatically uses the GPU for:
- Keypoint detection (kornia `ScaleSpaceDetector`)
- Descriptor matching (`torch.cdist` brute-force)
- Vocabulary k-means and TF-histogram encoding

No extra flags needed — GPU detection is automatic.

### COLMAP backend

COLMAP uses its own GPU detection. With the CUDA build installed and a
compatible GPU present, SIFT extraction and PatchMatch stereo run on GPU
automatically.

To force CPU-only COLMAP (slower but works without a GPU):
```powershell
# Download the "no-cuda" COLMAP build from the releases page
```

---

## 8. Troubleshooting

### `colmap` is not recognized

- Verify PATH: `$env:PATH` in PowerShell should contain the COLMAP folder.
- Restart VSCode and your terminal after editing PATH.
- Or pass `--colmap-bin "C:\full\path\to\colmap.exe"` directly.

### `ModuleNotFoundError: No module named 'cv2'`

The virtual environment is not active or `requirements.txt` was not installed:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `torch.cuda.is_available()` returns `False`

1. Confirm your GPU is CUDA-capable (`nvidia-smi` in PowerShell).
2. Ensure the CUDA Toolkit version matches the PyTorch CUDA build.
3. Reinstall PyTorch with the correct `--index-url` (see §3 above).

### COLMAP mapper produces 0 points

Common causes:
- Images have less than ~30% overlap — capture more photos.
- Images are too blurry or textureless.
- The workspace already contains a stale `database.db` — delete
  `colmap_workspace/` or pass `--colmap-workspace` to a fresh directory.

### Dense step fails: "no CUDA device"

`patch_match_stereo` requires a CUDA GPU. Use `--backend colmap` (sparse only)
or the Python `--dense` flag (CPU-based StereoSGBM) instead.

### PowerShell script execution blocked

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Long paths on Windows

If you see path-related errors deep inside COLMAP's workspace, enable long
paths in Windows:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

---

## Viewing PLY output

The output `.ply` files can be opened with:

| Application | Platform | Notes |
|-------------|----------|-------|
| [MeshLab](https://www.meshlab.net) | Windows / macOS / Linux | Free, recommended |
| [CloudCompare](https://cloudcompare.org) | Windows / macOS / Linux | Free, large clouds |
| [open3d](http://www.open3d.org) | Python | `--viz-interactive` flag uses this |
| Blender | Windows / macOS / Linux | Import via *File → Import → Stanford PLY* |
