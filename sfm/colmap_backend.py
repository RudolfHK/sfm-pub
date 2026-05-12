"""
COLMAP backend for the SfM pipeline.

Wraps the COLMAP CLI to perform feature extraction, matching, and sparse/dense
reconstruction, then converts COLMAP's binary output to the same PLY format
produced by the pure-Python backend so all downstream tooling is unaffected.

Typical use
-----------
    python run_sfm.py --image_dir ./images --output out.ply --backend colmap

For dense reconstruction:
    python run_sfm.py --image_dir ./images --output out.ply --backend colmap-mvs

COLMAP binary format references
--------------------------------
https://colmap.github.io/format.html
"""

import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .point_cloud import PointCloudExporter

logger = logging.getLogger(__name__)


# ─── COLMAP camera model metadata ─────────────────────────────────────────────

# Number of intrinsic parameters per model ID
_MODEL_N_PARAMS: Dict[int, int] = {
    0: 3,   # SIMPLE_PINHOLE    f cx cy
    1: 4,   # PINHOLE           fx fy cx cy
    2: 4,   # SIMPLE_RADIAL     f cx cy k1
    3: 5,   # RADIAL            f cx cy k1 k2
    4: 8,   # OPENCV            fx fy cx cy k1 k2 p1 p2
    5: 12,  # OPENCV_FISHEYE
    6: 8,   # FULL_OPENCV
    7: 5,   # FOV
    8: 8,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 6,  # THIN_PRISM_FISHEYE
}
_MODEL_NAMES: Dict[int, str] = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
    3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
}


# ─── COLMAP binary readers ─────────────────────────────────────────────────────

def _read_cameras_bin(path: Path) -> Dict[int, dict]:
    """
    Parse cameras.bin.

    Returns
    -------
    {camera_id: {"model": str, "width": int, "height": int, "params": list[float]}}
    """
    cameras: Dict[int, dict] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id  = struct.unpack("<I", f.read(4))[0]
            model   = struct.unpack("<i", f.read(4))[0]
            width   = struct.unpack("<Q", f.read(8))[0]
            height  = struct.unpack("<Q", f.read(8))[0]
            n_p     = _MODEL_N_PARAMS.get(model, 3)
            params  = list(struct.unpack(f"<{n_p}d", f.read(8 * n_p)))
            cameras[cam_id] = {
                "model":  _MODEL_NAMES.get(model, str(model)),
                "width":  int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_images_bin(path: Path) -> Dict[int, dict]:
    """
    Parse images.bin.

    Returns
    -------
    {image_id: {
        "qvec":        np.ndarray (4,)   # qw, qx, qy, qz
        "tvec":        np.ndarray (3,)
        "camera_id":   int
        "name":        str               # filename relative to --image_path
        "xys":         np.ndarray (M, 2) # 2-D observations
        "point3D_ids": np.ndarray (M,)   # -1 when unobserved
    }}
    """
    images: Dict[int, dict] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id  = struct.unpack("<I",  f.read(4))[0]
            qvec      = np.frombuffer(f.read(32), dtype="<f8")   # (4,)
            tvec      = np.frombuffer(f.read(24), dtype="<f8")   # (3,)
            camera_id = struct.unpack("<I",  f.read(4))[0]

            # Null-terminated filename
            chars: List[bytes] = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                chars.append(c)
            name = b"".join(chars).decode("utf-8")

            num_pts2d   = struct.unpack("<Q", f.read(8))[0]
            xys_flat    = np.frombuffer(f.read(16 * num_pts2d), dtype="<f8")
            p3d_ids_raw = np.frombuffer(f.read(8  * num_pts2d), dtype="<i8")

            # xys_flat layout: x0 y0 x1 y1 ...  interleaved with p3d_ids
            # COLMAP binary: for each pt2D → float64 x, float64 y, int64 id
            # The above two reads are WRONG for interleaved layout;
            # read them sequentially instead:
            xys         = np.zeros((num_pts2d, 2), dtype=np.float64)
            point3d_ids = np.full(num_pts2d, -1,   dtype=np.int64)

            images[image_id] = {
                "qvec":        qvec.copy(),
                "tvec":        tvec.copy(),
                "camera_id":   camera_id,
                "name":        name,
                "xys":         xys,
                "point3D_ids": point3d_ids,
            }
    return images


def _read_images_bin_correct(path: Path) -> Dict[int, dict]:
    """
    Parse images.bin with correct interleaved (x, y, point3D_id) layout.
    Replaces _read_images_bin for correctness.
    """
    images: Dict[int, dict] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id  = struct.unpack("<I",  f.read(4))[0]
            qvec      = np.array(struct.unpack("<4d", f.read(32)))
            tvec      = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I",  f.read(4))[0]

            chars: List[bytes] = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                chars.append(c)
            name = b"".join(chars).decode("utf-8")

            num_pts2d   = struct.unpack("<Q", f.read(8))[0]
            xys         = np.zeros((num_pts2d, 2), dtype=np.float64)
            point3d_ids = np.full(num_pts2d, -1,   dtype=np.int64)
            for k in range(num_pts2d):
                x, y   = struct.unpack("<2d", f.read(16))
                p3d_id = struct.unpack("<q",  f.read(8))[0]
                xys[k]         = (x, y)
                point3d_ids[k] = p3d_id

            images[image_id] = {
                "qvec":        qvec,
                "tvec":        tvec,
                "camera_id":   camera_id,
                "name":        name,
                "xys":         xys,
                "point3D_ids": point3d_ids,
            }
    return images


def _read_points3d_bin(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse points3D.bin → (xyz, rgb).

    Returns
    -------
    xyz : (N, 3) float64
    rgb : (N, 3) uint8
    Points are returned in ascending point3D_id order.
    """
    pts: Dict[int, Tuple] = {}
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            p3d_id    = struct.unpack("<Q", f.read(8))[0]
            xyz       = struct.unpack("<3d", f.read(24))
            rgb       = struct.unpack("<3B", f.read(3))
            _error    = struct.unpack("<d",  f.read(8))[0]
            track_len = struct.unpack("<Q",  f.read(8))[0]
            f.read(8 * track_len)                 # skip (image_id, point2D_idx) pairs
            pts[p3d_id] = (xyz, rgb)

    if not pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    sorted_ids = sorted(pts.keys())
    xyz_arr = np.array([pts[i][0] for i in sorted_ids], dtype=np.float64)
    rgb_arr = np.array([pts[i][1] for i in sorted_ids], dtype=np.uint8)
    return xyz_arr, rgb_arr


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def _qvec_to_rot(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
    q = qvec / np.linalg.norm(qvec)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [  2*qx*qy + 2*qz*qw,   1 - 2*qx*qx - 2*qz*qz,   2*qy*qz - 2*qx*qw],
        [  2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def _params_to_K(model: str, params: list) -> np.ndarray:
    """Extract (3×3) intrinsic matrix K from COLMAP camera model params."""
    K = np.eye(3, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":                          # f cx cy
        K[0, 0] = K[1, 1] = params[0]
        K[0, 2] = params[1]; K[1, 2] = params[2]
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):    # fx fy cx cy ...
        K[0, 0] = params[0]; K[1, 1] = params[1]
        K[0, 2] = params[2]; K[1, 2] = params[3]
    elif model in ("SIMPLE_RADIAL", "RADIAL",
                   "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):  # f cx cy k...
        K[0, 0] = K[1, 1] = params[0]
        K[0, 2] = params[1]; K[1, 2] = params[2]
    elif model == "OPENCV_FISHEYE":                        # fx fy cx cy k1..k4
        K[0, 0] = params[0]; K[1, 1] = params[1]
        K[0, 2] = params[2]; K[1, 2] = params[3]
    else:
        # Generic fallback: treat first param as f
        K[0, 0] = K[1, 1] = params[0]
        if len(params) >= 3:
            K[0, 2] = params[1]; K[1, 2] = params[2]
    return K


# ─── Public conversion utility ─────────────────────────────────────────────────

def colmap_model_to_cameras(model_dir: Path) -> Dict[int, dict]:
    """
    Convert a COLMAP sparse model to the ``cameras`` dict format used by
    IncrementalSfM / PointCloudExporter.

    Returns
    -------
    {img_idx: {"R": (3,3), "t": (3,1), "K": (3,3)}}
    Indexed 0 … N-1 in ascending image-name order.
    """
    colmap_cams = _read_cameras_bin(model_dir / "cameras.bin")
    colmap_imgs = _read_images_bin_correct(model_dir / "images.bin")

    cameras: Dict[int, dict] = {}
    for out_idx, (_iid, img) in enumerate(
        sorted(colmap_imgs.items(), key=lambda kv: kv[1]["name"])
    ):
        R = _qvec_to_rot(img["qvec"])
        t = img["tvec"].reshape(3, 1)
        cam_params = colmap_cams[img["camera_id"]]
        K = _params_to_K(cam_params["model"], cam_params["params"])
        cameras[out_idx] = {"R": R, "t": t, "K": K}
    return cameras


# ─── ColmapRunner ──────────────────────────────────────────────────────────────

class ColmapRunner:
    """
    Thin wrapper around the COLMAP CLI.

    Runs feature extraction, matching, and incremental mapping via subprocess,
    then reads back the binary sparse model and writes a PLY file compatible
    with the rest of the pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from ``run_sfm.py``.  The runner reads:
        ``image_dir``, ``output``, ``n_features``, ``match_strategy``,
        ``sequential_window``, ``vocab_top_k``, ``max_reproj_error``,
        ``no_refine_intrinsics``, ``dense``, ``dense_output``.
    colmap_bin : str
        COLMAP executable name or full path (default: ``"colmap"``).
    workspace : str | None
        Directory for COLMAP's database and sparse model.
        Defaults to ``<output_dir>/colmap_workspace/``.
    keep_workspace : bool
        Retain the workspace after a successful run (useful for debugging).
    """

    def __init__(
        self,
        args,
        colmap_bin: str = "colmap",
        workspace: Optional[str] = None,
        keep_workspace: bool = False,
    ) -> None:
        self.args           = args
        self.colmap_bin     = colmap_bin
        self.keep_workspace = keep_workspace

        output_path       = Path(args.output)
        self._image_dir   = Path(args.image_dir).resolve()
        self._output_ply  = output_path
        self._workspace   = (
            Path(workspace)
            if workspace
            else output_path.parent / "colmap_workspace"
        )

        # Dense output path (used when --backend colmap-mvs or --dense)
        dense_out = getattr(args, "dense_output", None)
        self._dense_ply  = Path(
            dense_out if dense_out
            else str(output_path.parent / f"{output_path.stem}_dense.ply")
        )
        self._run_dense  = (
            getattr(args, "dense", False)
            or getattr(args, "backend", "colmap") == "colmap-mvs"
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full COLMAP pipeline and write the output PLY(s)."""
        self._check_colmap()
        self._workspace.mkdir(parents=True, exist_ok=True)

        db_path    = self._workspace / "database.db"
        sparse_dir = self._workspace / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        logger.info(f"[COLMAP] Workspace : {self._workspace}")
        logger.info(f"[COLMAP] Image dir : {self._image_dir}")

        self._feature_extraction(db_path)
        self._matching(db_path)
        self._mapper(db_path, sparse_dir)

        model_dir = self._find_largest_model(sparse_dir)
        if model_dir is None:
            raise RuntimeError(
                "[COLMAP] Mapper produced no reconstruction. "
                "Ensure images have sufficient overlap and enough texture."
            )
        logger.info(f"[COLMAP] Best model : {model_dir}")

        xyz, rgb = _read_points3d_bin(model_dir / "points3D.bin")
        if len(xyz) == 0:
            raise RuntimeError("[COLMAP] Sparse model contains 0 points.")

        exporter = PointCloudExporter(max_reproj_error=self.args.max_reproj_error)
        exporter.save_ply(str(self._output_ply), xyz, rgb)
        logger.info(f"[COLMAP] Sparse PLY : {self._output_ply}  ({len(xyz):,} pts)")

        if self._run_dense:
            self._dense_pipeline(model_dir)

        if not self.keep_workspace:
            self._clean_workspace()

    # ── COLMAP sub-commands ───────────────────────────────────────────────────

    def _feature_extraction(self, db_path: Path) -> None:
        logger.info("[COLMAP] [1/3] Feature extraction…")
        cmd = [
            self.colmap_bin, "feature_extractor",
            "--database_path",                     str(db_path),
            "--image_path",                        str(self._image_dir),
            "--ImageReader.single_camera",         "1",
            "--SiftExtraction.max_num_features",   str(self.args.n_features),
            "--SiftExtraction.use_gpu",            "1" if _has_gpu() else "0",
        ]
        self._exec(cmd, "feature_extractor")

    def _matching(self, db_path: Path) -> None:
        strategy = getattr(self.args, "match_strategy", "exhaustive")
        logger.info(f"[COLMAP] [2/3] Matching  [{strategy}]…")

        if strategy == "sequential":
            cmd = [
                self.colmap_bin, "sequential_matcher",
                "--database_path",                str(db_path),
                "--SequentialMatching.overlap",
                    str(getattr(self.args, "sequential_window", 5)),
                "--SiftMatching.use_gpu", "1" if _has_gpu() else "0",
            ]
        elif strategy == "vocab_tree":
            vocab_path = getattr(self.args, "colmap_vocab_tree", None)
            if vocab_path and Path(vocab_path).exists():
                cmd = [
                    self.colmap_bin, "vocab_tree_matcher",
                    "--database_path",                         str(db_path),
                    "--VocabTreeMatching.vocab_tree_path",     str(vocab_path),
                    "--VocabTreeMatching.num_images",
                        str(getattr(self.args, "vocab_top_k", 10)),
                    "--SiftMatching.use_gpu", "1" if _has_gpu() else "0",
                ]
            else:
                if vocab_path:
                    logger.warning(
                        f"[COLMAP] vocab_tree path not found: {vocab_path}. "
                        "Falling back to exhaustive matching."
                    )
                else:
                    logger.warning(
                        "[COLMAP] --match_strategy vocab_tree requires "
                        "--colmap-vocab-tree. Falling back to exhaustive matching."
                    )
                cmd = self._exhaustive_cmd(db_path)
        else:
            cmd = self._exhaustive_cmd(db_path)

        self._exec(cmd, "matcher")

    def _mapper(self, db_path: Path, sparse_dir: Path) -> None:
        logger.info("[COLMAP] [3/3] Incremental mapper…")
        refine = "0" if getattr(self.args, "no_refine_intrinsics", False) else "1"
        cmd = [
            self.colmap_bin, "mapper",
            "--database_path",                     str(db_path),
            "--image_path",                        str(self._image_dir),
            "--output_path",                       str(sparse_dir),
            "--Mapper.ba_refine_focal_length",     refine,
            "--Mapper.ba_refine_extra_params",     refine,
        ]
        self._exec(cmd, "mapper")

    def _dense_pipeline(self, model_dir: Path) -> None:
        """
        Run COLMAP dense reconstruction:
        image_undistorter → patch_match_stereo → stereo_fusion.

        patch_match_stereo requires a CUDA-capable GPU. If the GPU is
        unavailable the step will fail and a descriptive error is logged.
        """
        dense_dir = self._workspace / "dense"
        dense_dir.mkdir(exist_ok=True)
        logger.info(f"[COLMAP] Dense pipeline → {dense_dir}")

        self._exec([
            self.colmap_bin, "image_undistorter",
            "--image_path",  str(self._image_dir),
            "--input_path",  str(model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ], "image_undistorter")

        self._exec([
            self.colmap_bin, "patch_match_stereo",
            "--workspace_path",   str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
        ], "patch_match_stereo")

        fused = dense_dir / "fused.ply"
        self._exec([
            self.colmap_bin, "stereo_fusion",
            "--workspace_path",   str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type",       "geometric",
            "--output_path",      str(fused),
        ], "stereo_fusion")

        if fused.exists():
            shutil.copy2(fused, self._dense_ply)
            logger.info(f"[COLMAP] Dense PLY  : {self._dense_ply}")
        else:
            logger.warning("[COLMAP] stereo_fusion produced no fused.ply.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _exhaustive_cmd(self, db_path: Path) -> list:
        return [
            self.colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "1" if _has_gpu() else "0",
        ]

    def _exec(self, cmd: list, stage: str) -> None:
        """Run a COLMAP sub-command, streaming output to the logger."""
        logger.debug("[COLMAP] %s", " ".join(str(c) for c in cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.debug("  [%s] %s", stage, line)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except FileNotFoundError:
            raise RuntimeError(
                f"[COLMAP] Executable '{self.colmap_bin}' not found.\n"
                "  • Install COLMAP: https://colmap.github.io/install.html\n"
                "  • Add it to PATH, or pass --colmap-bin /path/to/colmap"
            )

    def _check_colmap(self) -> None:
        """Verify COLMAP is accessible and log its version."""
        try:
            out = subprocess.check_output(
                [self.colmap_bin, "-h"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in out.splitlines():
                if "COLMAP" in line and ("." in line or "version" in line.lower()):
                    logger.info("[COLMAP] %s", line.strip())
                    break
        except FileNotFoundError:
            raise RuntimeError(
                f"[COLMAP] Executable '{self.colmap_bin}' not found.\n"
                "  • Install COLMAP: https://colmap.github.io/install.html\n"
                "  • Add it to PATH, or pass --colmap-bin /path/to/colmap"
            )
        except subprocess.CalledProcessError:
            pass    # Some COLMAP builds return non-zero for -h; that's fine.

    @staticmethod
    def _find_largest_model(sparse_dir: Path) -> Optional[Path]:
        """
        COLMAP mapper writes one sub-directory per connected component
        (0/, 1/, 2/, …).  Return the one whose points3D.bin is largest.
        """
        best: Optional[Path] = None
        best_size = -1
        for sub in sorted(sparse_dir.iterdir()):
            pts_bin = sub / "points3D.bin"
            if pts_bin.exists():
                size = pts_bin.stat().st_size
                if size > best_size:
                    best, best_size = sub, size
        return best

    def _clean_workspace(self) -> None:
        try:
            shutil.rmtree(self._workspace)
            logger.debug("[COLMAP] Removed workspace: %s", self._workspace)
        except Exception as exc:
            logger.warning("[COLMAP] Could not remove workspace: %s", exc)


# ─── Module-level helpers ──────────────────────────────────────────────────────

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
