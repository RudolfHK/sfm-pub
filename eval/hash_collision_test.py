#!/usr/bin/env python3
"""Demonstrate whether the checkpoint hash can miss a changed image set.

`_image_set_hash` in run_sfm.py digests only file *paths* and file *sizes*, not
file contents. This constructs the case that distinguishes "content-aware" from
"metadata-only": two different scenes written as uncompressed BMPs of identical
dimensions, so every filename and every byte count is unchanged while every
pixel differs.

If the resumed run reports a cache hit, the pipeline silently reconstructs scene
A's features against scene B's images.

Usage
-----
    python eval/hash_collision_test.py --image-dir <dir> --work-dir <scratch> \
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
_RE_SKIP_FEAT = re.compile(r"Skipped \(loaded from checkpoint\)")
_RE_STALE = re.compile(r"Checkpoint \S+ is stale")


def _write_bmps(srcs: list[Path], dst: Path, size: tuple[int, int]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(srcs):
        img = cv2.imread(str(p))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"img_{i:03d}.bmp"), img)


def _run(image_dir: Path, out: Path, ckpt: Path, extra: list[str]) -> str:
    cmd = [sys.executable, "-X", "utf8", str(_ROOT / "run_sfm.py"),
           "--image_dir", str(image_dir), "--output", str(out),
           "--checkpoint-dir", str(ckpt), "--n_features", "2000", *extra]
    p = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    src = Path(args.image_dir)
    imgs = sorted(p for p in src.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(imgs) < 8:
        print("need at least 8 source images", file=sys.stderr)
        return 1

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    scene = work / "scene"
    ckpt = work / "ckpt"
    out = work / "out.ply"
    size = (960, 540)

    # Scene A -> populate the checkpoint.
    _write_bmps(imgs[:4], scene, size)
    sizes_a = sorted((p.name, p.stat().st_size) for p in scene.iterdir())
    _run(scene, out, ckpt, [])

    # Scene B -> same filenames, same byte counts, entirely different pixels.
    shutil.rmtree(scene)
    _write_bmps(imgs[4:8], scene, size)
    sizes_b = sorted((p.name, p.stat().st_size) for p in scene.iterdir())

    log = _run(scene, out, ckpt, ["--resume"])
    cache_hit = bool(_RE_SKIP_FEAT.search(log))
    stale = bool(_RE_STALE.search(log))

    res = {
        "identical_names_and_sizes": sizes_a == sizes_b,
        "sizes_scene_a": sizes_a,
        "sizes_scene_b": sizes_b,
        "resume_reported_cache_hit": cache_hit,
        "resume_reported_stale": stale,
        "verdict": (
            "CONFIRMED — stale features reused for a completely different scene"
            if cache_hit and not stale else
            "NOT REPRODUCED — the change was detected"
        ),
    }
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
