#!/usr/bin/env python3
"""Build an evenly-spaced N-image subset of a dataset directory.

The Buddha images are ordered around the object, so taking every k-th image
keeps full 360-degree coverage while shrinking the pair count. Files are
hard-linked when possible (instant, no disk cost) and copied otherwise.

Usage
-----
    python eval/make_subset.py SRC_DIR DST_DIR --n 20
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def select(src: Path, n: int) -> list[Path]:
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in _EXT)
    if n >= len(imgs):
        return imgs
    # Evenly spaced indices across the full sequence, endpoints included.
    step = (len(imgs) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [imgs[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    chosen = select(src, args.n)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for p in chosen:
        target = dst / p.name
        try:
            os.link(p, target)
        except OSError:
            shutil.copy2(p, target)

    print(f"{len(chosen)} images -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
