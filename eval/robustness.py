#!/usr/bin/env python3
"""Degenerate-input and unavailable-backend behaviour tests.

Each case asserts a *behaviour*, not a result: the pipeline must either succeed
or fail with a clear, actionable message and a non-zero exit code. A traceback,
a hang, or — worst — a confident wrong answer counts as a failure.

Usage
-----
    python eval/robustness.py --image-dir <a-valid-image-dir> \
        --results-dir eval_results [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _run(extra: list[str], image_dir: str, out: Path,
         timeout: float = 900) -> dict:
    cmd = [sys.executable, str(_ROOT / "run_sfm.py"),
           "--image_dir", image_dir, "--output", str(out), *extra]
    try:
        p = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return {"rc": p.returncode, "out": (p.stdout or "") + (p.stderr or ""),
                "timeout": False}
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "", "timeout": True}


def _verdict(res: dict, expect: str, keywords: list[str]) -> dict:
    """
    expect: 'clean_error'  -> non-zero exit, message present, no traceback
            'success'      -> zero exit
    """
    log = res["out"]
    has_tb = "Traceback (most recent call last)" in log
    kw_hit = next((k for k in keywords if k.lower() in log.lower()), None)

    if res["timeout"]:
        return {**res, "verdict": "FAIL", "why": "timed out"}
    if expect == "success":
        ok = res["rc"] == 0
        return {**res, "verdict": "PASS" if ok else "FAIL",
                "why": "exit 0" if ok else f"exit {res['rc']}"}

    ok = res["rc"] not in (0, None) and kw_hit is not None and not has_tb
    why = []
    if res["rc"] in (0, None):
        why.append(f"expected non-zero exit, got {res['rc']}")
    if kw_hit is None:
        why.append(f"no diagnostic matching {keywords}")
    if has_tb:
        why.append("raw traceback leaked to user")
    return {**res, "verdict": "PASS" if ok else "FAIL",
            "why": "; ".join(why) or f"clean error: '{kw_hit}'"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True,
                    help="A directory of real images to derive cases from.")
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    src = Path(args.image_dir)
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in _EXT)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out = results_dir / "_robustness.ply"
        fast = ["--n_features", "2000", "--checkpoint-dir", str(tmp / "ck")]

        # 1 — directory does not exist
        cases["missing_directory"] = _verdict(
            _run(fast, str(tmp / "does_not_exist"), out),
            "clean_error", ["not found", "Image directory"])

        # 2 — empty directory
        (tmp / "empty").mkdir()
        cases["empty_directory"] = _verdict(
            _run(fast, str(tmp / "empty"), out),
            "clean_error", ["Need", "found 0"])

        # 3 — a single image (below the 2-image minimum)
        one = tmp / "one"; one.mkdir()
        shutil.copy2(imgs[0], one / imgs[0].name)
        cases["single_image"] = _verdict(
            _run(fast, str(one), out), "clean_error", ["Need", "at least 2"])

        # 4 — two identical images: zero baseline, must not invent geometry
        dup = tmp / "dup"; dup.mkdir()
        shutil.copy2(imgs[0], dup / "a.png")
        shutil.copy2(imgs[0], dup / "b.png")
        r = _run(fast, str(dup), out)
        cases["identical_images"] = _verdict(
            r, "clean_error",
            ["No pairs", "aborting", "No 3-D points", "degenerate", "baseline"])
        cases["identical_images"]["note"] = (
            "Zero-baseline stereo is unreconstructable; producing a cloud here "
            "would be a silent wrong answer.")

        # 5 — a corrupt file among valid ones
        corrupt = tmp / "corrupt"; corrupt.mkdir()
        for p in imgs[:3]:
            shutil.copy2(p, corrupt / p.name)
        (corrupt / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
        r = _run(fast, str(corrupt), out)
        cases["corrupt_image"] = _verdict(r, "success", [])
        cases["corrupt_image"]["note"] = (
            "Expected: skip the unreadable file with a warning and reconstruct "
            "from the rest.")

        # 6 — non-overlapping images (two halves of unrelated content)
        noov = tmp / "nooverlap"; noov.mkdir()
        import cv2
        im = cv2.imread(str(imgs[0]))
        h, w = im.shape[:2]
        cv2.imwrite(str(noov / "left.png"), im[:, : w // 3])
        cv2.imwrite(str(noov / "right.png"), np.flipud(im[:, 2 * w // 3:]))
        r = _run(fast, str(noov), out)
        cases["no_overlap"] = _verdict(
            r, "clean_error",
            ["No pairs", "aborting", "No 3-D points", "verification"])

        # 7 — backends that are not installed in this environment.
        # Each case needs enough images to actually reach the code path:
        # DINOv2 short-circuits to exhaustive matching when n <= top_k + 1,
        # and PnP only runs once a third camera is registered.
        # Evenly spaced, not the first k: in the Buddha set consecutive file
        # numbers are not consecutive viewpoints, so imgs[:6] barely overlaps
        # and would never register a third camera.
        from make_subset import select

        def _subset(name: str, k: int) -> Path:
            d = tmp / name
            d.mkdir()
            for p in select(src, k):
                shutil.copy2(p, d / p.name)
            return d

        two = _subset("two", 2)
        twelve = _subset("twelve", min(14, len(imgs)))

        for label, flags, kws, where in [
            ("feature_backend_superpoint",
             ["--feature-backend", "superpoint"], ["torch", "kornia", "install"], two),
            ("feature_backend_disk",
             ["--feature-backend", "disk"], ["torch", "kornia", "install"], two),
            ("match_loftr",
             ["--match_strategy", "loftr"], ["torch", "kornia", "install"], two),
            ("retrieval_dinov2",
             ["--retrieval", "dinov2"], ["torch", "install"], twelve),
            ("backend_colmap",
             ["--backend", "colmap"], ["COLMAP executable not found"], two),
        ]:
            cases[label] = _verdict(
                _run([*fast, *flags], str(where), out), "clean_error", kws)
            cases[label]["n_images_used"] = len(list(where.iterdir()))

        # 8 — optional solvers that should degrade gracefully, not crash
        for label, flags, kws in [
            ("ba_backend_pyceres", ["--ba-backend", "pyceres"],
             ["pyceres", "falling back", "fallback", "scipy"]),
            ("pnp_backend_poselib", ["--pnp-backend", "poselib"],
             ["poselib", "falling back", "fallback", "cv2"]),
        ]:
            r = _run([*fast, *flags], str(twelve), out)
            log = r["out"]
            hit = next((k for k in kws if k.lower() in log.lower()), None)
            # These must SUCCEED (documented fallback) and say what they did.
            m = re.search(r"Cameras\s*:\s*(\d+)\s*registered", log)
            n_cams = int(m.group(1)) if m else 0
            reached = n_cams >= 3  # PnP only runs from the third camera on
            ok = r["rc"] == 0 and hit is not None
            cases[label] = {
                **r,
                "verdict": "PASS" if ok else "FAIL",
                "why": (f"exit 0, announced fallback via {hit!r}" if ok
                        else f"exit {r['rc']}, fallback message {hit!r}"),
                "announced_fallback": hit is not None,
                "n_cameras_registered": n_cams,
                "code_path_reached": reached,
            }

    for c in cases.values():
        c["log_tail"] = "\n".join(c.pop("out", "").strip().splitlines()[-12:])

    n_fail = sum(1 for c in cases.values() if c["verdict"] == "FAIL")
    print(f"{len(cases) - n_fail}/{len(cases)} robustness cases PASS\n")
    for name, c in cases.items():
        print(f"  [{c['verdict']}] {name:28s} {c['why']}")

    if args.json:
        Path(args.json).write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
