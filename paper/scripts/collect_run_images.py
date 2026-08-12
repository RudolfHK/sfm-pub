#!/usr/bin/env python3
"""Uebernimmt weitere Diagnosebilder der beiden Visualisierungslaeufe.

Die Verzeichnisse `sfm_visualization_*` stehen in `.gitignore`; das Paper darf
nicht auf sie zeigen. Dieses Skript kopiert die zusaetzlich in der Langfassung
verwendeten Bilder nach `paper/figures/run_a` bzw. `run_b` und schneidet die
Rekonstruktionsschritte auf dieselbe linke Teilansicht zu wie die bereits
vorhandenen Bilder (Kasten 22, 55 bis 685, 1004; per Vorlagensuche bestimmt).

    python paper/scripts/collect_run_images.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_A = REPO / "sfm_visualization_20260802_153526"
RUN_B = REPO / "sfm_visualization_20260803_102114"
OUT_A = REPO / "paper" / "figures" / "run_a"
OUT_B = REPO / "paper" / "figures" / "run_b"

CROP = (22, 55, 685, 1004)   # linke Teilansicht der Schrittbilder

COPY = [
    (RUN_B / "01_features/features_00006._c.png", OUT_B / "abb03d_sift_keypoints_00006.png"),
    (RUN_B / "01_features/density_00006._c.png", OUT_B / "abb03e_feature_density_00006.png"),
    (RUN_B / "02_matching/matches_005_009.png", OUT_B / "abb04d_matches_005_009.png"),
    (RUN_B / "02_matching/epipolar_005_009.png", OUT_B / "abb05b_epipolar_005_009.png"),
    (RUN_B / "03_reconstruction/reprojection_errors_00031._c.png",
     OUT_B / "abb07d_reprojection_errors_00031.png"),
    (RUN_B / "03_reconstruction/reprojection_errors_00062._c.png",
     OUT_B / "abb07e_reprojection_errors_00062.png"),
    (RUN_A / "03_reconstruction/bundle_adjustment_convergence.png",
     OUT_A / "abb13b_ba_convergence_lauf_a.png"),
    (RUN_A / "04_pointcloud/pointcloud_6views.png",
     OUT_A / "abb13c_pointcloud_6views_lauf_a.png"),
    (RUN_A / "02_matching/match_matrix.png", OUT_A / "abb13d_match_matrix_lauf_a.png"),
]

CROPS = [
    (RUN_B / "03_reconstruction/step_003_camera_registered.png",
     OUT_B / "abb06f_step003_crop.png"),
    (RUN_B / "03_reconstruction/step_020_camera_registered.png",
     OUT_B / "abb06g_step020_crop.png"),
    (RUN_B / "03_reconstruction/step_050_camera_registered.png",
     OUT_B / "abb06h_step050_crop.png"),
]


def main() -> int:
    from PIL import Image

    missing = 0
    for src, dst in COPY:
        if not src.is_file():
            print(f"  fehlt: {src}")
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"kopiert: {dst.relative_to(REPO)}")

    for src, dst in CROPS:
        if not src.is_file():
            print(f"  fehlt: {src}")
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.crop(CROP).save(dst)
        print(f"zugeschnitten: {dst.relative_to(REPO)}")

    if missing:
        print(f"  {missing} Quelldatei(en) nicht gefunden")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
