#!/usr/bin/env python3
"""Prueft die Kennzahlen der Langfassung gegen die Artefakte im Repository.

Jede Zahl, die `paper/paper_l.md` im Fliesstext oder in einer Tabelle nennt und
die aus einem Artefakt stammt, wird hier gegen dieses Artefakt nachgerechnet.
Das Skript ersetzt kein Lesen des Textes, faengt aber jede Zahl ab, die beim
Umschreiben aus dem Tritt geraet.

    python paper/scripts/check_numbers_l.py

Rueckgabewert 0, wenn alle Pruefungen bestehen, sonst 1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper" / "scripts"))

from make_figures_l import parse_log  # noqa: E402


def colmap_points():
    tl, err = [], []
    with open(REPO / "paper_out/colmap_txt/points3D.txt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 9:
                continue
            err.append(float(p[7]))
            tl.append((len(p) - 8) // 2)
    return np.array(tl), np.array(err)


def main() -> int:
    md = (REPO / "paper/paper_l.md").read_text(encoding="utf-8")
    d = parse_log()
    run = json.loads((REPO / "eval_results/n67_base.run.json").read_text("utf-8"))
    s = run["summary"]
    cj = json.loads((REPO / "paper_out/colmap.cameras.json").read_text("utf-8"))
    tl, err = colmap_points()
    focal = json.loads((REPO / "paper/figures/l/focal_experiment.json").read_text("utf-8"))
    sweep = {round(r["f_error_pct"], 1): r for r in focal["sweep"]}
    variants = {r["variant"]: r for r in focal["solver_variants"]}

    checks = [
        # Referenzlauf n67_base
        ("468.252 Merkmale", d["n_keypoints"] == 468252),
        ("6.989 Merkmale je Bild", round(468252 / 67) == 6989),
        ("2.211 geprüfte Paare", d["pairs_tested"] == 2211),
        ("820 Paare nach Ratio-Test", d["pairs_raw"] == 820),
        ("643 verifizierte Paare", d["pairs_verified"] == 643),
        ("Startpaar 2.605 Punkte", d["seed_points"] == 2605),
        ("45.176 triangulierte Punkte", d["pts_before_outlier"] == 45176),
        ("42.502 nach Ausreißerentfernung", d["pts_after_outlier"] == 42502),
        ("39.721 exportierte Punkte", d["pts_exported"] == s["n_points"] == 39721),
        ("2.674 entfernte Punkte", 45176 - 42502 == 2674),
        ("2.781 im Exportfilter", 42502 - 39721 == 2781),
        ("132.637 Beobachtungen im BA", d["ba_obs"][-1] == 132637),
        ("107.140 exportierte Beobachtungen", s["n_observations"] == 107140),
        ("14 BA-Runden", len(d["ba_cams"]) == 14),
        ("BA-RMSE 4,91 auf 9,19 px",
         abs(d["ba_rmse_init"][0] - 4.906) < 5e-3
         and abs(d["ba_rmse_init"][-1] - 9.187) < 5e-3),
        ("BA-Minimum 3,99 px bei 12 Kameras",
         abs(min(d["ba_rmse_init"]) - 3.986) < 5e-3
         and d["ba_cams"][int(np.argmin(d["ba_rmse_init"]))] == 12),
        ("größte BA-Verbesserung 0,021 px",
         abs(max(a - b for a, b in zip(d["ba_rmse_init"], d["ba_rmse_final"]))
             - 0.021) < 5e-4),
        ("Reprojektions-RMSE 2,08 px", abs(s["reprojection"]["rmse_px"] - 2.083) < 5e-3),
        ("Tracklänge 2,70", abs(s["mean_track_length"] - 2.697) < 5e-3),
        ("Laufzeit 1.135,5 s", abs(run["wall_s"] - 1135.5) < 0.1),
        ("Merkmale 41,8 s", abs(s["stage_times_s"]["features"] - 41.8) < 0.05),
        ("Matching 1.036,6 s", abs(s["stage_times_s"]["matching"] - 1036.6) < 0.1),
        ("Verifikation und Rekonstruktion 46,9 s",
         abs(s["stage_times_s"]["verification"]
             + s["stage_times_s"]["reconstruction"] - 46.9) < 0.1),
        ("Spitzenspeicher 1.470 MB", abs(run["peak_rss_mb"] - 1469.7) < 0.1),
        ("135.932 Parameter", 5 + 67 * 6 + 45175 * 3 == 135932),
        ("265.274 Residuen", 2 * 132637 == 265274),
        ("Füllgrad 0,010 %",
         abs(100 * (2 * 132637 * 14) / (265274 * 135932) - 0.0103) < 5e-4),
        # COLMAP
        ("COLMAP 35.538 Punkte", cj["n_points"] == 35538),
        ("COLMAP Tracklänge 4,69", abs(cj["mean_track_length"] - 4.692) < 5e-3),
        ("COLMAP 166.740 Beobachtungen", int(tl.sum()) == 166740),
        ("COLMAP Fehler je Punkt 0,33 px", abs(err.mean() - 0.330) < 5e-4),
        ("COLMAP 1,1 % Zweibildpunkte", abs(100 * (tl == 2).mean() - 1.089) < 5e-3),
        ("COLMAP 36,5 % ab fünf Beobachtungen",
         abs(100 * (tl >= 5).mean() - 36.52) < 0.05),
        ("COLMAP Median der Tracklänge 4", np.median(tl) == 4),
        ("COLMAP Brennweite 1.857,5 px", abs(cj["shared_K"][0][0] - 1857.5) < 0.05),
        ("56 % mehr Beobachtungen", abs(100 * (166740 / 107140 - 1) - 55.6) < 0.5),
        ("11 % weniger Punkte", abs(100 * (1 - 35538 / 39721) - 10.5) < 0.5),
        ("Matching-Verhältnis 4,9", abs(1036.6 / 212.7 - 4.87) < 0.02),
        ("Gesamtverhältnis 3,9", abs(1135.5 / 293.4 - 3.87) < 0.02),
        # Brennweitenexperiment
        ("Experiment: 1.214 Punkte bei wahrer Brennweite",
         sweep[0.0]["n_pts"] == 1214),
        ("Experiment: 790 Punkte bei +47 %", sweep[47.0]["n_pts"] == 790),
        ("Experiment: Rotationsfehler 0,18 Grad bei 0 %",
         abs(sweep[0.0]["rot_err_med_deg"] - 0.18) < 5e-3),
        ("Experiment: RMSE 2,86 px bei 0 %", abs(sweep[0.0]["rmse_px"] - 2.86) < 5e-3),
        ("Experiment: RMSE-Band 5,50 bis 6,39 px",
         abs(min(r["rmse_px"] for k, r in sweep.items() if k != 0.0) - 5.50) < 5e-3
         and abs(max(r["rmse_px"] for r in sweep.values()) - 6.39) < 5e-3),
        ("Experiment: Löser holt auf 6,9 %",
         abs(variants["x_scale='jac' und strenge Toleranzen"]["f_end_err_pct"]
             - 6.89) < 0.02),
        ("Experiment: Rotationsfehler 0,13 Grad",
         abs(variants["x_scale='jac' und strenge Toleranzen"]["rot_err_med_deg"]
             - 0.13) < 5e-3),
        ("Experiment: Voreinstellung bleibt bei 47 %",
         abs(variants["Voreinstellung"]["f_end_err_pct"] - 47.0) < 0.05),
        # Textformalitäten
        ("kein Geviertstrich", md.count("—") == 0),
        ("kein Halbgeviertstrich", md.count("–") == 0),
        ("Abbildungen fortlaufend nummeriert",
         [n for k, n in re.findall(r"\*\*(Fig)\. (\d+):\*\*", md)]
         == [str(i) for i in range(1, 28)]),
    ]

    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  ok   " if ok else "  FEHLER ") + name)
    print(f"\n{len(checks) - len(bad)} von {len(checks)} Prüfungen bestanden.")
    if bad:
        print("nicht bestanden: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
