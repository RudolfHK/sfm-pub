#!/usr/bin/env python3
"""Prueft die Kennzahlen von `paper/paper_l2.md` gegen die Artefakte.

Jede Zahl, die das Paper nennt und die aus einer Messung stammt, wird hier
gegen das erzeugende Artefakt nachgerechnet.  Das Skript ersetzt kein Lesen des
Textes, faengt aber jede Zahl ab, die beim Umschreiben aus dem Tritt geraet.

    python paper/scripts/check_numbers_l2.py

Rueckgabewert 0, wenn alle Pruefungen bestehen, sonst 1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(path: str):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def gt(row, *path):
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> int:
    md = (REPO / "paper" / "paper_l2.md").read_text(encoding="utf-8")

    abl = {r["config"]: r for r in load("eval_results/fix_impact_n67/fix_impact.json")}
    repro = load("eval_results/repro_n20/reproducibility.json")
    mesh = load("eval_results/mesh_eval.json")
    quality = load("run_l2/mesh_final_quality.json")
    run = load("run_l2/cameras.json")
    planar = load("eval_results/planar_check.json")

    def row(name, *path):
        return gt(abl[name], *path)

    checks = [
        # ── Ablation, Tab. 6 ─────────────────────────────────────────────
        ("vorher: 38.332 Punkte", row("before", "summary", "n_points") == 38332),
        ("vorher: 103.176 Beobachtungen",
         row("before", "summary", "n_observations") == 103176),
        ("vorher: Tracklänge 2,69",
         abs(row("before", "summary", "mean_track_length") - 2.69) < 5e-3),
        ("vorher: RMSE 2,117 px",
         abs(row("before", "summary", "reprojection", "rmse_px") - 2.117) < 5e-4),
        ("vorher: Brennweitenfehler 47,03 %",
         abs(row("before", "gt", "focal", "error_pct") - 47.026) < 5e-3),
        ("vorher: Rotationsfehler 7,756 Grad",
         abs(row("before", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 7.756) < 5e-4),
        ("vorher: Positionsfehler 2,663 %",
         abs(row("before", "gt", "alignment", "lsq",
                 "position_err_pct_of_extent", "median") - 2.663) < 5e-4),
        ("+Brennweite: 55.793 Punkte", row("+focal", "summary", "n_points") == 55793),
        ("+Brennweite: Rotationsfehler 0,159 Grad",
         abs(row("+focal", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 0.159) < 5e-4),
        ("+Brennweite: Brennweitenfehler 0,60 %",
         abs(row("+focal", "gt", "focal", "error_pct") - 0.600) < 5e-3),
        ("+Löser: Rotationsfehler 7,216 Grad",
         abs(row("+ba", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 7.216) < 5e-4),
        ("+Tracks: Tracklänge 2,80",
         abs(row("+tracks", "summary", "mean_track_length") - 2.80) < 5e-3),
        ("+Tracks: Rotationsfehler 8,229 Grad",
         abs(row("+tracks", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 8.229) < 5e-4),
        ("+Schleifen ändert nichts",
         row("+loop", "summary", "n_points") == row("before", "summary", "n_points")
         and row("+loop", "summary", "n_observations")
         == row("before", "summary", "n_observations")),
        ("nachher: 51.444 Punkte", row("after", "summary", "n_points") == 51444),
        ("nachher: 159.478 Beobachtungen",
         row("after", "summary", "n_observations") == 159478),
        ("nachher: Tracklänge 3,10",
         abs(row("after", "summary", "mean_track_length") - 3.10) < 5e-3),
        ("nachher: RMSE 0,655 px",
         abs(row("after", "summary", "reprojection", "rmse_px") - 0.655) < 5e-4),
        ("nachher: Brennweitenfehler 0,59 %",
         abs(row("after", "gt", "focal", "error_pct") - 0.593) < 5e-3),
        ("nachher: Rotationsfehler 0,174 Grad",
         abs(row("after", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 0.174) < 5e-4),
        ("nachher: Positionsfehler 0,057 %",
         abs(row("after", "gt", "alignment", "lsq",
                 "position_err_pct_of_extent", "median") - 0.057) < 5e-4),
        ("Faktor 49 beim Rotationsfehler",
         abs(row("before", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             / row("+focal", "gt", "alignment", "lsq", "rotation_err_deg", "median")
             - 48.8) < 1.0),
        ("Löser kostet das Achtzehnfache",
         abs(gt(abl["+ba"], "summary", "stage_times_s", "reconstruction")
             / gt(abl["before"], "summary", "stage_times_s", "reconstruction")
             - 18.2) < 0.5),

        # ── Reproduzierbarkeit, Tab. 8 ───────────────────────────────────
        ("drei Läufe byte-gleich", repro["identical_ply"] is True),
        ("gleiche Zählungen", repro["identical_counts"] is True),
        ("Wiederholung: 12.491 Punkte",
         all(r["points"] == 12491 for r in repro["runs"])),
        ("Wiederholung: 30.188 Beobachtungen",
         all(r["observations"] == 30188 for r in repro["runs"])),
        ("Wiederholung: RMSE 0,7579 px",
         all(abs(r["rmse_px"] - 0.7579) < 5e-5 for r in repro["runs"])),

        # ── planare Szene, Tab. 11 ───────────────────────────────────────
        ("planar ohne Rückfall verworfen",
         any(r["scene"] == "planar" and not r["planar_fallback"]
             and r["accepted"] is False for r in planar)),
        ("planar mit Rückfall: 758 Inlier",
         any(r["scene"] == "planar" and r["planar_fallback"]
             and r.get("n_inliers") == 758 for r in planar)),
        ("planar mit Rückfall: Rotationsfehler unter 0,05 Grad",
         any(r["scene"] == "planar" and r["planar_fallback"]
             and r.get("rotation_error_deg", 9) < 0.05 for r in planar)),
        ("volumetrisch unverändert",
         len({r.get("n_inliers") for r in planar if r["scene"] == "volumetric"}) == 1),

        # ── Referenzlauf ─────────────────────────────────────────────────
        ("Referenzlauf: 67 Kameras", run["n_cameras_registered"] == 67),
        ("Referenzlauf: 51.444 Punkte", run["n_points"] == 51444),
        ("Referenzlauf: Brennweite 1.871,9 px",
         abs(run["shared_K"][0][0] - 1871.9) < 0.1),
        ("Referenzlauf: RMSE 0,655 px nach Filter",
         abs(run["reprojection"]["rmse_px"] - 0.655) < 5e-4),
        ("Referenzlauf: RMSE 0,968 px vor Filter",
         abs(run["reprojection_before_export_filter"]["rmse_px"] - 0.968) < 5e-4),
        ("Referenzlauf: Matching 285,8 s",
         abs(run["stage_times_s"]["matching"] - 285.8) < 0.1),
        ("Referenzlauf: Merkmale 45,6 s",
         abs(run["stage_times_s"]["features"] - 45.6) < 0.1),
        ("Referenzlauf: dichte Stufe 25,9 s",
         abs(run["stage_times_s"]["dense"] - 25.9) < 0.1),
        ("Referenzlauf: Oberfläche 49,1 s",
         abs(run["stage_times_s"]["mesh"] - 49.1) < 0.1),
        ("Referenzlauf: Rekonstruktion 1.593,7 s",
         abs(run["stage_times_s"]["reconstruction"]
             + run["stage_times_s"]["reconstruction_pass2"] - 1593.7) < 0.2),
        ("Referenzlauf: Brennweitensuche und Schleifenschluss 15,1 s",
         abs(run["stage_times_s"]["focal_search"]
             + run["stage_times_s"]["loop_closure"] - 15.1) < 0.2),
        ("Matching 3,6-mal schneller",
         abs(1036.6 / run["stage_times_s"]["matching"] - 3.63) < 0.05),

        # ── Mesh, Tab. 13 ────────────────────────────────────────────────
        ("Mesh: 102.425 Dreiecke", mesh["n_faces"] == 102425),
        ("Mesh: 53.048 Knoten", mesh["n_vertices"] == 53048),
        ("Mesh: eine Komponente",
         gt(mesh, "self", "topology", "n_components") == 1),
        ("Mesh: 3.903 Randkanten",
         gt(mesh, "self", "topology", "n_boundary_edges") == 3903),
        ("Mesh: Euler-Charakteristik -114",
         gt(mesh, "self", "topology", "euler_characteristic") == -114),
        ("Mesh: Median 3,29 Punktabstände",
         abs(gt(mesh, "self", "accuracy", "median_rel") - 3.29) < 5e-3),
        ("Mesh: 95-Prozent-Quantil 10,96",
         abs(gt(mesh, "self", "accuracy", "p95_rel") - 10.96) < 5e-3),
        ("Mesh: Abdeckung 98,7 %",
         abs(100 * gt(mesh, "self", "completeness", "coverage") - 98.68) < 0.05),
        ("Mesh: Abdeckung 91,7 % bei einem Punktabstand",
         abs(100 * gt(mesh, "self", "completeness", "coverage_1x") - 91.74) < 0.05),
        ("Mesh: Seitenverhältnis 1,27",
         abs(gt(mesh, "self", "geometry", "aspect_median") - 1.27) < 5e-3),
        ("Mesh: 3,3 % schlanke Dreiecke",
         abs(100 * gt(mesh, "self", "geometry", "sliver_frac") - 3.3) < 0.05),
        ("Mesh: 47,9 % extrapoliert",
         abs(100 * gt(mesh, "reference", "extrapolated_area_frac") - 47.9) < 0.05),
        ("Mesh: 2,88 Punktabstände zur Referenz",
         abs(gt(mesh, "reference", "mesh_to_reference", "median_in_spacing")
             - 2.88) < 5e-3),
        ("Mesh: Urteil WARN", quality["verdict"] == "WARN"),
        ("Photometrie trennt nicht",
         abs(gt(mesh, "photometric", "surface", "median_std")
             - gt(mesh, "photometric", "offset_10x", "median_std")) < 0.5),

        # ── Formalien ────────────────────────────────────────────────────
        ("kein Geviertstrich", md.count("—") == 0),
        ("kein Halbgeviertstrich", md.count("–") == 0),
        ("Abbildungen fortlaufend nummeriert",
         [m for m in re.findall(r"\*\*Fig\. (\d+):\*\*", md)]
         == [str(i) for i in range(1, 15)]),
        ("Tabellen fortlaufend nummeriert",
         [m for m in re.findall(r"\*\*Tab\. (\d+):\*\*", md)]
         == [str(i) for i in range(1, 15)]),
        ("keine Platzhalter mehr", "PLATZHALTER" not in md),
    ]

    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  ok      " if ok else "  FEHLER  ") + name)
    print(f"\n{len(checks) - len(bad)} von {len(checks)} Prüfungen bestanden.")
    if bad:
        print("nicht bestanden: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
