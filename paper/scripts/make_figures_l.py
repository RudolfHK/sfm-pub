#!/usr/bin/env python3
"""Erzeugt die neuen Abbildungen der Langfassung `paper/paper_l.md`.

Jede Abbildung entsteht aus einem Artefakt im Repository. Die Quelle steht im
Kopf der jeweiligen Funktion und im Anhang der Langfassung. Schematische
Abbildungen ohne Messbezug sind als solche gekennzeichnet.

    python paper/scripts/make_figures_l.py --all
    python paper/scripts/make_figures_l.py funnel growth

Voraussetzungen: matplotlib, numpy. Fuer die Abbildungen mit Ground-Truth-Bezug
zusaetzlich `--gt-dir` (Verzeichnis mit den `*_P.txt` des Buddha-Datensatzes).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures" / "l"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

# Farbschema, an den Style Guide angelehnt
BLUE = "#23355D"      # GFaI-Blau, Ueberschriften
OWN = "#e76f51"       # eigene Pipeline
REF = "#1f6f8b"       # COLMAP
GT_C = "#4d4d4d"      # Ground Truth
ACC = "#e9c46a"       # Hervorhebung
GREY = "#adb5bd"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titlecolor": BLUE,
    "axes.labelsize": 10,
    "axes.edgecolor": "#666666",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "figure.dpi": 110,
})

LOG_N67 = REPO / "eval_results" / "n67_base.log"
CAM_OWN = REPO / "eval_results" / "n67_base.cameras.json"
CAM_COLMAP = REPO / "paper_out" / "colmap.cameras.json"
COLMAP_PTS = REPO / "paper_out" / "colmap_txt" / "points3D.txt"
FOCAL_JSON = OUT / "focal_experiment.json"


def de(value, digits: int = 0) -> str:
    """Zahl mit deutschem Tausenderpunkt und Dezimalkomma."""
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"geschrieben: {path.relative_to(REPO)}")


# ── Protokoll-Auswertung ─────────────────────────────────────────────────────

def parse_log(path: Path = LOG_N67) -> dict:
    """Kamerazahl, Punktzahl und BA-Runden aus dem Protokoll eines Laufs."""
    text = path.read_text(encoding="utf-8", errors="replace")
    steps = [(int(c), int(p)) for p, c in
             re.findall(r"total: (\d+), cameras: (\d+)", text)]
    new_pts = [int(x) for x in re.findall(r"New 3-D pts: (\d+)", text)]
    corr = [int(x) for x in re.findall(r"\((\d+) 2D-3D correspondences", text)]
    seed = re.search(r"Initialised: (\d+) 3-D points from (\d+) cameras", text)
    seed_pair = re.search(r"Seed pair: images (\d+) \S+ (\d+)", text)
    ba = [(int(c), float(r)) for r, c, in
          re.findall(r"BA  init RMSE: ([\d.]+) px  \((\d+) cams", text)]
    ba_final = [float(x) for x in re.findall(r"BA final RMSE: ([\d.]+) px", text)]
    ba_pts = [int(x) for x in re.findall(r"BA  init RMSE: [\d.]+ px  \(\d+ cams, (\d+) pts", text)]
    ba_obs = [int(x) for x in re.findall(r"pts, (\d+) obs\)", text)]
    kps = re.search(r"([\d,]+) keypoints across", text)
    pairs_tested = re.search(r"matching \[CPU FLANN\]: (\d+) pairs", text, re.I)
    pairs_raw = re.search(r"done: (\d+)/(\d+) pairs with", text)
    pairs_ver = re.search(r"Verification done: (\d+)/(\d+) pairs accepted", text)
    outlier = re.search(r"Outlier removal: dropped (\d+)/(\d+) points", text)
    export = re.search(r"Outlier filter: keeping (\d+)/(\d+) points", text)
    return {
        "steps": steps,
        "new_pts": new_pts,
        "corr": corr,
        "seed_points": int(seed.group(1)) if seed else None,
        "seed_images": [int(seed_pair.group(1)), int(seed_pair.group(2))]
        if seed_pair else None,
        "ba_cams": [c for c, _ in ba],
        "ba_rmse_init": [r for _, r in ba],
        "ba_rmse_final": ba_final,
        "ba_points": ba_pts,
        "ba_obs": ba_obs,
        "n_keypoints": int(kps.group(1).replace(",", "")) if kps else None,
        "pairs_tested": int(pairs_tested.group(1)) if pairs_tested else None,
        "pairs_raw": int(pairs_raw.group(1)) if pairs_raw else None,
        "pairs_verified": int(pairs_ver.group(1)) if pairs_ver else None,
        "pts_before_outlier": int(outlier.group(2)) if outlier else None,
        "pts_after_outlier": (int(outlier.group(2)) - int(outlier.group(1))
                              if outlier else None),
        "pts_exported": int(export.group(1)) if export else None,
    }


# ── Abb. Trichter: was jede Stufe verwirft ───────────────────────────────────

def fig_funnel() -> None:
    """Quelle: eval_results/n67_base.log und n67_base.run.json."""
    d = parse_log()
    run = json.loads((REPO / "eval_results" / "n67_base.run.json").read_text())
    obs_final = run["summary"]["n_observations"]

    panels = [
        ("Bildpaare", [
            (d["pairs_tested"], "geprüfte Paare\n$N(N-1)/2$"),
            (d["pairs_raw"], "≥ 15 Rohzuordnungen\nnach Ratio-Test"),
            (d["pairs_verified"], "geometrisch\nverifiziert"),
        ]),
        ("Beobachtungen", [
            (d["ba_obs"][-1], "im letzten\nBundle Adjustment"),
            (obs_final, "nach beiden\nAusreißerfiltern"),
        ]),
        ("3-D-Punkte", [
            (d["pts_before_outlier"], "trianguliert"),
            (d["pts_after_outlier"], "nach Ausreißer-\nentfernung, 8 px"),
            (d["pts_exported"], "exportiert,\n4 px"),
        ]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    for ax, (title, rows) in zip(axes, panels):
        base = rows[0][0]
        ax.set_title(title)
        for k, (val, label) in enumerate(rows):
            w = val / base
            y = -k
            ax.add_patch(plt.Rectangle((0.5 - w / 2, y - 0.34), w, 0.68,
                                       facecolor=OWN if k == 0 else
                                       (REF if k == len(rows) - 1 else ACC),
                                       alpha=0.85, edgecolor="white", lw=1.5))
            ax.text(0.5, y + 0.08, de(val), ha="center",
                    va="center", fontsize=12, fontweight="bold", color="white")
            ax.text(0.5, y - 0.16, f"{de(100 * w, 1)} %", ha="center", va="center",
                    fontsize=9, color="white")
            ax.text(1.02, y, label, ha="left", va="center", fontsize=8.5,
                    color="#333333")
        ax.set_xlim(-0.05, 1.75)
        ax.set_ylim(-len(rows) + 0.4, 0.6)
        ax.axis("off")
    fig.suptitle("Filterkaskade des Referenzlaufs: 67 Bilder, 468.252 SIFT-Merkmale",
                 color=BLUE, fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_l02_trichter.png")


# ── Abb. Wachstum der Rekonstruktion ─────────────────────────────────────────

def fig_growth() -> None:
    """Quelle: eval_results/n67_base.log."""
    d = parse_log()
    cams = np.array([2] + [c for c, _ in d["steps"]])
    pts = np.array([d["seed_points"]] + [p for _, p in d["steps"]])
    new = np.array(d["new_pts"])
    corr = np.array(d["corr"])
    step = np.arange(1, len(new) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax1.plot(cams, pts, "-", color=OWN, lw=2, label="3-D-Punkte insgesamt")
    ax1.set_xlabel("registrierte Kameras")
    ax1.set_ylabel("3-D-Punkte", color=OWN)
    ax1.tick_params(axis="y", labelcolor=OWN)
    ax1.set_title("Die Wolke wächst, der Fehler wächst mit")

    ax1b = ax1.twinx()
    ax1b.plot(d["ba_cams"], d["ba_rmse_init"], "o--", color=BLUE, lw=1.6,
              ms=5, label="Reprojektions-RMSE vor BA")
    ax1b.plot(d["ba_cams"], d["ba_rmse_final"], "s:", color=REF, lw=1.4,
              ms=4, label="nach BA")
    ax1b.set_ylabel("Reprojektions-RMSE (px)", color=BLUE)
    ax1b.tick_params(axis="y", labelcolor=BLUE)
    ax1b.grid(False)
    ax1b.set_ylim(0, max(d["ba_rmse_init"]) * 1.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")

    ax2.bar(step, new, color=GREY, width=0.85, label="neue 3-D-Punkte")
    ax2.set_xlabel("Registrierungsschritt")
    ax2.set_ylabel("neue 3-D-Punkte je Schritt")
    ax2.set_title("Späte Kameras sehen fast nur noch bekannte Struktur")
    ax2b = ax2.twinx()
    ax2b.plot(step, corr, "-", color=REF, lw=1.6,
              label="verfügbare 2D-3D-Korrespondenzen")
    ax2b.set_ylabel("2D-3D-Korrespondenzen bei der Registrierung", color=REF)
    ax2b.tick_params(axis="y", labelcolor=REF)
    ax2b.grid(False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
    ax2.annotate(f"Mittel {de(new.mean(), 0)} neue Punkte je Schritt",
                 (0.03, 0.92), xycoords="axes fraction", fontsize=8.5,
                 color="#444444")
    fig.tight_layout()
    _save(fig, "fig_l03_wachstum.png")


# ── Ground-Truth-Auswertung ──────────────────────────────────────────────────

def _aligned(cam_json: Path, gt_dir: Path):
    from gt_pose_eval import load_gt, rot_angle_deg, umeyama
    gt = load_gt(gt_dir)
    est = json.loads(cam_json.read_text(encoding="utf-8"))
    rows = [c for c in est["cameras"] if c["image_name"].split(".")[0] in gt]
    rows.sort(key=lambda c: c["image_name"])
    names = [c["image_name"].split(".")[0] for c in rows]
    C_est = np.array([c["center"] for c in rows], float)
    R_est = [np.array(c["R"], float) for c in rows]
    C_gt = np.array([gt[n]["C"] for n in names])
    R_gt = [gt[n]["R"] for n in names]
    s, R_a, t_a = umeyama(C_est, C_gt)
    C_al = (s * (R_a @ C_est.T).T) + t_a
    C_all = np.array([g["C"] for g in gt.values()])
    extent = float(np.linalg.norm(C_all.max(0) - C_all.min(0)))
    pos = 100.0 * np.linalg.norm(C_al - C_gt, axis=1) / extent
    rot = np.array([rot_angle_deg(R_gt[i] @ (R_est[i] @ R_a.T).T)
                    for i in range(len(rows))])
    return {"names": names, "C_al": C_al, "C_gt": C_gt, "pos": pos, "rot": rot,
            "scale": s}


def fig_trajectories(gt_dir: Path) -> None:
    """Quelle: eval_results/n67_base.cameras.json, paper_out/colmap.cameras.json."""
    own = _aligned(CAM_OWN, gt_dir)
    ref = _aligned(CAM_COLMAP, gt_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, data, label, colour in ((axes[0], own, "eigene Pipeline", OWN),
                                    (axes[1], ref, "COLMAP", REF)):
        gt_xy = data["C_gt"][:, [0, 2]]
        est_xy = data["C_al"][:, [0, 2]]
        for a, b in zip(gt_xy, est_xy):
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=colour, lw=1.0,
                    alpha=0.8, zorder=1)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "o", mfc="none", mec=GT_C, ms=8,
                mew=1.2, label="Ground Truth", zorder=2)
        ax.plot(est_xy[:, 0], est_xy[:, 1], ".", color=colour, ms=9,
                label=label, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (Weltkoordinaten, auf GT ausgerichtet)")
        ax.set_ylabel("z")
        ax.set_title(f"{label}: Median {de(np.median(data['pos']), 2)} %, "
                     f"Maximum {de(data['pos'].max(), 2)} % der Ausdehnung")
        ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_l04_trajektorien.png")


def fig_capture_geometry(gt_dir: Path) -> None:
    """Quelle: Ground-Truth-Projektionsmatrizen des Buddha-Datensatzes."""
    from gt_pose_eval import load_gt
    gt = load_gt(gt_dir)
    names = sorted(gt)
    C = np.array([gt[n]["C"] for n in names])
    R = [gt[n]["R"] for n in names]

    # Szenenzentrum als Schnitt der optischen Achsen
    d = np.array([r[2] for r in R])
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, dd in zip(C, d):
        P = np.eye(3) - np.outer(dd, dd)
        A += P
        b += P @ c
    centre = np.linalg.solve(A, b)

    v = C - centre
    radius = np.linalg.norm(v, axis=1)
    elev = np.degrees(np.arcsin(-v[:, 1] / radius))   # y zeigt nach unten
    azim = np.degrees(np.arctan2(v[:, 2], v[:, 0]))
    idx = np.arange(len(names))
    # Rangkorrelation zwischen Dateinamenreihenfolge und Azimut
    rank_i = np.argsort(np.argsort(idx))
    rank_a = np.argsort(np.argsort(azim))
    rho = float(np.corrcoef(rank_i, rank_a)[0, 1])

    fig = plt.figure(figsize=(12.4, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    p = ax1.scatter(C[:, 0], C[:, 2], -C[:, 1], c=idx, cmap="viridis", s=34,
                    depthshade=False)
    ax1.scatter([centre[0]], [centre[2]], [-centre[1]], color=OWN, s=120,
                marker="*", label="Objektmitte")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_zlabel("Höhe")
    ax1.set_title("Kamerazentren der Ground Truth")
    ax1.view_init(elev=22, azim=-58)
    cb = fig.colorbar(p, ax=ax1, pad=0.1, shrink=0.7)
    cb.set_label("Bildindex", fontsize=9)
    ax1.legend(fontsize=8.5, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    s = ax2.scatter(azim, elev, c=idx, cmap="viridis", s=42, edgecolor="white",
                    linewidth=0.5)
    for i in range(0, len(names), 8):
        ax2.annotate(names[i], (azim[i], elev[i]), textcoords="offset points",
                     xytext=(5, 4), fontsize=7, color="#444444")
    ax2.set_xlabel("Azimut um die Objektmitte (Grad)")
    ax2.set_ylabel("Höhenwinkel (Grad)")
    ax2.set_title("Aufnahmen decken die Halbkugel ab, nicht einen Ring")
    ax2.set_xlim(-190, 190)
    ax2.text(0.02, 0.04,
             f"Rangkorrelation Bildindex zu Azimut: {de(rho, 2)}\n"
             f"Abstand zur Objektmitte {de(radius.min(), 2)} bis "
             f"{de(radius.max(), 2)}, Median {de(np.median(radius), 2)}",
             transform=ax2.transAxes, fontsize=8.5, color="#333333",
             bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.4"))
    fig.colorbar(s, ax=ax2, pad=0.02).set_label("Bildindex", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_l12_aufnahmegeometrie.png")


def fig_per_camera(gt_dir: Path) -> None:
    """Quelle: dieselben Dateien wie fig_trajectories."""
    own = _aligned(CAM_OWN, gt_dir)
    ref = _aligned(CAM_COLMAP, gt_dir)
    x = np.arange(len(own["names"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.4, 6.0), sharex=True)
    ax1.plot(x, own["rot"], "-o", color=OWN, ms=3.5, lw=1.3, label="eigene Pipeline")
    ax1.plot(x, ref["rot"], "-s", color=REF, ms=3.0, lw=1.3, label="COLMAP")
    ax1.set_yscale("log")
    ax1.set_ylabel("Rotationsfehler (Grad)")
    ax1.set_title("Fehler je Kamera, Bilder in Reihenfolge der Dateinamen")
    ax1.legend(fontsize=9, ncol=2, loc="upper left")
    for d, c in ((own, OWN), (ref, REF)):
        ax1.axhline(np.median(d["rot"]), color=c, ls=":", lw=1.0, alpha=0.7)

    ax2.plot(x, own["pos"], "-o", color=OWN, ms=3.5, lw=1.3)
    ax2.plot(x, ref["pos"], "-s", color=REF, ms=3.0, lw=1.3)
    ax2.set_yscale("log")
    ax2.set_ylabel("Positionsfehler (% der Ausdehnung)")
    ax2.set_xlabel("Bildindex")
    step = 5
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([own["names"][i] for i in range(0, len(x), step)],
                        rotation=45, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_l05_fehler_je_kamera.png")


# ── Abb. Selbstauskunft gegen Wahrheit ───────────────────────────────────────

# EVALUATION_RESULTS.md, Abschnitt B.1
RUNS = [
    # name, registrierte Kameras, Bilder, RMSE px, Rotationsfehler Median Grad
    ("n06_mini", 6, 6, 1.417, 4.08),
    ("n20_base", 13, 20, 1.525, 4.19),
    ("n20_dense", 7, 20, 1.286, 7.05),
    ("n20_feat12k", 13, 20, 1.497, 3.79),
    ("n20_feat4k", 12, 20, 1.642, 3.51),
    ("n20_localba", 13, 20, 1.597, 4.41),
    ("n20_repeat", 13, 20, 1.489, 4.07),
    ("n20_repeat2", 14, 20, 1.550, 5.12),
    ("n20_repeat3", 6, 20, 1.125, 6.04),
    ("n20_seq", 4, 20, 0.695, 4.15),
    ("n20_trackmerge", 12, 20, 1.530, 4.63),
    ("n67_base", 67, 67, 2.083, 6.29),
    ("n67_user", 67, 67, 1.600, 6.23),
]


def fig_blindness() -> None:
    """Quelle: EVALUATION_RESULTS.md, Abschnitt B.1; COLMAP aus points3D.txt."""
    rmse = np.array([r[3] for r in RUNS])
    rot = np.array([r[4] for r in RUNS])
    size = np.array([r[1] for r in RUNS])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sc = ax.scatter(rmse, rot, s=18 + 3.2 * size, c=size, cmap="YlOrRd",
                    edgecolor="#7f3b1f", linewidth=0.8, alpha=0.9, zorder=3)
    for name, cams, _, x, y in RUNS:
        if name in ("n67_base", "n67_user", "n20_seq", "n20_dense", "n20_repeat3"):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=8, color="#5a3a1f")
    # COLMAP: mittlerer Fehler je Punkt aus points3D.txt, andere Bezugsgroesse
    ax.scatter([0.330], [0.11], marker="*", s=260, color=REF, zorder=4,
               edgecolor="white", linewidth=0.8)
    ax.annotate("COLMAP (67 Bilder)\n0,33 px je Punkt, 0,11°", (0.330, 0.11),
                textcoords="offset points", xytext=(12, 6), fontsize=9, color=REF)
    ax.set_yscale("log")
    ax.set_xlabel("Selbstauskunft: Reprojektions-RMSE (px)")
    ax.set_ylabel("Wahrheit: Rotationsfehler gegen GT, Median (Grad)")
    ax.set_title("Der Reprojektionsfehler ordnet die Läufe nicht nach ihrer Genauigkeit")
    ax.set_xlim(0, 2.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("registrierte Kameras", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_l06_selbstauskunft.png")


def fig_reproducibility() -> None:
    """Quelle: EVALUATION_RESULTS.md, Abschnitte A.3 und B.3."""
    labels = ["n20_base", "n20_repeat", "n20_repeat2", "n20_repeat3"]
    cams = [13, 13, 14, 6]
    pts = [5522, 6283, 6118, 3756]
    warm = ["cold", "warm1", "warm2"]
    warm_cams = [14, 13, 14]
    warm_pts = [5822, 6281, 6429]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2))
    x = np.arange(len(labels))
    b = ax1.bar(x, cams, 0.55, color=[OWN if c > 6 else "#b23a48" for c in cams])
    ax1.bar_label(b, fmt="%d", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("registrierte Kameras von 20")
    ax1.set_ylim(0, 20)
    ax1.axhline(20, color=GT_C, ls="--", lw=1.0)
    ax1.set_title("Vier byte-identische Aufrufe, drei verschiedene Ergebnisse")
    for xi, p in zip(x, pts):
        ax1.text(xi, 1.0, f"{p:,} Pkt.".replace(",", "."), ha="center",
                 fontsize=8, color="white", rotation=90, va="bottom")

    x2 = np.arange(len(warm))
    b2 = ax2.bar(x2, warm_cams, 0.5, color=[GREY, REF, REF])
    ax2.bar_label(b2, fmt="%d", fontsize=9)
    for xi, p in zip(x2, warm_pts):
        ax2.text(xi, 0.7, f"{p:,} Pkt.".replace(",", "."), ha="center",
                 fontsize=8, color="white", rotation=90, va="bottom")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["kalt", "warm 1", "warm 2"], fontsize=9)
    ax2.set_ylabel("registrierte Kameras von 20")
    ax2.set_ylim(0, 20)
    ax2.set_title("Auch mit identischem Cache für Merkmale und Matches")
    fig.tight_layout()
    _save(fig, "fig_l07_reproduzierbarkeit.png")


# ── Abb. Brennweitenexperiment ───────────────────────────────────────────────

def fig_focal() -> None:
    """Quelle: paper/figures/l/focal_experiment.json (paper/scripts/focal_experiment.py)."""
    data = json.loads(FOCAL_JSON.read_text(encoding="utf-8"))
    sw = data["sweep"]
    f_err = np.array([r["f_error_pct"] for r in sw])
    rot = np.array([r["rot_err_med_deg"] for r in sw])
    rmse = np.array([r["rmse_px"] for r in sw])
    npts = np.array([r["n_pts"] for r in sw])
    ba = data["ba_only"]
    f_start = np.array([r["f_start_err_pct"] for r in ba])
    f_end = np.array([r["f_end_err_pct"] for r in ba])
    var = data.get("solver_variants", [])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 4.6))

    ax1.plot(f_err, rot, "-o", color=OWN, lw=2, ms=6, label="wahrer Rotationsfehler")
    ax1.set_xlabel("angenommene Brennweite, Abweichung vom wahren Wert (%)")
    ax1.set_ylabel("Rotationsfehler gegen GT (Grad)", color=OWN)
    ax1.tick_params(axis="y", labelcolor=OWN)
    ax1.axvline(0, color=GT_C, ls="--", lw=1.0)
    ax1.axvline(47.0, color=ACC, ls="-.", lw=1.4)
    ax1.text(47.0, ax1.get_ylim()[1] * 0.42, " Schätzung\n der Pipeline",
             fontsize=8.5, color="#8a6d1f")
    ax1b = ax1.twinx()
    ax1b.plot(f_err, rmse, "-s", color=BLUE, lw=1.6, ms=5,
              label="Selbstauskunft: RMSE")
    ax1b.set_ylabel("Reprojektions-RMSE (px)", color=BLUE)
    ax1b.tick_params(axis="y", labelcolor=BLUE)
    ax1b.set_ylim(0, max(rmse) * 1.35)
    ax1b.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")
    ax1.set_title("Der Reprojektionsfehler sättigt, der wahre Fehler nicht")

    ax2.bar(f_err, npts, width=8.0, color=GREY, label="akzeptierte 3-D-Punkte")
    ax2.set_xlabel("angenommene Brennweite, Abweichung vom wahren Wert (%)")
    ax2.set_ylabel("akzeptierte 3-D-Punkte von 1.500")
    ax2.axvline(0, color=GT_C, ls="--", lw=1.0)
    best = int(np.argmin(np.abs(f_err)))
    ax2.annotate(f"{npts[best]}", (f_err[best], npts[best]),
                 textcoords="offset points", xytext=(-8, 6), fontsize=9,
                 color="#333333")
    ax2b = ax2.twinx()
    ax2b.plot(f_start, f_start, "-", color=GT_C, lw=3.0, alpha=0.35,
              label="unveränderter Startwert")
    ax2b.plot(f_start, f_end, "o", color=REF, ms=6,
              label="Brennweite nach BA (Struktur korrekt)")
    ax2b.set_ylabel("Brennweitenfehler nach BA (%)", color=REF)
    ax2b.tick_params(axis="y", labelcolor=REF)
    ax2b.grid(False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower center")
    ax2.set_title("Die Punktzahl reagiert, das Bundle Adjustment nicht")

    # dritte Tafel: Loeservarianten bei +47 %
    labels = ["Vor-\neinstellung", "x_scale\n= 'jac'", "strenge\nToleranzen",
              "beides\nzusammen"]
    err = [r["f_end_err_pct"] for r in var]
    rot = [r["rot_err_med_deg"] for r in var]
    rmse = [r["rmse_final_px"] for r in var]
    x = np.arange(len(var))
    cols = [GREY, GREY, ACC, REF]
    b = ax3.bar(x, err, 0.6, color=cols)
    ax3.axhline(47.0, color=OWN, ls="--", lw=1.3)
    ax3.text(len(var) - 1.0, 50.5, "Startwert +47,0 %", fontsize=8.5, color=OWN,
             ha="center")
    ax3.axhline(0, color=GT_C, lw=1.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels[:len(var)], fontsize=8.5)
    ax3.set_ylabel("Brennweitenfehler nach BA (%)")
    ax3.set_ylim(-9, 58)
    ax3.set_title("Nur beides zusammen bewegt die Brennweite")
    for xi, (e, r_, m) in zip(x, zip(err, rot, rmse)):
        ax3.text(xi, e + 1.8, f"{de(e, 1)} %", ha="center", fontsize=9.5,
                 fontweight="bold")
        ax3.text(xi, 2.0, f"RMSE {de(m, 1)} px\nRot. {de(r_, 2)}°", ha="center",
                 fontsize=8, color="#333333")
    fig.tight_layout()
    _save(fig, "fig_l08_brennweitenexperiment.png")


# ── Abb. Tracklaengen bei COLMAP ─────────────────────────────────────────────

def fig_colmap_tracks() -> None:
    """Quelle: paper_out/colmap_txt/points3D.txt."""
    tl, err = [], []
    with open(COLMAP_PTS, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 9:
                continue
            err.append(float(p[7]))
            tl.append((len(p) - 8) // 2)
    tl = np.array(tl)
    err = np.array(err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2))
    bins = np.arange(1.5, min(tl.max(), 16) + 1.5, 1.0)
    ax1.hist(tl, bins=bins, color=REF, alpha=0.9, edgecolor="white")
    ax1.set_xlabel("Beobachtungen je 3-D-Punkt")
    ax1.set_ylabel("Anzahl Punkte")
    ax1.set_title(f"COLMAP: mittlere Tracklänge {de(tl.mean(), 2)}, "
                  f"nur {de(100 * (tl == 2).mean(), 1)} % Zweibildpunkte")
    ax1.axvline(tl.mean(), color=OWN, ls="--", lw=1.4)
    ax1.text(tl.mean() + 0.15, ax1.get_ylim()[1] * 0.86,
             f"Mittel {de(tl.mean(), 2)}", fontsize=9, color=OWN)
    ax1.text(2.6, ax1.get_ylim()[1] * 0.62,
             "eigene Pipeline: 2,70", fontsize=9, color=OWN)
    ax1.annotate("", xy=(2.7, ax1.get_ylim()[1] * 0.58),
                 xytext=(2.7, ax1.get_ylim()[1] * 0.05),
                 arrowprops=dict(arrowstyle="->", color=OWN, lw=1.4))

    ks = np.arange(2, 13)
    means = [err[tl == k].mean() if (tl == k).any() else np.nan for k in ks]
    counts = [(tl == k).sum() for k in ks]
    ax2.plot(ks, means, "-o", color=REF, lw=1.8, ms=5)
    ax2.set_xlabel("Beobachtungen je 3-D-Punkt")
    ax2.set_ylabel("mittlerer Reprojektionsfehler (px)")
    ax2.set_title("Der Punktfehler bleibt über alle Tracklängen unter 0,5 px")
    ax2.set_ylim(0, np.nanmax(means) * 1.45)
    for k, m, c in zip(ks, means, counts):
        if k in (2, 3, 5, 8, 12):
            ax2.annotate(f"n={c:,}".replace(",", "."), (k, m),
                         textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color="#444444")
    fig.tight_layout()
    _save(fig, "fig_l09_colmap_tracks.png")


# ── Abb. Triangulationswinkel ────────────────────────────────────────────────

def fig_triangulation_angle() -> None:
    """Analytisch, schematisch. Parameter: f = 1860,9 px, sigma = 0,5 px."""
    f = 1860.9
    sigma = 0.5
    ang = np.linspace(0.5, 45, 400)
    rel = sigma / (f * np.sin(np.radians(ang)))  # sigma_Z / Z bei Basislinie b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # linkes Panel: Schema zweier Sehstrahlen
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Schnitt zweier Sehstrahlen")
    for (cx, cy), col in (((-1.0, 0.0), OWN), ((1.0, 0.0), REF)):
        ax1.plot([cx], [cy], "o", color=col, ms=9)
        ax1.plot([cx, 0.0], [cy, 3.0], "-", color=col, lw=1.6)
        for d in (-1, 1):
            ax1.plot([cx, 0.0 + d * 0.16], [cy, 3.0], "-", color=col, lw=0.7,
                     alpha=0.5)
    ax1.plot([0], [3.0], "o", color="#333333", ms=6)
    ax1.add_patch(plt.Polygon([[-0.16, 2.72], [0.16, 2.72], [0.16, 3.28],
                               [-0.16, 3.28]], closed=True, facecolor=ACC,
                              alpha=0.55, edgecolor="none"))
    ax1.annotate("Unsicherheits-\nbereich", (0.2, 3.35), fontsize=8.5,
                 color="#8a6d1f")
    ax1.annotate("", xy=(-0.42, 2.1), xytext=(0.42, 2.1),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax1.text(0.0, 1.85, "Triangulationswinkel α", ha="center", fontsize=9)
    ax1.text(-1.0, -0.35, "Kamera 1", ha="center", fontsize=9, color=OWN)
    ax1.text(1.0, -0.35, "Kamera 2", ha="center", fontsize=9, color=REF)
    ax1.set_xlim(-1.9, 1.9)
    ax1.set_ylim(-0.7, 4.0)

    ax2.plot(ang, rel * 1e3, color=BLUE, lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Triangulationswinkel α (Grad)")
    ax2.set_ylabel(r"relative Tiefenunsicherheit $\sigma_Z/Z$ ($\times 10^{-3}$)")
    ax2.set_title("Tiefenunsicherheit bei 0,5 px Messrauschen und f = 1.861 px")
    for a, lab, col in ((2.0, "Mindestwinkel der\nTriangulation, 2°", OWN),
                        (5.0, "Mindestwinkel des\nStartpaares, 5°", REF)):
        ax2.axvline(a, color=col, ls="--", lw=1.3)
        y = sigma / (f * np.sin(np.radians(a))) * 1e3
        ax2.annotate(lab, (a, y), textcoords="offset points", xytext=(10, 0),
                     fontsize=8.5, color=col)
        ax2.plot([a], [y], "o", color=col, ms=6)
    fig.tight_layout()
    _save(fig, "fig_l10_triangulationswinkel.png")


# ── Abb. Besetzungsstruktur der Jacobi-Matrix ────────────────────────────────

def fig_sparsity() -> None:
    """Erzeugt mit sfm.bundle_adjustment._build_sparsity_v2, schematische Groesse."""
    from sfm.bundle_adjustment import _build_sparsity_v2

    rng = np.random.default_rng(0)
    n_cams, n_pts = 8, 120
    cam_idx, pt_idx = [], []
    for p in range(n_pts):
        first = rng.integers(0, n_cams - 1)
        span = rng.integers(2, 4)
        for c in range(first, min(first + span, n_cams)):
            cam_idx.append(c)
            pt_idx.append(p)
    cam_idx = np.array(cam_idx)
    pt_idx = np.array(pt_idx)
    J = _build_sparsity_v2(n_cams, n_pts, cam_idx, pt_idx, True, False)

    dense = np.asarray(J.todense())
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.imshow(dense, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Parameter: 5 Intrinsik | 6 je Kamera | 3 je Punkt")
    ax.set_ylabel("Residuen (2 je Beobachtung)")
    ax.set_title(f"Besetzungsstruktur: {de(100 * dense.mean(), 2)} % der Einträge "
                 "sind besetzt")
    ax.axvline(5, color=OWN, lw=1.2)
    ax.axvline(5 + 6 * n_cams, color=OWN, lw=1.2)
    box = dict(fc="white", ec="none", alpha=0.85, boxstyle="round,pad=0.2")
    ax.text(5 + 3 * n_cams, 0.05 * dense.shape[0], "Kameras", color=OWN,
            ha="center", va="top", fontsize=9, bbox=box)
    ax.text(5 + 6 * n_cams + n_pts * 1.5, 0.05 * dense.shape[0], "Punkte",
            color=OWN, ha="center", va="top", fontsize=9, bbox=box)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "fig_l11_jacobi.png")


FIGURES = {
    "funnel": (fig_funnel, False),
    "growth": (fig_growth, False),
    "trajectories": (fig_trajectories, True),
    "percam": (fig_per_camera, True),
    "capture": (fig_capture_geometry, True),
    "blindness": (fig_blindness, False),
    "repro": (fig_reproducibility, False),
    "focal": (fig_focal, False),
    "colmap_tracks": (fig_colmap_tracks, False),
    "angle": (fig_triangulation_angle, False),
    "sparsity": (fig_sparsity, False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", choices=sorted(FIGURES) + [[]][0:0],
                    help="Namen der zu erzeugenden Abbildungen")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gt-dir", default=None,
                    help="Verzeichnis mit den *_P.txt des Buddha-Datensatzes")
    args = ap.parse_args()

    names = sorted(FIGURES) if (args.all or not args.names) else args.names
    for name in names:
        fn, needs_gt = FIGURES[name]
        if needs_gt:
            if not args.gt_dir:
                print(f"  übersprungen: {name} (benötigt --gt-dir)")
                continue
            fn(Path(args.gt_dir))
        else:
            fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
