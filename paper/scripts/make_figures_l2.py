#!/usr/bin/env python3
"""Figures for `paper/paper_l2.md`: the improvements and the mesh stage.

Every figure is built from an artefact of the measured runs, never from a
hand-entered number. Sources are named in each function's docstring and in the
appendix of the paper.

    python paper/scripts/make_figures_l2.py --all \\
        --mesh <mesh.obj> --cloud <mesh_prepared_cloud.ply> \\
        --dense <dense.ply> --mesh-eval eval_results/mesh_eval.json \\
        --ablation eval_results/fix_impact_n67/fix_impact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures" / "l2"

BLUE = "#23355D"
OWN = "#e76f51"
REF = "#1f6f8b"
GT_C = "#4d4d4d"
ACC = "#e9c46a"
GREEN = "#2a9d8f"
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


def de(value, digits: int = 0) -> str:
    """German thousands separator and decimal comma."""
    return f"{value:,.{digits}f}".translate(str.maketrans({",": ".", ".": ","}))


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"geschrieben: {path.relative_to(REPO)}")


# ── shared loading ────────────────────────────────────────────────────────────

def load_mesh(path: Path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()
    return mesh


def load_cloud(path: Path):
    import open3d as o3d
    return o3d.io.read_point_cloud(str(path))


def _shade(ax, verts, tris, normals, view, colors=None, cmap=None,
           vmin=None, vmax=None):
    """Painter's-algorithm rendering of a mesh with simple diffuse shading."""
    from matplotlib.collections import PolyCollection

    elev, azim = view
    e, a = np.radians(elev), np.radians(azim)
    # Camera basis: right, up, forward.
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(up_world, fwd)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    proj = np.column_stack([verts @ right, verts @ up])
    depth = verts @ fwd
    tri_depth = depth[tris].mean(axis=1)
    order = np.argsort(tri_depth)[::-1]

    polys = proj[tris[order]]
    face_n = normals[tris[order]].mean(axis=1)
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, 0.5, 0.75])
    light /= np.linalg.norm(light)
    lam = np.clip(np.abs(face_n @ light), 0.0, 1.0) * 0.75 + 0.25

    if colors is None:
        rgba = np.zeros((len(polys), 4))
        rgba[:, :3] = np.array([0.80, 0.78, 0.74]) * lam[:, None]
        rgba[:, 3] = 1.0
    else:
        cvals = colors[tris[order]].mean(axis=1)
        norm = plt.Normalize(vmin if vmin is not None else np.percentile(cvals, 2),
                             vmax if vmax is not None else np.percentile(cvals, 98))
        rgba = plt.get_cmap(cmap or "viridis")(norm(cvals))
        rgba[:, :3] *= (0.55 + 0.45 * lam)[:, None]

    ax.add_collection(PolyCollection(polys, facecolors=rgba, edgecolors="none"))
    ax.set_xlim(proj[:, 0].min(), proj[:, 0].max())
    ax.set_ylim(proj[:, 1].min(), proj[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    return proj


# ── Fig: mesh from several directions ─────────────────────────────────────────

def fig_mesh_views(mesh_path: Path) -> None:
    """Quelle: das Mesh des Referenzlaufs."""
    mesh = load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    centre = verts.mean(axis=0)
    verts = verts - centre

    views = [(15, 0), (15, 90), (15, 180), (15, 270), (80, 0), (-60, 0)]
    titles = ["Vorderseite", "rechts", "Rückseite", "links", "Draufsicht", "Untersicht"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.0))
    for ax, view, title in zip(axes.ravel(), views, titles):
        _shade(ax, verts, tris, normals, view)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"Rekonstruierte Oberfläche: {de(len(tris))} Dreiecke, {de(len(verts))} Knoten",
        color=BLUE, fontsize=12, y=0.98)
    fig.tight_layout()
    _save(fig, "fig_l2_mesh_ansichten.png")


# ── Fig: deviation between mesh and cloud ─────────────────────────────────────

def fig_mesh_deviation(mesh_path: Path, cloud_path: Path) -> None:
    """Quelle: Mesh und vorbereitete Punktwolke desselben Laufs."""
    from scipy.spatial import cKDTree

    mesh = load_mesh(mesh_path)
    cloud = load_cloud(cloud_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    pts = np.asarray(cloud.points)

    tree = cKDTree(pts)
    d, _ = tree.query(verts, k=1, workers=-1)
    nn, _ = tree.query(pts[np.random.default_rng(0).choice(
        len(pts), min(20000, len(pts)), replace=False)], k=2, workers=-1)
    spacing = float(np.median(nn[:, 1]))
    d_rel = d / spacing

    fig = plt.figure(figsize=(12.4, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    _shade(ax1, verts - verts.mean(0), tris, normals, (15, 0),
           colors=d_rel, cmap="turbo", vmin=0, vmax=3)
    ax1.set_title("Abweichung zur Punktwolke, Vorderseite")

    ax2 = fig.add_subplot(1, 3, 2)
    _shade(ax2, verts - verts.mean(0), tris, normals, (15, 180),
           colors=d_rel, cmap="turbo", vmin=0, vmax=3)
    ax2.set_title("Rückseite")

    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, 3))
    cb = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.03, pad=0.02)
    cb.set_label("Abstand zur Wolke in Punktabständen", fontsize=9)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(np.clip(d_rel, 0, 6), bins=60, color=REF, alpha=0.9)
    ax3.axvline(1.0, color=GREEN, ls="--", lw=1.4)
    ax3.axvline(3.0, color=OWN, ls="--", lw=1.4)
    ax3.text(1.05, ax3.get_ylim()[1] * 0.9, "1 Punktabstand", fontsize=8.5, color=GREEN)
    ax3.text(3.05, ax3.get_ylim()[1] * 0.8, "3 Punktabstände", fontsize=8.5, color=OWN)
    ax3.set_xlabel("Abstand Knoten zu Wolke (Punktabstände)")
    ax3.set_ylabel("Anzahl Knoten")
    ax3.set_title(f"Median {de(np.median(d_rel), 2)}, "
                  f"{de(100 * (d_rel > 3).mean(), 1)} % über 3")
    _save(fig, "fig_l2_mesh_abweichung.png")


# ── Fig: stages of the mesh pipeline ──────────────────────────────────────────

def fig_mesh_stages(dense_path: Path, cloud_path: Path, mesh_path: Path) -> None:
    """Quelle: dichte Wolke, vorbereitete Wolke und Mesh desselben Laufs."""
    dense = load_cloud(dense_path)
    prep = load_cloud(cloud_path)
    mesh = load_mesh(mesh_path)

    d_pts = np.asarray(dense.points)
    p_pts = np.asarray(prep.points)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    centre = p_pts.mean(axis=0)

    rng = np.random.default_rng(0)

    def scatter(ax, pts, colors, title):
        p = pts - centre
        if len(p) > 60000:
            sel = rng.choice(len(p), 60000, replace=False)
            p = p[sel]
            colors = colors[sel] if colors is not None else None
        # Same view as the shaded panels: azimuth 0, elevation 15.
        e, a = np.radians(15), 0.0
        fwd = np.array([np.cos(e), 0.0, np.sin(e)])
        right = np.array([0.0, 1.0, 0.0])
        up = np.cross(fwd, right)
        xy = np.column_stack([p @ right, p @ up])
        order = np.argsort(-(p @ fwd))
        ax.scatter(xy[order, 0], xy[order, 1], s=0.6,
                   c=(colors[order] if colors is not None else GREY),
                   marker=".", linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6))
    scatter(axes[0], d_pts,
            np.asarray(dense.colors) if dense.has_colors() else None,
            f"dichte Wolke\n{de(len(d_pts))} Punkte")
    scatter(axes[1], p_pts,
            np.asarray(prep.colors) if prep.has_colors() else None,
            f"vorbereitete Wolke\n{de(len(p_pts))} Punkte, mit Normalen")
    _shade(axes[2], verts - centre, tris, normals, (15, 0))
    axes[2].set_title(f"bereinigtes Mesh\n{de(len(tris))} Dreiecke", fontsize=10)
    fig.suptitle("Von der dichten Wolke zur Oberfläche", color=BLUE, fontsize=12,
                 y=1.0)
    fig.tight_layout()
    _save(fig, "fig_l2_mesh_stufen.png")


# ── Fig: cross-section ────────────────────────────────────────────────────────

def fig_mesh_section(mesh_path: Path, cloud_path: Path) -> None:
    """Quelle: Mesh und vorbereitete Wolke; Schnitt durch die Objektmitte."""
    mesh = load_mesh(mesh_path)
    cloud = load_cloud(cloud_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    pts = np.asarray(cloud.points)
    centre = pts.mean(axis=0)
    extent = np.percentile(np.abs(pts - centre), 98, axis=0)

    # Slab through the centre, perpendicular to the axis with the largest spread
    # so the section cuts across the object rather than along it.
    axis = int(np.argmax(extent))
    others = [i for i in range(3) if i != axis]
    thickness = 0.02 * float(extent[axis])

    slab_pts = pts[np.abs(pts[:, axis] - centre[axis]) < thickness]

    # Mesh edges crossing the slab plane, as a polyline set.
    v = verts[:, axis] - centre[axis]
    segs = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = tris[:, a], tris[:, b]
        cross = (v[ia] * v[ib]) < 0
        if not cross.any():
            continue
        w = v[ia][cross] / (v[ia][cross] - v[ib][cross])
        p = verts[ia][cross] + (verts[ib][cross] - verts[ia][cross]) * w[:, None]
        segs.append(p)
    sec = np.vstack(segs) if segs else np.zeros((0, 3))

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    if len(slab_pts):
        ax.scatter(slab_pts[:, others[0]] - centre[others[0]],
                   slab_pts[:, others[1]] - centre[others[1]],
                   s=3, color=REF, alpha=0.55, label="Punkte der Wolke im Schnitt")
    if len(sec):
        ax.scatter(sec[:, others[0]] - centre[others[0]],
                   sec[:, others[1]] - centre[others[1]],
                   s=2, color=OWN, label="Schnittlinie des Mesh")
    ax.set_aspect("equal")
    ax.set_xlabel("Achse %d" % others[0])
    ax.set_ylabel("Achse %d" % others[1])
    ax.set_title("Schnitt durch das Modell: liegt die Fläche auf den Daten?")
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, "fig_l2_mesh_schnitt.png")


# ── Fig: triangle quality ─────────────────────────────────────────────────────

def fig_mesh_quality(mesh_eval_path: Path) -> None:
    """Quelle: eval_results/mesh_eval.json (eval/mesh_eval.py)."""
    rep = json.loads(Path(mesh_eval_path).read_text(encoding="utf-8"))
    geo = rep["self"]["geometry"]
    topo = rep["self"]["topology"]
    acc = rep["self"]["accuracy"]
    comp = rep["self"].get("completeness", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    keys = [("min_angle_deg_mean", "mittlerer\nkleinster Winkel"),
            ("aspect_ratio_mean", "mittleres\nSeitenverhältnis"),
            ("radius_ratio_mean", "mittleres\nRadienverhältnis")]
    have = [(lbl, geo[k]) for k, lbl in keys if k in geo]
    if have:
        labels = [h[0] for h in have]
        vals = [h[1] for h in have]
        bars = ax1.bar(range(len(vals)), vals, 0.55, color=REF)
        ax1.bar_label(bars, fmt="%.2f", fontsize=9)
        ax1.set_xticks(range(len(vals)))
        ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Dreiecksqualität")

    names, values, colors = [], [], []
    for label, value, good in (
        ("kantenmannigfaltig", topo.get("is_edge_manifold"), True),
        ("knotenmannigfaltig", topo.get("is_vertex_manifold"), True),
        ("wasserdicht", topo.get("is_watertight"), True),
        ("orientierbar", topo.get("is_orientable"), True),
    ):
        names.append(label)
        values.append(1.0 if value else 0.0)
        colors.append(GREEN if value else OWN)
    ax2.barh(range(len(names)), values, 0.55, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlim(0, 1.35)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["nein", "ja"])
    for i, v in enumerate(values):
        ax2.text(v + 0.04, i, "erfüllt" if v else "nicht erfüllt",
                 va="center", fontsize=9, color=GREEN if v else OWN)
    ax2.set_title(f"Topologie: {topo.get('n_components')} Komponente(n), "
                  f"Euler {topo.get('euler_characteristic')}")
    ax2.grid(False)

    fig.suptitle(
        f"Mesh mit {de(topo.get('n_faces', 0))} Dreiecken, Median-Abstand zur "
        f"Wolke {de(acc.get('median_in_spacing', 0), 2)} Punktabstände, "
        f"Abdeckung {de(100 * comp.get('coverage_frac', 0), 1)} %",
        color=BLUE, fontsize=11, y=1.03)
    fig.tight_layout()
    _save(fig, "fig_l2_mesh_qualitaet.png")


# ── Fig: photometric consistency ──────────────────────────────────────────────

def fig_photometric(mesh_eval_path: Path) -> None:
    """Quelle: eval_results/mesh_eval.json, Abschnitt 'photometric'."""
    rep = json.loads(Path(mesh_eval_path).read_text(encoding="utf-8"))
    ph = rep["photometric"]
    order = ["surface"] + sorted(k for k in ph if k.startswith("offset_"))
    labels = {"surface": "auf der\nrekonstruierten Fläche"}
    vals, errs, names, ns = [], [], [], []
    for key in order:
        entry = ph.get(key)
        if not isinstance(entry, dict) or not entry.get("n_usable"):
            continue
        vals.append(entry["median_std"])
        errs.append(entry["p90_std"] - entry["median_std"])
        factor = key.split("_")[1].rstrip("x") if key != "surface" else None
        names.append(labels.get(key, f"um {factor} Punktabstände\nversetzt"))
        ns.append(entry["n_usable"])

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(vals))
    colors = [GREEN] + [ACC] * (len(vals) - 1)
    bars = ax.bar(x, vals, 0.55, color=colors, yerr=errs, capsize=4,
                  error_kw={"ecolor": "#888888", "lw": 1.0})
    ax.bar_label(bars, fmt="%.1f", fontsize=10, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Farbstreuung über die Ansichten (Grauwerte 0 bis 255)")
    ax.set_title("Photometrische Probe unter den Ground-Truth-Posen")
    for xi, n in zip(x, ns):
        ax.text(xi, 0.4, f"n = {de(n)}", ha="center", fontsize=8, color="white")
    ax.set_ylim(0, max(vals) * 1.45)
    _save(fig, "fig_l2_photometrie.png")


# ── Fig: ablation of the fixes ────────────────────────────────────────────────

def fig_ablation(ablation_path: Path) -> None:
    """Quelle: eval_results/fix_impact_n67/fix_impact.json (eval/fix_impact.py)."""
    rows = json.loads(Path(ablation_path).read_text(encoding="utf-8"))

    def get(row, *path):
        cur = row
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    labels = {"before": "vorher", "+focal": "+ Brennweite",
              "+ba": "+ Löser", "+tracks": "+ Tracks",
              "+loop": "+ Schleifen", "after": "alle"}
    names = [labels.get(r["config"], r["config"]) for r in rows]
    rot = [get(r, "gt", "alignment", "lsq", "rotation_err_deg", "median") for r in rows]
    foc = [abs(get(r, "gt", "focal", "error_pct") or 0.0) for r in rows]
    track = [get(r, "summary", "mean_track_length") for r in rows]
    cams = [get(r, "summary", "n_cameras_registered") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(rows))

    b = axes[0].bar(x, rot, 0.6, color=[OWN if i == 0 else
                                        (GREEN if i == len(rows) - 1 else ACC)
                                        for i in range(len(rows))])
    axes[0].bar_label(b, fmt="%.2f", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_title("Rotationsfehler gegen GT (Grad, Median)")

    b = axes[1].bar(x, foc, 0.6, color=[OWN if i == 0 else
                                        (GREEN if i == len(rows) - 1 else ACC)
                                        for i in range(len(rows))])
    axes[1].bar_label(b, fmt="%.1f", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_title("Brennweitenfehler (%, Betrag)")

    b = axes[2].bar(x, track, 0.6, color=[OWN if i == 0 else
                                          (GREEN if i == len(rows) - 1 else ACC)
                                          for i in range(len(rows))])
    axes[2].bar_label(b, fmt="%.2f", fontsize=9)
    axes[2].set_title("mittlere Tracklänge")
    axes[2].axhline(4.69, color=REF, ls="--", lw=1.2)
    axes[2].text(0.02, 4.75, "COLMAP 4,69", fontsize=8.5, color=REF)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    fig.suptitle(
        "Wirkung der einzelnen Änderungen, 67 Bilder, sonst gleiche Einstellungen "
        f"(registrierte Kameras: {cams[0]} vorher, {cams[-1]} nachher)",
        color=BLUE, fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_l2_ablation.png")



# ── Fig: pose error before and after, against ground truth ────────────────────

def _aligned(cam_json: Path, gt_dir: Path):
    """Estimated camera centres and orientations after Sim(3) onto the GT."""
    import sys
    sys.path.insert(0, str(REPO / "eval"))
    from gt_pose_eval import load_gt, rot_angle_deg, umeyama

    gt = load_gt(gt_dir)
    est = json.loads(Path(cam_json).read_text(encoding="utf-8"))
    rows = [c for c in est["cameras"] if c["image_name"].split(".")[0] in gt]
    rows.sort(key=lambda c: c["image_name"])
    names = [c["image_name"].split(".")[0] for c in rows]
    C_est = np.array([c["center"] for c in rows], float)
    R_est = [np.array(c["R"], float) for c in rows]
    C_gt = np.array([gt[n]["C"] for n in names])
    R_gt = [gt[n]["R"] for n in names]
    s_, R_a, t_a = umeyama(C_est, C_gt)
    C_al = (s_ * (R_a @ C_est.T).T) + t_a
    C_all = np.array([g["C"] for g in gt.values()])
    extent = float(np.linalg.norm(C_all.max(0) - C_all.min(0)))
    pos = 100.0 * np.linalg.norm(C_al - C_gt, axis=1) / extent
    rot = np.array([rot_angle_deg(R_gt[i] @ (R_est[i] @ R_a.T).T)
                    for i in range(len(rows))])
    return {"names": names, "C_al": C_al, "C_gt": C_gt, "pos": pos, "rot": rot}


def fig_pose_before_after(before_cams: Path, after_cams: Path, gt_dir: Path) -> None:
    """Quelle: die Kameradateien der Zeilen 'vorher' und 'nachher'."""
    before = _aligned(before_cams, gt_dir)
    after = _aligned(after_cams, gt_dir)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))
    for ax, data, label, colour in ((axes[0], before, "vorher", OWN),
                                    (axes[1], after, "nachher", GREEN)):
        gt_xy = data["C_gt"][:, [0, 2]]
        est_xy = data["C_al"][:, [0, 2]]
        for a, b in zip(gt_xy, est_xy):
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=colour, lw=1.0, alpha=0.85,
                    zorder=1)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "o", mfc="none", mec=GT_C, ms=8, mew=1.2,
                label="Ground Truth", zorder=2)
        ax.plot(est_xy[:, 0], est_xy[:, 1], ".", color=colour, ms=9, label=label,
                zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (auf GT ausgerichtet)")
        ax.set_ylabel("z")
        ax.set_title(f"{label}: Median {de(np.median(data['pos']), 2)} %, "
                     f"Maximum {de(data['pos'].max(), 2)} %")
        ax.legend(fontsize=9, loc="upper right")

    ax = axes[2]
    x = np.arange(len(after["names"]))
    ax.plot(x, before["rot"], "-o", color=OWN, ms=3.0, lw=1.1, label="vorher")
    ax.plot(x, after["rot"], "-s", color=GREEN, ms=3.0, lw=1.1, label="nachher")
    ax.axhline(np.median(before["rot"]), color=OWN, ls=":", lw=1.0)
    ax.axhline(np.median(after["rot"]), color=GREEN, ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Bildindex")
    ax.set_ylabel("Rotationsfehler (Grad)")
    ax.set_title(f"Median {de(np.median(before['rot']), 2)}° auf "
                 f"{de(np.median(after['rot']), 2)}°")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_l2_posen_vorher_nachher.png")


# ── Fig: track length distribution before and after ───────────────────────────

def fig_tracks_before_after(before_cams: Path, after_cams: Path) -> None:
    """Quelle: die Kameradateien beider Zeilen; Tracklänge je 3-D-Punkt."""
    def lengths(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d["mean_track_length"], d["n_points"], d["n_observations"]

    mb, pb, ob = lengths(before_cams)
    ma, pa, oa = lengths(after_cams)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))
    b = ax1.bar([0, 1, 2], [mb, ma, 4.69], 0.55, color=[OWN, GREEN, REF])
    ax1.bar_label(b, fmt="%.2f", fontsize=10)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["vorher", "nachher", "COLMAP"], fontsize=10)
    ax1.set_ylabel("mittlere Beobachtungen je 3-D-Punkt")
    ax1.set_title("Tracklänge")

    x = np.arange(2)
    w = 0.38
    b1 = ax2.bar(x - w / 2, [pb, pa], w, color=GREY, label="3-D-Punkte")
    b2 = ax2.bar(x + w / 2, [ob, oa], w, color=BLUE, label="Beobachtungen")
    ax2.bar_label(b1, fmt="%d", fontsize=8.5)
    ax2.bar_label(b2, fmt="%d", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["vorher", "nachher"], fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_title("Punkte und Beobachtungen")
    fig.tight_layout()
    _save(fig, "fig_l2_tracks.png")


FIGURES = {
    "pose_before_after": ("before_cams", "after_cams", "gt_dir"),
    "tracks_before_after": ("before_cams", "after_cams"),
    "mesh_views": ("mesh",),
    "mesh_deviation": ("mesh", "cloud"),
    "mesh_stages": ("dense", "cloud", "mesh"),
    "mesh_section": ("mesh", "cloud"),
    "mesh_quality": ("mesh_eval",),
    "photometric": ("mesh_eval",),
    "ablation": ("ablation",),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mesh")
    ap.add_argument("--cloud")
    ap.add_argument("--dense")
    ap.add_argument("--mesh-eval")
    ap.add_argument("--ablation")
    ap.add_argument("--before-cams")
    ap.add_argument("--after-cams")
    ap.add_argument("--gt-dir")
    args = ap.parse_args()

    have = {"mesh": args.mesh, "cloud": args.cloud, "dense": args.dense,
            "mesh_eval": args.mesh_eval, "ablation": args.ablation,
            "before_cams": args.before_cams, "after_cams": args.after_cams,
            "gt_dir": args.gt_dir}
    names = list(FIGURES) if (args.all or not args.names) else args.names

    for name in names:
        needs = FIGURES[name]
        missing = [n for n in needs if not have.get(n)]
        if missing:
            print(f"  übersprungen: {name} (benötigt --{', --'.join(missing)})")
            continue
        fn = globals()["fig_" + name]
        fn(*[Path(have[n]) for n in needs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
