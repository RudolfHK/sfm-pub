#!/usr/bin/env python3
"""Kontrolliertes Brennweitenexperiment auf synthetischen Korrespondenzen.

Die Evaluation zeigt, dass die Pipeline auf dem Buddha-Datensatz eine um 47 %
zu grosse Brennweite ansetzt und das Bundle Adjustment sie nie korrigiert.
Zwei Erklaerungen wurden bereits ausgeschlossen (Schranken des Optimierers,
Parameterskalierung). Offen blieb die dritte: Die Rekonstruktion ist bei der
falschen Brennweite bereits in sich konsistent.

Dieses Skript prueft genau das, und zwar getrennt von Merkmalsextraktion und
Matching. Die Beobachtungen werden analytisch erzeugt, mit den echten
Ground-Truth-Kameraposen des Datensatzes und der wahren Brennweite. Danach
laufen dieselbe GeometricVerifier- und IncrementalSfM-Klasse wie im
Produktivbetrieb, aber mit einer angenommenen Brennweite, die um die wahre
variiert wird.

Zwei Versuchsreihen:

A  "sweep"     Ganze Rekonstruktion je angenommener Brennweite. Gemessen werden
               die Selbstauskunft der Pipeline (Reprojektionsfehler, Punktzahl)
               und der wahre Fehler gegen die Ground Truth (Rotation, Position).

B  "ba_only"   Struktur und Posen werden korrekt vorgegeben, allein die
               Intrinsik startet falsch. Damit zeigt sich, ob das Bundle
               Adjustment die Brennweite ueberhaupt zurueckholen kann, wenn es
               nicht in einem selbstkonsistenten Minimum sitzt.

Beide Reihen setzen die Zufallszahlengeneratoren von NumPy und OpenCV, so dass
zwei Aufrufe dasselbe Ergebnis liefern; genau das fehlt der Pipeline selbst.

    python paper/scripts/focal_experiment.py --gt-dir <buddha> \
        --out paper/figures/l/focal_experiment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from gt_pose_eval import load_gt, rot_angle_deg, umeyama  # noqa: E402
from sfm.bundle_adjustment import BundleAdjuster  # noqa: E402
from sfm.geometric_verification import GeometricVerifier  # noqa: E402
from sfm.reconstruction import IncrementalSfM  # noqa: E402
from sfm.utils import reprojection_error  # noqa: E402

# Aufnahmeparameter des Datensatzes
W, H = 2736, 1540
F_TRUE = 1860.8968          # Mittel der GT-Brennweiten, px
F_GUESS = 2736.0            # was estimate_intrinsics ohne EXIF ansetzt: max(W, H)

# Versuchsparameter
NOISE_PX = 0.5              # Standardabweichung des Keypoint-Rauschens
N_PTS = 1500                # Punkte auf der synthetischen Oberflaeche
STRIDE = 3                  # jede dritte GT-Kamera, ergibt 23 Ansichten
RADIUS = 0.6                # Radius der Punktwolke in Weltkoordinaten
VIS_ANGLE_DEG = 75.0        # Sichtbarkeitskegel um die Flaechennormale
FACTORS = [0.70, 0.85, 1.00, 1.15, 1.30, 1.4703, 1.70]

log = logging.getLogger("focal_experiment")


# ── Szene ────────────────────────────────────────────────────────────────────

def build_scene(gt_dir: Path, rng: np.random.Generator):
    """GT-Posen einlesen und eine synthetische Oberflaeche im Szenenzentrum bauen."""
    gt = load_gt(gt_dir)
    keys = sorted(gt)[::STRIDE]
    poses = [(gt[k]["R"], (-gt[k]["R"] @ gt[k]["C"]).reshape(3, 1), gt[k]["C"])
             for k in keys]

    # Szenenzentrum als Schnittpunkt der optischen Achsen (kleinste Quadrate)
    C = np.array([p[2] for p in poses])
    d = np.array([p[0][2] for p in poses])
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, dd in zip(C, d):
        P = np.eye(3) - np.outer(dd, dd)
        A += P
        b += P @ c
    centre = np.linalg.solve(A, b)

    normals = rng.normal(size=(N_PTS, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    radii = RADIUS * (1.0 + 0.15 * rng.normal(size=(N_PTS, 1)))
    X = centre + normals * radii
    return keys, poses, X, normals, centre


def _project(R, t, K, X):
    Xc = (R @ X.T + t).T
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    return np.stack([u, v], 1), z


def make_observations(poses, X, normals, K_true, rng):
    """Analytische Merkmale und exakte Korrespondenzen je Bildpaar."""
    feats, widx = {}, {}
    cos_thr = np.cos(np.radians(VIS_ANGLE_DEG))
    for i, (R, t, C) in enumerate(poses):
        px, z = _project(R, t, K_true, X)
        view = C - X
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        facing = (view * normals).sum(1) > cos_thr
        vis = (facing & (z > 0.05)
               & (px[:, 0] > 4) & (px[:, 0] < W - 4)
               & (px[:, 1] > 4) & (px[:, 1] < H - 4))
        kps = px[vis] + rng.normal(0.0, NOISE_PX, (int(vis.sum()), 2))
        feats[i] = {
            "keypoints": kps.astype(np.float32),
            "descriptors": np.zeros((len(kps), 128), np.float32),
            "image_path": Path(f"synthetic_{i:03d}.png"),
            "image_shape": (H, W, 3),
        }
        widx[i] = np.where(vis)[0]

    matches = {}
    for i in range(len(poses)):
        inv_i = {int(w): k for k, w in enumerate(widx[i])}
        for j in range(i + 1, len(poses)):
            common = np.intersect1d(widx[i], widx[j])
            if len(common) < 20:
                continue
            inv_j = {int(w): k for k, w in enumerate(widx[j])}
            matches[(i, j)] = np.array(
                [[inv_i[int(w)], inv_j[int(w)]] for w in common], dtype=np.int32)
    return feats, matches, widx


def K_of(f: float) -> np.ndarray:
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], float)


# ── Reihe A: vollstaendige Rekonstruktion je Brennweite ──────────────────────

def run_reconstruction(f_assumed, feats, matches, poses, seed):
    np.random.seed(seed)
    cv2.setRNGSeed(seed)

    K = K_of(f_assumed)
    t0 = time.time()
    verified = GeometricVerifier(ransac_threshold=1.0, min_inliers=15).verify_all(
        feats, matches, K)
    sfm = IncrementalSfM(features=feats, verified_pairs=verified, K=K,
                         max_reproj_error=4.0, ba_interval=5)
    cams, pts, obs, _ = sfm.reconstruct()
    wall = time.time() - t0

    K_end = sfm.K
    errs = np.array([
        reprojection_error(pts[pi], np.array([x, y]), K_end,
                           cams[img]["R"], cams[img]["t"])
        for img, pi, x, y in obs if img in cams and pi < len(pts)
    ])
    errs = errs[np.isfinite(errs)]

    idx = sorted(cams)
    out = {
        "f_assumed": float(f_assumed),
        "f_error_pct": 100.0 * (f_assumed - F_TRUE) / F_TRUE,
        "n_pairs_verified": len(verified),
        "n_cams": len(cams),
        "n_pts": len(pts),
        "n_obs": len(obs),
        "wall_s": wall,
        "f_after_ba": float(K_end[0, 0]),
        "f_moved_px": float(K_end[0, 0] - f_assumed),
        "rmse_px": float(np.sqrt(np.mean(errs ** 2))) if errs.size else None,
        "median_px": float(np.median(errs)) if errs.size else None,
    }
    if len(idx) >= 3:
        C_est = np.array([-cams[i]["R"].T @ cams[i]["t"].reshape(3) for i in idx])
        C_gt = np.array([poses[i][2] for i in idx])
        s, R_a, t_a = umeyama(C_est, C_gt)
        C_al = (s * (R_a @ C_est.T).T) + t_a
        extent = float(np.linalg.norm(C_gt.max(0) - C_gt.min(0)))
        pos = 100.0 * np.linalg.norm(C_al - C_gt, axis=1) / extent
        rot = np.array([rot_angle_deg(poses[i][0] @ (cams[i]["R"] @ R_a.T).T)
                        for i in idx])
        out |= {"pos_err_med_pct": float(np.median(pos)),
                "pos_err_max_pct": float(pos.max()),
                "rot_err_med_deg": float(np.median(rot)),
                "rot_err_max_deg": float(rot.max())}
    return out


# ── Reihe B: nur das Bundle Adjustment, korrekte Struktur ────────────────────

def _obs_rmse(cameras, pts, observations, K):
    errs = np.array([
        reprojection_error(pts[pi], np.array([x, y]), K,
                           cameras[img]["R"], cameras[img]["t"])
        for img, pi, x, y in observations
    ])
    errs = errs[np.isfinite(errs)]
    return float(np.sqrt(np.mean(errs ** 2))) if errs.size else float("nan")


def _patch_solver(**overrides):
    """Kontextmanager: schreibt Argumente in jeden least_squares-Aufruf des BA.

    Damit laesst sich pruefen, ob die Loesereinstellung und nicht das Modell
    verhindert, dass die Brennweite sich bewegt. Das Repository bleibt
    unveraendert.
    """
    import contextlib

    import sfm.bundle_adjustment as ba_mod

    @contextlib.contextmanager
    def _ctx():
        original = ba_mod.least_squares
        if not overrides:
            yield
            return

        def patched(*a, **kw):
            kw.update(overrides)
            return original(*a, **kw)

        ba_mod.least_squares = patched
        try:
            yield
        finally:
            ba_mod.least_squares = original

    return _ctx()


def run_ba_only(f_start, poses, X, widx, feats, seed, solver=None, adjuster=None):
    """Posen und Punkte korrekt, allein die Brennweite startet falsch.

    Damit laesst sich unterscheiden, ob das Bundle Adjustment die Brennweite
    grundsaetzlich nicht bewegen kann oder ob es sie nur dann stehen laesst,
    wenn die Rekonstruktion bereits zu ihr passt.
    """
    np.random.seed(seed)
    cv2.setRNGSeed(seed)

    cameras = {i: {"R": poses[i][0], "t": poses[i][1], "K": K_of(f_start)}
               for i in range(len(poses))}
    observations = []
    for i in range(len(poses)):
        kps = feats[i]["keypoints"].astype(np.float64)
        for local, world in enumerate(widx[i]):
            observations.append((i, int(world), float(kps[local, 0]),
                                 float(kps[local, 1])))

    rmse_init = _obs_rmse(cameras, X, observations, K_of(f_start))
    ba = adjuster if adjuster is not None else BundleAdjuster()

    # Abbruchmeldung des Loesers mitschneiden
    import sfm.bundle_adjustment as ba_mod
    captured: list[str] = []

    class _Grab(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Grab()
    ba_mod.logger.addHandler(handler)
    ba_mod.logger.setLevel(logging.INFO)
    t0 = time.time()
    try:
        with _patch_solver(**(solver or {})):
            cams_opt, X_opt, K_ref, _ = ba.adjust(
                cameras, X.copy(), observations, K_of(f_start),
                refine_intrinsics=True)
    finally:
        ba_mod.logger.removeHandler(handler)
    wall = time.time() - t0
    termination = next((m.split("condition")[0].split(", ")[-1].strip()
                        for m in reversed(captured) if "BA final RMSE" in m), None)
    f_end = float(K_ref[0, 0]) if K_ref is not None else float(f_start)
    K_end = K_ref if K_ref is not None else K_of(f_start)
    rmse_final = _obs_rmse(cams_opt, X_opt, observations, K_end)

    # Wie stark hat das BA statt der Brennweite die Geometrie veraendert?
    C_est = np.array([-cams_opt[i]["R"].T @ cams_opt[i]["t"].reshape(3)
                      for i in range(len(poses))])
    C_gt = np.array([poses[i][2] for i in range(len(poses))])
    s, R_a, t_a = umeyama(C_est, C_gt)
    C_al = (s * (R_a @ C_est.T).T) + t_a
    extent = float(np.linalg.norm(C_gt.max(0) - C_gt.min(0)))
    rot = np.array([rot_angle_deg(poses[i][0] @ (cams_opt[i]["R"] @ R_a.T).T)
                    for i in range(len(poses))])
    return {
        "f_start": float(f_start),
        "f_start_err_pct": 100.0 * (f_start - F_TRUE) / F_TRUE,
        "f_end": f_end,
        "f_end_err_pct": 100.0 * (f_end - F_TRUE) / F_TRUE,
        "n_obs": len(observations),
        "wall_s": wall,
        "termination": termination,
        "rmse_init_px": rmse_init,
        "rmse_final_px": rmse_final,
        "scale_after_ba": float(s),
        "pos_err_med_pct": float(np.median(
            100.0 * np.linalg.norm(C_al - C_gt, axis=1) / extent)),
        "rot_err_med_deg": float(np.median(rot)),
        "shape_rms_change": float(np.sqrt(np.mean(
            np.sum((X_opt - X) ** 2, axis=1)))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, help="Verzeichnis mit *_P.txt")
    ap.add_argument("--out", default=str(REPO / "paper/figures/l/focal_experiment.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.setLevel(logging.INFO)

    rng = np.random.default_rng(args.seed)
    keys, poses, X, normals, centre = build_scene(Path(args.gt_dir), rng)
    feats, matches, widx = make_observations(poses, X, normals, K_of(F_TRUE), rng)

    n_kps = sum(len(f["keypoints"]) for f in feats.values())
    track = np.zeros(len(X), int)
    for i in widx:
        track[widx[i]] += 1
    seen = track[track > 0]
    log.info("Szene: %d Kameras, %d Punkte, %d Beobachtungen, %d Paare",
             len(poses), len(X), n_kps, len(matches))
    log.info("mittlere Sichtbarkeit je Punkt: %.2f Kameras", seen.mean())

    result = {
        "setup": {
            "gt_dir": str(args.gt_dir), "seed": args.seed,
            "n_cameras": len(poses), "camera_stride": STRIDE,
            "image_size": [W, H], "f_true_px": F_TRUE, "f_guess_px": F_GUESS,
            "n_surface_points": int(N_PTS), "noise_px": NOISE_PX,
            "visibility_cone_deg": VIS_ANGLE_DEG,
            "scene_centre": [float(v) for v in centre],
            "n_keypoints": int(n_kps), "n_matchable_pairs": len(matches),
            "mean_visibility": float(seen.mean()),
            "images": [str(k) for k in keys],
        },
        "sweep": [],
        "ba_only": [],
        "solver_variants": [],
    }

    for factor in FACTORS:
        f = F_TRUE * factor
        r = run_reconstruction(f, feats, matches, poses, args.seed)
        result["sweep"].append(r)
        log.info("f=%.1f px (%+.1f %%): %d Kameras, %d Punkte, RMSE %.2f px, "
                 "Rotationsfehler %.2f Grad, BA verschiebt f um %.3f px",
                 f, r["f_error_pct"], r["n_cams"], r["n_pts"],
                 r["rmse_px"] or float("nan"),
                 r.get("rot_err_med_deg", float("nan")), r["f_moved_px"])

    for factor in FACTORS:
        r = run_ba_only(F_TRUE * factor, poses, X, widx, feats, args.seed)
        result["ba_only"].append(r)
        log.info("BA allein: Start f=%.1f px (%+.1f %%) endet bei %.1f px (%+.1f %%), "
                 "RMSE %.2f -> %.2f px, Rotationsfehler danach %.2f Grad",
                 r["f_start"], r["f_start_err_pct"], r["f_end"], r["f_end_err_pct"],
                 r["rmse_init_px"], r["rmse_final_px"], r["rot_err_med_deg"])

    # Reihe C: dieselbe Aufgabe wie Reihe B bei +47 %, aber mit veraenderter
    # Einstellung des Loesers. Zeigt, ob die Brennweite unbeweglich ist oder
    # nur der voreingestellte Abbruch sie stehen laesst.
    # max_nfev wird im BundleAdjuster mit der Parameterzahl multipliziert; der
    # Faktor 1 entspricht hier rund 4.500 Funktionsauswertungen und damit einem
    # Vielfachen dessen, was die Voreinstellung bis zum Abbruch verbraucht.
    strict = dict(ftol=1e-12, gtol=1e-12, xtol=1e-12, max_nfev=1)
    variants = [
        ("Voreinstellung", {}, None),
        ("x_scale='jac'", {"x_scale": "jac"}, None),
        ("strenge Toleranzen", {}, BundleAdjuster(**strict)),
        ("x_scale='jac' und strenge Toleranzen", {"x_scale": "jac"},
         BundleAdjuster(**strict)),
    ]
    for label, solver, adjuster in variants:
        r = run_ba_only(F_GUESS, poses, X, widx, feats, args.seed,
                        solver=solver, adjuster=adjuster)
        r["variant"] = label
        result["solver_variants"].append(r)
        log.info("Loeservariante %-38s f: %.1f -> %.1f px (%+.1f %%), "
                 "RMSE %.2f -> %.2f px, %s, %.1f s", label, r["f_start"],
                 r["f_end"], r["f_end_err_pct"], r["rmse_init_px"],
                 r["rmse_final_px"], r["termination"], r["wall_s"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("geschrieben: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
