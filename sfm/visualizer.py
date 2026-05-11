"""
sfm/visualizer.py — Optional visualization layer for the SfM pipeline.

ZERO COST when enabled=False: every hook method returns instantly.
Heavy dependencies (matplotlib, networkx, imageio, open3d) are imported
exclusively inside __init__ when enabled=True, so this module is safe
to import anywhere without pulling in visualization libraries.

Usage
-----
    viz = SfMVisualizer(enabled=True, output_dir="sfm_visualization")
    viz.on_pipeline_start(image_paths, K, args)
    ...
    # each hook is a one-liner no-op when enabled=False
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SfMVisualizer:
    """
    Event-driven visualization harness for the SfM pipeline.

    All public methods are lifecycle hooks called by the pipeline at key events.
    When enabled=False (the default) every hook is a guaranteed no-op — no
    imports, no allocations, no file I/O.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        enabled: bool = False,
        output_dir: str = "sfm_visualization",
        n_samples: int = 3,
        fmt: str = "png",
        interactive: bool = False,
        save_video: bool = False,
        dpi: int = 150,
        seed: int = 42,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            return

        # Heavy imports — ONLY executed when visualization is active
        import matplotlib
        matplotlib.use("Agg")               # headless / no GUI popup
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec

        self._plt      = plt
        self._cm       = cm
        self._mcolors  = mcolors
        self._GS       = GridSpec

        self._out      = Path(output_dir)
        self._n_samp   = n_samples
        self._fmt      = fmt
        self._interact = interactive
        self._video    = save_video
        self._dpi      = dpi
        self._rng      = np.random.default_rng(seed)

        # ── Pipeline-state accumulators ───────────────────────────────────
        self._features:     Optional[dict]       = None
        self._all_matches:  Optional[dict]       = None
        self._verified:     Optional[dict]       = None
        self._observations: Optional[list]       = None
        self._cameras:      dict                 = {}
        self._pts3d:        np.ndarray           = np.zeros((0, 3))
        self._colors:       Optional[np.ndarray] = None
        self._K:            Optional[np.ndarray] = None
        self._seed_pair:    Optional[tuple]      = None
        self._step_hist:    List[dict]           = []   # per-registration
        self._ba_hist:      List[dict]           = []   # per-BA round
        self._n_saved:      int                  = 0

        for sub in ("00_summary", "01_features", "02_matching",
                    "03_reconstruction", "04_pointcloud"):
            (self._out / sub).mkdir(parents=True, exist_ok=True)

        logger.info(f"[VIZ] Enabled  →  {self._out.resolve()}/")

    # ── Lifecycle hooks (all no-op when disabled) ─────────────────────────────

    def on_pipeline_start(self, image_paths: list, K: np.ndarray, args) -> None:
        if not self.enabled:
            return
        self._image_paths = list(image_paths)
        self._K = K.copy()
        import time
        self._t0 = time.time()
        logger.info(f"[VIZ] Pipeline start: {len(image_paths)} images")

    def on_all_features_done(self, features: dict) -> None:
        if not self.enabled:
            return
        self._features = features
        try:
            self._render_features()
        except Exception as exc:
            logger.warning(f"[VIZ] Feature rendering failed: {exc}")

    def on_all_matching_done(self, all_matches: dict, features: dict) -> None:
        if not self.enabled:
            return
        self._all_matches = all_matches
        self._features = features

    def on_geometric_verification_done(
        self,
        all_matches: dict,
        verified: dict,
        features: dict,
    ) -> None:
        if not self.enabled:
            return
        self._all_matches = all_matches
        self._verified    = verified
        self._features    = features
        try:
            self._render_matching()
        except Exception as exc:
            logger.warning(f"[VIZ] Matching rendering failed: {exc}")

    def on_seed_pair_selected(self, id_a: int, id_b: int) -> None:
        if not self.enabled:
            return
        self._seed_pair = (id_a, id_b)

    def on_camera_registered(
        self,
        image_id,                    # int or (int, int) for seed pair
        cameras_snap: dict,
        pts_snap: np.ndarray,
        n_new_pts: int,
    ) -> None:
        if not self.enabled:
            return
        self._cameras = cameras_snap
        self._pts3d   = pts_snap
        step = len(self._step_hist) + 1
        is_seed = isinstance(image_id, tuple)
        self._step_hist.append({
            "step":       step,
            "image_id":   image_id,
            "is_seed":    is_seed,
            "n_cameras":  len(cameras_snap),
            "n_pts":      len(pts_snap),
            "n_new":      n_new_pts,
        })
        try:
            self._render_registration_step(step, image_id, cameras_snap,
                                           pts_snap, n_new_pts, is_seed)
        except Exception as exc:
            logger.warning(f"[VIZ] Registration step {step} failed: {exc}")

    def on_bundle_adjustment_run(
        self,
        ba_step:      int,
        rmse_before:  float,
        rmse_after:   float,
        n_cameras:    int,
        n_pts_before: int,
        n_pts_after:  int,
    ) -> None:
        if not self.enabled:
            return
        self._ba_hist.append({
            "ba_step":      ba_step,
            "rmse_before":  rmse_before,
            "rmse_after":   rmse_after,
            "n_cameras":    n_cameras,
            "n_pts_before": n_pts_before,
            "n_pts_after":  n_pts_after,
        })

    def on_reconstruction_complete(
        self,
        cameras:      dict,
        points_3d:    np.ndarray,
        observations: list,
        features:     dict,
        K:            np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        self._cameras      = cameras
        self._pts3d        = points_3d
        self._observations = observations
        self._features     = features
        self._K            = K
        try:
            self._render_ba_convergence()
            self._render_point_lifecycle()
            self._render_camera_poses()
            sample_idxs = self._sample_image_indices(self._n_samp, registered=cameras)
            for idx in sample_idxs:
                try:
                    self._render_reprojection_errors(idx)
                except Exception as exc:
                    logger.warning(f"[VIZ] Reproj errors for {idx}: {exc}")
        except Exception as exc:
            logger.warning(f"[VIZ] Reconstruction summary failed: {exc}")

    def on_pipeline_complete(
        self,
        cameras:   dict,
        points_3d: np.ndarray,
        colors:    np.ndarray,
        stats:     dict,
    ) -> None:
        if not self.enabled:
            return
        self._cameras = cameras
        self._pts3d   = points_3d
        self._colors  = colors
        try:
            self._render_pointcloud_views()
        except Exception as exc:
            logger.warning(f"[VIZ] Point cloud views failed: {exc}")
        try:
            self._render_summary_dashboard(stats)
        except Exception as exc:
            logger.warning(f"[VIZ] Summary dashboard failed: {exc}")
        if self._video:
            try:
                self._save_reconstruction_video()
            except Exception as exc:
                logger.warning(f"[VIZ] Reconstruction video failed: {exc}")
            try:
                self._save_turntable_video()
            except Exception as exc:
                logger.warning(f"[VIZ] Turntable video failed: {exc}")
        if self._interact:
            try:
                self._open_interactive_viewer()
            except Exception as exc:
                logger.warning(f"[VIZ] Interactive viewer failed: {exc}")
        self._print_summary()

    # ── Feature visualizations ────────────────────────────────────────────────

    def _render_features(self) -> None:
        feats = self._features
        if not feats:
            return
        idxs = self._sample_image_indices(self._n_samp)
        for idx in idxs:
            self._render_keypoint_overlay(idx)
            self._render_density_heatmap(idx)
        self._render_feature_statistics()

    def _render_keypoint_overlay(self, idx: int) -> None:
        feat = self._features[idx]
        kps  = feat["keypoints"]  # (N, 2) float32
        img  = self._load_image_rgb(feat["image_path"])
        stem = Path(feat["image_path"]).stem

        fig, ax = self._plt.subplots(figsize=(12, 8))
        if img is not None:
            ax.imshow(img)
        else:
            h, w = feat.get("image_shape", (480, 640, 3))[:2]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_facecolor("#cccccc")

        if len(kps) > 0:
            # Color by descending index (proxy for SIFT response strength)
            colors = self._cm.RdYlBu(np.linspace(0, 1, len(kps)))
            ax.scatter(kps[:, 0], kps[:, 1], c=colors, s=12,
                       linewidths=0, alpha=0.7)

        extractor = "kornia/GPU" if self._features[idx].get("_backend") else "SIFT"
        ax.set_title(
            f"Image: {stem}  |  Keypoints: {len(kps):,}  |  Extractor: {extractor}",
            fontsize=11,
        )
        ax.axis("off")
        self._save_fig(fig, self._out / "01_features" / f"features_{stem}.{self._fmt}")

    def _render_density_heatmap(self, idx: int) -> None:
        feat = self._features[idx]
        kps  = feat["keypoints"]
        img  = self._load_image_rgb(feat["image_path"])
        stem = Path(feat["image_path"]).stem
        h, w = feat.get("image_shape", (480, 640, 3))[:2]

        fig, axes = self._plt.subplots(1, 2, figsize=(16, 6))

        for ax, bg in zip(axes, [img, None]):
            if bg is not None:
                ax.imshow(bg, alpha=0.5)
            else:
                ax.set_facecolor("#111111")

        if len(kps) > 0:
            heatmap, xedges, yedges = np.histogram2d(
                kps[:, 0], kps[:, 1],
                bins=(min(64, w // 10), min(48, h // 10)),
                range=[[0, w], [0, h]],
            )
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap.T, sigma=2)
            extent  = [0, w, h, 0]
            for ax in axes:
                ax.imshow(heatmap, extent=extent, cmap="hot",
                          alpha=0.6, interpolation="bilinear", aspect="auto")

        axes[0].set_title("Overlay on image", fontsize=10)
        axes[1].set_title("Heatmap only", fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Feature density — {stem}  ({len(kps):,} kps)", fontsize=12)

        self._save_fig(fig, self._out / "01_features" / f"density_{stem}.{self._fmt}")

    def _render_feature_statistics(self) -> None:
        feats = self._features
        if not feats:
            return

        counts = [len(feats[i]["keypoints"]) for i in sorted(feats)]
        names  = [Path(feats[i]["image_path"]).stem for i in sorted(feats)]
        n      = len(counts)

        fig, axes = self._plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Feature Extraction Statistics", fontsize=13, y=1.02)

        # Histogram of keypoints-per-image
        axes[0].hist(counts, bins=max(5, n // 3), color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Keypoints per image")
        axes[0].set_ylabel("# Images")
        axes[0].set_title("Keypoint count distribution")
        axes[0].axvline(np.mean(counts), color="red", ls="--", label=f"mean={np.mean(counts):.0f}")
        axes[0].legend(fontsize=9)

        # Bar chart: keypoints per image
        top_n  = min(n, 15)
        sorted_idx = np.argsort(counts)[::-1][:top_n]
        bar_colors = ["#d62728" if counts[i] < 100 else "steelblue" for i in sorted_idx]
        axes[1].barh(range(top_n), [counts[i] for i in sorted_idx],
                     color=bar_colors)
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels([names[i] for i in sorted_idx], fontsize=8)
        axes[1].set_xlabel("Keypoints")
        axes[1].set_title(f"Top-{top_n} images by keypoint count")
        axes[1].invert_yaxis()

        # Summary stats text
        axes[2].axis("off")
        total = sum(counts)
        summary = (
            f"Total images:   {n}\n"
            f"Total keypoints: {total:,}\n"
            f"Mean per image:  {np.mean(counts):.0f}\n"
            f"Min / Max:       {min(counts)} / {max(counts)}\n"
            f"Images < 100 kps: {sum(c<100 for c in counts)}"
        )
        axes[2].text(0.1, 0.5, summary, transform=axes[2].transAxes,
                     fontsize=11, verticalalignment="center",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))
        axes[2].set_title("Summary")

        fig.tight_layout()
        self._save_fig(fig, self._out / "01_features" / f"feature_statistics.{self._fmt}")

    # ── Matching visualizations ───────────────────────────────────────────────

    def _render_matching(self) -> None:
        if not self._verified:
            return
        pairs  = list(self._verified.keys())
        sample = self._sample_pair_indices(self._n_samp, pairs)
        for (i, j) in sample:
            try:
                self._render_match_pair(i, j)
            except Exception as exc:
                logger.warning(f"[VIZ] Match pair ({i},{j}) failed: {exc}")
            try:
                self._render_epipolar(i, j)
            except Exception as exc:
                logger.warning(f"[VIZ] Epipolar ({i},{j}) failed: {exc}")
        try:
            self._render_match_matrix()
        except Exception as exc:
            logger.warning(f"[VIZ] Match matrix failed: {exc}")
        try:
            self._render_connectivity_graph()
        except Exception as exc:
            logger.warning(f"[VIZ] Connectivity graph failed: {exc}")

    def _render_match_pair(self, i: int, j: int) -> None:
        feat_i = self._features[i]
        feat_j = self._features[j]
        img_i  = self._load_image_rgb(feat_i["image_path"])
        img_j  = self._load_image_rgb(feat_j["image_path"])

        hi, wi = feat_i.get("image_shape", (480, 640, 3))[:2]
        hj, wj = feat_j.get("image_shape", (480, 640, 3))[:2]
        if img_i is None:
            img_i = np.full((hi, wi, 3), 200, dtype=np.uint8)
        if img_j is None:
            img_j = np.full((hj, wj, 3), 200, dtype=np.uint8)

        # Composite side-by-side canvas
        h_out = max(hi, hj)
        canvas = np.zeros((h_out, wi + wj, 3), dtype=np.uint8)
        canvas[:hi, :wi]    = img_i
        canvas[:hj, wi:wi+wj] = img_j

        fig, ax = self._plt.subplots(figsize=(16, 7))
        ax.imshow(canvas)

        kps_i = self._features[i]["keypoints"]
        kps_j = self._features[j]["keypoints"]

        ver_data  = self._verified[(i, j)]
        inliers   = ver_data["inlier_matches"]          # (M, 2)
        raw       = self._all_matches.get((i, j), np.zeros((0, 2), dtype=np.int32))

        # Build inlier set for fast lookup
        inlier_set = {(int(m[0]), int(m[1])) for m in inliers}

        n_inliers  = len(inliers)
        n_total    = len(raw)
        n_outliers = n_total - n_inliers

        # Draw a random subset for clarity (max 200 lines)
        rng_draw = np.random.default_rng(0)
        draw_idx = rng_draw.choice(len(raw), min(200, len(raw)), replace=False)

        for k in draw_idx:
            ki, kj = int(raw[k, 0]), int(raw[k, 1])
            x1, y1 = float(kps_i[ki, 0]), float(kps_i[ki, 1])
            x2, y2 = float(kps_j[kj, 0]) + wi, float(kps_j[kj, 1])
            is_in  = (ki, kj) in inlier_set
            color  = (0.1, 0.8, 0.1, 0.6) if is_in else (0.9, 0.1, 0.1, 0.4)
            ax.plot([x1, x2], [y1, y2], "-", color=color, lw=0.7)

        pct = 100 * n_inliers / max(n_total, 1)
        si  = Path(feat_i["image_path"]).stem
        sj  = Path(feat_j["image_path"]).stem
        ax.set_title(
            f"{si}  ↔  {sj}   |   Total: {n_total}   "
            f"Inliers: {n_inliers} ({pct:.1f}%)   Outliers: {n_outliers}",
            fontsize=10,
        )
        ax.axis("off")
        out = self._out / "02_matching" / f"matches_{i:03d}_{j:03d}.{self._fmt}"
        self._save_fig(fig, out)

    def _render_match_matrix(self) -> None:
        if not self._verified or not self._features:
            return
        idxs = sorted(self._features.keys())
        n    = len(idxs)
        idx_map = {v: k for k, v in enumerate(idxs)}

        mat = np.zeros((n, n), dtype=np.int32)
        for (i, j), data in self._verified.items():
            r, c = idx_map[i], idx_map[j]
            mat[r, c] = data["n_inliers"]
            mat[c, r] = data["n_inliers"]

        fig, ax = self._plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
        masked = np.ma.masked_where(mat == 0, mat)
        im = ax.imshow(masked, cmap="Blues", aspect="auto")
        self._plt.colorbar(im, ax=ax, label="Inlier matches")

        labels = [Path(self._features[i]["image_path"]).stem for i in idxs]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

        thresh = mat.max() * 0.5 if mat.max() > 0 else 1
        for r in range(n):
            for c in range(n):
                if mat[r, c] > 0:
                    ax.text(c, r, str(mat[r, c]),
                            ha="center", va="center", fontsize=7,
                            color="white" if mat[r, c] > thresh else "black")

        ax.set_title("Inlier match matrix (image pairs)", fontsize=11)
        fig.tight_layout()
        self._save_fig(fig, self._out / "02_matching" / f"match_matrix.{self._fmt}")

    def _render_connectivity_graph(self) -> None:
        try:
            import networkx as nx
        except ImportError:
            logger.warning("[VIZ] networkx not installed — skipping connectivity graph")
            return

        if not self._verified or not self._features:
            return

        G = nx.Graph()
        for idx in self._features:
            G.add_node(idx, label=Path(self._features[idx]["image_path"]).stem)
        for (i, j), data in self._verified.items():
            G.add_edge(i, j, weight=data["n_inliers"])

        pos = nx.spring_layout(G, seed=42, k=2)

        n_inliers = [G[u][v]["weight"] for u, v in G.edges()]
        n_max     = max(n_inliers) if n_inliers else 1
        widths    = [1.0 + 4.0 * w / n_max for w in n_inliers]
        isolated  = list(nx.isolates(G))
        connected = [n for n in G.nodes() if n not in isolated]

        fig, ax = self._plt.subplots(figsize=(max(10, len(G.nodes) * 0.8), 8))

        nx.draw_networkx_nodes(G, pos, nodelist=connected,
                               node_color="steelblue", node_size=500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=isolated,
                               node_color="red", node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths,
                               edge_color=n_inliers, edge_cmap=self._cm.YlOrRd,
                               ax=ax, alpha=0.8)
        labels = {i: Path(self._features[i]["image_path"]).stem for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        ax.set_title(
            f"Image connectivity graph  "
            f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
            fontsize=11,
        )
        ax.axis("off")
        fig.tight_layout()
        self._save_fig(fig, self._out / "02_matching" / f"connectivity_graph.{self._fmt}")

    def _render_epipolar(self, i: int, j: int) -> None:
        ver = self._verified.get((i, j))
        if ver is None or "F" not in ver or ver["F"] is None:
            return
        F = ver["F"]

        feat_i = self._features[i]
        feat_j = self._features[j]
        img_i  = self._load_image_rgb(feat_i["image_path"])
        img_j  = self._load_image_rgb(feat_j["image_path"])
        hi, wi = feat_i.get("image_shape", (480, 640, 3))[:2]
        hj, wj = feat_j.get("image_shape", (480, 640, 3))[:2]

        inliers = ver["inlier_matches"]
        if len(inliers) == 0:
            return
        sample_k = self._rng.choice(len(inliers), min(10, len(inliers)), replace=False)

        kps_i = feat_i["keypoints"]
        kps_j = feat_j["keypoints"]

        fig, (ax1, ax2) = self._plt.subplots(1, 2, figsize=(16, 7))
        for ax, img, h, w in ((ax1, img_i, hi, wi), (ax2, img_j, hj, wj)):
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#cccccc")
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            ax.axis("off")

        colors_ep = self._cm.tab10(np.linspace(0, 1, len(sample_k)))

        for idx_k, c in zip(sample_k, colors_ep):
            ki = int(inliers[idx_k, 0])
            kj = int(inliers[idx_k, 1])

            pt_i = np.array([kps_i[ki, 0], kps_i[ki, 1], 1.0])
            pt_j = np.array([kps_j[kj, 0], kps_j[kj, 1], 1.0])

            ax1.scatter(*pt_i[:2], s=60, color=c, zorder=5)
            ax2.scatter(*pt_j[:2], s=60, color=c, zorder=5)

            # Epipolar line in image 2: l2 = F @ pt_i
            l2 = F @ pt_i
            ep = self._epipolar_endpoints(l2, wj, hj)
            if ep:
                (x0, y0), (x1y, y1y) = ep
                ax2.plot([x0, x1y], [y0, y1y], color=c, lw=1.2, alpha=0.8)

            # Epipolar line in image 1: l1 = F.T @ pt_j
            l1 = F.T @ pt_j
            ep = self._epipolar_endpoints(l1, wi, hi)
            if ep:
                (x0, y0), (x1y, y1y) = ep
                ax1.plot([x0, x1y], [y0, y1y], color=c, lw=1.2, alpha=0.8)

        si = Path(feat_i["image_path"]).stem
        sj = Path(feat_j["image_path"]).stem
        fig.suptitle(f"Epipolar geometry — {si}  ↔  {sj}", fontsize=11)
        fig.tight_layout()
        out = self._out / "02_matching" / f"epipolar_{i:03d}_{j:03d}.{self._fmt}"
        self._save_fig(fig, out)

    # ── Reconstruction visualizations ─────────────────────────────────────────

    def _render_registration_step(
        self,
        step:        int,
        image_id,
        cameras:     dict,
        pts3d:       np.ndarray,
        n_new:       int,
        is_seed:     bool = False,
    ) -> None:
        fig = self._plt.figure(figsize=(18, 7))
        gs  = self._GS(1, 3, figure=fig, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        cam_list   = sorted(cameras.keys())
        n_cams     = len(cam_list)
        step_color = self._cm.plasma(np.linspace(0.1, 0.9, max(n_cams, 1)))

        # ── Panel 1: top-down camera map (XZ plane) ────────────────────────
        ax1.set_facecolor("#0d1117")
        ax1.set_title("Camera positions (top-down)", fontsize=9, color="white")

        for ci, cam_idx in enumerate(cam_list):
            R = cameras[cam_idx]["R"]
            t = cameras[cam_idx]["t"].flatten()
            C = -(R.T @ t)
            color = step_color[ci]

            # Forward direction in world XZ
            fwd = R[2, :]  # third row of R = camera Z in world
            fwd_xz = np.array([fwd[0], fwd[2]])
            norm = np.linalg.norm(fwd_xz)
            if norm > 1e-6:
                fwd_xz /= norm

            is_new = cam_idx == image_id or (is_seed and ci >= n_cams - 2)
            size   = 120 if is_new else 60
            zorder = 5 if is_new else 3

            ax1.scatter(C[0], C[2], s=size, c=[color], zorder=zorder,
                        edgecolors="yellow" if is_new else "none", linewidths=1.5)
            ax1.annotate(
                str(cam_idx),
                (C[0], C[2]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=color,
            )
            scale = 0.2
            ax1.annotate(
                "",
                xy=(C[0] + scale * fwd_xz[0], C[2] + scale * fwd_xz[1]),
                xytext=(C[0], C[2]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

        if len(pts3d) > 0:
            ax1.scatter(pts3d[:, 0], pts3d[:, 2], s=1, c="white", alpha=0.15, zorder=1)

        ax1.set_xlabel("X", color="gray", fontsize=8)
        ax1.set_ylabel("Z", color="gray", fontsize=8)
        ax1.tick_params(colors="gray", labelsize=7)
        for sp in ax1.spines.values():
            sp.set_color("#333333")

        # ── Panel 2: point cloud side view (XY) ───────────────────────────
        ax2.set_facecolor("#0d1117")
        ax2.set_title("3-D point cloud (side view)", fontsize=9, color="white")
        if len(pts3d) > 0:
            prev_n = max(0, len(pts3d) - n_new)
            ax2.scatter(pts3d[:prev_n, 0], pts3d[:prev_n, 1],
                        s=1, c="steelblue", alpha=0.4, zorder=1)
            if n_new > 0:
                ax2.scatter(pts3d[prev_n:, 0], pts3d[prev_n:, 1],
                            s=8, c="lime", alpha=0.9, zorder=3,
                            label=f"+{n_new} new")
                ax2.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        ax2.set_xlabel("X", color="gray", fontsize=8)
        ax2.set_ylabel("Y", color="gray", fontsize=8)
        ax2.tick_params(colors="gray", labelsize=7)
        ax2.text(0.02, 0.98, f"Total: {len(pts3d):,} pts",
                 transform=ax2.transAxes, fontsize=8,
                 color="white", va="top")
        for sp in ax2.spines.values():
            sp.set_color("#333333")

        # ── Panel 3: registration progress bars ───────────────────────────
        ax3.set_title("Registration progress", fontsize=9)
        steps_so_far = self._step_hist
        if steps_so_far:
            xs      = [s["step"]     for s in steps_so_far]
            n_pts_v = [s["n_pts"]    for s in steps_so_far]
            n_new_v = [s["n_new"]    for s in steps_so_far]
            ax3.bar(xs, n_pts_v, color="steelblue", alpha=0.6, label="total pts")
            ax3.bar(xs, n_new_v, color="lime",       alpha=0.9, label="new pts")
            ax3.set_xlabel("Registration step", fontsize=8)
            ax3.set_ylabel("3-D points", fontsize=8)
            ax3.legend(fontsize=8)

            ax3b = ax3.twinx()
            n_cams_v = [s["n_cameras"] for s in steps_so_far]
            ax3b.plot(xs, n_cams_v, "o--", color="orange", lw=1.5, ms=5)
            ax3b.set_ylabel("Cameras registered", color="orange", fontsize=8)
            ax3b.tick_params(axis="y", colors="orange", labelsize=7)

        label = "Seed pair" if is_seed else f"Camera {image_id}"
        fig.suptitle(
            f"Step {step:03d} — {label}  "
            f"| Cameras: {n_cams}  | Points: {len(pts3d):,}",
            fontsize=11,
        )

        tag = "seed_pair" if is_seed else "camera_registered"
        out = self._out / "03_reconstruction" / f"step_{step:03d}_{tag}.{self._fmt}"
        self._save_fig(fig, out)

    def _render_ba_convergence(self) -> None:
        if not self._ba_hist:
            logger.debug("[VIZ] No BA history — skipping convergence plot")
            return

        fig, axes = self._plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Bundle Adjustment Convergence", fontsize=12)

        rounds    = [b["ba_step"]      for b in self._ba_hist]
        before    = [b["rmse_before"]  for b in self._ba_hist]
        after     = [b["rmse_after"]   for b in self._ba_hist]
        n_cams    = [b["n_cameras"]    for b in self._ba_hist]
        n_pts_b   = [b["n_pts_before"] for b in self._ba_hist]
        n_pts_a   = [b["n_pts_after"]  for b in self._ba_hist]

        ax = axes[0]
        ax.plot(rounds, before, "o--", color="tomato",    lw=2, ms=7, label="Before BA")
        ax.plot(rounds, after,  "o-",  color="steelblue", lw=2, ms=7, label="After BA")
        ax.fill_between(rounds, before, after, alpha=0.15, color="steelblue")
        for k, (r, c) in enumerate(zip(rounds, n_cams)):
            ax.annotate(f"C={c}", (r, before[k]), textcoords="offset points",
                        xytext=(3, 5), fontsize=7, color="gray")
        ax.set_xlabel("BA round")
        ax.set_ylabel("RMSE (px)")
        ax.set_title("Reprojection RMSE")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        width = 0.35
        xs    = np.array(rounds)
        ax2.bar(xs - width / 2, n_pts_b, width, label="Before BA", color="tomato",    alpha=0.7)
        ax2.bar(xs + width / 2, n_pts_a, width, label="After BA",  color="steelblue", alpha=0.7)
        ax2.set_xlabel("BA round")
        ax2.set_ylabel("Point count")
        ax2.set_title("Points before / after BA")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        self._save_fig(fig, self._out / "03_reconstruction" / f"bundle_adjustment_convergence.{self._fmt}")

    def _render_point_lifecycle(self) -> None:
        if self._observations is None or len(self._pts3d) == 0:
            return

        n_pts = len(self._pts3d)
        obs_count = np.zeros(n_pts, dtype=np.int32)
        for img_idx, pt_idx, x, y in self._observations:
            if pt_idx < n_pts:
                obs_count[pt_idx] += 1

        fig, axes = self._plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Point Lifecycle", fontsize=12)

        ax = axes[0]
        ax.hist(obs_count, bins=max(10, obs_count.max()), color="steelblue",
                edgecolor="white")
        ax.set_xlabel("Number of observing cameras")
        ax.set_ylabel("Number of 3-D points")
        ax.set_title("Observation coverage per point")
        ax.axvline(obs_count.mean(), color="red", ls="--",
                   label=f"mean={obs_count.mean():.1f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Reprojection error per point from observations
        if self._cameras and self._K is not None:
            err_per_pt = [[] for _ in range(n_pts)]
            K = self._K
            for img_idx, pt_idx, x, y in self._observations:
                if pt_idx >= n_pts or img_idx not in self._cameras:
                    continue
                cam  = self._cameras[img_idx]
                R, t = cam["R"], cam["t"].flatten()
                X_c  = R @ self._pts3d[pt_idx] + t
                if X_c[2] > 1e-6:
                    f = K[0, 0]
                    u = f * X_c[0] / X_c[2] + K[0, 2]
                    v = f * X_c[1] / X_c[2] + K[1, 2]
                    err_per_pt[pt_idx].append(
                        float(np.sqrt((u - x) ** 2 + (v - y) ** 2))
                    )
            mean_errs   = np.array([np.mean(e) if e else np.nan for e in err_per_pt])
            valid       = ~np.isnan(mean_errs)
            max_err = float(mean_errs[valid].max()) if valid.any() else 1.0
            axes[1].scatter(obs_count[valid], mean_errs[valid], s=5, alpha=0.4,
                            c=self._cm.coolwarm(mean_errs[valid] / max(max_err, 1.0)))
            axes[1].set_xlabel("Observations per point")
            axes[1].set_ylabel("Mean reprojection error (px)")
            axes[1].set_title("Error vs. coverage")
            axes[1].axhline(1.0, color="green",  ls="--", label="1 px")
            axes[1].axhline(2.0, color="orange", ls="--", label="2 px")
            axes[1].axhline(4.0, color="red",    ls="--", label="4 px")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        self._save_fig(fig, self._out / "03_reconstruction" / f"point_lifecycle.{self._fmt}")

    def _render_camera_poses(self) -> None:
        if not self._cameras:
            return

        fig = self._plt.figure(figsize=(14, 10))
        ax  = fig.add_subplot(111, projection="3d")

        cam_list  = sorted(self._cameras.keys())
        n_cams    = len(cam_list)
        step_cols = self._cm.plasma(np.linspace(0.1, 0.9, max(n_cams, 1)))

        centers = []
        for ci, cam_idx in enumerate(cam_list):
            R = self._cameras[cam_idx]["R"]
            t = self._cameras[cam_idx]["t"].flatten()
            C = -(R.T @ t)
            centers.append(C)
            color = step_cols[ci]
            is_seed = self._seed_pair and cam_idx in self._seed_pair

            ax.scatter(*C, s=80 if is_seed else 40, c=[color],
                       edgecolors="gold" if is_seed else "none",
                       linewidths=1.5, zorder=5)

            # Draw camera frustum axes (short stubs for X/Y/Z)
            scale = 0.12
            for axis, col in [(R.T[:, 0], "r"), (R.T[:, 1], "g"), (R.T[:, 2], "b")]:
                end = C + scale * axis
                ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]],
                        color=col, lw=0.8, alpha=0.6)

        # Camera trajectory
        if len(centers) > 1:
            traj = np.array(centers)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    "w--", lw=0.8, alpha=0.4)

        # Point cloud
        if len(self._pts3d) > 0:
            sample_pts = self._pts3d
            if len(sample_pts) > 5000:
                idx = self._rng.choice(len(sample_pts), 5000, replace=False)
                sample_pts = sample_pts[idx]
            ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2],
                       s=0.5, c="lightgray", alpha=0.3, zorder=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"Camera poses — {n_cams} cameras  |  {len(self._pts3d):,} points",
            fontsize=11,
        )
        self._save_fig(fig, self._out / "03_reconstruction" / f"camera_poses_final.{self._fmt}")

    def _render_reprojection_errors(self, img_idx: int) -> None:
        if img_idx not in self._cameras or self._observations is None:
            return
        feat = self._features[img_idx]
        img  = self._load_image_rgb(feat["image_path"])
        K    = self._K
        cam  = self._cameras[img_idx]
        R, t = cam["R"], cam["t"].flatten()

        # Get observations for this image
        obs = [(pt_idx, x, y)
               for (ii, pt_idx, x, y) in self._observations
               if ii == img_idx and pt_idx < len(self._pts3d)]

        if not obs:
            return

        h, w = feat.get("image_shape", (480, 640, 3))[:2]
        fig, ax = self._plt.subplots(figsize=(12, 8))
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_facecolor("#cccccc")

        f  = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        n_green = n_yellow = n_red = 0

        for pt_idx, x_obs, y_obs in obs:
            X_c = R @ self._pts3d[pt_idx] + t
            if X_c[2] <= 0:
                continue
            x_proj = f * X_c[0] / X_c[2] + cx
            y_proj = f * X_c[1] / X_c[2] + cy
            err    = float(np.sqrt((x_proj - x_obs) ** 2 + (y_proj - y_obs) ** 2))

            if err < 1.0:
                color = "limegreen";  n_green  += 1
            elif err < 2.0:
                color = "yellow";     n_yellow += 1
            else:
                color = "red";        n_red    += 1

            ax.annotate(
                "",
                xy     = (x_proj, y_proj),
                xytext = (x_obs, y_obs),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.2, mutation_scale=8),
            )

        stem = Path(feat["image_path"]).stem
        ax.set_title(
            f"Reprojection errors — {stem}  "
            f"| ✅<1px:{n_green}  🟡1-2px:{n_yellow}  🔴>2px:{n_red}",
            fontsize=9,
        )
        ax.axis("off")
        out = self._out / "03_reconstruction" / f"reprojection_errors_{stem}.{self._fmt}"
        self._save_fig(fig, out)

    # ── Point cloud visualizations ────────────────────────────────────────────

    def _render_pointcloud_views(self) -> None:
        if len(self._pts3d) == 0:
            return

        pts = self._pts3d
        if self._colors is not None and len(self._colors) == len(pts):
            c_norm = self._colors.astype(np.float32) / 255.0
        else:
            c_norm = np.full((len(pts), 3), 0.5)

        # Subsample for speed
        if len(pts) > 20_000:
            idx = self._rng.choice(len(pts), 20_000, replace=False)
            pts    = pts[idx]
            c_norm = c_norm[idx]

        views = [
            ("Front",  0,   0),
            ("Back",   0, 180),
            ("Left",   0,  90),
            ("Right",  0, -90),
            ("Top",   90,   0),
            ("Bottom",-90,   0),
        ]

        fig = self._plt.figure(figsize=(18, 11))
        for i, (title, elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(2, 3, i, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=c_norm, s=0.5, alpha=0.7)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        bb_min = pts.min(0)
        bb_max = pts.max(0)
        bb_str = (f"BBox: "
                  f"X[{bb_min[0]:.1f},{bb_max[0]:.1f}] "
                  f"Y[{bb_min[1]:.1f},{bb_max[1]:.1f}] "
                  f"Z[{bb_min[2]:.1f},{bb_max[2]:.1f}]")
        fig.suptitle(
            f"Point cloud — {len(self._pts3d):,} points\n{bb_str}",
            fontsize=11,
        )
        fig.tight_layout()
        self._save_fig(fig, self._out / "04_pointcloud" / f"pointcloud_6views.{self._fmt}")

    def _render_summary_dashboard(self, stats: dict) -> None:
        fig = self._plt.figure(figsize=(20, 10))
        gs  = self._GS(2, 4, figure=fig, hspace=0.45, wspace=0.4)

        # ── Header stats table ─────────────────────────────────────────────
        ax_txt = fig.add_subplot(gs[0, 0])
        ax_txt.axis("off")
        lines = [
            ("Input",         f"{stats.get('n_images', '?')} images"),
            ("Resolution",    stats.get("resolution", "?")),
            ("Cameras reg.",  f"{stats.get('n_cameras', '?')}"),
            ("3-D points",    f"{stats.get('n_points', 0):,}"),
            ("Mean reproj.",  f"{stats.get('rmse', 0):.2f} px"),
            ("BA rounds",     f"{len(self._ba_hist)}"),
            ("Total time",    f"{stats.get('elapsed', 0):.1f} s"),
        ]
        col_labels = ["Metric", "Value"]
        cell_data  = [[k, v] for k, v in lines]
        tbl = ax_txt.table(
            cellText=cell_data,
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            colWidths=[0.55, 0.45],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        ax_txt.set_title("Pipeline Summary", fontsize=10, pad=4)

        # ── Match matrix (mini) ────────────────────────────────────────────
        ax_mm = fig.add_subplot(gs[0, 1])
        if self._verified and self._features:
            idxs    = sorted(self._features.keys())
            n       = len(idxs)
            idx_map = {v: k for k, v in enumerate(idxs)}
            mat = np.zeros((n, n))
            for (i, j), d in self._verified.items():
                r, c = idx_map[i], idx_map[j]
                mat[r, c] = mat[c, r] = d["n_inliers"]
            ax_mm.imshow(np.ma.masked_where(mat == 0, mat), cmap="Blues", aspect="auto")
        ax_mm.set_title("Match matrix", fontsize=9)
        ax_mm.axis("off")

        # ── BA convergence ─────────────────────────────────────────────────
        ax_ba = fig.add_subplot(gs[0, 2:])
        if self._ba_hist:
            rounds = [b["ba_step"]     for b in self._ba_hist]
            before = [b["rmse_before"] for b in self._ba_hist]
            after  = [b["rmse_after"]  for b in self._ba_hist]
            ax_ba.plot(rounds, before, "o--", color="tomato",    label="Before BA", lw=1.5)
            ax_ba.plot(rounds, after,  "o-",  color="steelblue", label="After BA",  lw=1.5)
            ax_ba.set_xlabel("BA round", fontsize=8)
            ax_ba.set_ylabel("RMSE (px)", fontsize=8)
            ax_ba.legend(fontsize=8)
            ax_ba.grid(True, alpha=0.3)
        ax_ba.set_title("BA convergence", fontsize=9)

        # ── Top-down point cloud ───────────────────────────────────────────
        ax_pc = fig.add_subplot(gs[1, :])
        if len(self._pts3d) > 0:
            pts = self._pts3d
            c_norm = (self._colors.astype(np.float32) / 255.0
                      if self._colors is not None and len(self._colors) == len(pts)
                      else "steelblue")
            if len(pts) > 10_000:
                idx = self._rng.choice(len(pts), 10_000, replace=False)
                pts_d = pts[idx]
                c_d   = c_norm[idx] if isinstance(c_norm, np.ndarray) else c_norm
            else:
                pts_d, c_d = pts, c_norm
            ax_pc.scatter(pts_d[:, 0], pts_d[:, 2], s=0.5, c=c_d, alpha=0.5)
            # Overlay camera positions
            for cam_idx, cam in self._cameras.items():
                R = cam["R"]
                t = cam["t"].flatten()
                C = -(R.T @ t)
                ax_pc.scatter(C[0], C[2], s=60, c="yellow",
                              edgecolors="black", linewidths=0.5, zorder=5)
        ax_pc.set_xlabel("X")
        ax_pc.set_ylabel("Z")
        ax_pc.set_title("Top-down view (point cloud + cameras)", fontsize=9)
        ax_pc.set_aspect("equal", "datalim")

        fig.suptitle("SfM Pipeline Summary", fontsize=14, y=1.01)
        self._save_fig(fig, self._out / "00_summary" / f"pipeline_summary.{self._fmt}")

    # ── Video / interactive ───────────────────────────────────────────────────

    def _save_reconstruction_video(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError:
            logger.warning("[VIZ] imageio not installed — skipping reconstruction video")
            return

        frames_dir = self._out / "03_reconstruction"
        step_imgs  = sorted(frames_dir.glob("step_*.png"))
        if not step_imgs:
            return

        frames = []
        for p in step_imgs:
            try:
                frames.append(iio.imread(p))
            except Exception:
                pass
        if not frames:
            return

        out_path = self._out / "reconstruction_growth.gif"
        iio.imwrite(str(out_path), frames, duration=500, loop=0)
        logger.info(f"[VIZ] Reconstruction video saved → {out_path}")

    def _save_turntable_video(self) -> None:
        if len(self._pts3d) == 0:
            return
        try:
            import imageio.v3 as iio
        except ImportError:
            logger.warning("[VIZ] imageio not installed — skipping turntable video")
            return

        pts = self._pts3d
        c_norm = (self._colors.astype(np.float32) / 255.0
                  if self._colors is not None and len(self._colors) == len(pts)
                  else "steelblue")
        if len(pts) > 5000:
            idx    = self._rng.choice(len(pts), 5000, replace=False)
            pts    = pts[idx]
            c_norm = c_norm[idx] if isinstance(c_norm, np.ndarray) else c_norm

        frames = []
        fig = self._plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=c_norm, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

        for angle in range(0, 360, 6):
            ax.view_init(elev=20, azim=angle)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[:, :, :3])

        self._plt.close(fig)
        out_path = self._out / "04_pointcloud" / "pointcloud_turntable.gif"
        iio.imwrite(str(out_path), frames, duration=50, loop=0)
        logger.info(f"[VIZ] Turntable video saved → {out_path}")

    def _open_interactive_viewer(self) -> None:
        if len(self._pts3d) == 0:
            return
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("[VIZ] open3d not installed — skipping interactive viewer")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._pts3d)
        if self._colors is not None and len(self._colors) == len(self._pts3d):
            pcd.colors = o3d.utility.Vector3dVector(
                self._colors.astype(np.float64) / 255.0
            )

        geoms = [pcd]
        for cam_idx, cam in self._cameras.items():
            R = cam["R"]
            t = cam["t"].flatten()
            C = -(R.T @ t)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(C)
            sphere.paint_uniform_color([1.0, 0.8, 0.0])
            geoms.append(sphere)

        print("\n[VIZ] Interactive viewer controls:")
        print("      Left drag=rotate  |  Right drag=pan  |  Scroll=zoom  |  Q=quit\n")
        o3d.visualization.draw_geometries(
            geoms,
            window_name="SfM Point Cloud",
            width=1280,
            height=800,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_fig(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=self._dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        self._plt.close(fig)
        self._n_saved += 1

    def _load_image_rgb(self, path) -> Optional[np.ndarray]:
        """Load image as (H,W,3) uint8 RGB. Returns None if unavailable."""
        try:
            import cv2 as _cv2
            img = _cv2.imread(str(path))
            if img is not None:
                return img[:, :, ::-1].copy()
        except Exception:
            pass
        try:
            from PIL import Image as _PIL
            with _PIL.open(str(path)) as im:
                return np.array(im.convert("RGB"))
        except Exception:
            pass
        return None

    def _sample_image_indices(
        self, n: int, registered: Optional[dict] = None
    ) -> list:
        """Return up to n random image indices, preferring registered cameras."""
        if not self._features:
            return []
        pool = (list(registered.keys()) if registered
                else list(self._features.keys()))
        k = min(n, len(pool))
        return [int(i) for i in self._rng.choice(pool, k, replace=False)]

    def _sample_pair_indices(self, n: int, pairs: list) -> list:
        """Return up to n random pairs."""
        k = min(n, len(pairs))
        idx = self._rng.choice(len(pairs), k, replace=False)
        return [pairs[int(i)] for i in idx]

    @staticmethod
    def _epipolar_endpoints(
        line: np.ndarray, w: int, h: int
    ) -> Optional[Tuple[tuple, tuple]]:
        """Clip epipolar line [a,b,c] to image bounds; return two endpoints."""
        a, b, c = float(line[0]), float(line[1]), float(line[2])
        pts: list = []
        if abs(b) > 1e-10:
            y = -c / b
            if 0 <= y <= h:
                pts.append((0.0, y))
            y = (-c - a * w) / b
            if 0 <= y <= h:
                pts.append((float(w), y))
        if abs(a) > 1e-10:
            x = -c / a
            if 0 <= x <= w:
                pts.append((x, 0.0))
            x = (-c - b * h) / a
            if 0 <= x <= w:
                pts.append((x, float(h)))
        if len(pts) >= 2:
            return (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1])
        return None

    def _print_summary(self) -> None:
        logger.info(
            f"[VIZ] Visualization complete — {self._n_saved} figures saved "
            f"→ {self._out}/"
        )
        logger.info(
            f"[VIZ] Open {self._out / '00_summary' / f'pipeline_summary.{self._fmt}'}"
            " for full overview"
        )
