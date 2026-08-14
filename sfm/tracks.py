"""
Global feature tracks from the verified match graph.

Why
---
`paper/paper_l.md` §6.7 measured a mean track length of 2.70 against COLMAP's
4.69 on the same 67 images, with more than two thirds of all points resting on
exactly two views. A point seen by two cameras is fixed by two rays and has
nothing left over to detect an error with; a point seen by five ties five
cameras together and stiffens the whole block. The gap is the single largest
structural difference between the two systems on this dataset.

The reason for the short tracks is that observations are only ever created
along the pair that triangulated them. If image A matches B and B matches C on
the same physical corner, the pipeline learns A-B and B-C separately and never
concludes A-B-C. Chaining those matches transitively is exactly what a track
graph does.

What this module does
---------------------
Union-find over the inlier matches of the *verified* pairs, keyed by
``(image index, keypoint index)``. Every connected component is one candidate
track, i.e. one physical surface point.

A component that contains two keypoints of the same image is self-contradictory:
one surface point cannot appear twice in one picture, so at least one match in
that component is wrong. Such components are split rather than trusted: the
image with the conflict is dropped from the component, keeping the majority
chain. Nothing is believed on trust; the reconstruction still gates every
observation on the reprojection error before using it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

KeyPoint = Tuple[int, int]      # (image index, keypoint index)


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: Dict[KeyPoint, KeyPoint] = {}
        self.rank: Dict[KeyPoint, int] = {}

    def add(self, item: KeyPoint) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: KeyPoint) -> KeyPoint:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:      # path compression
            parent[item], item = root, parent[item]
        return root

    def union(self, a: KeyPoint, b: KeyPoint) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class TrackGraph:
    """
    Connected components of the verified match graph.

    Attributes
    ----------
    track_of : {(img, kp): track_id}
    members  : {track_id: [(img, kp), ...]}, sorted by image index
    """

    def __init__(
        self,
        verified_pairs: Dict[Tuple[int, int], dict],
        min_length: int = 2,
        max_length: int = 200,
    ) -> None:
        uf = _UnionFind()
        n_edges = 0
        for (i, j), data in verified_pairs.items():
            inl = data.get("inlier_matches")
            if inl is None or len(inl) == 0:
                continue
            for ki, kj in inl[:, :2].astype(np.int64):
                a = (int(i), int(ki))
                b = (int(j), int(kj))
                uf.add(a)
                uf.add(b)
                uf.union(a, b)
                n_edges += 1

        raw: Dict[KeyPoint, List[KeyPoint]] = defaultdict(list)
        for node in uf.parent:
            raw[uf.find(node)].append(node)

        self.track_of: Dict[KeyPoint, int] = {}
        self.members: Dict[int, List[KeyPoint]] = {}
        n_conflict = 0
        n_dropped_len = 0
        next_id = 0

        for component in raw.values():
            by_image: Dict[int, List[KeyPoint]] = defaultdict(list)
            for node in component:
                by_image[node[0]].append(node)

            conflicted = [img for img, nodes in by_image.items() if len(nodes) > 1]
            if conflicted:
                n_conflict += 1
                for img in conflicted:
                    del by_image[img]

            if len(by_image) < min_length or len(by_image) > max_length:
                n_dropped_len += 1
                continue

            nodes = [nodes[0] for _, nodes in sorted(by_image.items())]
            self.members[next_id] = nodes
            for node in nodes:
                self.track_of[node] = next_id
            next_id += 1

        lengths = np.array([len(v) for v in self.members.values()], dtype=np.int32)
        self.stats = {
            "n_edges": n_edges,
            "n_components": len(raw),
            "n_tracks": len(self.members),
            "n_conflicting": n_conflict,
            "n_rejected_length": n_dropped_len,
            "mean_length": float(lengths.mean()) if lengths.size else 0.0,
            "median_length": float(np.median(lengths)) if lengths.size else 0.0,
            "max_length": int(lengths.max()) if lengths.size else 0,
            "frac_len2": float((lengths == 2).mean()) if lengths.size else 0.0,
            "frac_len_ge3": float((lengths >= 3).mean()) if lengths.size else 0.0,
        }
        logger.info(
            "[TRACKS] %d verified matches → %d tracks, mean length %.2f, "
            "%.1f %% reach three or more views (%d components dropped for a "
            "repeated image, %d for length)",
            n_edges, self.stats["n_tracks"], self.stats["mean_length"],
            100.0 * self.stats["frac_len_ge3"], n_conflict, n_dropped_len,
        )

    # ── queries ───────────────────────────────────────────────────────────

    def track_id(self, img: int, kp: int) -> Optional[int]:
        return self.track_of.get((int(img), int(kp)))

    def observations_in(
        self, track_id: int, images: Iterable[int]
    ) -> List[KeyPoint]:
        """Members of a track that belong to the given images."""
        wanted = set(int(i) for i in images)
        return [node for node in self.members.get(track_id, ()) if node[0] in wanted]

    def length_histogram(self, max_bin: int = 12) -> Dict[int, int]:
        hist: Dict[int, int] = defaultdict(int)
        for nodes in self.members.values():
            hist[min(len(nodes), max_bin)] += 1
        return dict(sorted(hist.items()))
