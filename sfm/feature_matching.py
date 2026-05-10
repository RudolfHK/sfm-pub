"""
Stage 2 — Pairwise feature matching.

Uses FLANN (fast approximate NN) with:
  • Lowe's ratio test (default threshold 0.75)
  • Mutual-best / cross-check filter for extra robustness
"""

import logging
from itertools import combinations
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: pair key → (M, 2) int32 match array
MatchDict = Dict[Tuple[int, int], np.ndarray]


class FeatureMatcher:
    """
    Exhaustive pairwise matcher.

    Parameters
    ----------
    ratio_threshold : Lowe's ratio test threshold (0 < t < 1).
    cross_check     : Enforce mutual-best consistency.
    min_matches     : Drop pairs below this raw-match count.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
        min_matches: int = 15,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.min_matches = min_matches

        FLANN_INDEX_KDTREE = 1
        self._matcher = cv2.FlannBasedMatcher(
            {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
            {"checks": 50},
        )

    # ── public API ────────────────────────────────────────────────────────

    def match_pair(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> np.ndarray:
        """
        Match two descriptor matrices.

        Returns
        -------
        matches : (M, 2) int32  — (idx_in_desc1, idx_in_desc2) pairs
        """
        if len(desc1) < 2 or len(desc2) < 2:
            return np.zeros((0, 2), dtype=np.int32)

        good_12 = self._ratio_matches(desc1, desc2, forward=True)

        if not self.cross_check:
            return np.array(sorted(good_12), dtype=np.int32) if good_12 else np.zeros((0, 2), dtype=np.int32)

        good_21 = self._ratio_matches(desc2, desc1, forward=False)
        mutual = good_12 & good_21

        return np.array(sorted(mutual), dtype=np.int32) if mutual else np.zeros((0, 2), dtype=np.int32)

    def match_all(self, features: dict) -> MatchDict:
        """
        Exhaustive pairwise matching across all images.

        Returns
        -------
        matches : dict  (i, j) → (M, 2) int32   (i < j always)
        """
        indices = sorted(features.keys())
        all_pairs = list(combinations(indices, 2))
        n_pairs = len(all_pairs)
        logger.info(f"Matching {n_pairs} pairs (exhaustive, ratio={self.ratio_threshold}, cross_check={self.cross_check})…")

        matches: MatchDict = {}
        for k, (i, j) in enumerate(all_pairs):
            m = self.match_pair(
                features[i]["descriptors"],
                features[j]["descriptors"],
            )
            if len(m) >= self.min_matches:
                matches[(i, j)] = m
            else:
                logger.debug(
                    f"  Pair ({i},{j}): {len(m)} matches — below threshold, skipped"
                )

            if (k + 1) % max(1, n_pairs // 10) == 0:
                logger.info(f"  … {k+1}/{n_pairs} pairs matched")

        logger.info(
            f"Matching done: {len(matches)}/{n_pairs} pairs "
            f"with ≥{self.min_matches} raw matches"
        )
        return matches

    # ── internals ─────────────────────────────────────────────────────────

    def _ratio_matches(
        self, query: np.ndarray, train: np.ndarray, forward: bool
    ) -> set:
        """
        kNN (k=2) + Lowe's ratio test.

        If forward=True  → returns {(query_idx, train_idx)}
        If forward=False → returns {(train_idx, query_idx)}  (for cross-check)
        """
        raw = self._matcher.knnMatch(
            query.astype(np.float32), train.astype(np.float32), k=2
        )
        good = set()
        for m_list in raw:
            if len(m_list) < 2:
                continue
            m, n = m_list
            if m.distance < self.ratio_threshold * n.distance:
                if forward:
                    good.add((m.queryIdx, m.trainIdx))
                else:
                    good.add((m.trainIdx, m.queryIdx))
        return good
