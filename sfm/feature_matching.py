"""
Stage 2 — Pairwise feature matching.

Three strategies are available:

  FeatureMatcher   — exhaustive O(N²), best quality, slow for large N
  SequentialMatcher — sliding-window O(N·W), suited to ordered image sequences
  VocabTreeMatcher — bag-of-words retrieval, O(N·top_k), suited to unordered sets

All three share the same output type: MatchDict.
"""

import logging
from itertools import combinations
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias: pair key → (M, 2) int32 match array
MatchDict = Dict[Tuple[int, int], np.ndarray]


def _make_flann() -> cv2.FlannBasedMatcher:
    FLANN_INDEX_KDTREE = 1
    return cv2.FlannBasedMatcher(
        {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
        {"checks": 50},
    )


def _ratio_matches(
    matcher: cv2.FlannBasedMatcher,
    query: np.ndarray,
    train: np.ndarray,
    ratio: float,
    forward: bool,
) -> set:
    """kNN (k=2) + Lowe ratio test; returns {(q_idx, t_idx)} or swapped."""
    raw = matcher.knnMatch(query.astype(np.float32), train.astype(np.float32), k=2)
    good = set()
    for m_list in raw:
        if len(m_list) < 2:
            continue
        m, n = m_list
        if m.distance < ratio * n.distance:
            good.add((m.queryIdx, m.trainIdx) if forward else (m.trainIdx, m.queryIdx))
    return good


def _match_pair(
    matcher: cv2.FlannBasedMatcher,
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float,
    cross_check: bool,
) -> np.ndarray:
    """FLANN match with ratio test + optional cross-check; returns (M,2) int32."""
    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    good_12 = _ratio_matches(matcher, desc1, desc2, ratio, forward=True)
    if not cross_check:
        return np.array(sorted(good_12), dtype=np.int32) if good_12 else np.zeros((0, 2), dtype=np.int32)
    good_21 = _ratio_matches(matcher, desc2, desc1, ratio, forward=False)
    mutual  = good_12 & good_21
    return np.array(sorted(mutual), dtype=np.int32) if mutual else np.zeros((0, 2), dtype=np.int32)


class FeatureMatcher:
    """
    Exhaustive O(N²) pairwise FLANN matcher.

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
        self.cross_check     = cross_check
        self.min_matches     = min_matches
        self._matcher        = _make_flann()

    # ── public API ────────────────────────────────────────────────────────

    def match_pair(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> np.ndarray:
        """Match two descriptor matrices; returns (M, 2) int32."""
        return _match_pair(
            self._matcher, desc1, desc2,
            self.ratio_threshold, self.cross_check,
        )

    def match_all(self, features: dict) -> MatchDict:
        """
        Exhaustive pairwise matching across all images.

        Returns
        -------
        matches : dict  (i, j) → (M, 2) int32   (i < j always)
        """
        indices   = sorted(features.keys())
        all_pairs = list(combinations(indices, 2))
        n_pairs   = len(all_pairs)
        logger.info(
            f"Exhaustive matching: {n_pairs} pairs  "
            f"(ratio={self.ratio_threshold}, cross_check={self.cross_check})…"
        )

        matches: MatchDict = {}
        for k, (i, j) in enumerate(all_pairs):
            m = self.match_pair(
                features[i]["descriptors"],
                features[j]["descriptors"],
            )
            if len(m) >= self.min_matches:
                matches[(i, j)] = m
            else:
                logger.debug(f"  Pair ({i},{j}): {len(m)} matches — skipped")

            if (k + 1) % max(1, n_pairs // 10) == 0:
                logger.info(f"  … {k+1}/{n_pairs} pairs done")

        logger.info(
            f"Exhaustive matching done: {len(matches)}/{n_pairs} pairs "
            f"with ≥{self.min_matches} raw matches"
        )
        return matches


# ─── Sequential matcher ──────────────────────────────────────────────────────

class SequentialMatcher:
    """
    Sliding-window matcher: each image is matched only to the next `window`
    images (ordered by feature-dict index).  O(N × window) pairs instead of
    O(N²).  Best suited to video sequences or images captured along a path.

    Parameters
    ----------
    window          : Half-window size — image i is matched to i+1 … i+window.
    ratio_threshold : Lowe's ratio test threshold.
    cross_check     : Mutual-best filter.
    min_matches     : Minimum matches to keep a pair.
    """

    def __init__(
        self,
        window: int = 5,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
        min_matches: int = 15,
    ) -> None:
        self.window          = window
        self.ratio_threshold = ratio_threshold
        self.cross_check     = cross_check
        self.min_matches     = min_matches
        self._matcher        = _make_flann()

    def match_all(self, features: dict) -> MatchDict:
        indices = sorted(features.keys())
        n       = len(indices)
        logger.info(
            f"Sequential matching: {n} images, window={self.window}  "
            f"(≤{n * self.window} candidate pairs)…"
        )

        matches: MatchDict = {}
        for r, i in enumerate(indices):
            for j in indices[r + 1 : r + 1 + self.window]:
                m = _match_pair(
                    self._matcher,
                    features[i]["descriptors"],
                    features[j]["descriptors"],
                    self.ratio_threshold,
                    self.cross_check,
                )
                if len(m) >= self.min_matches:
                    matches[(i, j)] = m

        logger.info(f"Sequential matching done: {len(matches)} pairs retained")
        return matches


# ─── Vocabulary-tree (bag-of-words) matcher ──────────────────────────────────

class VocabTreeMatcher:
    """
    Approximate image retrieval via bag-of-words with TF-IDF weighting.

    Algorithm
    ---------
    1. Sample descriptors from all images and cluster them (k-means) to form
       a flat visual vocabulary of `n_words` visual words.
    2. Encode each image as a TF-IDF histogram over the vocabulary.
    3. Select the `top_k` most similar image pairs per image (cosine sim).
    4. Run FLANN matching only on those candidate pairs.

    Complexity: O(N · n_words) retrieval + O(N · top_k) FLANN runs.
    Automatically falls back to exhaustive matching when N ≤ top_k + 1.

    Parameters
    ----------
    n_words                : Vocabulary size (k-means cluster count).
    top_k                  : Number of nearest neighbours to retrieve per image.
    ratio_threshold        : Lowe's ratio test threshold.
    cross_check            : Mutual-best filter.
    min_matches            : Minimum matches to keep a pair.
    max_sample_per_image   : Max descriptors sampled per image for k-means.
    """

    def __init__(
        self,
        n_words: int = 256,
        top_k: int = 10,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
        min_matches: int = 15,
        max_sample_per_image: int = 1_000,
    ) -> None:
        self.n_words              = n_words
        self.top_k                = top_k
        self.ratio_threshold      = ratio_threshold
        self.cross_check          = cross_check
        self.min_matches          = min_matches
        self.max_sample_per_image = max_sample_per_image
        self._matcher             = _make_flann()

    # ── public API ────────────────────────────────────────────────────────

    def match_all(self, features: dict) -> MatchDict:
        indices = sorted(features.keys())
        n       = len(indices)

        if n <= self.top_k + 1:
            logger.info(
                f"VocabTree: dataset small ({n} images) — using exhaustive matching"
            )
            base = FeatureMatcher(self.ratio_threshold, self.cross_check, self.min_matches)
            return base.match_all(features)

        # 1. Build vocabulary
        logger.info(f"VocabTree: building {self.n_words}-word vocabulary from {n} images…")
        pool  = self._sample_descriptors(features, indices)
        vocab = self._build_vocab(pool)

        # 2. Encode images as TF-IDF histograms
        logger.info("VocabTree: encoding images with TF-IDF…")
        histograms = {idx: self._encode(features[idx]["descriptors"], vocab) for idx in indices}
        tfidf      = self._apply_idf(histograms, indices)

        # 3. Select top-K candidate pairs
        logger.info(f"VocabTree: retrieving top-{self.top_k} candidates per image…")
        candidates = self._select_candidates(tfidf, indices)
        logger.info(f"VocabTree: {len(candidates)} candidate pairs → running FLANN…")

        # 4. FLANN matching on candidates only
        matches: MatchDict = {}
        for i, j in candidates:
            m = _match_pair(
                self._matcher,
                features[i]["descriptors"],
                features[j]["descriptors"],
                self.ratio_threshold,
                self.cross_check,
            )
            if len(m) >= self.min_matches:
                matches[(i, j)] = m

        logger.info(
            f"VocabTree matching done: {len(matches)}/{len(candidates)} pairs retained"
        )
        return matches

    # ── internals ─────────────────────────────────────────────────────────

    def _sample_descriptors(self, features: dict, indices: list) -> np.ndarray:
        rng   = np.random.default_rng(0)
        parts = []
        for idx in indices:
            descs = features[idx]["descriptors"]
            if len(descs) > self.max_sample_per_image:
                sel   = rng.choice(len(descs), self.max_sample_per_image, replace=False)
                descs = descs[sel]
            parts.append(descs)
        return np.vstack(parts).astype(np.float32)

    def _build_vocab(self, pool: np.ndarray) -> np.ndarray:
        n_words  = min(self.n_words, len(pool))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, vocab = cv2.kmeans(
            pool, n_words, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        return vocab  # (n_words, 128)

    def _encode(self, descs: np.ndarray, vocab: np.ndarray) -> np.ndarray:
        """Assign descriptors to nearest vocab word; return frequency histogram."""
        hist = np.zeros(len(vocab), dtype=np.float32)
        if len(descs) == 0:
            return hist
        assign_matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 1}, {"checks": 20}
        )
        raw = assign_matcher.match(descs.astype(np.float32), vocab)
        for m in raw:
            hist[m.trainIdx] += 1.0
        return hist

    def _apply_idf(self, histograms: dict, indices: list) -> dict:
        n        = len(indices)
        n_words  = len(next(iter(histograms.values())))
        doc_freq = np.zeros(n_words, dtype=np.float32)
        for idx in indices:
            doc_freq += (histograms[idx] > 0).astype(np.float32)

        idf = np.log((n + 1.0) / (doc_freq + 1.0))

        tfidf = {}
        for idx in indices:
            h    = histograms[idx] * idf
            norm = float(np.linalg.norm(h))
            tfidf[idx] = h / norm if norm > 1e-10 else h
        return tfidf

    def _select_candidates(self, tfidf: dict, indices: list) -> list:
        mat = np.array([tfidf[idx] for idx in indices], dtype=np.float32)
        sim = mat @ mat.T  # cosine similarity

        pairs: set = set()
        for r in range(len(indices)):
            sim[r, r] = -1.0
            top = np.argsort(sim[r])[::-1][: self.top_k]
            for c in top:
                gi = indices[r]
                gj = indices[c]
                pairs.add((min(gi, gj), max(gi, gj)))

        return sorted(pairs)
