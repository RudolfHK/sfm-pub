"""
Stage 2 — Pairwise feature matching.

Three strategies are available:

  FeatureMatcher   — exhaustive O(N²), best quality, slow for large N
  SequentialMatcher — sliding-window O(N·W), suited to ordered image sequences
  VocabTreeMatcher — bag-of-words retrieval, O(N·top_k), suited to unordered sets

GPU acceleration
----------------
When PyTorch + CUDA are available, descriptor matching uses torch.cdist for the
full pairwise L2 distance matrix on the GPU, replacing CPU FLANN kNN.  The ratio
test and cross-check filter are then applied to the resulting distance tensors
without any Python loops.

Vocabulary building (k-means) and image similarity scoring (cosine similarity)
in VocabTreeMatcher also run on the GPU when available.

All GPU paths fall back to CPU FLANN automatically on OOM or import errors.
"""

import logging
from itertools import combinations
from typing import Dict, Tuple

import cv2
import numpy as np

from .device import get_device, has_gpu

logger = logging.getLogger(__name__)

# Type alias: pair key → (M, 2) int32 match array
MatchDict = Dict[Tuple[int, int], np.ndarray]


# ─── CPU FLANN helpers ────────────────────────────────────────────────────────

def _make_flann() -> cv2.FlannBasedMatcher:
    FLANN_INDEX_KDTREE = 1
    return cv2.FlannBasedMatcher(
        {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
        {"checks": 50},
    )


def _ratio_matches_cpu(
    matcher: cv2.FlannBasedMatcher,
    query: np.ndarray,
    train: np.ndarray,
    ratio: float,
    forward: bool,
) -> set:
    """kNN (k=2) + Lowe ratio test; returns set of (q_idx, t_idx) or swapped."""
    raw = matcher.knnMatch(query.astype(np.float32), train.astype(np.float32), k=2)
    # Vectorized filter — avoids Python loop over all match pairs
    valid = [
        (m[0].distance, m[0].queryIdx, m[0].trainIdx, m[1].distance)
        for m in raw
        if len(m) >= 2
    ]
    if not valid:
        return set()
    arr = np.array(valid, dtype=np.float32)
    mask = arr[:, 0] < ratio * arr[:, 3]
    pairs = arr[mask, 1:3].astype(np.int32)
    if forward:
        return {(int(p[0]), int(p[1])) for p in pairs}
    return {(int(p[1]), int(p[0])) for p in pairs}


def _match_pair_cpu(
    matcher: cv2.FlannBasedMatcher,
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float,
    cross_check: bool,
) -> np.ndarray:
    """FLANN match with ratio test + optional cross-check; returns (M,2) int32."""
    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=np.int32)
    good_12 = _ratio_matches_cpu(matcher, desc1, desc2, ratio, forward=True)
    if not cross_check:
        return (
            np.array(sorted(good_12), dtype=np.int32)
            if good_12
            else np.zeros((0, 2), dtype=np.int32)
        )
    good_21 = _ratio_matches_cpu(matcher, desc2, desc1, ratio, forward=False)
    mutual = good_12 & good_21
    return (
        np.array(sorted(mutual), dtype=np.int32)
        if mutual
        else np.zeros((0, 2), dtype=np.int32)
    )


# ─── GPU matching ─────────────────────────────────────────────────────────────

def _match_pair_gpu(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float,
    cross_check: bool,
) -> np.ndarray:
    """
    Brute-force L2 matching via torch.cdist on GPU.

    Computes the full (N×M) distance matrix once, applies the ratio test and
    optional mutual-consistency filter entirely as tensor operations — no Python
    loops over individual matches.

    Returns (M, 2) int32 array of (query_idx, train_idx) pairs.
    """
    import torch

    device = get_device()
    if device is None:
        raise RuntimeError("no torch device")

    d1 = torch.from_numpy(desc1.astype(np.float32)).to(device)  # (N, 128)
    d2 = torch.from_numpy(desc2.astype(np.float32)).to(device)  # (M, 128)

    dist = torch.cdist(d1, d2)  # (N, M)  L2 distances

    # Forward ratio test: for each query, top-2 train distances
    k = min(2, d2.shape[0])
    vals_f, idx_f = dist.topk(k, dim=1, largest=False)  # (N, k)

    if k < 2:
        # Only one train descriptor: ratio test is undefined, keep all
        mask_f = torch.ones(d1.shape[0], dtype=torch.bool, device=device)
    else:
        mask_f = vals_f[:, 0] < ratio * vals_f[:, 1]

    q_idx = torch.where(mask_f)[0]    # passing query indices
    t_idx = idx_f[mask_f, 0]          # their best train match

    if not cross_check:
        if len(q_idx) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return torch.stack([q_idx, t_idx], dim=1).cpu().numpy().astype(np.int32)

    # Backward ratio test (reuse transposed dist — no second cdist call)
    k2 = min(2, d1.shape[0])
    vals_b, idx_b = dist.T.topk(k2, dim=1, largest=False)  # (M, k)

    if k2 < 2:
        mask_b = torch.ones(d2.shape[0], dtype=torch.bool, device=device)
    else:
        mask_b = vals_b[:, 0] < ratio * vals_b[:, 1]

    q2_idx = torch.where(mask_b)[0]   # passing train (= backward query) indices
    t2_idx = idx_b[mask_b, 0]          # their best backward match (= forward query)

    if len(q_idx) == 0 or len(q2_idx) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # Mutual check: (q, t) must also appear as (t, q) in backward pass
    set_12 = set(zip(q_idx.cpu().tolist(), t_idx.cpu().tolist()))
    # backward pair (q2, t2) means d2[q2] → d1[t2], i.e., original (t2, q2)
    set_21 = set(zip(t2_idx.cpu().tolist(), q2_idx.cpu().tolist()))
    mutual = sorted(set_12 & set_21)

    if not mutual:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(mutual, dtype=np.int32)


def _match_pair(
    matcher: cv2.FlannBasedMatcher,
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float,
    cross_check: bool,
) -> np.ndarray:
    """
    Match two descriptor sets.  Uses GPU (torch.cdist) when available,
    falls back to CPU FLANN on OOM or when PyTorch is absent.
    """
    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=np.int32)

    if has_gpu():
        try:
            return _match_pair_gpu(desc1, desc2, ratio, cross_check)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM during matching — falling back to CPU FLANN")
            else:
                logger.debug(f"GPU matching error ({e}) — falling back to CPU FLANN")
        except Exception as e:
            logger.debug(f"GPU matching unavailable ({e}) — using CPU FLANN")

    return _match_pair_cpu(matcher, desc1, desc2, ratio, cross_check)


# ─── Exhaustive matcher ───────────────────────────────────────────────────────

class FeatureMatcher:
    """
    Exhaustive O(N²) pairwise matcher.

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
        backend   = "GPU" if has_gpu() else "CPU FLANN"
        logger.info(
            f"Exhaustive matching [{backend}]: {n_pairs} pairs  "
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
    4. Run GPU/FLANN matching only on those candidate pairs.

    GPU acceleration is used for k-means, image encoding, and cosine similarity
    when PyTorch + CUDA are available.

    Complexity: O(N · n_words) retrieval + O(N · top_k) match runs.
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
        gpu_tag = "GPU" if has_gpu() else "CPU"
        logger.info(
            f"VocabTree [{gpu_tag}]: building {self.n_words}-word vocabulary "
            f"from {n} images…"
        )
        pool  = self._sample_descriptors(features, indices)
        vocab = self._build_vocab(pool)

        # 2. Encode images as TF-IDF histograms
        logger.info("VocabTree: encoding images with TF-IDF…")
        histograms = {idx: self._encode(features[idx]["descriptors"], vocab) for idx in indices}
        tfidf      = self._apply_idf(histograms, indices)

        # 3. Select top-K candidate pairs
        logger.info(f"VocabTree: retrieving top-{self.top_k} candidates per image…")
        candidates = self._select_candidates(tfidf, indices)
        logger.info(f"VocabTree: {len(candidates)} candidate pairs → running matching…")

        # 4. Matching on candidates only
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
        """
        Build visual vocabulary via k-means.  Uses GPU when available,
        falls back to cv2.kmeans on CPU.
        """
        n_words = min(self.n_words, len(pool))

        if has_gpu():
            try:
                return self._build_vocab_gpu(pool, n_words)
            except Exception as e:
                logger.warning(f"GPU k-means failed ({e}), using CPU cv2.kmeans")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, vocab = cv2.kmeans(
            pool, n_words, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        return vocab  # (n_words, 128)

    def _build_vocab_gpu(self, pool: np.ndarray, n_words: int) -> np.ndarray:
        """Lloyd's k-means on GPU via PyTorch."""
        import torch

        device = get_device()
        if device is None or device.type == "cpu":
            raise RuntimeError("no CUDA device")

        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        X = torch.from_numpy(pool).float().to(device)  # (N, 128)

        # Random initialisation (good enough; PP init is O(N·K) calls)
        perm = torch.randperm(len(X), generator=gen, device=device)[:n_words]
        C    = X[perm].clone()  # (K, 128)

        for _ in range(20):
            # Assignment: nearest centroid for each descriptor
            dists  = torch.cdist(X, C)         # (N, K)
            labels = dists.argmin(dim=1)        # (N,)

            # Update: mean of each cluster
            new_C  = torch.zeros_like(C)
            counts = torch.zeros(n_words, dtype=X.dtype, device=device)
            new_C.scatter_add_(0, labels.unsqueeze(1).expand(-1, X.shape[1]), X)
            counts.scatter_add_(0, labels, torch.ones(len(X), dtype=X.dtype, device=device))

            nonempty = counts > 0
            new_C[nonempty] /= counts[nonempty].unsqueeze(1)
            new_C[~nonempty] = C[~nonempty]  # keep dead centroids

            shift = (new_C - C).norm(dim=1).max().item()
            C = new_C
            if shift < 0.1:
                break

        logger.debug(f"GPU k-means converged (shift={shift:.3f})")
        return C.cpu().numpy()  # (K, 128)

    def _encode(self, descs: np.ndarray, vocab: np.ndarray) -> np.ndarray:
        """
        Assign descriptors to nearest vocab word; return frequency histogram.
        GPU path (torch.cdist + bincount) when available, CPU FLANN fallback.
        """
        hist = np.zeros(len(vocab), dtype=np.float32)
        if len(descs) == 0:
            return hist

        if has_gpu():
            try:
                return self._encode_gpu(descs, vocab)
            except Exception:
                pass

        # CPU fallback via FLANN
        assign_matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 1}, {"checks": 20}
        )
        raw = assign_matcher.match(descs.astype(np.float32), vocab)
        idxs = np.array([m.trainIdx for m in raw], dtype=np.int32)
        np.add.at(hist, idxs, 1.0)
        return hist

    def _encode_gpu(self, descs: np.ndarray, vocab: np.ndarray) -> np.ndarray:
        """GPU-accelerated nearest-word assignment via torch.cdist + bincount."""
        import torch

        device = get_device()
        if device is None or device.type == "cpu":
            raise RuntimeError("no CUDA device")

        d = torch.from_numpy(descs.astype(np.float32)).to(device)
        v = torch.from_numpy(vocab.astype(np.float32)).to(device)
        assignments = torch.cdist(d, v).argmin(dim=1)  # (N,)
        hist = torch.bincount(assignments, minlength=len(vocab)).float()
        return hist.cpu().numpy()

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
        """
        Find top-K most similar pairs per image via cosine similarity.
        Uses GPU matrix multiply when available.
        """
        mat = np.array([tfidf[idx] for idx in indices], dtype=np.float32)

        if has_gpu():
            try:
                return self._select_candidates_gpu(mat, indices)
            except Exception:
                pass

        # CPU fallback
        sim = mat @ mat.T  # (N, N) cosine similarity
        pairs: set = set()
        for r in range(len(indices)):
            sim[r, r] = -1.0
            top = np.argsort(sim[r])[::-1][: self.top_k]
            for c in top:
                gi = indices[r]
                gj = indices[c]
                pairs.add((min(gi, gj), max(gi, gj)))
        return sorted(pairs)

    def _select_candidates_gpu(self, mat: np.ndarray, indices: list) -> list:
        """GPU cosine similarity via torch.mm."""
        import torch

        device = get_device()
        if device is None or device.type == "cpu":
            raise RuntimeError("no CUDA device")

        m = torch.from_numpy(mat).to(device)          # (N, K)
        sim = m @ m.T                                  # (N, N)
        sim.fill_diagonal_(-1.0)

        k   = min(self.top_k, sim.shape[1] - 1)
        top = sim.topk(k, dim=1).indices.cpu().numpy()  # (N, k)

        pairs: set = set()
        for r in range(len(indices)):
            for c in top[r]:
                gi, gj = indices[r], indices[int(c)]
                pairs.add((min(gi, gj), max(gi, gj)))
        return sorted(pairs)
