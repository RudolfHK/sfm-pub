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
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Dict, List, Optional, Tuple

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

    # Chunked cdist: process d1 in blocks of 1024 rows so peak GPU memory
    # stays at 1024×M×4 bytes (~32 MB for M=8000) instead of N×M×4 bytes
    # (~256 MB for N=M=8000), preventing OOM on 4 GB GPUs.
    _CHUNK = 1024
    k = min(2, d2.shape[0])
    all_vals: list = []
    all_idx:  list = []
    for _start in range(0, d1.shape[0], _CHUNK):
        _block = d1[_start : _start + _CHUNK]
        _d = torch.cdist(_block, d2)
        _v, _i = _d.topk(k, dim=1, largest=False)
        all_vals.append(_v)
        all_idx.append(_i)
    vals_f = torch.cat(all_vals, dim=0)  # (N, k)
    idx_f  = torch.cat(all_idx,  dim=0)  # (N, k)

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

    # Backward ratio test: chunk d2 vs d1
    k2 = min(2, d1.shape[0])
    all_vals_b: list = []
    all_idx_b:  list = []
    for _start in range(0, d2.shape[0], _CHUNK):
        _block = d2[_start : _start + _CHUNK]
        _d = torch.cdist(_block, d1)
        _v, _i = _d.topk(k2, dim=1, largest=False)
        all_vals_b.append(_v)
        all_idx_b.append(_i)
    vals_b = torch.cat(all_vals_b, dim=0)  # (M, k2)
    idx_b  = torch.cat(all_idx_b,  dim=0)  # (M, k2)

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
    workers         : Threads used for the pair loop.  0 picks a default from
                      the core count; 1 keeps the loop serial.
    seed            : Base seed for the per-pair RNG.  Negative disables
                      per-pair seeding and restores history-dependent results.

    Parallelism
    ----------
    Matching was measured at 91 % of a 67-image run and 4.9× slower than COLMAP
    on the same CPU, the same pair count and the same descriptors — the gap is
    that this loop ran serially while COLMAP used every core.  The work is
    embarrassingly parallel across pairs, and the heavy part (FLANN `knnMatch`)
    is OpenCV C++ that releases the GIL, so threads give real speedup without
    the cost of pickling several hundred MB of descriptors to subprocesses.

    Reproducibility
    ---------------
    FLANN's randomised KD-trees draw from OpenCV's RNG, which makes a pair's
    result depend on how many FLANN calls preceded it: matching one pair twice
    through the same matcher was measured returning 932 and then 915 matches.
    Serially that is merely hidden — a fixed process seed plus a fixed pair
    order reproduces the same sequence — but under threads the consumption
    order is nondeterministic and results vary run to run.

    Each pair is therefore seeded from its own index before matching, so a
    pair's result depends only on which pair it is.  Matching then reproduces
    exactly across runs *and* across worker counts, which is what makes the
    parallel path verifiable against the serial one at all.  Results are also
    assembled in serial pair order so downstream iteration order never changes.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
        min_matches: int = 15,
        workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.cross_check     = cross_check
        self.min_matches     = min_matches
        self.workers         = workers
        self.seed            = seed
        self._matcher        = _make_flann()
        # cv2.FlannBasedMatcher is not safe to share across threads; each
        # worker gets its own through thread-local storage.
        self._local          = threading.local()

    def _thread_matcher(self) -> cv2.FlannBasedMatcher:
        m = getattr(self._local, "matcher", None)
        if m is None:
            m = _make_flann()
            self._local.matcher = m
        return m

    def _resolve_workers(self, n_pairs: int) -> int:
        """Thread count to use: never more than there is work for."""
        if self.workers and self.workers > 0:
            n = self.workers
        else:
            n = os.cpu_count() or 1
        return max(1, min(n, n_pairs))

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

        # The GPU path already saturates the device and serialises on it, so
        # threads would only add contention there.
        n_workers = 1 if has_gpu() else self._resolve_workers(n_pairs)
        logger.info(
            f"Exhaustive matching [{backend}, {n_workers} worker"
            f"{'s' if n_workers != 1 else ''}]: {n_pairs} pairs  "
            f"(ratio={self.ratio_threshold}, cross_check={self.cross_check})…"
        )

        results: List[Optional[np.ndarray]] = [None] * n_pairs
        done = 0
        step = max(1, n_pairs // 10)

        def work(k: int) -> int:
            i, j = all_pairs[k]
            if self.seed >= 0:
                # Per-pair seed: makes this pair's FLANN result independent of
                # how many pairs ran before it, and of which thread runs it.
                cv2.setRNGSeed(self.seed + k)
            if has_gpu():
                m = self.match_pair(
                    features[i]["descriptors"], features[j]["descriptors"],
                )
            else:
                m = _match_pair_cpu(
                    self._thread_matcher(),
                    features[i]["descriptors"], features[j]["descriptors"],
                    self.ratio_threshold, self.cross_check,
                )
            results[k] = m
            return k

        if n_workers == 1:
            for k in range(n_pairs):
                work(k)
                if (k + 1) % step == 0:
                    logger.info(f"  … {k+1}/{n_pairs} pairs done")
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for _ in pool.map(work, range(n_pairs)):
                    done += 1
                    if done % step == 0:
                        logger.info(f"  … {done}/{n_pairs} pairs done")

        # Assemble in serial pair order so the result is order-identical to
        # the single-threaded path.
        matches: MatchDict = {}
        for k, (i, j) in enumerate(all_pairs):
            m = results[k]
            if m is not None and len(m) >= self.min_matches:
                matches[(i, j)] = m
            else:
                logger.debug(
                    f"  Pair ({i},{j}): {0 if m is None else len(m)} matches — skipped"
                )

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
        n_words: int = 4096,
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


# ─── DINOv2 retrieval-guided matcher ─────────────────────────────────────────

class DINOv2Matcher:
    """
    Image retrieval via DINOv2 CLS-token embeddings with FAISS (or numpy)
    approximate nearest-neighbour search, followed by standard SIFT matching
    on the retrieved candidate pairs.

    Activated with ``--retrieval dinov2``.  Requires ``torch`` (included in
    the ``gpu`` extra).  Uses FAISS for fast ANN when available; falls back to
    numpy cosine similarity automatically.

    Complexity: O(N) DINOv2 forward passes + O(N·top_k) SIFT match runs.

    Parameters
    ----------
    top_k           : Number of nearest neighbours to retrieve per image.
    ratio_threshold : Lowe ratio test threshold for descriptor matching.
    cross_check     : Mutual-best consistency filter.
    min_matches     : Minimum raw matches to keep a candidate pair.
    model_name      : DINOv2 variant to load from torch.hub.
    image_size      : Resize shorter side to this before embedding.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        top_k: int = 10,
        ratio_threshold: float = 0.75,
        cross_check: bool = True,
        min_matches: int = 15,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
    ) -> None:
        self.top_k           = top_k
        self.ratio_threshold = ratio_threshold
        self.cross_check     = cross_check
        self.min_matches     = min_matches
        self.model_name      = model_name
        self.image_size      = image_size
        self._matcher        = _make_flann()
        self._model          = None   # lazy-loaded on first call

    # ── public API ────────────────────────────────────────────────────────

    def match_all(self, features: dict) -> MatchDict:
        indices = sorted(features.keys())
        n       = len(indices)

        if n <= self.top_k + 1:
            logger.info(
                f"DINOv2: dataset small ({n} images) — using exhaustive matching"
            )
            base = FeatureMatcher(self.ratio_threshold, self.cross_check, self.min_matches)
            return base.match_all(features)

        logger.info(
            f"DINOv2 retrieval: extracting embeddings for {n} images "
            f"(model={self.model_name})…"
        )
        embeddings = self._extract_embeddings(features, indices)  # (N, D)
        candidates = self._retrieve_candidates(embeddings, indices)
        logger.info(
            f"DINOv2: {len(candidates)} candidate pairs → running SIFT matching…"
        )

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
            f"DINOv2 matching done: {len(matches)}/{len(candidates)} pairs retained"
        )
        return matches

    # ── internals ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load DINOv2 from torch.hub; cache on self._model."""
        if self._model is not None:
            return self._model
        try:
            import torch
            model = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                verbose=False,
            )
            device = get_device()
            if device is not None:
                model = model.to(device)
            model.eval()
            self._model = model
            logger.info(
                f"DINOv2: loaded {self.model_name} on "
                f"{device if device else 'cpu'}"
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"DINOv2 model could not be loaded: {exc}.  "
                "Install torch and ensure internet access for torch.hub."
            ) from exc

    def _preprocess(self, bgr: np.ndarray) -> "torch.Tensor":
        """Resize, normalise to ImageNet stats, return (1, 3, H, W) tensor."""
        import torch

        h, w = bgr.shape[:2]
        scale = self.image_size / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0

        mean = np.array(self._IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(self._IMAGENET_STD,  dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
        return tensor

    def _extract_embeddings(self, features: dict, indices: List[int]) -> np.ndarray:
        """Return (N, D) float32 L2-normalised DINOv2 CLS embeddings."""
        import torch

        model  = self._load_model()
        device = get_device()
        embs: List[np.ndarray] = []

        for idx in indices:
            path = features[idx]["image_path"]
            bgr  = cv2.imread(str(path))
            if bgr is None:
                embs.append(np.zeros(model.embed_dim, dtype=np.float32))
                continue

            inp = self._preprocess(bgr)
            if device is not None:
                inp = inp.to(device)

            with torch.no_grad():
                emb = model(inp)   # (1, D) — CLS token
            embs.append(emb.squeeze(0).cpu().float().numpy())

        mat = np.vstack(embs)   # (N, D)
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-10)
        return mat / norms      # L2-normalise → cosine sim = dot product

    def _retrieve_candidates(
        self, embeddings: np.ndarray, indices: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Find top-k nearest neighbours per image.
        Uses FAISS IndexFlatIP when available; falls back to numpy dot product.
        """
        k = min(self.top_k + 1, len(indices))   # +1 because self is always #1

        try:
            import faiss  # noqa: F401
            return self._retrieve_faiss(embeddings, indices, k)
        except ImportError:
            pass

        return self._retrieve_numpy(embeddings, indices, k)

    def _retrieve_faiss(
        self, embeddings: np.ndarray, indices: List[int], k: int
    ) -> List[Tuple[int, int]]:
        import faiss

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        _, I = index.search(embeddings.astype(np.float32), k)

        pairs: set = set()
        for r in range(len(indices)):
            for c in I[r]:
                if c == r:
                    continue
                gi, gj = indices[r], indices[int(c)]
                pairs.add((min(gi, gj), max(gi, gj)))
        return sorted(pairs)

    def _retrieve_numpy(
        self, embeddings: np.ndarray, indices: List[int], k: int
    ) -> List[Tuple[int, int]]:
        sim = embeddings @ embeddings.T   # (N, N) cosine similarity
        np.fill_diagonal(sim, -1.0)

        pairs: set = set()
        for r in range(len(indices)):
            top = np.argsort(sim[r])[::-1][: k - 1]
            for c in top:
                gi, gj = indices[r], indices[int(c)]
                pairs.add((min(gi, gj), max(gi, gj)))
        return sorted(pairs)


# ─── LightGlue matcher (SuperPoint + LightGlue) ───────────────────────────────

class LoFTRMatcher:
    """
    Detector-free dense matching using kornia's LoFTR.

    LoFTR produces semi-dense correspondences directly from image pairs using
    a Transformer architecture, without requiring any keypoint detection step.
    It excels on textureless surfaces where SIFT and DISK fail to detect
    repeatable keypoints.

    Requires ``torch`` and ``kornia>=0.7``.
    Activated by ``--match-strategy loftr`` in run_sfm.py.

    Parameters
    ----------
    min_matches     : Minimum confident matches to retain a pair.
    pretrained      : LoFTR pretrained variant ('outdoor' or 'indoor').
    candidate_pairs : Optional pre-filtered pair list.  None = exhaustive.
    """

    def __init__(
        self,
        min_matches: int = 15,
        pretrained: str = "outdoor",
        candidate_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.min_matches     = min_matches
        self.pretrained      = pretrained
        self.candidate_pairs = candidate_pairs
        self._matcher        = None   # lazy-loaded

    def _load_matcher(self):
        if self._matcher is not None:
            return self._matcher
        try:
            import torch
            import kornia.feature as KF

            device = get_device()
            lg = KF.LoFTR(pretrained=self.pretrained).eval()
            if device is not None:
                lg = lg.to(device)
            self._matcher = lg
            logger.info(
                f"LoFTR ({self.pretrained}) loaded on {device if device else 'cpu'}"
            )
            return lg
        except Exception as exc:
            raise RuntimeError(
                f"LoFTR could not be initialised: {exc}.  "
                "Install kornia>=0.7 and torch."
            ) from exc

    def match_all(self, features: dict) -> MatchDict:
        """Match all candidate pairs using LoFTR."""
        indices = sorted(features.keys())
        if self.candidate_pairs is not None:
            pairs = self.candidate_pairs
        else:
            if len(indices) > 50:
                logger.warning(
                    "LoFTR: %d images with no retrieval front-end — exhaustive "
                    "O(N²)=%d pairs.  Consider using --retrieval dinov2.",
                    len(indices), len(indices) * (len(indices) - 1) // 2,
                )
            pairs = list(combinations(indices, 2))

        n_pairs = len(pairs)
        logger.info(f"LoFTR matching: {n_pairs} pairs (min_matches={self.min_matches})…")

        loftr = self._load_matcher()
        matches: MatchDict = {}

        for k, (i, j) in enumerate(pairs):
            m = self._match_pair_loftr(loftr, features[i], features[j])
            if len(m) >= self.min_matches:
                matches[(i, j)] = m
            if (k + 1) % max(1, n_pairs // 10) == 0:
                logger.info(f"  … {k+1}/{n_pairs} pairs done")

        logger.info(f"LoFTR done: {len(matches)}/{n_pairs} pairs retained")
        return matches

    def _match_pair_loftr(self, loftr, feat_i: dict, feat_j: dict) -> np.ndarray:
        """Run LoFTR on one pair; return (M, 2) pseudo-keypoint-index match array.

        LoFTR produces floating-point coordinates, not integer keypoint indices.
        We create synthetic integer indices by appending the LoFTR matches as
        new keypoints to the feature dict so downstream code (geometric
        verification) can treat them like standard matches.
        """
        import torch
        import cv2 as _cv2

        device = get_device()

        def _to_gray_tensor(path):
            bgr = _cv2.imread(str(path))
            if bgr is None:
                return None
            gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
            t = torch.from_numpy(gray).float().div(255.0).unsqueeze(0).unsqueeze(0)
            return t.to(device) if device else t

        t_i = _to_gray_tensor(feat_i["image_path"])
        t_j = _to_gray_tensor(feat_j["image_path"])
        if t_i is None or t_j is None:
            return np.zeros((0, 2), dtype=np.int32)

        try:
            with torch.no_grad():
                out = loftr({"image0": t_i, "image1": t_j})
        except Exception as exc:
            logger.debug(f"LoFTR pair failed ({exc})")
            return np.zeros((0, 2), dtype=np.int32)

        conf  = out["confidence"].cpu().numpy()     # (M,)
        kps0  = out["keypoints0"].cpu().numpy()     # (M, 2) float
        kps1  = out["keypoints1"].cpu().numpy()     # (M, 2) float

        # Keep only confident matches (LoFTR default threshold is 0.2)
        good = conf > 0.2
        if not good.any():
            return np.zeros((0, 2), dtype=np.int32)

        kps0 = kps0[good]; kps1 = kps1[good]

        # Append LoFTR-generated keypoints to the feature dicts so that
        # geometric verification can index them.  Use negative base offsets
        # to distinguish from SIFT keypoints.
        n_existing_i = len(feat_i["keypoints"])
        n_existing_j = len(feat_j["keypoints"])

        feat_i["keypoints"] = np.vstack([
            feat_i["keypoints"],
            kps0.astype(np.float32),
        ])
        feat_j["keypoints"] = np.vstack([
            feat_j["keypoints"],
            kps1.astype(np.float32),
        ])

        n_new = int(good.sum())
        qi = np.arange(n_existing_i, n_existing_i + n_new, dtype=np.int32)
        ti = np.arange(n_existing_j, n_existing_j + n_new, dtype=np.int32)
        return np.stack([qi, ti], axis=1)   # (M, 2)


class LightGlueMatcher:
    """
    Pairwise matcher using kornia's LightGlue attention network.

    LightGlue replaces the ratio-test FLANN/GPU path when SuperPoint
    features are used.  It takes per-image (keypoints, descriptors) pairs
    and returns direct confident matches with learned score filtering.

    Requires ``kornia>=0.7`` and ``torch``.

    Activated when ``--feature-backend superpoint`` is combined with
    ``--match-strategy`` in run_sfm.py (LightGlueMatcher is selected
    automatically by run_sfm when the feature backend is superpoint).

    Parameters
    ----------
    min_matches  : Minimum matches to retain a pair.
    depth        : LightGlue Transformer depth (default 9 for full model).
    """

    def __init__(
        self,
        min_matches: int = 15,
        depth: int = 9,
        candidate_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.min_matches      = min_matches
        self.depth            = depth
        self.candidate_pairs  = candidate_pairs   # None = exhaustive
        self._matcher         = None   # lazy-loaded

    def _load_matcher(self):
        if self._matcher is not None:
            return self._matcher
        try:
            import torch
            import kornia.feature as KF

            device = get_device()
            lg = KF.LightGlue("superpoint").eval()
            if device is not None:
                lg = lg.to(device)
            self._matcher = lg
            logger.info(f"LightGlue loaded on {device if device else 'cpu'}")
            return lg
        except Exception as exc:
            raise RuntimeError(
                f"LightGlue could not be initialised: {exc}.  "
                "Install kornia>=0.7 and torch."
            ) from exc

    def match_all(self, features: dict) -> MatchDict:
        """
        Match all candidate pairs using LightGlue.

        If ``candidate_pairs`` was set at construction, only those pairs are
        matched (O(k·N)); otherwise exhaustive O(N²) matching is used.
        """
        indices = sorted(features.keys())
        if self.candidate_pairs is not None:
            pairs = self.candidate_pairs
        else:
            if len(indices) > 50:
                logger.warning(
                    "LightGlue: %d images with no retrieval front-end — falling back "
                    "to exhaustive O(N²)=%d pairs.  Pass candidate_pairs from "
                    "DINOv2Matcher or VocabTreeMatcher to avoid O(N²) GPU compute.",
                    len(indices), len(indices) * (len(indices) - 1) // 2,
                )
            pairs = list(combinations(indices, 2))

        n_pairs = len(pairs)
        logger.info(
            f"LightGlue matching: {n_pairs} pairs "
            f"(min_matches={self.min_matches})…"
        )

        lg      = self._load_matcher()
        matches: MatchDict = {}

        for k, (i, j) in enumerate(pairs):
            m = self._match_pair_lg(lg, features[i], features[j])
            if len(m) >= self.min_matches:
                matches[(i, j)] = m
            if (k + 1) % max(1, n_pairs // 10) == 0:
                logger.info(f"  … {k+1}/{n_pairs} pairs done")

        logger.info(
            f"LightGlue done: {len(matches)}/{n_pairs} pairs retained"
        )
        return matches

    def _match_pair_lg(
        self, lg, feat_i: dict, feat_j: dict
    ) -> np.ndarray:
        """Run LightGlue on one pair; return (M, 2) int32 match array."""
        import torch

        device = get_device()

        kps_i   = feat_i["keypoints"].astype(np.float32)   # (N, 2)
        descs_i = feat_i["descriptors"].astype(np.float32) # (N, 256)
        kps_j   = feat_j["keypoints"].astype(np.float32)
        descs_j = feat_j["descriptors"].astype(np.float32)

        if len(kps_i) < 4 or len(kps_j) < 4:
            return np.zeros((0, 2), dtype=np.int32)

        def to_tensor(arr):
            t = torch.from_numpy(arr).unsqueeze(0)   # (1, N, D)
            return t.to(device) if device else t

        try:
            with torch.no_grad():
                out = lg({
                    "image0": {
                        "keypoints": to_tensor(kps_i),
                        "descriptors": to_tensor(descs_i),
                    },
                    "image1": {
                        "keypoints": to_tensor(kps_j),
                        "descriptors": to_tensor(descs_j),
                    },
                })
        except Exception as exc:
            logger.debug(f"LightGlue pair failed ({exc})")
            return np.zeros((0, 2), dtype=np.int32)

        # out["matches0"][0]: (N_i,) — for each kp in image0, matched kp idx in image1 or -1
        matches0 = out["matches0"][0].cpu().numpy()   # (N_i,)
        valid    = matches0 >= 0
        if not valid.any():
            return np.zeros((0, 2), dtype=np.int32)

        qi = np.where(valid)[0].astype(np.int32)
        ti = matches0[valid].astype(np.int32)
        return np.stack([qi, ti], axis=1)   # (M, 2)
