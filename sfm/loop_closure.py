"""
Geometric loop closure and registration rescue, without a retrieval network.

The problem
-----------
`paper/paper_l.md` section 7 recorded two related failures of the incremental
loop. Registration accumulates tension because every new camera is hung off the
existing block and nothing ever re-checks whether two cameras that ended up
close together actually share observations; and the run stops with images
unregistered as soon as no candidate has enough 2D-3D correspondences left.

The shipped `--loop-closure` only bridges *disconnected components* and needs
DINOv2 or a vocabulary tree, so on a machine without PyTorch it is unreachable.

The idea here
-------------
After a first reconstruction the poses are already known well enough to say
which images *should* see each other, even if the match graph never connected
them. Two cameras are proposed as a candidate pair when

  * their centres are close relative to their distance from the scene, and
  * their viewing directions differ by less than a threshold, and
  * their optical axes pass near a common point, and
  * no verified edge between them exists yet.

Unregistered images cannot be scored this way, so they get the complementary
treatment: the pairs with the most raw matches against registered images are
re-verified with relaxed thresholds. Both candidate sets are small, in the tens,
so matching them costs a fraction of the exhaustive stage that already ran.

Nothing here needs a library the pipeline does not already use, and nothing is
believed on trust: proposed pairs go through the ordinary geometric
verification before they are allowed into the scene graph.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PairKey = Tuple[int, int]


def _camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ np.asarray(t, dtype=np.float64).reshape(3)).ravel()


def _viewing_direction(R: np.ndarray) -> np.ndarray:
    """World-space optical axis (third row of R, by the world-to-camera form)."""
    return np.asarray(R, dtype=np.float64)[2].ravel()


def propose_loop_pairs(
    cameras: Dict[int, dict],
    existing: Iterable[PairKey],
    max_angle_deg: float = 45.0,
    max_centre_ratio: float = 0.6,
    max_pairs: int = 200,
) -> List[PairKey]:
    """
    Camera pairs that the current poses say should overlap but which have no
    verified edge.

    Parameters
    ----------
    cameras          : {img_idx: {'R', 't'}} of the registered cameras.
    existing         : pair keys already present in the scene graph.
    max_angle_deg    : largest angle between optical axes still counted as
                       overlapping. 45° keeps genuinely shared viewpoints and
                       drops opposite sides of the object.
    max_centre_ratio : largest distance between the two centres, as a fraction
                       of the median distance from camera to scene centre.
    max_pairs        : cap, so a large scene cannot explode the matching cost.

    Returns
    -------
    Candidate pairs (i < j), best first.
    """
    idx = sorted(cameras)
    if len(idx) < 3:
        return []

    have: Set[PairKey] = {(min(a, b), max(a, b)) for a, b in existing}

    centres = np.array([_camera_centre(cameras[i]["R"], cameras[i]["t"]) for i in idx])
    dirs = np.array([_viewing_direction(cameras[i]["R"]) for i in idx])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-12)

    # Scene centre as the least-squares intersection of the optical axes; it
    # gives the natural length unit for "close together".
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    try:
        scene = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        scene = centres.mean(axis=0)
    radius = float(np.median(np.linalg.norm(centres - scene, axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        return []

    cos_thr = float(np.cos(np.radians(max_angle_deg)))
    dist_thr = max_centre_ratio * radius

    scored: List[Tuple[float, PairKey]] = []
    for a in range(len(idx)):
        for b_ in range(a + 1, len(idx)):
            i, j = idx[a], idx[b_]
            key = (min(i, j), max(i, j))
            if key in have:
                continue
            cos_ang = float(np.dot(dirs[a], dirs[b_]))
            if cos_ang < cos_thr:
                continue
            dist = float(np.linalg.norm(centres[a] - centres[b_]))
            if dist > dist_thr:
                continue
            # Prefer pairs that are close and similarly oriented, but not
            # degenerate: a tiny baseline carries no depth information.
            if dist < 1e-6 * radius:
                continue
            score = cos_ang * (1.0 - dist / max(dist_thr, 1e-12))
            scored.append((score, key))

    scored.sort(key=lambda s: -s[0])
    return [key for _, key in scored[:max_pairs]]


def propose_rescue_pairs(
    unregistered: Sequence[int],
    registered: Sequence[int],
    all_matches: Dict[PairKey, np.ndarray],
    top_k: int = 8,
) -> List[PairKey]:
    """
    For each unregistered image, its strongest raw-match partners among the
    registered ones.

    These pairs were matched before and rejected by geometric verification, or
    they survived but carried too few inliers to register the image. Either way
    they are the only evidence available for those images, so they get a second
    pass at relaxed thresholds rather than being dropped silently.
    """
    reg = set(registered)
    proposals: List[PairKey] = []
    for img in unregistered:
        scored: List[Tuple[int, PairKey]] = []
        for key, matches in all_matches.items():
            if img not in key:
                continue
            other = key[0] if key[1] == img else key[1]
            if other not in reg:
                continue
            scored.append((len(matches), key))
        scored.sort(key=lambda s: -s[0])
        proposals.extend(key for _, key in scored[:top_k])
    # Deduplicate, keep order.
    seen: Set[PairKey] = set()
    unique: List[PairKey] = []
    for key in proposals:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def close_loops(
    cameras: Dict[int, dict],
    features: dict,
    all_matches: Dict[PairKey, np.ndarray],
    verified: Dict[PairKey, dict],
    matcher,
    verifier,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    unregistered: Optional[Sequence[int]] = None,
    max_angle_deg: float = 45.0,
    max_centre_ratio: float = 0.6,
    max_pairs: int = 200,
    relaxed_ratio: Optional[float] = None,
) -> Dict[str, object]:
    """
    Find, match and verify the missing edges; return what was added.

    ``matcher`` must expose ``match_pair_indices(features, pairs, ratio=None)``
    or ``match_pairs``; only the pairs handed in are matched. ``verifier`` is
    the ordinary :class:`GeometricVerifier`.

    The caller decides what to do with the result. The intended use is to merge
    the new edges into the scene graph and run the reconstruction again, which
    is cheap compared with matching and keeps this module free of any
    reconstruction state.
    """
    stats: Dict[str, object] = {
        "loop_candidates": 0, "rescue_candidates": 0,
        "matched": 0, "verified": 0, "new_edges": [],
    }

    loop_pairs = propose_loop_pairs(
        cameras, verified.keys(),
        max_angle_deg=max_angle_deg,
        max_centre_ratio=max_centre_ratio,
        max_pairs=max_pairs,
    )
    stats["loop_candidates"] = len(loop_pairs)

    rescue_pairs: List[PairKey] = []
    if unregistered:
        rescue_pairs = propose_rescue_pairs(
            list(unregistered), list(cameras.keys()), all_matches,
        )
        rescue_pairs = [p for p in rescue_pairs if p not in verified]
        stats["rescue_candidates"] = len(rescue_pairs)

    candidates = list(dict.fromkeys(loop_pairs + rescue_pairs))
    if not candidates:
        logger.info("[LOOP] No candidate pairs proposed.")
        return stats

    logger.info(
        "[LOOP] %d candidate pairs proposed (%d from pose overlap, %d to rescue "
        "unregistered images); matching them.",
        len(candidates), len(loop_pairs), len(rescue_pairs),
    )

    new_matches = matcher.match_pair_indices(features, candidates, ratio=relaxed_ratio)
    stats["matched"] = len(new_matches)

    added: List[PairKey] = []
    for key, matches in new_matches.items():
        if key in verified:
            continue
        i, j = key
        result = verifier.verify_pair(
            features[i]["keypoints"], features[j]["keypoints"],
            matches, K, dist_coeffs=dist_coeffs,
        )
        if result is not None:
            verified[key] = result
            all_matches[key] = matches
            added.append(key)

    stats["verified"] = len(added)
    stats["new_edges"] = added
    logger.info(
        "[LOOP] %d of %d matched pairs passed verification and entered the "
        "scene graph.", len(added), len(new_matches),
    )
    return stats
