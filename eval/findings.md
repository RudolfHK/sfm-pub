## C. Status of the standing weakness claims (`SFM_QUALITY_REPORT.md`)

That report was a code reading dated 2026-06-01. Each claim below was re-checked
against the current tree; "demonstrated" means a run in `eval_results/` exhibits
the behaviour, not merely that the code looks that way.

| ID | Claim | Status | Evidence |
|---|---|---|---|
| W-01 | pyceres BA silently runs zero iterations (`break` in the residual loop) | **Fixed** | The loop now `continue`s per observation with an explanatory comment, [sfm/bundle_adjustment.py:736-741](sfm/bundle_adjustment.py#L736-L741). Untestable at runtime — pyceres is not installed. |
| W-02 | `_register_image` double-undistorts points passed to PnP | **Fixed** | Zero distortion is passed alongside pre-undistorted points, [sfm/reconstruction.py:598-628](sfm/reconstruction.py#L598-L628). |
| W-03 | BA-refined per-camera intrinsics never written back | **Partially fixed** | `update_from_ba` is now called, but only with `k1`/`k2` — the refined focal and principal point are still dropped, [sfm/reconstruction.py:1027-1033](sfm/reconstruction.py#L1027-L1033). Low practical impact today because the focal never moves at all (D2). |
| W-04 | scipy BA assumes square pixels (fx = fy) | **Fixed** | `--ba-separate-focal` adds independent fx/fy, [sfm/bundle_adjustment.py:163-190](sfm/bundle_adjustment.py#L163-L190). |
| W-05 | Planarity check's SVD indexing relies on coincidence | **Still present, and the check is mislabelled** | `_sv[-1]/_sv[-2]` equals σ₂/σ₁ only because the matrix has exactly two columns. More importantly it operates on 2-D *image* points, which are planar by construction, so it detects **collinearity**, not the planar-scene degeneracy the comment claims, [sfm/geometric_verification.py:196-207](sfm/geometric_verification.py#L196-L207). |
| W-06 | No homography fallback for planar scenes | **Partially addressed** | Planar pairs are now detected and *skipped* rather than handled, [sfm/geometric_verification.py:170-190](sfm/geometric_verification.py#L170-L190). Rejecting a pair avoids a wrong pose but still loses the data. |
| W-07 | Kornia LAF scale × 6.0 is unvalidated | **Untestable here** | kornia not installed. |
| W-08 | `_accept_batch` uses the new camera's K for both cameras | **Fixed** | `K2=K_j` is passed explicitly, [sfm/reconstruction.py:481](sfm/reconstruction.py#L481). |
| W-09 | MVS "fusion" is not true multi-view fusion | **Still present** | `--mvs-fusion` filters on a positive-depth visibility count only, [sfm/mvs.py:232-252](sfm/mvs.py#L232-L252); there is no photometric consistency test. |
| W-10 | EXIF orientation tags ignored | **Fixed** | `PIL.ImageOps.exif_transpose` is applied on load, [sfm/utils.py:43-53](sfm/utils.py#L43-L53). |
| W-11 | LightGlue always exhaustive when `candidate_pairs=None` | **Untestable here** | torch not installed. |
| W-12 | Global BA only, no local BA | **Fixed** | `--local-ba-window` implemented; measured at N=20 it costs +1.3 s and changed nothing else materially (see A.1). |
| W-13 | No loop-closure detection | **Partially fixed** | `--loop-closure` exists but requires `--retrieval dinov2` or `--match_strategy vocab_tree`; the DINOv2 path cannot run without torch, so on this machine loop closure is reachable only via vocab_tree. |
| W-14 | Checkpoint hash uses only filename and size | **Still present — demonstrated** | `_image_set_hash` digests paths and `getsize` only, [run_sfm.py:673-685](run_sfm.py#L673-L685). `eval/hash_collision_test.py` builds two different scenes as same-dimension BMPs (identical names, identical 1 555 254-byte sizes) and the resumed run reports a cache hit with no staleness warning: scene A's features are reused for scene B. |
| W-15 | PLY writer uses float64 | **Fixed** | Header declares `property float`, [sfm/point_cloud.py:310-312](sfm/point_cloud.py#L310-L312); measured 15.0 bytes/point (3×f32 + 3×u8). |
| W-16 | Mesh vertex colour uses k=1 NN | **Still present** | [sfm/mesh/postprocess.py:7](sfm/mesh/postprocess.py#L7). |
| W-17 | GPU brute-force matcher allocates O(N²) | **Fixed** | Chunked `torch.cdist` in 1024-row blocks, [sfm/feature_matching.py:125-165](sfm/feature_matching.py#L125-L165). Untestable at runtime — torch not installed. |

Nine of seventeen are fixed, three partially, three still present, and two cannot
be exercised in this environment. The report's headline correctness bugs (W-01,
W-02) are genuinely gone.

---

## D. Defects found by this evaluation

Ranked by impact on result trustworthiness. Every item is backed by a run in
`eval_results/`.

### D1 — Output is not reproducible (critical)

Four runs of a byte-identical command on the same 20 images registered **13, 13,
14 and 6 cameras** and produced 5 522 – 6 283 points. A fifth equivalent run
(`n20_dense`, same reconstruction flags) registered 7. A user can therefore lose
more than half the reconstruction between two invocations with nothing changed.

The `--resume` experiment localises the cause: `warm1` and `warm2` load
**identical cached features and identical cached matches**, and still disagree
(13 vs 14 cameras, 6 281 vs 6 429 points). The variation is therefore produced
downstream of matching, in geometric verification / incremental SfM / BA.

Mechanism: `cv2.setRNGSeed` is never called anywhere in the repository, so every
`USAC_MAGSAC` and `solvePnPRansac` draws from OpenCV's process-global RNG, seeded
from system state. Note that the pipeline seeds its *numpy* RNGs consistently
(`np.random.default_rng(0)` in four places) — only the OpenCV side was missed.

**Fix**: call `cv2.setRNGSeed(seed)` once in `main()` and expose `--seed`.
This is a few lines and would make every other measurement in this report
sharper.

### D2 — Focal length is 47 % wrong and unrecoverable (critical)

Ground truth is **1 860.9 px**; every run estimates **2 736.0 px**, exactly the
image width, on all three scales. The Buddha PNGs carry no EXIF, so
`estimate_intrinsics` falls back to `focal = max(W, H)`
([sfm/utils.py:108-110](sfm/utils.py#L108-L110)) — a ~53° FoV guess that happens
to be badly wrong for this camera.

Bundle adjustment never repairs it. Across 67 cameras every BA call logs
`f=2736.0 (Δ±0.00)`. Two candidate explanations were tested:

- *Bounds clamp the focal* — **ruled out**: bounds are `[0.5·f₀, 2.0·f₀]` =
  [1368, 5472] px and contain the true value
  ([sfm/bundle_adjustment.py:403-404](sfm/bundle_adjustment.py#L403-L404)).
- *Parameter scaling freezes it* — **ruled out by experiment**.
  `eval/focal_scaling_probe.py` re-runs the pipeline in-process with
  `x_scale="jac"` injected into every `least_squares` call; the focal still does
  not move (47.026 % → 47.026 %). `x_scale` is not the limiting factor.

The remaining explanation, consistent with all the evidence, is that the
reconstruction is already self-consistent at the wrong focal: the seed pose,
every triangulation and every PnP were computed with f = 2736, so the structure
is projectively deformed to fit it and BA sits in a deep local minimum with
nothing to gain. Supporting evidence: on the full 67-image run the reprojection
RMSE is **2.08 px** and the median residual is **0.84 px** — comfortably inside
the 4 px acceptance threshold — while that same reconstruction carries a **6.29°
median rotation error** against ground truth.

**This is the single most important result of the evaluation**: the pipeline's
own quality signal (reprojection error) looks good precisely when the geometry
is wrong. Any tuning done against reprojection error alone is tuning against a
metric that cannot see this failure.

**Fix**: add a `--focal` / `--intrinsics` override — there is currently **no CLI
way to supply a known calibration** (`--help` exposes no such option) — and
consider a focal sweep at seed time, which is what COLMAP does when EXIF is
absent.

### D3 — Checkpoint cache is content-blind (high)

Demonstrated, not merely read: `eval/hash_collision_test.py` gives two entirely
different 4-image scenes the same filenames and the same byte sizes (uncompressed
BMPs of equal dimensions). The resumed run reports a cache hit, logs no staleness
warning, and reconstructs scene B using scene A's features. Any workflow that
edits images in place — re-exporting, colour-correcting, undistorting — will
silently produce results for the previous version of the data.

**Fix**: hash content (or at least mtime) in `_image_set_hash`
([run_sfm.py:673-685](run_sfm.py#L673-L685)).

### D4 — One unreadable image aborts the whole run (high)

A single corrupt file among valid images terminates the pipeline with an
uncaught `OSError` from [sfm/utils.py:59](sfm/utils.py#L59), after feature
extraction has already been paid for. On a 67-image, 19-minute run, one bad file
costs the entire job. Skipping it with a warning is the obvious behaviour.

### D5 — Unavailable optional backends leak tracebacks (medium)

`--feature-backend superpoint`, `--feature-backend disk`, `--match_strategy
loftr` and `--retrieval dinov2` all fail with a raw `ModuleNotFoundError`
traceback rather than the actionable "install X" message the argparse help
promises. By contrast `--backend colmap` (clean pre-flight check),
`--ba-backend pyceres` and `--pnp-backend poselib` (announced fallbacks) get this
right — so the pattern already exists in the codebase and is simply not applied
to the four feature/matching backends.

Note that `--retrieval dinov2` only reaches its failure once N > `--retrieval_top_k`
+ 1; below that it silently and legitimately falls back to exhaustive matching.

### D6 — Dense output is silently truncated at 500 000 points (medium)

Both dense runs returned *exactly* 500 000 points — the hard-coded
`max_dense_pts` default ([sfm/mvs.py:57](sfm/mvs.py#L57)), which has no CLI flag.
The dense cloud is therefore limited by a constant, not by the scene, and the
point count carries no information about reconstruction quality. Nothing in the
log says truncation occurred.

### D7 — ~10 % of sparse points are exact duplicates (medium)

10.7 % of points in `n20_base` and 8.1 % in `n67_base` have a nearest-neighbour
distance below 1e-9 — the same 3-D point stored more than once, i.e. tracks that
should have merged. `--track-merge`, which exists for this purpose, does not help:
10.68 % → 10.87 %, with 12 rather than 13 cameras registered. Given D1, that
difference is inside run-to-run noise, so the honest statement is that
`--track-merge` shows **no measurable benefit** on this dataset.

### D8 — Matching dominates cost and scales quadratically (medium, by design)

Matching is 91 % of the 67-image run (1 036.6 s of 1 134.4 s). Measured cost per
pair is essentially flat — 0.505 s/pair at N=6, 0.497 at N=20, 0.469 at N=67 —
so total matching time tracks the O(N²) pair count almost exactly. Extrapolating,
200 images would spend ~2.6 h in matching alone. The alternatives are weak on
this dataset: `sequential` halves the time but collapses the reconstruction to
4/20 cameras, because consecutive Buddha filenames are not consecutive
viewpoints; the retrieval-based options need torch.

---

## E. Verdicts

| Dimension | Verdict | Basis |
|---|---|---|
| Completeness | **Good at full density, poor when sparse** | 67/67 cameras on the full set; 13/20 on an evenly-spaced 20-image subset; 2/6 at 60° spacing. The pipeline needs dense angular sampling. |
| Pose accuracy | **Weak** | 6.29° median rotation error, 2.06 % median position error on the full 67-image set. A healthy SfM pipeline on a controlled 360° capture should be well under 1°. |
| Intrinsics | **Failing** | +47 % focal error, never corrected, no override available (D2). |
| Precision (self-consistency) | **Good but misleading** | 0.35–2.08 px reprojection RMSE across all 14 runs, median residual 0.84 px at N=67 — while the geometry is measurably deformed. The 5.4 % of residuals above 4 px at N=67 is the only self-reported hint that anything is wrong. |
| Point-cloud health | **Acceptable with caveats** | Sub-1 % flyers and outliers, 99.96 % of points genuinely coloured; but ~10 % exact duplicates (D7). |
| Efficiency | **Acceptable at this scale, will not scale** | ~17 s/image end-to-end at N=67, 1.47 GB peak. Quadratic matching is the wall (D8). |
| Checkpointing | **Effective but unsafe** | 19.9× speedup, correct invalidation on parameter and image-set changes, but content-blind hashing (D3). |
| Robustness | **Mixed** | 8/13 cases pass; input validation is genuinely good, optional-backend failure handling is not (D4, D5). |
| Reproducibility | **Failing** | Up to 57 % variation in registered cameras between identical runs (D1). |

### Recommended order of work

1. **Seed the OpenCV RNG** (D1). Everything else is unmeasurable until results
   are repeatable — including whether any fix below actually helped.
2. **Add a focal/intrinsics override and stop trusting `max(W, H)`** (D2). This
   is the largest single source of geometric error measured here.
3. **Hash image content in the checkpoint key** (D3) and **skip unreadable
   images** (D4). Both are small and prevent silent or total data loss.
4. Route the four unavailable backends through the same pre-flight check
   `--backend colmap` already uses (D5), and expose `max_dense_pts` (D6).
5. Only then revisit track merging (D7) and matching scalability (D8).

### Note on the pipeline's own test suite

[integration_test.py](integration_test.py) still passes (6/6 cameras, 0.874 px
RMSE) with the `--export-cameras` addition. It cannot catch any defect above: it
injects analytic correspondences, so it bypasses features and matching, uses a
known-exact K, and re-runs are compared against fixed thresholds rather than
against each other. D1 and D2 in particular are invisible to it by construction.
