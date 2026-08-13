# `eval/` — SfM evaluation harness

Measures the pipeline's **efficiency** and the **accuracy of its results** against
ground-truth camera poses. Nothing here modifies the pipeline; the only change
made to `run_sfm.py` for this harness is the opt-in `--export-cameras PATH` flag,
which dumps registered poses, per-stage timings and reprojection statistics as
JSON.

The brief this harness implements is [../EVALUATION_PROMPT.md](../EVALUATION_PROMPT.md);
the measured output is [../EVALUATION_RESULTS.md](../EVALUATION_RESULTS.md).

## Why ground truth matters here

The AliceVision Buddha dataset ships a 3×4 projection matrix `NNNNN_P.txt` next
to every image in `buddha/`, and those images are byte-identical to the ones in
`imgs_only/`. So camera poses and focal length can be scored absolutely rather
than by reprojection error alone — which matters, because this pipeline reports
a healthy reprojection RMSE on a reconstruction that is measurably deformed.

## Full run

```bash
DATA=/path/to/dataset_buddha-master/dataset_buddha-master
SCRATCH=/path/to/scratch

# 1. Build the image subsets (6 / 20 / 67 + the dataset's own mini6)
python eval/make_subset.py "$DATA/imgs_only"    "$SCRATCH/buddha_06"    --n 6
python eval/make_subset.py "$DATA/imgs_only"    "$SCRATCH/buddha_20"    --n 20
python eval/make_subset.py "$DATA/imgs_only"    "$SCRATCH/buddha_67"    --n 67
python eval/make_subset.py "$DATA/buddha_mini6" "$SCRATCH/buddha_mini6" --n 6

# 2. Run the configuration matrix (sequential; ~1 h on a CPU-only machine)
python eval/run_matrix.py --data-root "$SCRATCH" --results-dir eval_results

# 3. Behavioural tests
python eval/robustness.py       --image-dir "$DATA/imgs_only" \
                                --results-dir eval_results \
                                --json eval_results/robustness.json
python eval/checkpoint_test.py  --image-dir "$SCRATCH/buddha_20" \
                                --results-dir eval_results \
                                --ckpt-dir "$SCRATCH/ckpt_ct" \
                                --json eval_results/checkpoint_test.json
python eval/hash_collision_test.py --image-dir "$SCRATCH/buddha_67" \
                                --work-dir "$SCRATCH/hashtest" \
                                --json eval_results/hash_collision.json

# 4. Aggregate everything into the report
python eval/report.py --results-dir eval_results \
                      --gt-dir "$DATA/buddha" --gt-dir-mini6 "$DATA/buddha_mini6" \
                      --out EVALUATION_RESULTS.md
```

`run_matrix.py` skips configurations that already have a `.run.json`, so an
interrupted matrix resumes where it stopped (`--force` to re-run, `--only a,b`
to select).

## Scripts

| Script | Answers |
|---|---|
| `make_subset.py` | Builds evenly-spaced N-image subsets (hard-links where possible). Even spacing matters: consecutive Buddha filenames are *not* consecutive viewpoints. |
| `run_config.py` | Runs one configuration, capturing wall time, peak RSS of the process tree, the log, and the exported poses into `<name>.run.json`. |
| `run_matrix.py` | The configuration matrix — three scales, the efficiency levers, repeated identical runs, dense+mesh, and the command documented in `run_sfm.py`'s own usage header. Sequential so timings stay comparable. |
| `gt_pose_eval.py` | Scores poses against `*_P.txt`: Sim(3) (Umeyama) alignment on camera centres, plus a RANSAC variant, then rotation error in degrees, position error as % of scene extent, and estimated vs ground-truth focal. |
| `ply_stats.py` | Point-cloud health: density, nearest-neighbour distribution, exact duplicates, flyers, outliers, colour validity. |
| `robustness.py` | Degenerate inputs (missing/empty dir, 1 image, identical images, corrupt file, no overlap) and unavailable backends. Asserts *behaviour*: clean actionable error, never a traceback or a silent wrong answer. |
| `checkpoint_test.py` | `--resume` speedup, cache invalidation correctness, and determinism isolation — two resumed runs consume identical cached features and matches, so any disagreement is generated downstream of matching. |
| `hash_collision_test.py` | Whether the checkpoint key can miss a changed image set: two different scenes as same-dimension BMPs give identical filenames and identical byte sizes. |
| `focal_scaling_probe.py` | Tests one hypothesis for the frozen focal length by injecting `x_scale="jac"` into every `least_squares` call in a child process. Result: hypothesis **rejected** — kept because a ruled-out explanation is worth recording. |
| `report.py` | Aggregates every artefact into `EVALUATION_RESULTS.md`, appending the hand-written analysis in `findings.md`. Re-runs nothing; anything missing renders as an explicit `n/a`. |

## Requirements

Core pipeline dependencies plus `psutil` (peak-memory sampling; the harness
degrades to `null` without it). `eval_results/` is gitignored.
