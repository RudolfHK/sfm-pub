[TOC]

# Visualization Guide

This guide explains every figure the SfM visualization suite produces, what
healthy output looks like, what pathological output looks like, and what to do
when something looks wrong.

---

## Introduction

### What the visualization suite shows

The SfM visualization suite generates diagnostic figures at every major stage
of the pipeline:

| Stage | What it captures |
|-------|-----------------|
| Feature extraction | How many keypoints were found and where they are |
| Feature matching | Which image pairs share features and how reliably |
| Geometric verification | Whether the fundamental matrix is well-conditioned |
| Registration steps | How the reconstruction grows camera-by-camera |
| Bundle adjustment | How the global reprojection error evolves |
| Final point cloud | The colorized sparse 3-D output from six views |
| Summary | A one-page dashboard of key health metrics |

Every figure is saved to disk and can be opened with any image viewer.  No GUI
is required during the pipeline run.

### How to enable it

```bash
# Minimal — save PNG figures to sfm_visualization/
python run_sfm.py --image_dir ./images --output out.ply --visualize

# More control
python run_sfm.py \
  --image_dir   ./images   \
  --output      out.ply    \
  --visualize              \
  --viz-samples 5          \   # sample 5 images/pairs instead of the default 3
  --viz-output  ./my_viz   \   # custom output directory
  --viz-format  jpg        \   # jpg / png / pdf
  --viz-dpi     200        \   # DPI for saved figures
  --viz-seed    0          \   # reproducible sampling
  --viz-save-video         \   # export GIF animations (requires imageio)
  --viz-interactive            # open open3d viewer at end (requires open3d)
```

When `--visualize` is **not** passed the flag has absolute zero cost: the
visualizer is never imported and no memory is allocated.

### Output folder structure

```
sfm_visualization/                    ← --viz-output root
├── 00_summary/
│   └── pipeline_summary.png          ← one-page dashboard (start here)
├── 01_features/
│   ├── features_<stem>.png           ← keypoint overlay (one per sampled image)
│   ├── density_<stem>.png            ← feature density heatmap
│   └── feature_statistics.png        ← dataset-wide keypoint statistics
├── 02_matching/
│   ├── matches_<A>_<B>.png           ← side-by-side match drawing
│   ├── epipolar_<A>_<B>.png          ← epipolar lines (one per sampled pair)
│   ├── match_matrix.png              ← N×N inlier count heatmap
│   └── connectivity_graph.png        ← image connectivity graph
├── 03_reconstruction/
│   ├── step_001_seed_pair.png        ← seed pair initialisation step
│   ├── step_002_camera_registered.png
│   ├── step_003_camera_registered.png
│   ├── …
│   ├── bundle_adjustment_convergence.png
│   ├── point_lifecycle.png
│   ├── camera_poses_final.png
│   └── reprojection_errors_<stem>.png  (one per sampled registered image)
├── 04_pointcloud/
│   ├── pointcloud_6views.png
│   └── pointcloud_turntable.gif      ← only with --viz-save-video
└── reconstruction_growth.gif         ← only with --viz-save-video
```

**Start with `00_summary/pipeline_summary.png`.**  If that looks healthy, dig
into the per-stage figures only when you want to understand a specific issue.

---

## Chapter 1 — Feature Extraction Outputs

Directory: `01_features/`

---

### `features_<image>.png` — Keypoint Overlay

**What it shows**

A scatter of colored dots over the image, one dot per detected keypoint.
The dots are colored by detection order using the `RdYlBu` colormap.
Because SIFT returns keypoints sorted by response strength (strongest first),
the mapping is:

| Color | Meaning |
|-------|---------|
| **Red / warm** | Keypoints with the highest response (strong corners, blobs, edges) |
| **Yellow** | Medium-strength keypoints |
| **Blue / cool** | Lowest-strength keypoints retained within the `--n_features` budget |

The title reports the total keypoint count and the extractor backend
(`SIFT` or `kornia/GPU`).

**Healthy result**

- Dots spread broadly across most of the image, not clustered in one region.
- A visible gradient from red (concentrated in sharp corners, edges, and
  textured patches) to blue (scattered across lower-contrast regions).
- At least a few hundred keypoints on the object of interest rather than
  almost all on background.
- Typical range: 2,000 – 8,000 keypoints per image for a well-textured scene.

**Unhealthy result**

| Symptom | Likely cause |
|---------|-------------|
| All dots in one corner | Strong vignetting, overexposure, or a dominant high-contrast border element |
| Very few dots (<500) on a complex object | Image is too blurry, underexposed, or heavily compressed |
| All dots on sky/floor, none on the object | The object has low texture; SIFT finds features wherever contrast exists |
| A single large red cluster and nothing else | One very dominant feature region (e.g., a bright window) suppressing everything else |

**What to do**

- *Too few features on the object:* Increase `--n_features`, shoot with better
  lighting, or add texture (chalk patterns, stickers) to the object surface.
- *Features only on background:* Crop or mask images to focus on the subject.
- *All clustered:* Check exposure/white balance; consider disabling or adjusting
  any in-camera sharpening.

---

### `density_<image>.png` — Feature Density Heatmap

**What it shows**

Two side-by-side panels showing a Gaussian-smoothed 2-D histogram of
keypoint positions, rendered with the `hot` colormap:

| Color (hot) | Meaning |
|-------------|---------|
| Black / very dark | No or very few keypoints |
| Dark red | Sparse keypoint coverage |
| Bright red/orange | Moderate density |
| Yellow / white | Very high density (hotspot) |

Left panel overlays the heatmap on the original image at 50 % opacity.
Right panel shows the heatmap alone on a black background, making subtle
density differences easier to read.

**How to use it**

1. **Find textureless regions.** Any black or very dark region on the heatmap
   means the pipeline has no features there.  Those image regions *cannot*
   contribute to the reconstruction; 3-D points inside them will be missing.

2. **Spot hotspots.** A single blazing white spot with black everywhere else
   means nearly all features cluster in one small area.  This is similar to
   the keypoint overlay diagnosis above.

3. **Assess overall uniformity.** A roughly uniform warm hue across the image
   is ideal; it means features are distributed to span the image, which gives
   bundle adjustment strong constraints everywhere.

**Healthy result** — Warm orange/red hue spread broadly over the image with
natural bright spots at the most textured areas (brickwork, tree foliage,
window frames) and acceptable cool areas over flat surfaces.

**Unhealthy result** — White hotspot in one corner, black everywhere else.
This image will dominate the feature budget and contribute very few 3-D points
outside that hotspot.

---

### `feature_statistics.png` — Dataset-wide Statistics

**What it shows**

Three panels summarising keypoint counts across the entire image set:

**Panel 1 — Keypoints-per-image histogram**

A histogram where the x-axis is the number of keypoints and the y-axis is
the number of images that fell in each bin.  A red dashed line marks the
dataset mean.

*What to look for:*
- The distribution should be roughly unimodal and centered.
- A long left tail (many images with very few keypoints) signals images that
  will struggle to register.
- A spike at zero or near-zero means some images extracted almost nothing —
  those images will almost certainly be skipped.

**Panel 2 — Top-N images by keypoint count (horizontal bar chart)**

Shows up to 15 images ranked by keypoint count.
**Red bars** mean the image has fewer than 100 keypoints — a critical warning.
Any image below this threshold is very unlikely to produce useful matches and
will probably fail to register.

*What to look for:*
- All bars the same color (steelblue) — healthy, no outliers.
- One or two red bars — investigate those images; they may be motion-blurred,
  heavily occluded, or nearly identical to their neighbors.
- Many red bars — the dataset has a systematic quality issue (bad lighting,
  wrong settings, overly compressed JPEGs).

**Panel 3 — Text summary box**

```
Total images:     N
Total keypoints:  M,MMM
Mean per image:   K
Min / Max:        low / high
Images < 100 kps: X
```

*Key health indicators:*

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Mean per image | >2,000 | 500–2,000 | <500 |
| Images < 100 kps | 0 | 1–2 | ≥3 |

---

## Chapter 2 — Feature Matching Outputs

Directory: `02_matching/`

---

### `matches_<A>_<B>.png` — Pairwise Match Drawing

**What it shows**

Two images placed side-by-side on a single canvas.  Lines are drawn between
matched keypoints; up to 200 random matches are drawn for clarity:

| Line color | Meaning |
|------------|---------|
| **Green** (semi-transparent) | Inliers — survived RANSAC geometric verification |
| **Red** (semi-transparent) | Outliers — raw matches rejected by RANSAC |

The title reports: `Total: N  Inliers: M (P%)  Outliers: O`

**Reading the inlier ratio**

The inlier ratio `P = M / N` is the single most important number in this plot.

| Inlier ratio | Interpretation |
|-------------|----------------|
| >70 % | Excellent — clean, consistent geometry |
| 50–70 % | Good — normal for most image pairs |
| 30–50 % | Acceptable — geometry is recovered but matches are noisy |
| <30 % | Poor — RANSAC may be fitting noise; reconstruction may be unreliable |
| <15 % | Likely wrong pair or degenerate configuration |

**Healthy result** — Predominantly green lines, roughly parallel and spanning
both images uniformly.  Lines cross cleanly without obvious clusters.

**Unhealthy results**

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Mostly red lines, low inlier count | Large viewpoint change, insufficient overlap, blur | Reshoot with smaller step between positions |
| Many green lines but all horizontal / all from one image region | Repetitive texture (brick, floor tiles, fence) | Increase `--min_inliers` threshold; use `--match_strategy sequential` |
| Lines that fan out wildly / cross at extreme angles | Moving objects in the scene, or degenerate planar scene | Remove moving objects; for planar scenes use homography verification |
| Zero lines shown | Pair was rejected after geometric verification; should not appear here | — |

A critical image pair (one that bridges two groups of photos) with a low
inlier ratio is a serious problem.  If two clusters of images only share one
connecting pair and that pair has a 20 % inlier ratio, the whole reconstruction
may split or drift at that junction.

---

### `match_matrix.png` — Match Heatmap

**What it shows**

An N×N symmetric matrix where:
- Row and column indices correspond to images (labeled by filename stem).
- Each cell's color intensity represents the number of verified inlier matches
  between that pair.
- White/zero cells (masked) mean the pair had no verified matches.
- Numbers in each cell show the exact inlier count.

The colormap is `Blues`: light blue = few matches, dark blue = many matches.

**Reading the matrix**

*Well-connected dataset:*
Many non-zero cells, especially near the diagonal (sequential images) and
potentially off-diagonal (image overlap at different angles).  The matrix looks
like a dense band or checkered pattern depending on the capture strategy.

*Fragmented dataset — dark rows AND columns:*
If an image's row and column are both mostly zero, that image shares no
verified pairs with any other image.  It **will not register**.  This is the
earliest warning signal before registration even begins.

*Two disconnected clusters:*
The matrix shows two dense blocks on the diagonal with an empty off-diagonal
region between them.  The pipeline will reconstruct two separate components
and only keep the largest.  You are missing coverage between the two clusters
— reshoot images that bridge the gap.

*Symmetric stripe close to zero for a single image:*
One image row is darker than its neighbors even though it is surrounded by
images that match well.  That image is likely heavily blurred, severely
overexposed, or taken from a very different viewpoint.

---

### `connectivity_graph.png` — Image Graph

**What it shows**

A spring-layout graph where:
- Each node is one image (labeled by filename stem).
- Edges connect verified image pairs.
- **Edge thickness**: scaled from 1× to 5× based on inlier count relative to
  the dataset maximum (thicker = more inliers).
- **Edge color**: `YlOrRd` colormap — yellow = few inliers, red = many inliers.
- **Blue nodes**: connected (have at least one verified pair).
- **Red nodes**: isolated (no verified pairs with any other image) — will not
  register.

**Healthy result**

All or nearly all nodes are blue and connected in one large component.  Edges
are thick and red/orange, indicating strong pairwise geometry across the
dataset.  The graph looks like a single cluster, possibly with a ring topology
for a 360° capture.

**Warning patterns**

| Pattern | Meaning |
|---------|---------|
| One or more **red** (isolated) nodes | Those images match nothing; they will be skipped entirely |
| Two or more **disconnected subgraphs** | Split reconstruction — missing coverage between groups |
| A single **thin bridge** between two dense clusters | One weak pair (low inliers) connecting two groups; if RANSAC rejects it, the reconstruction splits |
| **Star topology** (all edges from one node) | One central image connects everything; if it fails to register, the whole reconstruction fails |

A disconnected subgraph is not always fatal — the pipeline keeps the largest
component — but you will lose the images in smaller subgraphs.

---

### `epipolar_<A>_<B>.png` — Epipolar Geometry

**What it shows**

Two side-by-side images.  Up to 10 inlier keypoint pairs are sampled; each
pair is given a unique color from the `tab10` palette.

For each pair:
- A **colored dot** is plotted at the keypoint location in each image.
- An **epipolar line** is drawn in each image — the constraint that the
  corresponding point in the other image must lie on.

The epipolar line in image 2 for a point `p1` in image 1 is:

```
l₂ = F · p1
```

and the reverse:

```
l₁ = Fᵀ · p2
```

**How to verify the geometry**

This figure is only generated when the fundamental matrix `F` is stored in the
verification data.  A correct `F` means every blue dot in image 2 should lie
*exactly on* its same-colored epipolar line.

| Observation | Meaning |
|-------------|---------|
| Dots lie precisely on epipolar lines | Excellent geometry; `F` is well-estimated |
| Dots are within 1–3 pixels of their lines | Normal; within RANSAC threshold |
| Dots are 5–15 pixels from their lines | Poor geometry; `F` is noisy, possibly from a near-degenerate configuration |
| Epipolar lines are all roughly parallel | Camera motion was approximately translational (no rotation), or a pure-rotation pair — triangulation will fail |
| Lines converge to a point inside the image | That point is the epipole (one camera center projected into the other); this is geometrically correct if it looks right |
| Lines converge to a point far outside the image | Nearly pure rotation (very small baseline); expect poor triangulation for this pair |

**What to do if lines are far from dots**

This usually means the fundamental matrix was estimated from a noisy or small
inlier set.  Try lowering `--ransac_thr` (e.g. 0.5 px) for stricter
verification, or increase `--min_inliers` to require more evidence.

---

## Chapter 3 — Reconstruction Step Outputs

Directory: `03_reconstruction/`

---

### `step_<NNN>_<tag>.png` — Per-Step Registration Snapshots

A new figure is saved every time a camera is registered.  The filename
encodes the step number (`001`, `002`, …) and whether this is the seed pair
(`seed_pair`) or an incremental registration (`camera_registered`).

**The figure has three panels:**

---

#### Panel 1 — Camera Map (top-down, XZ plane)

Dark background.  Axes represent the world X (right) and Z (forward) directions
viewed from above.  The Y axis (up) is collapsed.

**What is shown:**
- **Colored dots** — one per registered camera.  Color follows the `plasma`
  colormap from early (dark blue/purple, step 1) to late (bright yellow, last
  step).  This makes the registration **order** immediately visible.
- **Numbered labels** — image index annotated next to each dot.
- **Arrows** — short black arrows showing each camera's approximate forward
  (optical axis) direction in the XZ plane.
- **Yellow border** — the camera registered *at this step* has a larger dot
  with a yellow outline.
- **White dots (alpha=0.15)** — the current 3-D point cloud projected onto XZ,
  giving a ghost outline of the scene footprint.

**How to spot drift**

Camera drift is the most important thing to check in these snapshots.
Compare the last few frames in the sequence:
- In a healthy reconstruction cameras follow a smooth, predictable path —
  arc, straight line, or ring matching the physical capture trajectory.
- If a camera's position suddenly jumps to an implausible location (the wrong
  side of the scene, or far away from all others), the PnP registration for
  that camera was incorrect.  Drift here compounds: the next camera registered
  may also be wrong because it links to the bad camera.

**Healthy camera distributions**

| Capture style | Expected pattern |
|--------------|-----------------|
| 360° orbit | Ring of dots with evenly spaced arrows pointing inward |
| Linear walkthrough | Line or shallow arc |
| Object-centric turntable | Nearly concentric ring around a central cluster |
| Grid survey (aerial) | Regular grid of dots with arrows all pointing the same way |

**Degenerate distributions**

| Pattern | Problem |
|---------|---------|
| All cameras on a line, facing the same direction | Potential planar degeneracy; may not triangulate well |
| One camera far from all others | That camera likely failed PnP; check its reprojection errors |
| Cameras stop being added even though unregisetred images remain | No more images could be registered — likely disconnected coverage or a bad seed pair |

---

#### Panel 2 — Point Cloud Growth (XY side view)

Dark background.  Axes represent world X (right) and Y (up).

- **Steelblue dots (s=1)** — points that existed before this registration step.
- **Lime green dots (s=8)** — points added by this step's triangulation, labeled
  `+N new`.
- The total point count is shown in the top-left corner.

**What to look for**

- **Healthy** — lime cloud grows consistently each step, expanding into new
  parts of the scene.  The total count increases smoothly.
- **Few new points** — this camera triangulated almost nothing.  Possible
  causes: small baseline to registered cameras, poor overlap, or few matches.
  This is not immediately fatal but reduces reconstruction density.
- **Many filtered points** — if total points *decrease* from one step to the
  next, outlier filtering removed more points than were added.  A large single-
  step drop often coincides with a bundle adjustment round that found many
  high-error points.
- **Degenerate cloud shape** — if the point cloud looks like a flat plane or
  collapses along one axis, the scene may be planar or the camera baseline is
  too small.

---

#### Panel 3 — Registration Progress Bars

Two overlapping charts:
- **Steelblue bars** — total 3-D point count at each registration step.
- **Lime bars** — number of new points added at each step.
- **Orange dashed line** (right y-axis) — cumulative cameras registered.

**How to read it**

- Lime bars should be consistently non-zero across steps; this means every
  new camera contributes new geometry.
- A lime bar that is close to zero means a particular camera added almost
  no new points.  That camera was registered (it had enough 2D-3D
  correspondences via existing points) but did not extend the scene.
- Steelblue bars that *dip* at a step indicate bundle adjustment removed a
  large number of outlier points.  A small dip is healthy; a large dip
  (more than ~20 % of the total) suggests the reconstruction had accumulated
  significant error or a bad camera registration slipped through.
- The orange line should increase monotonically and at a roughly steady rate.
  A plateau means multiple images could not be registered in a row.

---

### `bundle_adjustment_convergence.png` — BA Convergence

**What it shows**

Two panels covering every bundle adjustment round that ran during the
pipeline.

**Left panel — Reprojection RMSE**

- **Tomato dashed line** (`o--`) — reprojection RMSE *before* BA for this round.
- **Steelblue solid line** (`o-`) — reprojection RMSE *after* BA.
- **Blue fill** between the two lines — the improvement from each round.
- **Gray annotations** (`C=N`) — number of registered cameras at the time of
  each round.

The x-axis is the BA round number (1, 2, 3, …).  The y-axis is RMSE in pixels.

**Right panel — Point count before/after BA**

Grouped bar chart: tomato bars = points *before* each round, steelblue bars =
points *after* (i.e., after outlier removal).

---

**How to read the convergence curve**

*Ideal behavior:*
1. RMSE starts high on the first round (initial seed pair), then drops sharply.
2. Subsequent rounds show progressively smaller improvements — the curve
   flattens and approaches a stable value.
3. Before-BA and after-BA lines converge as the reconstruction matures.

*Acceptable RMSE thresholds (final after-BA value):*

| RMSE | Assessment |
|------|-----------|
| < 0.5 px | Excellent — near-perfect geometry |
| 0.5 – 1.0 px | Good — suitable for most applications |
| 1.0 – 2.0 px | Acceptable — reconstruction is usable but noisy |
| 2.0 – 4.0 px | Poor — significant geometric error; results may drift |
| > 4.0 px | Critical — reconstruction is likely unreliable |

*Warning signs:*

| Pattern | Diagnosis |
|---------|-----------|
| RMSE *increases* after a BA round | A bad camera was registered just before that round and BA locked it in; check step snapshots |
| RMSE starts high and never converges below 2 px | Poor initialization (bad seed pair) or too few images with good overlap |
| Large spikes in the before-BA line | Individual camera registrations are introducing high-error observations |
| After-BA line higher than before-BA on the same round | Numerical issue in optimization (rare); try adjusting `--max_reproj_error` |
| Point count drops to near-zero after a BA round | Overly aggressive outlier filtering; reduce `--max_reproj_error` |

The camera count annotations on the before-BA dots let you correlate RMSE
spikes with specific registration events — if RMSE spikes when `C=12`, it
was the 12th camera that caused the problem.

---

### `point_lifecycle.png` — Point Lifecycle

**What it shows**

Two panels about the 3-D points that survived to the final reconstruction.

**Left panel — Observation coverage histogram**

- X-axis: number of cameras that observe each 3-D point.
- Y-axis: number of 3-D points in each bin.
- Red dashed line: mean observations per point.

A 3-D point with more observations is more robustly triangulated and has a
more reliable position estimate.

*Healthy distribution:*
- Peak at 3–8 observations, long tail extending to higher values.
- Mean observation count ≥ 3.

*Warning signs:*

| Pattern | Meaning |
|---------|---------|
| Almost all points have exactly 2 observations | Points are weakly constrained; the reconstruction is just barely triangulated. Adding more images or overlap would help |
| Bimodal distribution with many at 2 and some at 10+ | Two-tier quality: a core set of well-observed points and many marginal ones |
| Mean < 2.5 | Very sparse coverage; reconstruction is likely fragile |

**Right panel — Error vs. observation coverage scatter**

- X-axis: number of observations per point.
- Y-axis: mean reprojection error for that point (in pixels).
- Color: `coolwarm` — blue = low error, red = high error.
- Reference lines: green = 1 px, orange = 2 px, red = 4 px.

*What to look for:*
- The cloud should be mostly below the green 1 px line.
- Points with few observations (leftmost column) should not systematically
  have higher error than well-observed points.  If they do, the triangulation
  angle was very shallow (small baseline) for those points.
- If a horizontal band of red points exists at a fixed error value independent
  of observation count, there may be a systematic calibration error (wrong
  focal length) or lens distortion not being modeled.
- Most red dots near x=2 means weakly-constrained points are high-error —
  these are candidates for stricter outlier filtering.

---

### `camera_poses_final.png` — Final Camera Poses (3-D)

**What it shows**

A 3-D scatter plot of all registered camera centers, the point cloud, and a
trajectory line connecting cameras in registration order.

- **Colored dots** — cameras, colored by registration order using the `plasma`
  colormap (dark blue/purple = registered first, bright yellow = registered
  last).
- **Gold-bordered dots** — the seed pair cameras (the two images used to
  initialise the reconstruction).
- **RGB axis stubs** per camera — three short line segments showing the camera's
  local X (red), Y (green), and Z (blue) axes in world space.  The blue (Z)
  stub points in the optical axis direction.
- **White dashed line** — trajectory connecting cameras in registration order.
- **Gray point cloud** (subsampled to max 5,000 points) — the scene geometry
  for context.

**What to look for**

*Healthy result:*
The camera centers form a smooth, geometrically sensible path that matches
the physical capture trajectory.  Axis stubs are consistently oriented (e.g.,
all pointing inward for a 360° orbit).  The trajectory line does not cross
itself unexpectedly.

*Diagnosing problems:*

| Observation | Diagnosis |
|-------------|-----------|
| One camera far from the cluster | That camera's PnP registration failed or found a bad solution; its reprojection errors will be high |
| Trajectory line doubles back unexpectedly | Registration order does not match capture order, or overlap is too low to find the correct registration sequence |
| All Z-axis stubs pointing in the same direction | Near-planar scene or one-sided capture; 3-D reconstruction will lack depth resolution |
| Seed pair (gold borders) very close together | Small baseline seed; triangulation uncertainty will be high at the start |
| Cameras clustered in two groups facing each other | Object captured from two opposing sides but missing intermediate views |

---

### `reprojection_errors_<image>.png` — Per-Image Reprojection Arrows

**What it shows**

The source image overlaid with arrows for each 3-D point observed in that
image.  Each arrow runs from the **observed** keypoint location (tail) to the
**projected** 3-D point location (head).

**Arrow colors:**

| Color | Error range | Assessment |
|-------|-------------|-----------|
| **Lime green** | < 1 px | Excellent — near-perfect agreement |
| **Yellow** | 1 – 2 px | Acceptable |
| **Red** | > 2 px | Poor — this observation is an outlier candidate |

The title shows the count of each category: `✅<1px:N  🟡1-2px:N  🔴>2px:N`

**How to read arrow length and direction**

- A **short green arrow** means the 3-D point reprojects almost exactly to
  where the keypoint was detected — the geometry is consistent.
- A **long red arrow** means the 3-D point projects far from the observation.
  A few scattered long arrows are normal; a large cluster of long red arrows
  in one region of the image is a red flag.

**Diagnostic patterns:**

| Pattern | Meaning |
|---------|---------|
| Most arrows green, few scattered yellow | Healthy result |
| Red arrows clustered in one corner | Possible local distortion (lens vignetting, strong distortion at image edges) not fully corrected by radial distortion model |
| All arrows point in the same direction | Systematic offset — possibly wrong principal point (`cx`, `cy`) estimate |
| Red arrows all in the background, green in foreground | Background points are poorly constrained (seen from fewer cameras or extreme angles) |
| Red arrows forming a circular pattern around the image center | Radial distortion not modeled — try removing `--no_refine_intrinsics` |
| Almost all arrows are red | This camera's registration failed; it may have registered to the wrong place |

---

## Chapter 4 — Point Cloud Outputs

Directory: `04_pointcloud/`

---

### `pointcloud_6views.png` — Six Orthographic Views

**What it shows**

A 2×3 grid of 3-D scatter plots, each showing the colorized sparse point cloud
from a standard orthographic direction.  Points are rendered at s=0.5 with
α=0.7 and are colored using the RGB values from the final PLY colorization
(or gray if colorization was not run).

Up to 20,000 points are subsampled for rendering speed; if the cloud has
more points, a random subset is drawn.

| Position | View name | Camera | What it shows |
|----------|-----------|--------|---------------|
| Top-left | Front | Horizontal, from front | Width vs. height |
| Top-center | Back | Horizontal, from rear | Shows back side |
| Top-right | Left | Horizontal, from left side | Depth vs. height |
| Bottom-left | Right | Horizontal, from right side | — |
| Bottom-center | Top | From directly above | Width vs. depth footprint |
| Bottom-right | Bottom | From directly below | Underside |

The figure title shows total point count and the 3-D bounding box:
`X[min, max] Y[min, max] Z[min, max]`.

**How to assess completeness**

- Compare across all six views.  Regions that appear **hollow or missing**
  from two or more views consistently indicate a gap in the reconstruction.
- The **top view** is most useful for understanding scene footprint and coverage.
- Look for **floating clusters** of points that are far from the main cloud —
  these are likely outliers that survived filtering (triangulated at a bad
  position due to wrong matches).
- **Color correctness**: if colorization ran, the colors should match the real
  scene.  Uniformly gray points mean colorization failed or `--no_filter` was
  used without a separate colorization pass.

**Assessing point density**

- A dense, solid-looking cloud (even at s=0.5) for a small scene is healthy.
- A very sparse cloud with visible individual dots per surface suggests either
  low `--n_features`, poor overlap, or aggressive outlier filtering.
- Flat surfaces (walls, floors) should appear as thin sheets in the side views,
  not thick blobs.  Thick blobs indicate high 3-D position noise.

---

### `pointcloud_turntable.gif` — 360° Animation (optional)

Generated only when `--viz-save-video` is passed.  Shows 60 frames rotating
the point cloud 360° around the vertical axis at a 20° elevation angle.
Useful for presenting the reconstruction, and for spotting outlier clusters
and missing regions that may not be obvious in static views.

---

### `reconstruction_growth.gif` — Pipeline Animation (optional)

Generated only when `--viz-save-video` is passed.  Concatenates every
`step_*.png` snapshot from `03_reconstruction/` into a single animated GIF
(500 ms per frame).  Lets you watch the reconstruction grow step by step
and immediately spot the frame where drift or a bad registration begins.

---

## Chapter 5 — Reading the Pipeline Summary

File: `00_summary/pipeline_summary.png`

**This is the first figure to open.**

The dashboard is a 2×4 grid:

---

**Top-left — Stats table**

| Row | What it means |
|-----|---------------|
| Input | Total images passed to `--image_dir` |
| Resolution | Width × height of the first image |
| Cameras reg. | How many images were successfully registered |
| 3-D points | Sparse point count in the final PLY |
| Mean reproj. | Final RMSE reprojection error in pixels |
| BA rounds | How many bundle adjustment runs executed |
| Total time | Wall-clock time for the whole pipeline |

**Health thresholds:**

| Metric | Good | Warning | Poor |
|--------|------|---------|------|
| Registration rate (cameras / total) | > 80 % | 60–80 % | < 60 % |
| Mean reprojection error | < 1.0 px | 1.0–2.0 px | > 2.0 px |
| Points per camera | > 500 | 100–500 | < 100 |
| BA rounds | 2–10 | 10–20 | > 20 |

*Registration rate* is the most important single metric.  If fewer than 60 %
of your images registered, the reconstruction covers less than 60 % of the
scene — you are missing large parts of the model.

*Points per camera* (divide 3-D points by cameras registered) is a rough
quality indicator.  A well-connected, densely-overlapping dataset typically
achieves 1,000–5,000 points per camera with 8,000 features per image.

*BA rounds > 20* is unusual and may indicate poor initialization or that the
pipeline is struggling to converge; consider checking the BA convergence
figure for spiking RMSE.

---

**Top-center — Mini match matrix**

A compact version of the full `match_matrix.png`.  If you see mostly white
(empty) rows, that tells you at a glance that matching failed for large
portions of the dataset.

---

**Top-right — BA convergence curve**

A compact version of the full `bundle_adjustment_convergence.png`.  The final
after-BA RMSE (rightmost steelblue dot) is the number to focus on.

---

**Bottom row — Top-down point cloud**

Full-width top-down view (XZ plane) of the colorized point cloud with camera
positions overlaid as yellow dots.  At a glance this shows:
- Whether cameras encircle the scene or are one-sided.
- Whether the point cloud fills the region bounded by cameras.
- Whether any cameras are anomalously far from the main cluster.

---

## Chapter 6 — Diagnostic Decision Tree

Use this tree when you suspect a problem.  Always start at the top.

```
MY RECONSTRUCTION LOOKS WRONG
│
├── Step 1: Open 00_summary/pipeline_summary.png
│   │
│   ├── Registration rate < 60%?
│   │   └──► Go to Step 2 (matching)
│   │
│   ├── Mean reprojection error > 2.0 px?
│   │   └──► Go to Step 4 (BA convergence)
│   │
│   └── Both look OK but output PLY has wrong shape/scale?
│       └──► Go to Step 5 (camera poses)
│
├── Step 2: Open 02_matching/match_matrix.png
│   │
│   ├── Dark rows / columns? (image has almost no matches)
│   │   ├──► Open 01_features/feature_statistics.png
│   │   │    Is that image in the "red bar" list (<100 kps)?
│   │   │    YES → image quality issue (blur, exposure, texture)
│   │   │          Reshoot or remove that image.
│   │   │    NO  → Features exist but nothing matches.
│   │   │          Try increasing --n_features or reducing --ratio.
│   │   └──► Open 01_features/features_<that_image>.png
│   │        Are features on the object or only on background?
│   │        BACKGROUND only → crop/mask images or add texture to subject.
│   │
│   └── Two disconnected blocks of matched images?
│       └──► Coverage gap between image groups.
│            Reshoot missing transition views.
│            Alternatively: try --match_strategy vocab_tree
│            to find non-sequential image pairs.
│
├── Step 3: Open 02_matching/connectivity_graph.png
│   │
│   ├── Red (isolated) nodes?
│   │   └──► Those images will never register. See Step 2 diagnosis above.
│   │
│   └── Thin single bridge between two clusters?
│       └──► That pair is your only connection between groups.
│            Open matches_<A>_<B>.png for that pair.
│            If inlier ratio < 30%, reshoot with more overlap.
│
├── Step 4: Open 03_reconstruction/bundle_adjustment_convergence.png
│   │
│   ├── RMSE never decreases below 2 px?
│   │   ├──► Poor seed pair: the two images used to initialize had bad geometry.
│   │   │    Try changing --ba_interval or add more overlapping images.
│   │   └──► Wrong focal length: if your camera's EXIF is missing or wrong,
│   │        estimate_intrinsics() will guess. Try specifying K manually if
│   │        you know the focal length.
│   │
│   ├── RMSE spikes at a specific BA round?
│   │   └──► Open step_XXX_camera_registered.png for the step where the spike
│   │        begins. Check Panel 1 (camera map) for a misplaced camera.
│   │        Check reprojection_errors_<that_image>.png for high red counts.
│   │
│   └── RMSE converges but final value is 1.0-2.0 px (acceptable range)?
│       └──► Reconstruction is usable. For better accuracy: add more images,
│            use a calibrated camera, or enable intrinsics refinement
│            (remove --no_refine_intrinsics).
│
├── Step 5: Open 03_reconstruction/camera_poses_final.png
│   │
│   ├── One camera far from all others?
│   │   └──► Open reprojection_errors_<that_image>.png.
│   │        Almost all red? → That camera's PnP failed.
│   │        Remove that image or increase --min_inliers.
│   │
│   ├── Cameras don't form the expected trajectory shape?
│   │   └──► Check step snapshots in order:
│   │        which step_XXX did the path start drifting?
│   │        That registration event is the root cause.
│   │
│   └── Cameras look right but point cloud shape is wrong?
│       └──► Check 04_pointcloud/pointcloud_6views.png.
│            Floating clusters → outlier filtering needed
│            (check --max_reproj_error, maybe lower it).
│            Missing regions → coverage gap (go to Step 2/3).
│
└── Step 6: Open 01_features/feature_statistics.png
    │
    └── Any images < 100 keypoints (red bars)?
        └──► Those images are the source of almost every problem.
             Fix the image quality or remove them from the dataset.
```

---

## Quick Reference: All Output Files

| File | When generated | Key metric to read |
|------|---------------|-------------------|
| `00_summary/pipeline_summary.png` | Always | Registration rate, mean RMSE |
| `01_features/features_<stem>.png` | Always (sampled images) | Keypoint count, spread |
| `01_features/density_<stem>.png` | Always (sampled images) | Coverage gaps, hotspots |
| `01_features/feature_statistics.png` | Always | Images < 100 kps |
| `02_matching/matches_<A>_<B>.png` | Always (sampled pairs) | Inlier ratio |
| `02_matching/match_matrix.png` | Always | Dark rows = unregisterable images |
| `02_matching/connectivity_graph.png` | Always | Isolated nodes, thin bridges |
| `02_matching/epipolar_<A>_<B>.png` | When F matrix is available | Points on lines? |
| `03_reconstruction/step_XXX_*.png` | Every registration event | Camera drift, new point count |
| `03_reconstruction/bundle_adjustment_convergence.png` | After reconstruction | Final RMSE, spikes |
| `03_reconstruction/point_lifecycle.png` | After reconstruction | Mean observations, error scatter |
| `03_reconstruction/camera_poses_final.png` | After reconstruction | Outlier cameras, trajectory shape |
| `03_reconstruction/reprojection_errors_<stem>.png` | After reconstruction (sampled) | Red arrow clusters |
| `04_pointcloud/pointcloud_6views.png` | After export | Completeness, outlier clusters |
| `04_pointcloud/pointcloud_turntable.gif` | Only with --viz-save-video | — |
| `reconstruction_growth.gif` | Only with --viz-save-video | Step where drift begins |
