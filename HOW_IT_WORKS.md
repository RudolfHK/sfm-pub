# How It Works: Turning Photos Into 3D

*A complete, human-readable guide to the Structure from Motion pipeline in this repository.*

---

## Introduction — "Turning Photos Into 3D"

Close one eye and hold a finger in front of your face. Now close the other eye and open the first. Your finger seems to jump sideways, even though it didn't move. That apparent shift — *parallax* — is how your brain estimates depth. Two eyes, slightly apart, see the world from fractionally different angles. Your brain triangulates those two views into a single 3D percept without you thinking about it at all.

Photogrammetry does the same thing with photographs. Give it two or more pictures of an object taken from different positions, and it can measure the parallax between matching points across those pictures, and from that parallax reconstruct the geometry of the real scene. Every time you see a surveyor's drone produce a 3D map of a building site, or a scientist scanning a fossil for digital preservation, photogrammetry is at work.

**Structure from Motion (SfM)** extends photogrammetry in an important way: it doesn't need you to know *where* the cameras were when you took the photos. It figures that out too — entirely from the images themselves. The "structure" is the 3D point cloud of the scene; the "motion" is the trajectory of the camera as you walked around taking pictures. SfM recovers both, simultaneously, from a bag of photos taken in any order with any camera.

The pipeline in this repository takes a folder of photographs and produces a colored 3D point cloud saved as a `.ply` file that you can open in MeshLab, CloudCompare, or Blender. To do that, it runs six major stages:

> **Feature extraction** → **Feature matching** → **Geometric verification** → **Incremental reconstruction** → **Bundle adjustment** → **Point cloud export**

Each stage builds on the last, refines the estimates of the previous ones, and together they produce something that feels almost magical: depth from flat photographs. Let's take it apart.

---

## Chapter 1 — How Cameras See the World

### The Camera as a Projector Running Backwards

Imagine shining a flashlight at a wall in a dark room. The flashlight casts a cone of light; objects in that cone cast shadows on the wall. A camera works exactly like this flashlight — but in reverse. Instead of light going *out* from a point and hitting a wall, light from the world comes *in* through a tiny hole (the aperture) and hits a sensor (the wall). Every 3D point in the world gets squashed down to a 2D pixel on the sensor.

This projection loses one dimension — depth. A pixel at position (300, 200) in a photo tells you a ray went from that sensor location through the lens into the world, but not *how far down that ray* the object was. That lost depth is exactly what SfM works to recover.

### Camera Intrinsics — The Camera's Inner Geometry

Every camera has a personality described by its **intrinsic parameters**:

- **Focal length (f)**: how strongly the lens bends light. A longer focal length means a narrower field of view (telephoto); shorter means wider. In pixel units, focal length is the distance between the lens and sensor, measured in pixels. Typical smartphones: 1500–4000 px. A 50mm lens on a full-frame sensor: roughly 3600 px.

- **Principal point (cx, cy)**: ideally the dead center of the image, where the optical axis pierces the sensor. In practice it's close to center but rarely exact.

Together these four numbers form the **camera matrix K**, a 3×3 table that converts 3D camera coordinates into 2D pixel coordinates. Think of K as the camera's "ruler" — it converts "meters in front of me" into "pixels on my sensor."

```
        ┌         ┐
    K = │ f  0  cx│
        │ 0  f  cy│
        │ 0  0   1│
        └         ┘
```

> 🔍 **In the code:** `sfm/utils.py` → `estimate_intrinsics()` builds K from the image shape, setting `f = max(width, height)` (a reasonable guess for a "normal" lens). If the image has EXIF metadata with a `FocalLengthIn35mmFilm` tag, `read_exif_focal_px()` converts that to pixels for a better starting estimate.

### Lens Distortion — Why Straight Lines Bow

Real lenses aren't perfect. A wide-angle lens bends light more at the edges than the center, making straight lines in the world bow outward (barrel distortion) or inward (pincushion distortion). This is **radial distortion**, described by two coefficients k1 and k2.

```
      No distortion:         Barrel distortion (k1 < 0):
      ┌─────────────┐         ╭─────────────╮
      │             │         │             │
      │             │   →     │             │
      │             │         │             │
      └─────────────┘         ╰─────────────╯
```

The distortion formula says: the further a point is from the image center, the more it gets pushed outward (or inward). Mathematically: the distorted position is the ideal position multiplied by `(1 + k1·r² + k2·r⁴)`, where r is the distance from the image center. Tiny coefficients, large visual effect.

> ⚠️ **Common misconception:** Ignoring distortion doesn't just make your output slightly blurry — it actively breaks the geometry. A camera matrix K assumes straight rays; if the real rays are bent, the math that recovers 3D structure will be wrong in ways that compound across the reconstruction.

> 🔍 **In the code:** `sfm/bundle_adjustment.py` → `_project_distorted()` applies the k1/k2 Brown-Conrady radial distortion model during bundle adjustment, allowing the pipeline to simultaneously refine the distortion coefficients alongside camera poses and 3D points.

### Camera Extrinsics — Where the Camera Is in the World

Beyond the camera's internal optics, we need to know where the camera sat in 3D space and which way it was pointing — its **extrinsic parameters**: a rotation matrix R (3×3) and a translation vector t (3×1). Together they describe the rigid transformation from world coordinates to camera coordinates.

Think of it this way: R is the camera's orientation (did you tilt it? spin it?), and t is its position (where were you standing?). Knowing R and t for every photo lets you understand how the views relate to each other — which is the foundation of triangulation.

> 💡 **Key insight:** SfM starts knowing neither K nor R nor t for any of the photos in your pile. It figures them all out from scratch using only pixel values. That's what makes it remarkable.

---

## Chapter 2 — Finding Interesting Points (Feature Extraction)

### Why We Can't Just Compare Every Pixel

Comparing images pixel by pixel fails immediately. Take a photo of a mug from the left, then from the right. The mug's handle might appear at pixel (200, 300) in the first image and pixel (450, 180) in the second — completely different locations. The pixels have moved, the size has changed slightly, the lighting is different, and the angle has shifted. A naive comparison finds nothing.

What survives these changes? **Distinctive structure** — sharp corners, unique blob patterns, the junctions of edges. These are the places where a patch of image is unlike its neighbors in a way that persists across viewpoints. The strategy: find these places, describe them in a way that survives transformation, then match descriptions.

### Keypoints — The Landmarks of an Image

Imagine walking into a town square you've never seen. If someone asks you to describe a landmark, you'd naturally pick the ornate fountain, the clock tower, or the unusual statue — not "the section of gray pavement in the northwest corner." You'd choose things that are distinctive, visible from multiple directions, and memorable.

A **keypoint** is exactly this: a detected location in an image that is:
- **Distinctive**: stands out from its surroundings
- **Repeatable**: can be found again in another photo of the same place
- **Localizable**: its position can be measured precisely

The algorithm looks for "blobs" — regions that are bright on a dark background or vice versa — across multiple scales of the image. Scaling the image progressively down and looking for persistent structures is called **scale-space analysis**. Features that survive in multiple scales are the ones stable enough to track.

### Descriptors — A Feature's Fingerprint

Finding the keypoint's location isn't enough — we need to *describe* what it looks like so we can recognize it in another image. A **descriptor** is a compact numerical summary of the patch around the keypoint.

The SIFT descriptor (Scale-Invariant Feature Transform, David Lowe 2004) works like this:

1. **Take a 16×16 pixel patch** around the keypoint.
2. **Divide it into a 4×4 grid** of cells — 16 cells total.
3. **In each cell**, measure the direction and strength of edges (gradients) at each pixel.
4. **Build an 8-bin histogram** of those directions, weighted by edge strength.
5. **Concatenate** all 16 histograms → 16 × 8 = **128 numbers**.
6. **Normalize** the 128-number vector to make it robust to lighting changes.

The result is a 128-dimensional "fingerprint" for the patch. Two patches from different photos of the same real-world point will produce very similar 128-D vectors, even if the viewpoint has changed somewhat. Two patches from different things will produce very different vectors.

> 💡 **The orientation trick:** Before computing the descriptor, SIFT rotates the patch to align with the dominant gradient direction in the keypoint's neighborhood. This makes the descriptor rotation-invariant — a corner described from a tilted camera produces the same numbers as from an upright one.

```
Patch around keypoint:        Gradient directions:     8-bin histogram per cell:
┌───────────────────┐         ↖ ↑ ↑ ↗ ↑ ↖ ↑ ↑          ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ . . * * . . . .  │         ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑     →    │▓▓▓│ │▓  │ │▓▓ │ │ ▓▓│
│ . * * * * . . .  │         ↗ → → ↘ → ↗ → →          └───┘ └───┘ └───┘ └───┘
│ . . * * * * . .  │         → → ↘ ↓ ↘ → ↘ →          ┌───┐ ┌───┐ ┌───┐ ┌───┐
│ . . . * * . . .  │         ↘ ↓ ↓ ↓ ↓ ↓ ↘ ↘          │▓  │ │  ▓│ │▓  │ │  ▓│
└───────────────────┘         ↓ ↓ ↓ ← ↙ ↓ ↙ ↙          └───┘ └───┘ └───┘ └───┘
   (keypoint at center)       (simplified)                128 numbers total
```

> 🔍 **In the code:** `sfm/feature_extraction.py` → `FeatureExtractor` class. The default backend is OpenCV's CPU SIFT implementation (`cv2.SIFT_create`) with 8,000 features per image, 3 octave layers, and a contrast threshold of 0.04 to reject very weak keypoints. If a CUDA GPU is available, it tries kornia's GPU detector first, then OpenCV CUDA SURF. The output for each image is a dict containing `keypoints` (N×2 float32 pixel positions) and `descriptors` (N×128 float32 normalized vectors).

---

## Chapter 3 — Recognizing the Same Point Across Images (Feature Matching)

### The Core Problem: Finding Twins

We now have thousands of 128-D fingerprints for every image. The matching stage asks: for each fingerprint in image A, is there a "twin" fingerprint in image B that likely comes from the same real-world point?

This is a nearest-neighbor problem in 128-dimensional space. For each query descriptor in image A, find the descriptor in image B with the smallest Euclidean distance — the most similar fingerprint.

### Lowe's Ratio Test — "Only Match If You're Clearly the Best"

Naive nearest-neighbor matching produces many false matches — two unrelated patches that happen to look similar. David Lowe's insight: a *good* match should be unambiguously better than the second-best match. If the nearest and second-nearest descriptors have almost the same distance, the match is ambiguous and probably wrong.

The **ratio test**: only accept a match if the distance to the nearest neighbor divided by the distance to the second-nearest neighbor is below 0.75. In plain English: "the best match must be at least 25% better than the runner-up." This simple filter eliminates most false matches.

```
Query descriptor Q:

  Best match   (dist = 0.4)  ←  Accept if 0.4/0.6 < 0.75  ✓
  Second best  (dist = 0.6)

  Best match   (dist = 0.5)  ←  Reject if 0.5/0.52 > 0.75  ✗  (too ambiguous)
  Second best  (dist = 0.52)
```

### Cross-Check — "You Match Me, I Match You, or No Deal"

Even after the ratio test, some matches are one-sided: A thinks B is its best match, but B's own best match is some entirely different descriptor. These are suspicious. The **cross-check** filter discards any match that isn't mutual — both A→B and B→A must agree.

> 🔍 **In the code:** `sfm/feature_matching.py` → `FeatureMatcher.match_all()`. The code uses OpenCV's FLANN (Fast Library for Approximate Nearest Neighbors) with a KD-tree index (5 trees, 50 checks) for speed on high-dimensional descriptors. The ratio threshold is 0.75 and cross-check is enabled by default.

### Matching Strategies — Scaling Up

For N images, exhaustive matching checks every pair — N×(N-1)/2 pairs total. For 10 images that's 45 pairs; for 100 images it's 4,950. This quickly becomes a bottleneck.

The repository offers three strategies:

**Exhaustive** (`FeatureMatcher`): Check all pairs. Best quality, O(N²) time. Fine for small datasets (< ~50 images).

**Sequential** (`SequentialMatcher`): Only match each image to the next W images in sequence. Ideal for video or images captured in a walking path. O(N × W) — vastly faster.

**Vocabulary Tree** (`VocabTreeMatcher`): Build a compact visual vocabulary by clustering all descriptors with k-means into N_words "visual words." Encode each image as a TF-IDF histogram (images that use rare visual words share more meaningful structure). Compute cosine similarity between all image histograms, retrieve only the top-K most similar pairs, and run FLANN only on those. O(N × top_K) FLANN runs instead of O(N²).

> 💡 **Why TF-IDF for images?** The same trick used in search engines applies here. A "visual word" that appears in almost every image (like a blur or a common texture) carries little information. A visual word that appears in only 2 out of 100 images is highly discriminating. TF-IDF downweights common words and upweights rare ones, making the similarity score more meaningful.

---

## Chapter 4 — Geometric Verification and Pose Estimation

### Beyond Matching: Proving the Geometry Makes Sense

Raw feature matches — even after ratio test and cross-check — still contain errors. Some matches are just wrong. Before using them to compute camera geometry, we need to **verify** that the matches as a group are geometrically consistent.

### The Fundamental Matrix — The Universal Law of Two-Camera Geometry

Here's a remarkable fact: any two cameras looking at the same 3D scene must obey a geometric law. Given a point in image A, its corresponding point in image B must lie somewhere along a specific *line* in image B — called the **epipolar line**. This isn't optional; it's a consequence of projective geometry.

Think of it like this: you and a friend are both looking at the same sculpture from different positions. If you point at the sculpture's nose, your friend knows you're pointing along a ray that starts at your eyes and goes through their line of sight somewhere. They don't know exactly where along your ray the nose is, but they know the nose in *their* image must lie on the corresponding epipolar line.

```
Camera A            3D Point X           Camera B
    *────────────────────●────────────────────*
    │ Ray from A through X                    │
    │                   │             Epipolar│
    image A:            │             line in │
    ┌────────┐          │             image B:│
    │   *    │ point x  │          ┌──────────┤
    │ (matches) ────────┼──────────┤   /      │
    └────────┘          │          │  / x'    │
                        │          │ /        │
                                   └──────────┘
                                   (x' must lie on this line)
```

The **Fundamental Matrix F** (3×3) encodes this constraint. It's the relationship between the two images' coordinate systems. A correct match (x in A, x' in B) satisfies: `x'^T · F · x = 0`. False matches violate this equation.

The **Essential Matrix E** is the version of F that accounts for calibration — it encodes pure rotation and translation without the distortion of different focal lengths. E = K'^T · F · K. From E we can extract the actual relative rotation R and translation direction t between the two cameras.

> ⚠️ **Common misconception:** The translation from E is only a *direction* — not a magnitude. SfM can recover the shape of a scene but not its absolute scale. A scene reconstructed from photos of a toy car and a real car could be identical — you'd need a known distance (a ruler in the scene, GPS coordinates, etc.) to fix the scale.

### RANSAC — "Vote for the Most Popular Explanation, Ignore the Cheaters"

Imagine you're trying to pass a law, but some of the ballots have been stuffed by fraudsters. One approach: if you can find a large majority that all agree on the same law, you can be fairly confident that's the real vote, even without knowing exactly which ballots were fraudulent.

**RANSAC** (Random Sample and Consensus) works the same way:

1. **Pick a random tiny subset** of matches (8 points for the Fundamental Matrix).
2. **Fit a model** (compute F) from only those 8 points.
3. **Test the model** against all matches: how many are consistent with this F? (These are **inliers**.)
4. **Repeat** many times with different random subsets.
5. **Keep the model with the most inliers.**

Because fraudulent matches (outliers) are random, the chance that your random 8 all happen to be outliers is small. Eventually you'll draw a subset that's all genuine matches, and the F computed from it will be supported by the whole genuine population.

> 💡 **USAC-MAGSAC:** The code uses a modern variant called USAC_MAGSAC (introduced in OpenCV 4.5) instead of the classical RANSAC. MAGSAC doesn't require you to choose an inlier/outlier threshold in advance — it marginates over a range of thresholds and produces more robust estimates. It's meaningfully more accurate on real data.

### Two-Stage Verification

The code runs a **two-stage verification** for each matched pair:

**Stage 1 — F estimation (USAC_MAGSAC):** Use all matches to find the Fundamental Matrix. This tolerates different focal lengths and unknown calibration.

**Stage 2 — E estimation (RANSAC on F-inliers):** Using only the F-inliers, compute the Essential Matrix. This is more constrained (it requires the calibration K) but gives us R and t directly.

**Pose recovery:** Decompose E into the four possible (R, t) combinations. Use a **cheirality check** — project the matched 3D points and see which (R, t) puts them in front of both cameras. Only one of the four solutions passes this test.

> 🔍 **In the code:** `sfm/geometric_verification.py` → `GeometricVerifier.verify_pair()`. Note the pre-shuffling of matches before RANSAC — this works around a specific crash in some OpenCV 4.x builds on certain datasets. The result dict contains F, E, R, t, and the list of inlier match indices.

### Incremental Registration — How New Cameras Are Added

After the seed pair is initialized, new cameras are added one at a time using **PnP** (Perspective-n-Point):

> *"You know where some 3D points are in world coordinates. You can see them in a new photo. So where must the camera be?"*

Given N correspondences between 3D world points and 2D image pixels, PnP solves for the 6 degrees of freedom (rotation + translation) of the camera. The code uses **EPnP** (Efficient Perspective-n-Point), which needs only 6 correspondences to get a solution, plus RANSAC to handle outliers.

> 🔍 **In the code:** `sfm/reconstruction.py` → `IncrementalSfM._register_image()`. PnP is called with `cv2.solvePnPRansac()` using the pre-computed undistorted keypoints (distortion has been removed before calling PnP, so the camera matrix K can be applied cleanly).

---

## Chapter 5 — Building the 3D Points (Triangulation)

### Two Rays, One Point

Once you know where two cameras are and what they're looking at, triangulation is conceptually simple: each camera defines a ray from its center through the matched pixel. Where those two rays meet in 3D space is the location of the real-world point.

```
    Camera 1                Camera 2
        *                       *
        │╲                     /│
        │  ╲                 /  │
        │    ╲             /    │
        │      ╲         /      │
        │        ╲     /        │
        │          ╲ /          │
        │           ●           │
        │        (3D point)     │
```

In practice, rays from real (noisy) measurements never perfectly intersect. There's always a gap. The triangulation algorithm finds the **midpoint of the shortest segment** between the two rays — the 3D point that is closest to both. OpenCV's `cv2.triangulatePoints` computes this using the **DLT** (Direct Linear Transform) method.

### The Baseline Problem

The gap between the two camera positions is called the **baseline**. Triangulation quality depends critically on it:

```
  Wide baseline (good):             Narrow baseline (bad):
  
  *           *                       * *
   ╲         /                        ╲╲
    ╲       /                          ╲╲
     ╲     /                            ╲╲
      ╲   /   ← small angular           ╲╲← large angular
       ╲ /     uncertainty              ╲╲  uncertainty
        ●                                ●  (poorly localized)
```

When cameras are far apart (wide baseline), small errors in the pixel positions produce small errors in the 3D position. When cameras are very close (narrow baseline), the two rays are nearly parallel — tiny pointing errors translate into enormous depth uncertainty.

The code enforces a **minimum triangulation angle** of 1°. Pairs with smaller angles are rejected regardless of their reprojection error.

> ⚠️ **Common misconception:** "More overlap is always better." Too much overlap (cameras almost on top of each other) gives poor depth resolution. The sweet spot is typically 20–60° of overlap.

### Reprojection Error — "Project It Back and Check"

After triangulating a 3D point, we can validate it by **reprojecting** it back into both cameras and measuring how far the reprojected location is from the original detected keypoint. This distance in pixels is the **reprojection error**.

```
  1. Detect keypoint at (250, 180) in image A
  2. Triangulate → 3D point at (1.23, -0.45, 3.67) in world
  3. Project 3D point through camera A → (252, 181) in pixels
  4. Reprojection error = √((252-250)² + (181-180)²) = √5 ≈ 2.24 px
```

A well-triangulated point from good matches has a reprojection error of less than 2 pixels. The code uses a threshold of 4px for initial triangulation — loose enough to accept imperfect initial estimates — with BA tightening them later.

> 🔍 **In the code:** `sfm/reconstruction.py` → `_triangulate_dlt()` does the triangulation, and `_accept_point()` runs the quality checks: positive depth in both cameras, minimum triangulation angle ≥ 1°, and reprojection error ≤ `max_reproj_error` in both views.

---

## Chapter 6 — Polishing Everything at Once (Bundle Adjustment)

### The Inevitable Accumulation of Error

Every estimate so far is slightly wrong. The focal length is approximate. The RANSAC poses are noisy. The triangulated 3D points are slightly off. And these errors **accumulate** as you add more cameras — each new camera's pose is estimated from 3D points that were themselves triangulated from poses that themselves have error. The whole reconstruction slowly drifts.

Bundle Adjustment (BA) is the answer: a global optimization that simultaneously adjusts *every* camera pose *and* every 3D point until the whole system is as self-consistent as possible.

### The Tangled Net Analogy

Imagine a tangled fishing net lying on the floor. You know roughly what shape it should be, but every knot is slightly in the wrong place. Each person grabs a section of the net and pulls gently, adjusting it until their section lies flat. Everyone adjusts simultaneously, and the pulls propagate through the net. After many rounds of small adjustments, the net lies flat.

Bundle Adjustment is this collective adjustment. The "net" is the set of all camera parameters and 3D points. The "pulling flat" criterion is: minimize the total reprojection error — the sum over all observations of (distance between observed keypoint and reprojected 3D point)².

```
     Camera A              Camera B              Camera C
      pose R_A, t_A         pose R_B, t_B         pose R_C, t_C
         │    ╲              │    │              ╱    │
         │      ╲            │    │            ╱      │
         ↓        ↓          ↓    ↓          ↓        ↓
      observed  observed  obs   obs       obs      observed
       pixel     pixel    px    px        px        pixel
         ▲        ▲          ▲    ▲          ▲        ▲
         └────────┘          └────┘          └────────┘
              3D point P_1        3D point P_2
         (jointly adjust all cameras + all points to minimize total discrepancy)
```

### What's Being Minimized

The optimization minimizes the sum of squared distances between each observed 2D keypoint and the 2D projection of its corresponding 3D point through the corresponding camera. In words: "make every 3D point, when projected through every camera that can see it, land as close as possible to the original detected keypoint."

The code uses the **Brown-Conrady radial distortion model** in the projection: projected position = `f * (1 + k1·r² + k2·r⁴) · normalized_coordinates + principal_point`. So the optimization simultaneously refines focal length f, distortion k1 and k2, all camera rotations and translations, and all 3D point positions. That's potentially thousands of parameters adjusted together.

### Levenberg-Marquardt — Finding the Bottom of a Bumpy Valley

The reprojection error landscape is a high-dimensional, bumpy valley. We want to roll downhill to the lowest point. Pure gradient descent ("always go downhill") can be slow and get stuck. Newton's method ("jump to the predicted minimum") can overshoot on curved surfaces.

**Levenberg-Marquardt** is a clever hybrid: it uses Newton's method near the minimum (where the surface is well-approximated by a bowl) and gradient descent far from the minimum (where Newton might overshoot). It adaptively switches between them. The scipy TRF (Trust Region Reflective) solver used here is a closely related and highly robust variant.

### Robust Loss — "Don't Let One Bad Point Ruin Everything"

If even one gross outlier is still in the point set (a match that was wrong all along), its reprojection error might be 50 pixels — which, when squared, utterly dominates the optimization and pulls everything toward fitting that one bad point.

The solution: use a **Huber loss** instead of squared error. For errors below a threshold (2 pixels in this code), it behaves like squared error (sensitive to small errors). For large errors, it grows linearly instead of quadratically — dramatically reducing the influence of gross outliers.

```
  Squared loss: error² — explodes for outliers, dominates optimization
  Huber loss:
     ┌─────────────────────────────────────────────────
     │      *                        (linear growth)
     │     *
     │    *                    ●
     │   *               ●
     │  ● ●●            (quadratic near zero)
     └────────────────────────────────────────────────
     0   threshold      large outlier
```

### Sparse Structure — Making BA Fast

The Jacobian matrix of partial derivatives used in LM has a special structure: each residual depends only on *one* camera and *one* 3D point — not on all cameras and all points simultaneously. This gives the Jacobian a block-sparse structure that can be exploited to solve the linear system in O(cameras × features) time instead of O(total_parameters³). The code explicitly builds this sparse pattern using `scipy.sparse.lil_matrix`.

> 🔍 **In the code:** `sfm/bundle_adjustment.py` → `BundleAdjuster.adjust()`. Key details: the shared intrinsics [f, k1, k2] appear at positions 0–2 of the parameter vector (when `refine_intrinsics=True`), followed by camera poses (6 params each: Rodrigues rotation vector + translation vector), followed by 3D point coordinates (3 params each). The sparsity matrix `_build_sparsity_v2()` marks exactly which parameters affect which residuals. The divergence guard checks that the final RMSE is no more than 3× the initial RMSE — if BA somehow made things worse, it reverts.

> 🔍 **In the code:** `sfm/reconstruction.py` → `_run_ba()`. BA runs every `ba_interval` cameras (default: every 5) and again at the very end. After BA, the refined K and dist_coeffs are read back, and the undistorted keypoint cache is recomputed if distortion changed — so subsequent PnP calls use the updated distortion model.

---

## Chapter 7 — The Final Point Cloud

### What You Have After SfM

After reconstruction and final BA, you have:
- **Camera poses**: the estimated R and t for every registered image
- **3D point positions**: the world coordinates of all successfully triangulated points
- **Observations**: which 3D point was seen from which camera at which pixel

What you *don't* have (yet) is color, density, or a surface. Just a **sparse point cloud** — thousands of floating dots corresponding to wherever SIFT features were found.

### Why "Sparse"?

SIFT only finds features where there's distinctive texture — corners, edges, unique blobs. Uniform surfaces (walls, sky, water, skin) produce very few or no features. A room with white walls gives SfM a handful of points on door frames and light switches, and nothing at all on the walls themselves. This is the fundamental limitation of SfM: it produces detail proportional to texture richness.

```
  Dense (real-world surface):     Sparse SfM output:
  ████████████████████████       . .  .    .  .
  ████████████████████████       .   .  .    .
  ████████████████████████     .   .     .  .  .
  ████████████████████████       .  .  .    .
  (every point on surface)       (only distinctive regions)
```

### Coloring the Points

Each 3D point was observed from multiple cameras. To color it, the code samples the pixel color from each observation (using bilinear interpolation for sub-pixel accuracy), skips observations with high reprojection error, and averages the valid RGB samples.

> 🔍 **In the code:** `sfm/point_cloud.py` → `PointCloudExporter.colorize()`. Images are loaded lazily (only when a point needs color sampling from that view) and cached to avoid redundant I/O. The averaging uses BGR→RGB conversion since OpenCV natively uses BGR.

### Outlier Filtering

A final pass removes 3D points whose mean reprojection error is too high. The threshold is the minimum of `max_reproj_error` and `(median + 3σ)` across all point errors — an adaptive, statistically motivated cutoff.

### The PLY Format

PLY (Polygon File Format, or Stanford Triangle Format) is a simple binary format for 3D geometry. The code writes **binary little-endian PLY** — `float32` XYZ coordinates followed by `uint8` RGB color bytes for each vertex. This format is universally supported by MeshLab, CloudCompare, Blender, and open3d.

> 🔍 **In the code:** `sfm/point_cloud.py` → `PointCloudExporter.save_ply()`. It builds a NumPy structured array with the exact dtype layout matching the PLY spec, then writes it in one shot — much faster than a loop.

### From Sparse to Dense: Multi-View Stereo

SfM gives you sparse points and camera poses. **Multi-View Stereo (MVS)** goes further: it uses the known camera geometry to do dense matching — comparing entire rows of pixels between rectified stereo image pairs to get depth at *every* pixel, not just where SIFT found features.

The code includes an MVS stage using **StereoSGBM** (Semi-Global Block Matching):

1. **Stereo rectify** two cameras: warp both images so that corresponding points lie on the same horizontal scan line.
2. **Compute disparity**: for each pixel in image 1, find how many pixels to the right (or left) the matching pixel is in image 2. Disparity is inversely proportional to depth.
3. **Reproject**: convert the disparity map to a 3D point cloud using OpenCV's `reprojectImageTo3D`.
4. **Filter**: remove invalid disparities, points behind the camera, and distance outliers.
5. **Transform** from the rectified camera frame back to world coordinates.

> 🔍 **In the code:** `sfm/mvs.py` → `MVSDensifier.densify()`. Activate with `--dense` on the command line. The SGBM parameters (numDisparities=256, blockSize=5) are tuned for typical photography; you might need to adjust them for specific scenes.

---

## Chapter 8 — How All the Pieces Connect

### A Full Walkthrough: 30 Photos of a Garden Statue

Imagine you've walked around a stone garden statue and taken 30 overlapping photos from different angles, roughly every 12°. Here's what happens:

**Stage 1 — Images load, intrinsics estimated**

The pipeline reads the first image (say, 4000×3000 px). It tries to read EXIF metadata: your phone reports 26mm equivalent focal length. `read_exif_focal_px()` converts this to roughly 3600 pixels. The camera matrix K is built. dist_coeffs starts at zero — we'll refine them.

**Stage 2 — Feature extraction (per-image, independent)**

For each of the 30 images, `FeatureExtractor` converts to grayscale and runs SIFT. Each image yields perhaps 4,000–8,000 keypoints with their 128-D descriptors. Total: ~200,000 keypoints across all 30 images.

**Stage 3 — Matching (pairwise)**

With 30 images, exhaustive matching checks 30×29/2 = 435 pairs. For each pair, FLANN finds nearest-neighbor descriptor matches, the ratio test culls them, and cross-check keeps only mutual matches. Pairs with fewer than 15 matches are dropped. Perhaps 200 pairs survive with strong matches.

**Stage 4 — Geometric verification (pairwise)**

For each of the 200 surviving pairs, USAC_MAGSAC estimates F, culling non-epipolar matches. Then E is estimated from F-inliers, and pose (R, t) is extracted. Pairs with fewer than 15 verified inliers are dropped. Perhaps 150 pairs survive — each now tagged with a relative pose and a set of verified inlier match indices.

**Stage 5 — Seed pair selection**

The pair with the most inliers is selected as the starting point. Say it's images 0 and 1 (adjacent photos, lots of overlap). Camera 0 is placed at the origin (R=I, t=0) — this defines the coordinate system. Camera 1's R and t are taken from the verified pose.

**Stage 5 — Initial triangulation**

The inlier matches between cameras 0 and 1 are triangulated: each match produces a candidate 3D point. Points that fail depth tests or have excessive reprojection error are discarded. Say 180 of 220 matches survive → 180 initial 3D points.

**Stage 5 — Incremental registration (loop)**

Image 2 is picked next (most 2D-3D correspondences through its overlapping matches with cameras 0 and 1). PnP RANSAC uses the known 3D points to estimate camera 2's pose. New matches between camera 2 and cameras 0 and 1 that have no 3D point yet are triangulated → more 3D points.

This continues around the statue: cameras 3, 4, 5... each adding its pose and triangulating more points. After every 5 new cameras, BA runs.

**Bundle Adjustment (periodic)**

After cameras 0–5 are registered, BA runs. It adjusts all 6 poses and all current 3D points simultaneously. The focal length tightens slightly. k1 becomes slightly negative (a hint of barrel distortion from the phone lens). The RMSE drops from 1.2px to 0.7px. The now-improved poses make the next triangulations more accurate.

The loop continues: cameras 6–10 register and triangulate. BA runs again. And so on.

**Final BA + cleanup**

After all 30 cameras are registered, a final BA runs over the complete reconstruction. Outlier points (mean reprojection error > threshold) are removed. The dense point cloud of the statue is now geometrically consistent.

**Stage 6 — Export**

Each 3D point is colored by averaging its RGB observations (skipping high-error views). The binary PLY is written. You open it in MeshLab and see a colored point cloud of the statue — every visible surface documented in 3D from your 30 photos.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INPUTS: folder of JPG/PNG images                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Intrinsics estimation                                     │
│  • EXIF focal length OR heuristic f = max(W, H)                    │
│  • Initial K matrix, dist_coeffs = [0, 0, 0, 0]                   │
│  OUTPUT: K (3×3)                                                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Feature extraction  (per image, independent)             │
│  • SIFT: up to 8,000 keypoints per image                           │
│  • 128-D L2-normalized descriptors                                  │
│  OUTPUT: features dict: {img_idx → {keypoints, descriptors, ...}} │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Feature matching  (pairwise)                              │
│  • FLANN nearest-neighbor search in 128-D                           │
│  • Lowe ratio test (< 0.75) + cross-check                          │
│  • Strategy: exhaustive / sequential / vocab-tree                   │
│  OUTPUT: all_matches dict: {(i,j) → (M,2) match array}            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Geometric verification  (pairwise)                        │
│  • F estimation via USAC_MAGSAC → inlier mask                      │
│  • E estimation via RANSAC on F-inliers                             │
│  • Pose recovery: R, t from E (cheirality check)                   │
│  OUTPUT: verified dict: {(i,j) → {F, E, R, t, inlier_matches}}    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Incremental reconstruction                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Select seed pair (most inliers) → triangulate → cameras[0,1]│   │
│  └────────────────────────┬────────────────────────────────────┘   │
│                           │                                         │
│  ┌────────────────────────▼────────────────────────────────────┐   │
│  │  LOOP: while unregistered images remain:                    │   │
│  │    1. Pick image with most 2D-3D correspondences            │   │
│  │    2. PnP RANSAC → new camera pose                          │   │
│  │    3. Triangulate new 3D points                             │   │
│  │    4. If N_new_cameras % ba_interval == 0: run BA ──────┐  │   │
│  └────────────────────────┬────────────────────────────────│──┘   │
│                           │                 ┌──────────────┘       │
│  ┌────────────────────────▼────────────────▼───────────────────┐  │
│  │  Bundle Adjustment: jointly optimize                         │  │
│  │    • shared [f, k1, k2] (when refine_intrinsics=True)       │  │
│  │    • all camera [rvec, tvec]                                 │  │
│  │    • all 3D point [X, Y, Z]                                  │  │
│  │    → refined K, dist_coeffs fed back into reconstruction    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Final BA → outlier removal                                         │
│  OUTPUT: cameras dict, points_3d array, observations list          │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6a: Sparse point cloud export                                │
│  • Statistical outlier filtering (median + 3σ)                     │
│  • Colorize: average RGB from valid observations                    │
│  • Write binary little-endian PLY                                   │
│  OUTPUT: output.ply (sparse colored point cloud)                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ (if --dense)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6b: MVS densification (optional)                             │
│  • Select consecutive camera pairs                                  │
│  • Stereo rectify + StereoSGBM disparity                           │
│  • Reproject to 3D → transform to world frame                      │
│  OUTPUT: output_dense.ply (dense colored point cloud)              │
└─────────────────────────────────────────────────────────────────────┘
```

### The Feedback Loops

Notice that BA doesn't just run once at the end — it runs periodically throughout the incremental registration, and the results feed back:

- **Refined K**: updated focal length → undistorted keypoints are recomputed → subsequent PnP calls use better calibration
- **Refined dist_coeffs**: updated k1/k2 → future triangulations use better geometry
- **Refined poses**: updated camera positions → future triangulations produce more accurate 3D points

This tightening feedback loop is what separates SfM from a naive "chain of relative poses" — it prevents error from accumulating indefinitely.

---

## Chapter 9 — What Can Go Wrong and Why

### "Too Few Witnesses" — Insufficient Matches

**Symptom:** Very few pairs pass geometric verification. Cameras fail to register.

**Why:** If two images share fewer than ~15 geometrically verified matches, we can't reliably estimate their relative pose. This happens when images have too little overlap (photos taken too far apart), too much blur, or too different an exposure.

**Real-world analogy:** You're trying to identify a suspect, but only three witnesses can agree on any detail. That's not enough to convict.

**Fix:** Take more photos with 60–80% overlap. Ensure good exposure and focus.

---

### "Nothing to Grab Onto" — Textureless Surfaces

**Symptom:** Correct number of cameras registered, but very few 3D points. Reconstruction is hollow.

**Why:** SIFT needs distinctive texture to find keypoints. Smooth surfaces — white walls, calm water, blue sky, skin — are invisible to feature detectors.

**Real-world analogy:** Trying to find landmarks on a featureless salt flat. There are none.

**Fix:** Scatter distinctive reference objects (textured cards, pattern sheets) on smooth surfaces. Or use specialized methods designed for textureless objects.

---

### "A Mirror Shows a Different World" — Reflections and Transparency

**Symptom:** Weird ghostly points floating in space, or reconstruction failure near shiny surfaces.

**Why:** A reflection shows a virtual image of objects behind the camera. Its "position" in 3D is behind the mirror surface — not where the mirror physically is. Matching reflections across photos produces geometrically impossible correspondences.

**Real-world analogy:** Trying to navigate using a funhouse mirror as a window.

**Fix:** Avoid photographing shiny or reflective surfaces. Polarizing filters can help with windows and water.

---

### "Telephone" — Drift in Long Sequences

**Symptom:** The reconstruction curves or bends. A model of a complete 360° loop doesn't close — the start and end are displaced.

**Why:** Each camera pose is estimated from the previous one's 3D points. Small errors accumulate. After 100 cameras, the error has compounded significantly. Without "loop closure" — detecting when you've returned to a starting location — the drift is unchecked.

**Real-world analogy:** Drawing a map by walking and writing down each turn and distance. Small measurement errors compound over a long journey until your map is badly wrong.

**Fix:** Increase BA frequency. Ensure the vocabulary-tree matcher detects revisited locations. Consider loop-closure algorithms.

---

### "All in a Line" — Degenerate Camera Configurations

**Symptom:** Reconstruction fails or produces extreme outliers even with plenty of matches.

**Why:** If all cameras lie on a straight line, or all look at the scene from the same direction (pure forward motion), the Essential Matrix can't distinguish between several very different 3D configurations — the problem is mathematically underdetermined.

```
  Bad configuration:              Good configuration:
  ● ─── ● ─── ● ─── ●             ●       ●
  (all collinear, views            │ ╲   ╱ │
   barely differ)                  ●───●───●
                                  (spread around scene)
```

**Real-world analogy:** Trying to judge the depth of a distant object when you can only move left and right, never closer.

**Fix:** Circle around objects. Vary height and angle. Mix distant and close-up shots.

---

### "A Crooked Foundation" — Bad Initialization

**Symptom:** The reconstruction of later cameras is systematically distorted. BA doesn't help.

**Why:** If the seed pair has inaccurate relative pose (e.g., the epipolar geometry was poorly conditioned, or RANSAC found a degenerate solution), all subsequent triangulation and registration is built on flawed initial 3D points.

**Real-world analogy:** Building a skyscraper on a foundation poured at a 2° tilt. Every floor is slightly more off-level than the one below.

**Fix:** The code picks the seed pair with the most verified inliers, which usually gives the most stable pose. Increasing `min_inliers` raises the bar further. Manual seed-pair selection can help for tricky datasets.

---

### "The Net Is Too Tangled" — BA Divergence

**Symptom:** BA makes things worse. RMSE increases. The divergence guard reverts to the initial parameters.

**Why:** BA is a non-convex optimization. If the initial estimates are very poor (large reprojection errors), LM might converge to a local minimum worse than the starting point. This can also happen if too many outlier 3D points remain in the set, overwhelming the Huber loss.

**Real-world analogy:** The tangled fishing net analogy, but someone cut half the strings. Now pulling one section just tangled everything else more.

**Fix:** The code's divergence guard (revert if final RMSE > 3× initial RMSE) is the first safety net. Running outlier removal before BA, increasing `f_scale` for the Huber loss, or reducing `ba_interval` to run BA more frequently all help keep estimates close to a good basin.

---

### "Distortion is Lying to You" — Uncorrected Lens Distortion

**Symptom:** Systematic curved distortions in the reconstruction, especially near image borders. Lines that should be straight aren't. BA doesn't fully converge.

**Why:** If the true lens distortion is significant (wide-angle or fisheye lenses) but k1/k2 are set to zero, every pixel near the image border has a slightly wrong position. The F/E matrices are computed from these wrong positions, contaminating the initial poses.

**Fix:** The code starts with k1=k2=0 and lets BA refine them. For very wide-angle lenses (fisheye), pre-calibration with a checkerboard gives much better initial estimates.

---

## Chapter 10 — Glossary of Key Terms

**Baseline**
The physical distance between two camera positions. A wider baseline gives better depth resolution (triangulation angle) but reduces the number of shared visible points between the two views.

**Bundle Adjustment (BA)**
A global optimization that simultaneously refines all camera poses and all 3D point positions to minimize the total reprojection error across all observations. "Bundle" refers to the bundles of light rays connecting cameras to 3D points.

**Camera extrinsics**
The position (translation t) and orientation (rotation R) of a camera in 3D space. Extrinsics tell you *where* the camera is and *which way* it's pointing, as opposed to its internal optical properties.

**Camera intrinsics**
The internal optical properties of a camera: focal length, principal point (image center), and lens distortion coefficients. These describe how the camera maps 3D points to 2D pixels, independent of where the camera is in the world.

**Dense reconstruction / MVS**
Multi-View Stereo: a dense depth-estimation technique that extends SfM's sparse points to a full depth map at (nearly) every pixel, using the known camera poses from SfM. This repository implements MVS via StereoSGBM.

**Descriptor**
A compact numerical summary (here: 128 floating-point numbers) of the appearance of the image patch around a keypoint. Designed to be similar for the same real-world point viewed from different angles, and different for distinct points.

**Epipolar geometry**
The geometric relationship between two cameras viewing the same scene. Given a point in one image, its corresponding point in the other image must lie on a specific line (the epipolar line) — a constraint that allows geometric verification of matches.

**Essential Matrix (E)**
A 3×3 matrix encoding the relative rotation and (direction of) translation between two calibrated cameras. It encapsulates the epipolar constraint for cameras with known intrinsics K. From E you can extract R and t.

**Fundamental Matrix (F)**
A 3×3 matrix encoding the epipolar geometry between two uncalibrated cameras. Any correct match (x in image A, x' in image B) satisfies x'^T · F · x = 0. It's the uncalibrated version of the Essential Matrix.

**Homography**
A 3×3 transformation matrix that maps points from one image plane to another, valid when the scene is planar or when cameras rotate without translating. Not the same as epipolar geometry, which applies to 3D scenes viewed from different positions.

**Inlier**
A data point (match, observation) that is consistent with the geometric model being estimated. Inliers support the model; outliers contradict it. RANSAC explicitly identifies and separates inliers from outliers.

**Keypoint**
A detected location in an image that is distinctive, repeatable, and localizable — a corner, blob, or edge junction that can be found again in another photo of the same scene.

**Outlier**
A data point inconsistent with the geometric model — a wrong match, a mis-triangulated point, or a gross measurement error. Robust methods (RANSAC, Huber loss) limit the damage outliers can do.

**Pose**
The combination of rotation and translation describing where a camera is in 3D space and which way it's facing. A 6-degrees-of-freedom quantity.

**RANSAC (Random Sample Consensus)**
An algorithm for fitting a model (e.g., Fundamental Matrix) to data that contains outliers. It randomly samples the minimum number of points needed to fit the model, counts how many other points agree (inliers), and repeats until a good model is found.

**Reprojection error**
For a 3D point X observed as pixel x in camera C: project X through C's camera matrix to get a predicted pixel x̂, then measure the distance ‖x - x̂‖ in pixels. A small reprojection error means the 3D point is geometrically consistent with the observation.

**Scale ambiguity**
SfM can only recover the 3D shape of a scene up to an unknown overall scale. A reconstruction from photos of a toy and a full-size object could be identical. Absolute scale requires a known distance in the scene.

**Sparse point cloud**
The SfM output: a set of 3D points at locations where reliable feature correspondences existed across multiple views. Only textured, distinctive regions are represented — featureless areas have no points.

**Triangulation**
Computing the 3D location of a point from its 2D projections in two or more cameras with known poses. Geometrically: finding where the rays from each camera through the observed pixel meet in 3D space.

---

## Appendix — The Pipeline at a Glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     SfM PIPELINE AT A GLANCE                            │
├──────────┬───────────────────────┬───────────────────┬───────────────────┤
│  STAGE   │  WHAT IT DOES         │  INPUT → OUTPUT   │  KEY FAILURE RISK │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 1. Intro-│ Estimate camera focal │ Image files       │ Wrong focal gives │
│ intrinsics│ length from EXIF or  │ → K matrix        │ bad scale, slow BA│
│          │ image dimensions      │                   │ convergence       │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 2. Feature│ Detect distinctive   │ Images            │ Blur, low contrast│
│ extraction│ keypoints; compute   │ → {keypoints,     │ → few/no features │
│          │ 128-D SIFT descriptors│   descriptors}    │                   │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 3. Feature│ Find likely          │ Descriptor sets   │ Textureless scenes│
│  matching │ same-3D-point pairs  │ → raw match lists │ No overlap between│
│          │ via FLANN + ratio +   │                   │ adjacent images   │
│          │ cross-check          │                   │                   │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 4. Geo-  │ Compute F/E matrices  │ Raw matches       │ Too few matches;  │
│ metric   │ via USAC-MAGSAC;      │ → verified pairs  │ degenerate config │
│ verifica-│ extract R, t;         │   with inliers    │ (planar scene,    │
│ tion     │ reject bad matches    │                   │ all cameras same  │
│          │                       │                   │ direction)        │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 5. Incre-│ Initialize from seed  │ Verified pairs    │ Bad seed pair;    │
│ mental   │ pair; register each   │ + features        │ drift; reflections│
│ recon-   │ new camera via PnP;   │ → cameras dict    │ textureless areas │
│ struction│ triangulate new 3D    │   points_3d array │ give sparse cloud │
│          │ points; run BA every  │   observations    │                   │
│          │ N cameras             │                   │                   │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 6. Bundle│ Global optimization:  │ All cameras +     │ Diverges if init  │
│ adjust-  │ jointly refine f,k1,  │ points + obs      │ too far from      │
│ ment     │ k2, all poses, all 3D │ → refined cameras │ truth; outliers   │
│          │ points to minimize    │   points K dist   │ overwhelming Huber│
│          │ reprojection error    │                   │ loss              │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 7a. Sparse│Filter outliers;      │ points_3d +       │ Wrong K gives bad │
│ PLY      │ sample RGB from images│ cameras + features│ reprojection →    │
│ export   │ for each point;       │ → output.ply      │ ugly colors       │
│          │ write binary PLY      │                   │                   │
├──────────┼───────────────────────┼───────────────────┼───────────────────┤
│ 7b. MVS  │ Rectify stereo pairs; │ cameras + images  │ Narrow baseline;  │
│ dense    │ SGBM disparity;       │ → dense .ply      │ Textureless areas;│
│ (--dense)│ reproject to world    │                   │ occlusions; slow  │
│          │ coords; filter; merge │                   │ on large datasets │
└──────────┴───────────────────────┴───────────────────┴───────────────────┘

                         ┌─────────────┐
                         │  FULL FLOW  │
                         └──────┬──────┘
               JPGs/PNGs        │
                  │             │
                  ▼             │
          ┌─────────────┐       │
          │  Features   │       │
          │  (per img)  │       │
          └──────┬──────┘       │
                 │              │
                 ▼              │
          ┌─────────────┐       │
          │   Matches   │       │
          │  (pairs)    │       │
          └──────┬──────┘       │
                 │              │
                 ▼              │
          ┌─────────────┐       │
          │  Verified   │       │
          │   Pairs     │       │
          └──────┬──────┘       │
                 │    ┌─────────┘
                 ▼    ▼
          ┌─────────────────┐
          │  Incremental    │◄──────────┐
          │  Registration   │           │
          │   + BA (loop)   │──────────►│ (refines K, dist,
          └────────┬────────┘           │  poses fed back)
                   │
                   ▼
          ┌─────────────────┐
          │  Sparse cloud   │──►  output.ply
          └────────┬────────┘
                   │ (--dense)
                   ▼
          ┌─────────────────┐
          │  MVS / SGBM     │──►  output_dense.ply
          └─────────────────┘
```

---

*Built with OpenCV, NumPy, SciPy, and a lot of linear algebra that ultimately just means: "make the numbers agree with each other as well as possible."*
