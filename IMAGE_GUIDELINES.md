# IMAGE_GUIDELINES.md — Shooting for Structure from Motion

A practical guide to capturing images that reconstruct well.

---

## Table of Contents

1. [Overview](#1-overview)
2. [✅ Ideal Image Characteristics](#2--ideal-image-characteristics)
3. [❌ Bad Image Examples — What to Avoid](#3--bad-image-examples--what-to-avoid)
4. [📷 Recommended Shooting Strategies](#4--recommended-shooting-strategies)
5. [Camera & Lens Recommendations](#5-camera--lens-recommendations)
6. [Quick Reference Checklist](#6-quick-reference-checklist)
7. [⚠️ Troubleshooting: Why Did My Reconstruction Fail?](#7-️-troubleshooting-why-did-my-reconstruction-fail)

---

## 1. Overview

Structure from Motion works by finding the **same physical points** across many photographs taken from different positions, then using the geometry of those matches to simultaneously figure out where each camera was and where each point sits in 3D space.

Every step in that chain depends on your images:

```
Good images → Reliable feature matches → Stable geometry → Dense point cloud
Bad images  → Sparse / wrong matches  → Drift or failure
```

The pipeline never sees your object directly — it only sees pixel gradients. Anything that makes it hard to find repeatable, geometrically consistent features (blur, reflections, flat textures, extreme lighting) will degrade or destroy the reconstruction.

**The three things SfM needs most from your images:**

| Need | Why it matters |
|------|---------------|
| **Overlap** | Each 3D point must be visible in ≥ 3 images for robust triangulation |
| **Texture** | Features are detected at gradient boundaries — no gradients, no features |
| **Consistency** | The same point must *look* the same across views for descriptors to match |

Think of it like a jigsaw puzzle: SfM is trying to assemble a 3D model from fragments of evidence. More fragments, more overlap between them, and more distinctive markings on each piece all make the assembly more reliable.

---

## 2. ✅ Ideal Image Characteristics

### Overlap

SfM requires that each 3D point appears in multiple images so it can be triangulated. More overlap gives more matches per pair, making the geometry more stable and the registration of each new camera easier.

> 💡 **Rule of thumb:** Aim for **60–80% overlap** between consecutive shots. If you are walking around an object, each step forward should leave most of the previous frame still visible.

| Overlap | Effect |
|---------|--------|
| < 40%   | Too few shared features; cameras frequently fail to register |
| 40–60%  | Acceptable for simple, highly textured objects |
| **60–80%** | **Recommended for most subjects** |
| > 80%   | Diminishing returns; large image sets with little new information |

**Practical tip:** After your shoot, flip through consecutive frames as a quick animation. If the viewpoint jumps dramatically between frames, you need more shots.

---

### Coverage Angle

Walking a single ring around an object at eye level captures the *sides* but leaves the top and bottom nearly unobserved. A reconstruction built from only one elevation looks like a hollow shell viewed from that ring — the roof and underside will be missing or highly uncertain.

**Recommended elevation rings:**

| Subject height | Rings to shoot | Elevations |
|---------------|---------------|------------|
| Small object (< 30 cm) | 3 | ~30°, ~60°, ~80° (near top-down) |
| Medium object (30 cm–2 m) | 3–4 | 20°, 45°, 70°, + top-down if accessible |
| Building / statue | 2–3 | Ground level, mid-rise, aerial if available |

**Practical tip:** Think of your target as sitting inside a dome. You want to populate that dome with camera positions, not just trace a line around its equator.

---

### Texture

SIFT and similar detectors find keypoints at *edges and corners* — places where pixel intensity changes rapidly. Textureless, uniform surfaces have no such changes and produce zero usable features.

| Surface type | Suitability | Notes |
|-------------|-------------|-------|
| Rough stone, bark, fabric | ✅ Excellent | Rich, natural texture |
| Painted metal, matte plastic | ✅ Good | Usually enough micro-texture |
| Smooth white ceramic | ⚠️ Marginal | May need artificial texture added |
| Glass, mirror, polished chrome | ❌ Poor | Reflections change with viewpoint |
| Clear water, transparent plastic | ❌ Very poor | Near-zero surface features |

**Practical tip:** If you *must* scan a textureless object, temporarily apply a matte spray (available at art supply shops) or scatter a pattern of small stickers across the surface. Remove them in post-processing by masking those regions.

---

### Lighting

Consistent, diffuse lighting is your best friend. SfM matches descriptors built from gradient *directions*, not absolute brightness — but strong, moving shadows create *false* gradients that appear to move between frames, confusing the matcher.

> 💡 **Rule of thumb:** Overcast daylight is ideal. If you can, shoot outdoors on a lightly cloudy day or in open shade. Avoid direct sun, which casts sharp shadows that shift as you move.

| Lighting scenario | Quality | Problem |
|------------------|---------|---------|
| Overcast / diffuse outdoor | ✅ Best | Even illumination, no harsh shadows |
| Open shade (sun blocked) | ✅ Excellent | Same as above |
| Lightbox / ring flash (for small objects) | ✅ Good | Watch for specular highlights |
| Partly cloudy, moving shadows | ⚠️ Risky | Shadows shift between frames |
| Direct midday sun | ⚠️ Poor | Harsh shadows, specular surfaces |
| Indoor mixed artificial + window | ⚠️ Poor | Color cast changes, unpredictable |
| Changing ambient (dusk / sunset) | ❌ Bad | Exposure and color shift every frame |

**Practical tip:** If shooting indoors, keep windows consistently lit by closing blinds and using artificial lights that do not flicker. Shoot at a constant ISO/shutter/aperture so exposure does not drift.

---

### Focus & Sharpness

A blurry image is like a smeared fingerprint — the features are there, but SIFT cannot pin them down precisely. Both types of blur are damaging:

- **Motion blur** (camera shake or subject movement): smears edges in one direction, making keypoint localization inaccurate.
- **Depth-of-field blur** (shallow focus): makes background features undetectable while only the foreground is sharp. If the pipeline cannot match background features, it loses constraints on the camera's distance.

> 💡 **Rule of thumb:** Shoot at **f/8–f/11** on a camera lens for maximum depth of field. Use a tripod or at least brace your arms against your body. Minimum shutter speed: 1/(2 × focal_length_mm) for handheld work.

**Practical tip:** Review your images at 100% zoom before starting the pipeline. If you see motion blur on sharp edges (straight lines becoming streaks), reshoot.

---

### Resolution

Higher resolution lets SIFT detect finer features and localize keypoints more accurately, which improves triangulation precision.

| Resolution | Notes |
|-----------|-------|
| < 2 MP    | Generally insufficient — very few keypoints detected |
| 4–8 MP    | Minimum recommended for good results |
| **12–24 MP** | **Sweet spot** — good quality, manageable file size |
| > 36 MP   | Diminishing returns; processing is slower; consider downsampling to 20–24 MP |

**Practical tip:** If processing time is a concern, downsample images to 12–16 MP before ingestion. The pipeline scales down anyway; you control the quality floor.

---

### Number of Images

More images is better — up to a point. Too few means gaps in coverage; too many (with very low overlap differences) wastes compute without adding new geometric constraints.

> 💡 **Rule of thumb:** For a hand-held object, **60–120 images** is a typical good range. For a room or building exterior, **100–300 images**. For aerial: **200–500+** depending on area.

| Image count | Typical result |
|------------|---------------|
| < 15       | Often fails to register; seed pair unreliable |
| 15–40      | Feasible for simple, strongly textured objects |
| **40–150** | **Typical good range for most objects** |
| 150–500    | Good for large / complex subjects; BA becomes slower |
| > 500      | Consider hierarchical or guided matching to reduce pair count |

---

### Background

A clean, static background reduces the chance that the pipeline tracks *background* features instead of — or in addition to — your subject.

| Background type | Effect |
|----------------|--------|
| Plain matte surface (grey cardboard, cloth) | ✅ Best — no distraction |
| Static indoor clutter | ✅ Acceptable if it does not move |
| Moving people, cars, foliage | ❌ Bad — creates ghost features |
| Mirror or reflective floor | ❌ Bad — doubles apparent scene structure |
| Repeating pattern (tiled floor, bookshelves) | ⚠️ Risky — can cause false matches |

**Practical tip:** For small objects, use a lazy Susan on a plain matte cloth. For outdoor subjects, time your shoot to avoid pedestrian traffic, and use a mask or segmentation step to exclude sky/ground if the reconstruction drifts.

---

## 3. ❌ Bad Image Examples — What to Avoid

### Too Few Images or Wide Angular Gaps

**Why it breaks SfM:** Each new camera is registered using 2D–3D correspondences — matches between pixels in the new image and already-triangulated 3D points. If no previous camera covered that viewpoint closely enough, there are no shared 3D points, and registration fails. The pipeline reports "no more images can be registered."

**Fix:** Reduce your step size between frames. If you are walking a circle, take a step after every 10–15 degrees rather than every 30–45 degrees.

---

### Repetitive or Symmetric Textures

**Why it breaks SfM:** A brick wall, tire tread, or plaid fabric has hundreds of nearly identical feature patches. The matcher cannot distinguish "the third brick from the left" from "the fifth brick from the left," producing a cascade of false matches. RANSAC may settle on a *wrong* geometry that happens to satisfy many of these ambiguous correspondences.

**Fix:** If you cannot avoid a repetitive surface, include distinctive *non-repetitive* anchor points (a mark, a patch of different material) that give the matcher unique landmarks to work with.

---

### Reflective or Transparent Surfaces

**Why it breaks SfM:** A mirror reflects the *scene*, so what the camera sees changes dramatically with viewpoint. A shiny chrome ball reflects an entirely different environment from each angle. These are *not* features of the object — they are features of the lighting, which breaks the assumption that the same 3D point looks the same across views.

**Fix:** Apply a matte coating (temporary spray paint, talcum powder, chalk) before scanning. Alternatively, mask the reflective regions in each image so the pipeline ignores them.

---

### Motion Blur

**Why it breaks SfM:** SIFT localizes keypoints at sub-pixel accuracy. Motion blur turns a sharp corner into a smear, shifting the apparent keypoint position. Even a 2-pixel blur can increase the reprojection error enough to cause the match to be rejected as an outlier, thinning the inlier set below the threshold.

**Fix:** Use a tripod, bump your shutter speed, or brace against a wall. For handheld shoots, use burst mode and discard blurry frames by inspecting at 100% zoom.

---

### Inconsistent Lighting

**Why it breaks SfM:** SIFT descriptors capture gradient magnitudes. A shadow that falls differently in two consecutive frames creates a region where the gradients appear to have changed, even though the 3D geometry has not. The descriptor for the shadowed region in frame A will not match its descriptor in frame B, losing those correspondences entirely.

**Fix:** Shoot in stable light. For studio setups, keep lights fixed and turn off lamps that might shift. For outdoor, shoot in a window of 1–2 hours around the same time on the same day.

---

### Wide-Angle or Fisheye Lenses Without Calibration

**Why it breaks SfM:** The pipeline uses a pinhole camera model (straight lines stay straight). Wide-angle lenses — especially anything below ~24 mm equivalent — introduce radial distortion that bends straight lines into curves. The pipeline's fundamental matrix estimation assumes straight epipolar lines, so distortion causes systematic outliers in RANSAC and biased pose estimates.

**Fix:** Either provide a calibrated distortion model (and undistort images before feeding them in) or use a lens with focal length ≥ 24 mm equivalent. Do not mix wide and normal lens images in the same session.

---

### Images of Only One Side / Insufficient Coverage

**Why it breaks SfM:** Any surface that is never observed cannot be reconstructed. Worse, the *boundary* between observed and unobserved regions is geometrically unconstrained, so the reconstruction can drift or bend near those edges.

**Fix:** Plan your coverage in advance. For an object, check during the shoot that you have captured the back, bottom, and any recessed areas. Use a mirror or second camera position to see occluded regions if necessary.

---

### Foreground Clutter or Moving Objects

**Why it breaks SfM:** A person walking through the scene appears in one frame but not others. The pipeline will attempt to triangulate their position as a static 3D point, then fail because the reprojection into other frames is wrong. This creates a cloud of spurious outlier points and can contaminate the inlier sets used for pose estimation.

**Fix:** Clear the scene before shooting. If that is impossible (public space, busy environment), use a *mask* per image to exclude moving objects, or plan to run a statistical outlier filter aggressive enough to discard short-track points.

---

## 4. 📷 Recommended Shooting Strategies

### Small Object on a Turntable

Best for: figurines, shoes, food products, fossils, artifacts (< 50 cm).

**Setup:**
- Place the object on a matte grey or black lazy Susan
- Use a fixed camera on a tripod; rotate the object, not the camera
- Shoot in a lightbox or under diffuse studio lighting
- Disable auto-focus after the first frame (use manual focus or AF-lock)

**Shot plan:**

| Ring | Elevation | Images | Angular step |
|------|-----------|--------|-------------|
| 1    | ~20° (near horizontal) | 36 | 10° |
| 2    | ~45° | 24 | 15° |
| 3    | ~70° | 18 | 20° |
| 4    | 90° (top-down) | 1–4 | — |

**Total: ~80–85 images.** If the object has complex underside geometry, add a fifth ring at –20° by tilting the object on a foam wedge.

> 💡 Rotate the turntable in 10° increments and use a remote shutter or 2-second timer to avoid camera shake.

---

### Outdoor Object / Statue (Walk-Around Strategy)

Best for: sculptures, trees, vehicles, exterior architectural details.

**Setup:**
- Shoot on an overcast day or in open shade
- Keep the sun behind you (if it must be sunny), not to the side
- Use a consistent distance from the object (~2–5 m depending on size)

**Shot plan:**

| Ring | Approximate camera height | Images | Angular step |
|------|--------------------------|--------|-------------|
| 1 (low) | Hip height | 36 | 10° |
| 2 (mid) | Eye level | 36 | 10° |
| 3 (high) | Arms raised / step ladder | 24 | 15° |
| Detail | Close-up passes of key features | 20–40 | Varies |

**Total: ~120–140 images.** For larger statues (> 3 m), add a drone pass at 20° and 45° elevation.

> 💡 Walk slowly and overlap each step. If you can, keep the lens at a constant zoom setting and verify focus before each burst.

---

### Room or Architectural Interior

Best for: rooms, corridors, retail spaces.

**Setup:**
- Turn on every light source; close blinds to eliminate changing window light
- Shoot at a consistent exposure (set manually)
- Use a 35–50 mm equivalent lens

**Shot plan:**

1. **Perimeter walk** at standing height (eye level), one frame every ~0.5 m, maintaining 70% overlap
2. **Corner positions** — stand in each corner and shoot toward the opposite wall and toward the ceiling
3. **Detail passes** — zoom in or step closer to capture feature-rich surfaces (picture frames, furniture textures)
4. **Ceiling upshots** — tilt the camera ~45° upward at 4–6 positions

**Total: 100–250 images** depending on room size.

> ⚠️ Avoid images where a large portion of the frame is a blank white wall or ceiling. These add near-zero information and can confuse pose estimation.

---

### Aerial / Drone Capture

Best for: terrain, archaeological sites, rooftops, large structures.

**Setup:**
- Fly at a consistent altitude (or constant AGL for sloped terrain)
- Use **nadir** (straight down) + **oblique** (camera tilted 30–45°) passes
- Lock gimbal angle; do not vary between frames
- Choose a cloudy day or fly early morning / late evening for even light

**Shot plan:**

| Pass type | Camera angle | Overlap target | GSD target |
|-----------|-------------|---------------|-----------|
| Grid (nadir) | 90° (straight down) | 80% front, 70% side | 2–5 cm/px |
| Oblique N | 45° facing North | 70% front, 60% side | — |
| Oblique E | 45° facing East | 70% front, 60% side | — |
| Oblique S/W | 45° facing South/West | 70% front, 60% side | — |
| Perimeter orbit | 60° inward | 70% | — |

**Total: 300–600+ images** for a 1-hectare site at 2 cm/px. Use photogrammetry flight-planning software (OpenDroneMap, DroneDeploy) to automate waypoints.

> 💡 For structures with vertical facades (buildings, cliff faces), the nadir-only approach will miss walls entirely. Always add oblique or orbit passes.

---

## 5. Camera & Lens Recommendations

### Fixed Focal Length vs. Zoom

| Lens type | Recommendation | Reason |
|-----------|---------------|--------|
| **Prime (fixed focal length)** | ✅ Preferred | Constant focal length = stable intrinsics across all images |
| Zoom, locked at one end | ✅ Acceptable | Only if you *never* touch the zoom ring during the session |
| Zoom, varied during session | ❌ Avoid | Changing focal length changes K — calibration is invalid mid-session |

The SfM pipeline estimates a *single* intrinsic matrix K for the entire session (unless you feed per-image EXIF focal lengths). Even a small zoom change mid-session introduces systematic error in pose estimation.

---

### Avoiding Auto-Focus Changes Mid-Session

Auto-focus subtly changes the effective focal length and, more critically, moves the focus plane. This shifts feature positions relative to a fixed K, introducing errors proportional to depth-of-field variation.

**Best practice:** Half-press to lock AF on the subject before the first shot, then switch to manual focus for the rest of the session. Or use hyperfocal distance focusing (f/8 at the distance that puts your subject and background in focus simultaneously).

---

### RAW vs. JPEG

| Format | Notes |
|--------|-------|
| **JPEG** | Convenient; mild compression artifacts usually have no practical impact on SIFT at normal compression levels (quality ≥ 85) |
| **RAW** | Better tonal range; useful if you need to recover shadows or highlights in post. Export at ≥ quality 90 before feeding the pipeline |
| RAW + heavy post-processing | ⚠️ Risky if processing changes *vary between frames* (different tone curves, exposure adjustments) — treat each image consistently |

**Practical tip:** Shoot JPEG Fine or RAW and export at consistent settings. The pipeline does not benefit from the extra dynamic range that RAW provides — what matters is consistency.

---

### Phone Cameras — Are They Viable?

Short answer: **yes, with caveats.**

| Factor | Typical phone camera behavior | Impact |
|--------|------------------------------|--------|
| Focal length | Usually 24–26 mm equivalent (wide) | Some barrel distortion; undistort if possible |
| Auto-HDR / computational photography | May blend multiple frames | Creates motion-blended artifacts; disable HDR |
| Portrait mode / depth effect | Artificially blurs background | Disables feature detection in blurred zones |
| Auto-exposure, auto-white-balance | Changes between frames | Increases descriptor mismatch |
| Rolling shutter | Bends fast-moving content | Mostly harmless for static scenes |

**Best phone practices:**
- Lock exposure and white balance (most modern phones allow this in Pro mode)
- Disable portrait mode, HDR, and Night mode
- Use the primary (1×) camera, not ultra-wide or telephoto
- Shoot in well-lit conditions to minimize noise

---

## 6. Quick Reference Checklist

Print this out and run through it before every shoot.

### Before You Shoot

- [ ] Lighting is **diffuse and consistent** — overcast sky, lightbox, or static indoor lights
- [ ] Camera is set to **manual exposure** (or locked AE/AF)
- [ ] **Zoom locked** at one focal length (or prime lens in use)
- [ ] **Focus locked** at hyperfocal distance or manual focus set
- [ ] **Shutter speed** fast enough to avoid motion blur (≥ 1/200 s handheld)
- [ ] **Aperture** set to f/8–f/11 for deep depth of field
- [ ] Background is **matte, static**, and non-repetitive
- [ ] Subject surface has **visible texture** (no large untextured regions)
- [ ] Moving objects (people, vehicles, foliage) **cleared from the scene**

### During the Shoot

- [ ] Each frame overlaps the previous by **≥ 60%**
- [ ] Shooting **multiple elevation rings** (not just one horizontal orbit)
- [ ] Covering **all sides** — back, bottom, recessed areas
- [ ] Images are **in focus** (spot-check at 100% zoom periodically)
- [ ] **Not changing** focal length, focus, or exposure between frames
- [ ] Using a **remote shutter** or timer to avoid camera shake

### After the Shoot — Before Processing

- [ ] Reviewed images at **100% zoom** — deleted obviously blurry frames
- [ ] No fewer than **40 images** (ideally 60+) for a typical object
- [ ] Consistent file format and no mid-session **white-balance shifts**
- [ ] Ground-truth scale reference present if **metric accuracy** is required (ruler, scale bar)

---

## 7. ⚠️ Troubleshooting: Why Did My Reconstruction Fail?

### "Only 3 of my 40 images got registered"

| Likely cause | How to confirm | Fix |
|-------------|---------------|-----|
| Angular gaps too wide | Consecutive images share < 20% of area | Reshoot with 2× more frames, reduce step size |
| Subject too textureless | Few keypoints detected (< 200 per image in verbose log) | Apply matte spray; add artificial texture |
| Extreme lighting change | Matched pairs have very few inliers | Reshoot in stable light; normalize exposure |
| Wrong focal length in K | Estimated K is wildly off | Provide EXIF data or calibrate manually |

> 💡 Run with `--verbose` and check the "Feature matching" log: if most pairs have < 15 raw matches, the images are too dissimilar or the subject is untexturable.

---

### "Point cloud looks like noise / scattered"

| Likely cause | How to confirm | Fix |
|-------------|---------------|-----|
| False feature matches on repetitive texture | High inlier count but reprojection error also high | Increase `--ransac_thr` to filter bad matches; add unique anchor features |
| Moving foreground objects | Cloud has "ghost" points near known moving regions | Mask moving objects; remove those images |
| Reflective surface | Cloud has extra phantom points near reflections | Apply matte coating; mask reflective regions |
| Too few images → under-constrained triangulation | Many points with only 2 observations | Add more images from missing viewpoints |

---

### "One side of the object is missing"

| Likely cause | How to confirm | Fix |
|-------------|---------------|-----|
| No images from that elevation / angle | Camera coverage map has a gap | Add a new ring of shots from the missing direction |
| Occluded by another object or itself | The region is never visible in any frame | Reposition the object or use a mirror to image occluded surfaces |
| Low texture on that side | Few keypoints on the missing surface | Apply temporary texture; use a denser image grid |

---

### "Reconstruction drifted — the model is bent or warped"

This is the classic **accumulated drift** problem, where small pose errors compound over a long sequence.

| Likely cause | How to confirm | Fix |
|-------------|---------------|-----|
| No loop-closure images | The last camera does not see the first camera's features | Add images that bridge the start and end of your orbit (overlap the loop) |
| Too few inliers per pair → BA poorly constrained | Verbose log shows < 10 inliers for many pairs | Increase overlap; improve texture |
| Only one elevation ring | Model bends at the "poles" | Add top/bottom images to anchor the vertical axis |
| Scale ambiguity | Model is correct in shape but scale drifts | Include a ruler / scale bar visible in several frames |

> 💡 Bundle adjustment can correct small amounts of drift, but it cannot invent constraints that are absent from the images. If the model bends, the images themselves are under-constrained — adding more images from the drifted region is the only reliable fix.

---

*Last updated: May 2026. For pipeline-specific parameters (`--min_inliers`, `--ransac_thr`, etc.) see the [CLI reference in README.md](README.md#cli-reference).*
