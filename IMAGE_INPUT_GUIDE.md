# Image Input Guide

A practical guide to capturing photographs that produce good Structure from Motion reconstructions.
The quality of your output is limited primarily by the quality of your input images.

---

## 1. How the Pipeline Uses Your Images

The pipeline does not compare pixels directly. Instead it finds hundreds of distinctive
corner and blob features ("keypoints") in each image, builds a 128-number fingerprint
("descriptor") for each one, and then looks for the same fingerprint in other images to
identify which photos share the same physical points.

This means:

- **Distinctive texture** is essential — the pipeline cannot track a white wall.
- **Overlap** is essential — adjacent photos must share enough visible surface to match.
- **Blur, noise, and exposure problems** destroy the fingerprints before matching begins.

Every guideline in this document derives from one of those three constraints.

---

## 2. Essential Characteristics

### 2.1 Overlap

Adjacent images should share **60–80% of their field of view**.

```
Too little overlap (40%):         Correct overlap (70%):
┌──────────┐                      ┌──────────┐
│          │  ┌──────────┐        │          │
│          │  │          │        │  ┌───────┼──────┐
│          │  │          │        │  │ SHARED│      │
└──────────┘  └──────────┘        └──────────┘      │
                                     └──────────────┘
    Low inlier count →                  High inlier count →
    pair may be dropped                 robust verification
```

With 40% overlap, a typical image pair yields 15–30 verified inlier matches — barely enough.
With 70% overlap, the same pair yields 80–200 inliers, giving the reconstruction a solid
foundation. For smooth or repetitive surfaces, use 80–90% overlap.

**Practical rule:** Each point on your subject should appear in at least 3 images.

### 2.2 Coverage

Walk completely around objects. For rooms, photograph each wall, the ceiling, and the floor.
Avoid leaving gaps — if a region is only visible in one image it cannot be triangulated.

Coverage issues by scenario:

| Scenario | Common mistake | Fix |
|----------|---------------|-----|
| Object scan | Starting/ending gap leaves seam | Ensure first and last image overlap |
| Interior room | Only photographing walls at eye level | Add upward-tilted and downward-tilted passes |
| Outdoor building | Only shooting the facade | Add oblique shots from corners |
| Archaeological site | Nadir-only drone flight | Add oblique flights at 30–45° |

### 2.3 Texture

SIFT needs **distinctive local texture** to place keypoints. Surfaces that fail:

- **Uniform color** — white walls, grey concrete, clear sky
- **Repetitive patterns** — brick, tile, mesh, fabric with regular patterns
- **Specular surfaces** — polished metal, wet rock, glossy paint
- **Transparent surfaces** — glass, water

If your subject includes large textureless areas, place **reference markers** (printed A4 sheets
with distinct patterns) or **natural texture aids** (gravel, fabric scraps, patterned tape) on
those surfaces. Remove them in post-processing if needed.

### 2.4 Focus and Sharpness

Motion blur and out-of-focus images produce no usable features. Rules:

- Use a **fast shutter speed** in anything but static studio conditions (aim for ≥ 1/200 s outdoors).
- Use **continuous autofocus** off — it can hunt between frames. Lock focus or use manual.
- For handheld shooting: brace your arms against your body; exhale before pressing the shutter.
- **Depth of field:** use f/5.6–f/11. Very wide apertures (f/1.4–f/2.8) give shallow DoF; parts of the scene may be blurry.

Quick blur test: zoom to 100% in any image viewer. Feature edges should be a single pixel wide,
not a two-or-three-pixel gradient.

### 2.5 Lighting

Consistent lighting across all images prevents the descriptor from treating the same surface
as two different things:

- Shoot during **overcast conditions** or in **open shade** outdoors — soft, even illumination.
- Avoid **direct sun** creating harsh shadows that move as you walk around.
- Avoid **mixed light sources** (tungsten + daylight) — colour shifts fool the descriptor.
- In interiors, turn on all room lights and close blinds to minimize sunlight patches.
- **Night shooting** is possible but requires high-quality lighting rigs or very high ISO (which
  introduces noise that masks features).

### 2.6 Exposure

Auto-exposure can shift between frames. Use **manual exposure** or at least lock:
- ISO (set to the base value for your camera — typically 100 or 200)
- Aperture (locked)
- Allow shutter speed to adapt only within ±1 stop

**Avoid blowout** (completely white areas lose all texture). Check the histogram — no clipping
on the right side. For difficult lighting, bracket exposures and pick the best.

### 2.7 Resolution

Minimum: 4 megapixels (2560 × 1600). Recommended: 12–24 megapixels.

Higher resolution gives the feature detector more to work with, but very large images slow
extraction. If shooting at 50 MP, downsample to 24 MP before processing. The pipeline reads
full-resolution images; resizing is your responsibility.

### 2.8 Image Count

| Scene type | Minimum | Recommended |
|-----------|---------|-------------|
| Small object (< 30 cm) | 20 | 40–60 |
| Medium object (30 cm – 2 m) | 30 | 50–80 |
| Room interior | 40 | 80–120 |
| Building facade | 30 | 60–100 |
| Outdoor scene / site | 50 | 100–300 |

More images improve coverage and reduce per-image matching pressure, but increase processing
time quadratically (for exhaustive matching) or linearly (for sequential/vocab-tree). Use
`--match_strategy vocab_tree` for datasets above ~100 images.

---

## 3. Anti-Patterns to Avoid

### 3.1 All Images from the Same Direction

```
Bad (all from front):          Good (circular coverage):
    ↓ ↓ ↓ ↓ ↓                     ↙ ↓ ↘
    [obj]                        ←  [obj]  →
                                   ↖ ↑ ↗
```

Images taken from the same direction produce nearly identical views. The Essential Matrix
cannot distinguish between a small object close up and a large object far away when all views
are frontoparallel. Always encircle your subject.

### 3.2 The Straight Walkthrough

For room scans: walking in a straight line and shooting forward produces cameras all on a line.
This degenerate configuration makes triangulation angles very small. Add lateral movements
and turns. A figure-8 path through a room gives much better results than a linear traverse.

### 3.3 Zooming Between Shots

The pipeline assumes a fixed focal length (single shared camera matrix K) across all images.
If you zoom in or out between shots, some images will have the wrong focal length assigned.
Either:
- Lock zoom throughout the entire capture session, or
- Sort images by focal length EXIF and process groups separately.

### 3.4 Mixed Camera Models

Similar to zooming: images from different cameras (e.g., phone + DSLR) have different intrinsics.
The current Python backend uses a single shared K. For mixed-camera datasets use
`--backend colmap`, which supports per-camera intrinsics.

### 3.5 Reflective and Transparent Surfaces

Reflections create "virtual" keypoints whose apparent 3D position is behind the reflecting
surface — geometrically impossible, causing matching failures and outlier 3D points. Strategies:

- Photograph at angles that minimize reflections (< 20° from surface normal for glass).
- Use a polarizing filter on the lens.
- Mask reflective regions from images in post-processing before running the pipeline.

### 3.6 Moving Objects

Cars, people, waving foliage — anything that changes position between shots produces keypoints
that match nothing in adjacent images (the object moved) or match incorrectly (a car in the
same lane but different position). The RANSAC outlier rejection handles some of this, but
large moving objects covering >30% of multiple frames can cause registration failures.

Shoot early morning (fewer people and cars), use a long focal length (compresses background
motion), or remove frames with prominent movers.

### 3.7 Sky and Water

Uniform sky and open water produce zero keypoints. More importantly they fill large fractions
of frames, reducing the texture-bearing area. Aim your camera slightly downward to include
foreground texture. For aerial scans, avoid nadir (straight down) passes over water.

---

## 4. Shooting Strategies by Subject Type

### 4.1 Small Object — Turntable Approach

Best for objects 5–50 cm across (pottery, fossils, small sculptures, hardware).

**Setup:**
- Place the object on a textured surface (newspaper, printed pattern sheet).
- Use controlled indoor lighting — two diffuse sources at 45° from front, no direct sunlight.
- Fix your camera on a tripod at a consistent height and distance.

**Capture sequence:**
1. Equatorial ring: ~20–30 shots rotating 12–18° each, at eye-level to the object center.
2. High-angle ring: ~20 shots at 30° above horizontal, same rotation step.
3. Top shots: 5–8 shots looking nearly straight down at the top of the object.
4. (Optional) Low-angle ring: ~10 shots at 15° below horizontal for underside coverage.

**Total: 55–70 images.** Use a tripod-mounted turntable or rotate the object, not the camera.

### 4.2 Medium Object or Person

For objects 50 cm – 2 m (chairs, statues, cars, people in controlled poses).

**Procedure:**
1. Walk slowly in a complete circle at 1.5–2× the object's longest dimension.
2. Take a photo every ~10° of arc (36 images for a full circle).
3. Add a second circle at a different height (angled up or down ~20°).
4. For complex geometry (undercuts, overhangs), add a third pass at extreme angles.

**People:** Use a reference frame (distinctive background, textured floor) and ask the subject
to stay completely still. Shoot all angles within 60–90 seconds.

### 4.3 Building Exterior

For building facades 5–50 m across.

**Horizontal strips:**
- Walk parallel to the facade at a fixed distance (2–5× the facade height).
- Shoot every 1–2 m so consecutive images have 70% overlap.
- Repeat at multiple heights if the building is taller than ~3 floors.

**Corner shots:** Always include shots from each corner with ~120° coverage — these provide the
cross-connections that stop long facades from behaving like linear degenerate configurations.

**GSD:** At 10 m distance and 24 MP camera, ground sample distance is ~3 mm/pixel. Match
your distance to your required resolution.

### 4.4 Room Interior

**Entry sweep:** From the doorway, shoot the full room in a slow horizontal pan before moving.

**Figure-8 path:** Walk in a figure-8 through the room, shooting forward and sideways, capturing
all walls, major furniture, and the ceiling. Aim for each wall to appear in at least 5 images
from different positions.

**Close-up passes:** For detailed surfaces (fireplace, bookshelf, artwork), add a dedicated
close-up pass at 0.5–1 m from the surface.

**Lighting:** Switch on ALL lights (no shadows), close blinds. Take test shots and verify
histograms are not clipped.

### 4.5 Aerial / Drone

**Grid flight plan:** Overlapping rows with 70% forward overlap and 70% side overlap. At 100 m
altitude this gives roughly 4 cm/pixel GSD with a 20 MP camera.

**Oblique passes:** Pure nadir flights produce near-degenerate geometry for vertical walls.
Add one or more oblique flights at 40–60° gimbal angle.

**Wind:** Strong wind introduces camera shake and scene motion. Fly in calm conditions or
adjust shutter speed to compensate.

---

## 5. Camera and Lens Recommendations

| Scenario | Recommended lens | Notes |
|----------|-----------------|-------|
| Object scanning | 35–85 mm (FF equiv) | Avoid extreme wide-angle — barrel distortion reduces accuracy |
| Building exterior | 24–50 mm | Wide enough for close facades but not fisheye |
| Room interior | 24–35 mm | Accept that BA will need to fit k1/k2 distortion |
| Aerial / drone | Fixed focal length, 24–35 mm | Variable-aperture drone lenses introduce zoom artifacts |

Avoid:
- Fisheye lenses (> 120° FoV) — the pipeline's distortion model (k1, k2) is insufficient; use `--backend colmap` with a fisheye camera model.
- Macro lenses at minimum focus — depth of field is too shallow and parallax is extreme.

**EXIF data:** The pipeline reads `FocalLengthIn35mmFilm` from EXIF to initialize the focal
length (instead of the heuristic f = max(W,H)). Leave EXIF embedding enabled in your camera.

---

## 6. Quality Checklist

Before running the pipeline, verify:

- [ ] ≥ 20 images (for the simplest objects); more for complex scenes
- [ ] Each part of the subject visible in ≥ 3 images from clearly different angles
- [ ] 60–80% overlap between adjacent images (check a few pairs by eye)
- [ ] No motion blur (zoom to 100% and check edges)
- [ ] No exposure clipping (review histogram on a few images)
- [ ] Consistent focal length (zoom locked, single camera model)
- [ ] Consistent lighting across all images
- [ ] No large moving objects (people, cars) in the frame
- [ ] Textureless areas supplemented with reference markers or natural texture
- [ ] Images sorted in a sensible order for sequential matching (or use vocab_tree for unordered)

---

## 7. Troubleshooting Poor Reconstructions

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Very few cameras registered (< 50% of images) | Insufficient overlap or blur | Increase overlap; check focus |
| Reconstruction splits into disconnected components | Scene graph gaps | Add transition images bridging gap areas |
| Point cloud is very sparse | Textureless surfaces | Add reference markers; use `--n_features 16000` |
| Reconstruction drifts or curves | Long sequence without loop closure | Ensure first/last images overlap; use vocab_tree |
| High reprojection error (> 3 px) | Mixed focal lengths or strong distortion | Fix zoom; use `--backend colmap` for better distortion model |
| "Too few inliers" for many pairs | Insufficient matches | Lower `--ratio` to 0.7 or `--min_inliers` to 10 |
| Dense cloud has holes | SGBM baseline too narrow | Use covisibility-based pair selection (default since R-04) |
| Reflective "ghost" points | Mirrors or windows in scene | Mask reflective surfaces before processing |

For systematic diagnosis of reconstruction quality, see `VISUALIZATION_GUIDE.md`.

---

## 8. Camera Settings Reference

### 8.1 Recommended Settings by Scenario

| Scenario | ISO | Aperture | Shutter speed | Notes |
|----------|-----|----------|---------------|-------|
| Outdoor overcast | 100–400 | f/8 | 1/250–1/1000 | Lock manual exposure |
| Outdoor sunny | 100 | f/11 | 1/500–1/2000 | Harsh shadows: shoot in shade |
| Indoor studio | 100–200 | f/8 | 1/200 | Use diffuse continuous lights |
| Indoor ambient | 400–800 | f/5.6 | 1/60–1/200 | Brace camera; check for blur |
| Drone / aerial | 100–200 | f/5.6–f/8 | 1/1000+ | Eliminate motion blur |
| Macro (< 10 cm) | 100–200 | f/11–f/16 | 1/200 | Use focus rail, not autofocus |

**ISO:** Keep as low as practical. High ISO (>3200) introduces luminance noise that
creates spurious gradient directions in SIFT patches, reducing descriptor distinctiveness.

**Aperture:** The sweet spot for sharpness is f/5.6–f/11 for most lenses. Stopping down
to f/16+ causes diffraction softening on most sensors. Wide apertures (f/1.4–f/2.8)
give thin depth of field; the SIFT detector will place keypoints in the sharp region but
not in blurry areas, creating uneven coverage.

**Shutter speed:** The rule of thumb for handheld shooting is 1/(2 × focal_length_mm).
For a 50 mm lens: 1/100 s minimum. For a 200 mm telephoto: 1/400 s minimum.

### 8.2 White Balance

Set white balance to a **fixed preset** (Daylight, Cloudy, Flash, Tungsten) rather than
Auto. Auto white balance can shift between frames when the camera pans from a bright wall
to a dark window, making the same textured surface look different in two consecutive images.
The SIFT descriptor is not color-aware (it operates on luminance), but color is used for
point cloud colorization — inconsistent white balance produces ugly color seams.

---

## 9. Pre-Processing Your Images

### 9.1 File Format

The pipeline reads JPEG, PNG, TIFF, and BMP. JPEG with quality ≥ 90 is acceptable.
Quality < 80 introduces block artifacts that create spurious high-frequency texture,
confusing the SIFT detector.

For archival or scientific work, shoot RAW and export to 16-bit TIFF. The pipeline
handles TIFF natively; the extra bit depth doesn't improve feature matching but
avoids any JPEG compression artifacts.

### 9.2 Renaming and Ordering

For `--match_strategy sequential`, images must be sorted in **spatial order** (the order
in which consecutive images were taken). The pipeline sorts by filename. Ensure your
filenames sort in the correct sequence:

- Camera-assigned names (`IMG_1234.JPG`, `DSC_5678.JPG`) are usually already in
  timestamp order.
- If you merged images from multiple shooting sessions, rename them to reflect spatial
  order: `001.jpg`, `002.jpg`, … using a batch rename tool.

For `--match_strategy vocab_tree` or `exhaustive`, order does not matter.

### 9.3 Removing Bad Frames

Before running the pipeline, remove:

- **Blurry frames** (motion blur, defocus): zoom to 100% and discard any image where
  feature edges are not sharp.
- **Duplicate frames**: images within 2–3% of each other's position add redundancy but
  inflate processing time. Keep one in three if you over-captured.
- **Sky-only or floor-only frames** that contain no part of the subject (these produce
  zero features relevant to the reconstruction).
- **Frames with large occlusions**: a person walking through the frame mid-capture can
  usually be discarded if an adjacent frame shows the same area unobstructed.

### 9.4 Downsampling Large Images

For cameras above 24 MP, consider downsampling to 12–24 MP before processing:

```bash
# Using ImageMagick
mogrify -resize 6000x4000> -quality 95 ./images/*.jpg

# Using Python + Pillow
python -c "
from pathlib import Path
from PIL import Image

for p in Path('images').glob('*.jpg'):
    img = Image.open(p)
    if max(img.size) > 6000:
        img.thumbnail((6000, 4000), Image.LANCZOS)
        img.save(p, quality=95)
"
```

Processing 50 MP images takes ~3× longer than 20 MP images with minimal quality gain,
because the pipeline's SIFT feature cap (`--n_features 8000`) limits how many extra
features are used regardless of resolution.

### 9.5 EXIF Data

Keep EXIF data intact. The pipeline reads `FocalLengthIn35mmFilm` (tag 0xA405) to
initialize the focal length. If this tag is missing (some RAW exporters strip EXIF),
the pipeline falls back to `f = max(W, H)` pixels — a reasonable heuristic but less
accurate than the EXIF value.

Do **not** strip EXIF before processing. Strip after, if privacy is a concern.

---

## 10. Verifying Your Dataset Before Processing

Run this quick sanity check before starting a long pipeline run:

```bash
# Count images and check extensions
ls images/ | wc -l
ls images/*.jpg images/*.png images/*.JPG 2>/dev/null | wc -l

# Check image dimensions are consistent
python -c "
from pathlib import Path
from PIL import Image

sizes = {}
for p in sorted(Path('images').glob('*.[jJpP][pPnN][gG]'))[:5]:
    img = Image.open(p)
    sizes[p.name] = img.size
for name, sz in sizes.items():
    print(f'{name}: {sz[0]}x{sz[1]}')
"

# Quick blur check on a sample (requires opencv-python)
python -c "
import cv2, glob, sys
imgs = sorted(glob.glob('images/*.jpg'))[:10]
for p in imgs:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    status = 'OK' if lap > 100 else 'BLURRY'
    print(f'{status:6s}  {p}  (sharpness={lap:.0f})')
"
```

Sharpness values (Laplacian variance) above 100 are generally acceptable; below 50
indicates likely blur.

---

## 11. Understanding Scale Ambiguity

Structure from Motion recovers 3D structure up to an **unknown scale factor**. The output
point cloud has the correct shape but no absolute size — a model of a coffee mug and a
model of a building could look identical in MeshLab unless you know the scale.

If you need metric scale, add at least one **known distance** to the scene:
- Place a ruler or scale bar in a few images.
- Measure two clearly identifiable points and note the distance.
- Use GCPs (Ground Control Points) with GPS coordinates for aerial surveys.

After reconstruction, you can apply a uniform scale factor in MeshLab (`Filters → Normals,
Curvatures and Orientation → Transform: Scale, Normalize`) to convert the model to real
units.

The reprojection error reported by the pipeline is in **pixels**, not metric units.
A 0.8 px reprojection RMSE on a 3000-pixel-wide image corresponds to roughly 0.03% of
the image width — this is independent of the physical scale.

