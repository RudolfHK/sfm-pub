"""
Focal length from EXIF, with provenance.

Why this module exists
----------------------
`paper/paper_l.md` traced the single largest geometric error of the pipeline to
one line: with no usable EXIF the focal fell back to ``max(W, H)``, which on the
Buddha set is 2736 px against a true 1860.9 px (+47.0 %), and that error
propagated into the seed pose, every triangulation and every PnP.

The fallback itself was only half the problem. The other half was that only one
EXIF route was ever tried, ``FocalLengthIn35mmFilm`` (tag 0xA405), which many
cameras simply do not write. Three further routes are standard and are
implemented here, so that a file which *does* carry the information is no longer
treated like a file that does not.

Routes, in order of trust
-------------------------
1. ``FocalLengthIn35mmFilm``: the camera has already done the conversion.
   f_px = f_35 / diag_35mm * diag_px, with diag_35mm = sqrt(36² + 24²).
2. ``FocalLength`` with ``FocalPlaneXResolution`` and
   ``FocalPlaneResolutionUnit``: the sensor pitch is stated directly, so
   f_px = f_mm * FocalPlaneXResolution / unit_in_mm, scaled when the stored
   pixel width differs from the width we actually load.
3. ``FocalLength`` with a sensor width looked up from ``Make``/``Model``:
   f_px = f_mm / sensor_width_mm * W_px. Only a short table of common bodies is
   carried; an unknown model means this route is skipped rather than guessed.
4. Nothing usable. The caller then decides, and `run_sfm.py` starts the focal
   search instead of trusting the ``max(W, H)`` heuristic.

Every result carries a `source` string so the log and the camera export can say
where the number came from. A focal that is off by a factor is worse than no
focal at all, so every route is sanity-checked against a plausible field of
view (roughly 5° to 150° diagonal) before it is returned.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Diagonal of the 35 mm full frame, the reference for "35 mm equivalent".
SENSOR_DIAG_35MM = math.sqrt(36.0 ** 2 + 24.0 ** 2)

# EXIF tag numbers (PIL returns the numeric keys from ``_getexif``).
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_FOCAL_LENGTH = 0x920A          # mm, rational
TAG_FOCAL_PLANE_X_RES = 0xA20E     # pixels per resolution unit
TAG_FOCAL_PLANE_UNIT = 0xA210      # 1 none, 2 inch, 3 cm, 4 mm, 5 um
TAG_EXIF_IMAGE_WIDTH = 0xA002
TAG_FOCAL_35 = 0xA405              # mm, 35 mm equivalent

# Resolution unit -> millimetres per unit.
_UNIT_MM: Dict[int, float] = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}

# Sensor width in mm for bodies that write FocalLength but no focal-plane
# resolution.  Kept deliberately short: a wrong entry is worse than no entry.
SENSOR_WIDTH_MM: Dict[Tuple[str, str], float] = {
    ("apple", "iphone 12"): 5.76,
    ("apple", "iphone 13"): 5.76,
    ("apple", "iphone 14"): 5.76,
    ("apple", "iphone 15"): 9.80,
    ("dji", "fc220"): 6.17,
    ("dji", "fc330"): 6.17,
    ("dji", "fc6310"): 13.20,
    ("gopro", "hero8 black"): 6.17,
    ("gopro", "hero9 black"): 6.17,
    ("canon", "canon eos 5d mark iii"): 36.0,
    ("canon", "canon eos 6d"): 35.8,
    ("canon", "canon eos 80d"): 22.3,
    ("nikon", "nikon d750"): 35.9,
    ("nikon", "nikon d5300"): 23.5,
    ("sony", "ilce-6000"): 23.5,
    ("sony", "ilce-7m3"): 35.6,
    ("panasonic", "dmc-gh4"): 17.3,
}

# Plausibility gate on the diagonal field of view.
_MIN_FOV_DEG = 5.0
_MAX_FOV_DEG = 150.0


@dataclass(frozen=True)
class FocalEstimate:
    """A focal length in pixels together with where it came from."""

    focal_px: float
    source: str
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"focal_px": self.focal_px, "source": self.source,
                "detail": self.detail}


def _to_float(value) -> Optional[float]:
    """EXIF rationals arrive as tuples, Fractions or plain numbers."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _fov_deg(focal_px: float, w: int, h: int) -> float:
    diag = math.sqrt(float(w) ** 2 + float(h) ** 2)
    return math.degrees(2.0 * math.atan(diag / (2.0 * focal_px)))


def _plausible(focal_px: Optional[float], w: int, h: int, route: str) -> bool:
    if focal_px is None or not math.isfinite(focal_px) or focal_px <= 0.0:
        return False
    fov = _fov_deg(focal_px, w, h)
    if not (_MIN_FOV_DEG <= fov <= _MAX_FOV_DEG):
        logger.debug("  EXIF route %s rejected: f=%.1f px implies %.1f° FoV",
                     route, focal_px, fov)
        return False
    return True


def read_exif(path) -> Dict[int, object]:
    """Return the raw numeric EXIF dict, or an empty dict."""
    try:
        from PIL import Image as _PIL
        with _PIL.open(str(path)) as img:
            exif = img._getexif()
        return dict(exif) if exif else {}
    except Exception:
        return {}


def focal_from_exif(path, image_shape) -> Optional[FocalEstimate]:
    """
    Focal length in pixels for one image, or None when EXIF says nothing.

    Parameters
    ----------
    path        : image file path.
    image_shape : (h, w, ...) of the image *as loaded*, which may differ from
                  the EXIF pixel dimensions after downscaling or rotation.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    exif = read_exif(path)
    if not exif:
        return None

    # ── Route 1: 35 mm equivalent ─────────────────────────────────────────
    f35 = _to_float(exif.get(TAG_FOCAL_35))
    if f35:
        diag_px = math.sqrt(float(w) ** 2 + float(h) ** 2)
        focal = f35 / SENSOR_DIAG_35MM * diag_px
        if _plausible(focal, w, h, "35mm-equivalent"):
            return FocalEstimate(focal, "exif_focal35",
                                 f"FocalLengthIn35mmFilm = {f35:.1f} mm")

    f_mm = _to_float(exif.get(TAG_FOCAL_LENGTH))

    # ── Route 2: focal-plane resolution ───────────────────────────────────
    if f_mm:
        fp_res = _to_float(exif.get(TAG_FOCAL_PLANE_X_RES))
        unit = exif.get(TAG_FOCAL_PLANE_UNIT)
        unit_mm = _UNIT_MM.get(int(unit)) if unit is not None else None
        if fp_res and unit_mm:
            # Pixels per millimetre on the sensor, as recorded by the camera.
            px_per_mm = fp_res / unit_mm
            exif_w = _to_float(exif.get(TAG_EXIF_IMAGE_WIDTH)) or float(w)
            # Rescale when the loaded image is not the recorded size.
            focal = f_mm * px_per_mm * (float(w) / exif_w)
            if _plausible(focal, w, h, "focal-plane"):
                return FocalEstimate(
                    focal, "exif_focal_plane",
                    f"FocalLength = {f_mm:.2f} mm, "
                    f"{px_per_mm:.1f} px/mm on the sensor")

    # ── Route 3: sensor width from Make/Model ─────────────────────────────
    if f_mm:
        make = str(exif.get(TAG_MAKE, "")).strip().lower()
        model = str(exif.get(TAG_MODEL, "")).strip().lower()
        sensor_w = SENSOR_WIDTH_MM.get((make, model))
        if sensor_w is None and make:
            # Model strings often repeat the make; try the bare model too.
            for (mk, md), value in SENSOR_WIDTH_MM.items():
                if md == model and mk in make:
                    sensor_w = value
                    break
        if sensor_w:
            focal = f_mm / sensor_w * float(w)
            if _plausible(focal, w, h, "sensor-table"):
                return FocalEstimate(
                    focal, "exif_sensor_table",
                    f"FocalLength = {f_mm:.2f} mm, sensor width "
                    f"{sensor_w:.2f} mm ({make} {model})")

    return None


def summarise(estimates: Dict[int, Optional[FocalEstimate]]) -> str:
    """One log line describing which route produced how many focals."""
    counts: Dict[str, int] = {}
    for est in estimates.values():
        key = est.source if est is not None else "none"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
