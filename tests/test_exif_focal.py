"""Tests for the EXIF focal-length routes.

The Buddha dataset carries no EXIF at all, so none of these paths is exercised
by the measured runs. They are the fix for the first finding of
`paper/paper_l.md` and therefore need their own evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sfm.exif_focal import (SENSOR_DIAG_35MM, FocalEstimate, focal_from_exif,
                            summarise)


@pytest.fixture()
def image_factory(tmp_path):
    """Write a small JPEG carrying the requested EXIF tags."""
    from PIL import Image

    def make(name: str, tags: dict, size=(4000, 3000)):
        path = tmp_path / name
        Image.new("RGB", size, (128, 128, 128)).save(path, "JPEG")
        if tags:
            import piexif  # noqa: F401  (optional; skipped when absent)
        return path, size

    return make


def _write_exif(path, tags: dict) -> bool:
    """Attach EXIF via piexif when available; return False when it is not."""
    try:
        import piexif
    except ImportError:
        return False
    exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    for tag, value in tags.items():
        if tag in (piexif.ImageIFD.Make, piexif.ImageIFD.Model):
            exif["0th"][tag] = value
        else:
            exif["Exif"][tag] = value
    piexif.insert(piexif.dump(exif), str(path))
    return True


def test_no_exif_returns_none(tmp_path):
    from PIL import Image

    path = tmp_path / "plain.png"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(path)
    assert focal_from_exif(path, (480, 640, 3)) is None


def test_focal35_route(tmp_path):
    """Route 1: the camera already states the 35 mm equivalent."""
    piexif = pytest.importorskip("piexif")
    from PIL import Image

    path = tmp_path / "f35.jpg"
    w, h = 4000, 3000
    Image.new("RGB", (w, h), (128, 128, 128)).save(path, "JPEG")
    assert _write_exif(path, {piexif.ExifIFD.FocalLengthIn35mmFilm: 50})

    est = focal_from_exif(path, (h, w, 3))
    assert est is not None
    assert est.source == "exif_focal35"
    expected = 50.0 / SENSOR_DIAG_35MM * math.hypot(w, h)
    assert est.focal_px == pytest.approx(expected, rel=1e-9)


def test_focal_plane_route(tmp_path):
    """Route 2: focal length in mm plus the sensor's pixel pitch."""
    piexif = pytest.importorskip("piexif")
    from PIL import Image

    path = tmp_path / "fp.jpg"
    w, h = 4000, 3000
    Image.new("RGB", (w, h), (128, 128, 128)).save(path, "JPEG")
    # 4000 px across a 23.5 mm sensor -> 170.2 px/mm, stated per inch.
    px_per_mm = w / 23.5
    per_inch = int(round(px_per_mm * 25.4))
    ok = _write_exif(path, {
        piexif.ExifIFD.FocalLength: (35, 1),
        piexif.ExifIFD.FocalPlaneXResolution: (per_inch, 1),
        piexif.ExifIFD.FocalPlaneResolutionUnit: 2,     # inch
        piexif.ExifIFD.PixelXDimension: w,
    })
    assert ok

    est = focal_from_exif(path, (h, w, 3))
    assert est is not None
    assert est.source == "exif_focal_plane"
    assert est.focal_px == pytest.approx(35.0 * px_per_mm, rel=2e-3)


def test_sensor_table_route(tmp_path):
    """Route 3: focal length in mm plus a known body."""
    piexif = pytest.importorskip("piexif")
    from PIL import Image

    path = tmp_path / "table.jpg"
    w, h = 6000, 4000
    Image.new("RGB", (w, h), (128, 128, 128)).save(path, "JPEG")
    ok = _write_exif(path, {
        piexif.ImageIFD.Make: b"SONY",
        piexif.ImageIFD.Model: b"ILCE-6000",
        piexif.ExifIFD.FocalLength: (30, 1),
    })
    assert ok

    est = focal_from_exif(path, (h, w, 3))
    assert est is not None
    assert est.source == "exif_sensor_table"
    assert est.focal_px == pytest.approx(30.0 / 23.5 * w, rel=1e-6)


def test_implausible_focal_is_rejected(tmp_path):
    """A focal implying a 1° field of view is a broken tag, not a calibration."""
    piexif = pytest.importorskip("piexif")
    from PIL import Image

    path = tmp_path / "bad.jpg"
    w, h = 4000, 3000
    Image.new("RGB", (w, h), (128, 128, 128)).save(path, "JPEG")
    assert _write_exif(path, {piexif.ExifIFD.FocalLengthIn35mmFilm: 5000})
    assert focal_from_exif(path, (h, w, 3)) is None


def test_summarise_counts_routes():
    estimates = {
        0: FocalEstimate(1000.0, "exif_focal35"),
        1: FocalEstimate(1000.0, "exif_focal35"),
        2: None,
    }
    text = summarise(estimates)
    assert "exif_focal35: 2" in text
    assert "none: 1" in text
