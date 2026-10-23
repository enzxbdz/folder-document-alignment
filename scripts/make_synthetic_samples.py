"""Create SYNTHETIC geometric test pages with known transforms. Not real photographs.

Usage:  python -m scripts.make_synthetic_samples [output_dir] [count]
Default output: sample_data/synthetic/ (git-ignored). Each file name records the applied transform.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


def make_text_page(w: int = 900, h: int = 1200, seed: int = 0) -> np.ndarray:
    """White page with dark horizontal bars standing in for text lines (deterministic)."""
    rng = np.random.default_rng(seed)
    page = np.full((h, w, 3), 245, np.uint8)
    y, margin = int(h * 0.08), int(w * 0.10)
    while y < h * 0.92:
        x = margin
        while x < w - margin - 40:
            seg = int(rng.integers(40, 160))
            x2 = min(x + seg, w - margin)
            cv2.rectangle(page, (x, y), (x2, y + 9), (25, 25, 25), -1)
            x = x2 + int(rng.integers(8, 20))
        y += 34
    return page


def rotate_ccw(img: np.ndarray, deg: float, bg=(245, 245, 245)) -> np.ndarray:
    """Independent reference rotation, positive = counter-clockwise, enlarged canvas."""
    h, w = img.shape[:2]
    t = math.radians(abs(deg))
    nw, nh = int(math.ceil(w * math.cos(t) + h * math.sin(t))), int(math.ceil(h * math.cos(t) + w * math.sin(t)))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def place_on_desk(page: np.ndarray, dst_quad, size=(1400, 1800), desk=(40, 55, 70)) -> np.ndarray:
    """Project the page onto a darker 'desk' so its four corners land on dst_quad (tl,tr,br,bl)."""
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    m = cv2.getPerspectiveTransform(src, np.float32(dst_quad))
    canvas = np.full((size[1], size[0], 3), desk, np.uint8)
    warped = cv2.warpPerspective(page, m, size, borderMode=cv2.BORDER_TRANSPARENT, dst=canvas)
    return warped


def two_pages_scene(page: np.ndarray, size=(1600, 1000), desk=(40, 55, 70)) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), desk, np.uint8)
    small = cv2.resize(page, (620, 800))
    canvas[100:900, 80:700] = small
    canvas[100:900, 860:1480] = small
    return canvas


def noise_image(w: int = 800, h: int = 1000, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def blank_image(w: int = 800, h: int = 1000) -> np.ndarray:
    return np.full((h, w, 3), 250, np.uint8)


def encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def main(out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = [-12, -6, -3, 0, 2, 5, 9, 14]
    for i in range(count):
        a = angles[i % len(angles)]
        page = rotate_ccw(make_text_page(seed=i), a)
        (out_dir / f"synthetic_{i:03d}_rot{a:+d}.jpg").write_bytes(encode(page))
    quad = [(260, 230), (1180, 300), (1230, 1560), (200, 1480)]
    (out_dir / "synthetic_perspective_desk.jpg").write_bytes(encode(place_on_desk(make_text_page(), quad)))
    (out_dir / "synthetic_two_pages_ambiguous.jpg").write_bytes(encode(two_pages_scene(make_text_page())))
    (out_dir / "synthetic_blank.png").write_bytes(encode(blank_image(), ".png"))
    print(f"Wrote {count + 3} SYNTHETIC images to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_data/synthetic"),
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
