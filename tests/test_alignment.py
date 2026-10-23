import itertools
import math

import cv2
import numpy as np
import pytest

from app.config import ProcessingConfig
from app.services import image_alignment as ia
from scripts.make_synthetic_samples import (blank_image, encode, make_text_page, noise_image, place_on_desk,
                                            rotate_ccw, two_pages_scene)

CFG = ProcessingConfig()
DESK_QUAD = [(260, 230), (1180, 300), (1230, 1560), (200, 1480)]


def tilt_ccw(img) -> float:
    """Independent check: principal axis of dark pixels, folded to [-45, 45), + = counter-clockwise."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(g < 100)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = vecs[:, int(np.argmax(vals))]
    theta_cw = math.degrees(math.atan2(vy, vx))
    return ((-theta_cw + 45) % 90) - 45


@pytest.mark.parametrize("method", ["projection_profile", "min_area_rect", "consensus"])
@pytest.mark.parametrize("angle", [-9.0, -4.0, 4.0, 9.0])
def test_sign_convention_and_correction(method, angle):
    cfg = ProcessingConfig(skew_method=method)
    src = rotate_ccw(make_text_page(), angle)
    assert tilt_ccw(src) == pytest.approx(angle, abs=0.6)             # the fixture really is rotated that way
    est = ia.estimate_skew(src, cfg)
    assert est.angle_deg == pytest.approx(angle, abs=0.6)             # + = counter-clockwise content
    out = ia.align_image(encode(src), cfg)
    assert out.status == "processed" and out.correction_applied == "rotation"
    fixed = cv2.imdecode(np.frombuffer(out.output_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert abs(tilt_ccw(fixed)) < 0.7                                  # moved to upright, not away from it


@pytest.mark.parametrize("angle", [0.0, 0.1, -0.2])
def test_near_zero_skew_is_not_rotated_and_not_marked_corrected(angle):
    data = encode(rotate_ccw(make_text_page(), angle))
    out = ia.align_image(data, CFG)
    assert out.status == "processed"
    assert out.correction_applied == "none" and out.unchanged_copy and out.output_bytes == data


def test_rotation_does_not_crop_content():
    src = rotate_ccw(make_text_page(), 9.0)
    out = ia.align_image(encode(src), CFG)
    fixed = cv2.imdecode(np.frombuffer(out.output_bytes, np.uint8), cv2.IMREAD_COLOR)
    before, after = (src.min(axis=2) < 100).sum(), (fixed.min(axis=2) < 100).sum()
    assert 0.9 < after / before < 1.1
    assert fixed.shape[0] >= src.shape[0] * 0.95


@pytest.mark.parametrize("perm", list(itertools.permutations(range(4))))
def test_corner_ordering_is_deterministic(perm):
    rect = np.array([[10, 20], [300, 30], [310, 400], [5, 390]], float)   # tl, tr, br, bl
    got = ia.order_corners(rect[list(perm)])
    assert np.array_equal(got, rect)


def test_warp_maps_corners_to_corners():
    quad = np.array([[200, 150], [900, 210], [940, 1100], [160, 1050]], float)
    img = np.full((1300, 1100, 3), 128, np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), col in zip(quad, colors):
        cv2.circle(img, (int(x), int(y)), 9, col, -1)
    out = ia.warp_to_rectangle(img, ia.order_corners(quad[::-1]), CFG)    # input order must not matter
    h, w = out.shape[:2]
    for (x, y), col in zip([(5, 5), (w - 6, 5), (w - 6, h - 6), (5, h - 6)], colors):
        assert np.abs(out[y, x].astype(int) - col).sum() < 120
    exp_w = max(math.dist(quad[0], quad[1]), math.dist(quad[3], quad[2]))
    exp_h = max(math.dist(quad[0], quad[3]), math.dist(quad[1], quad[2]))
    assert abs(w - exp_w) <= 1 and abs(h - exp_h) <= 1


def test_perspective_page_on_desk_is_rectified():
    scene = place_on_desk(make_text_page(), DESK_QUAD)
    out = ia.align_image(encode(scene), CFG)
    assert out.status == "processed" and out.perspective_action == "applied"
    assert "perspective" in out.correction_applied
    w, h = out.output_size
    assert abs(w - 1033) <= 25 and abs(h - 1261) <= 25                 # sizes implied by the known quad
    fixed = cv2.imdecode(np.frombuffer(out.output_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert abs(tilt_ccw(fixed)) < 1.0


def test_ambiguous_geometry_goes_to_review_not_corrected():
    out = ia.align_image(encode(two_pages_scene(make_text_page())), CFG)
    assert out.status == "review_required"
    assert out.perspective_action == "rejected" and out.output_bytes is None


@pytest.mark.parametrize("img", [blank_image(), noise_image()], ids=["blank", "noise"])
def test_empty_or_noisy_images_need_review(img):
    out = ia.align_image(encode(img), CFG)
    assert out.status == "review_required" and out.output_bytes is None


def test_malformed_and_limits():
    assert ia.align_image(b"not an image", CFG).status == "failed"
    good = encode(make_text_page())
    assert ia.align_image(good[:25], CFG).status == "failed"
    assert ia.align_image(good, ProcessingConfig(max_input_pixels=1000)).status == "unsupported_skipped"
    png_header_lie = bytearray(encode(make_text_page(), ".png"))
    png_header_lie[16:24] = (60000).to_bytes(4, "big") * 2           # claims 60000x60000, never decoded
    assert ia.align_image(bytes(png_header_lie), CFG).status == "unsupported_skipped"


def test_processing_is_deterministic():
    data = encode(rotate_ccw(make_text_page(), 6.0))
    assert ia.align_image(data, CFG).output_bytes == ia.align_image(data, CFG).output_bytes
