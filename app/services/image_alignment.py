"""Classical-CV alignment pipeline (OpenCV + NumPy). No HTTP and no filesystem access.

Sign convention (used everywhere, including the audit CSV):
    angle_deg > 0  means the page content is rotated COUNTER-CLOCKWISE (as viewed) from upright.
    The correction therefore rotates the image CLOCKWISE by angle_deg.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import ProcessingConfig

PROCESSED, REVIEW, FAILED, UNSUPPORTED = "processed", "review_required", "failed", "unsupported_skipped"


@dataclass
class AlignmentOutcome:
    status: str
    output_bytes: Optional[bytes] = None
    output_ext: str = ""
    unchanged_copy: bool = False          # True: caller must write the original bytes unchanged
    estimator: str = ""
    angle_deg: Optional[float] = None
    perspective_action: str = ""          # applied | skipped | rejected
    perspective_reason: str = ""
    correction_applied: str = "none"      # none | rotation | perspective | perspective+rotation
    note: str = ""
    output_size: Optional[tuple] = None   # (width, height)


@dataclass
class QuadDetection:
    action: str                           # applied | skipped | rejected
    reason: str
    quad: Optional[np.ndarray] = None     # 4x2 float64, ordered tl,tr,br,bl, full-resolution coords


@dataclass
class SkewEstimate:
    method: str
    angle_deg: Optional[float]            # None => rejected
    reason: str = ""


# ---------- format sniffing (never trust extensions or client MIME) ----------

def sniff_format(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    return None


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def read_dimensions(data: bytes, fmt: str) -> Optional[tuple]:
    """Read (width, height) from the header without decoding pixels (decompression-bomb guard)."""
    if fmt == "png":
        if len(data) >= 24 and data[12:16] == b"IHDR":
            return struct.unpack(">II", data[16:24])
        return None
    i, n = 2, len(data)
    while i + 9 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
        elif marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
        elif marker in (0xD9, 0xDA):
            return None
        else:
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in _SOF:
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + max(seglen, 2)
    return None


# ---------- geometry ----------

def order_corners(pts: np.ndarray) -> Optional[np.ndarray]:
    """Deterministic tl, tr, br, bl ordering. Ties broken by x, then y. None if degenerate."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s, d = p.sum(axis=1), p[:, 1] - p[:, 0]
    tl = int(np.lexsort((p[:, 1], p[:, 0], s))[0])
    br = int(np.lexsort((-p[:, 1], -p[:, 0], -s))[0])
    tr = int(np.lexsort((p[:, 1], -p[:, 0], d))[0])
    bl = int(np.lexsort((-p[:, 1], p[:, 0], -d))[0])
    idx = [tl, tr, br, bl]
    return p[idx] if len(set(idx)) == 4 else None


def _dist(a, b) -> float:
    return float(np.hypot(*(np.asarray(a) - np.asarray(b))))


def _rect_size(q: np.ndarray) -> tuple:
    tl, tr, br, bl = q
    return max(_dist(tl, tr), _dist(bl, br)), max(_dist(tl, bl), _dist(tr, br))


def _quad_problem(q: np.ndarray, cfg: ProcessingConfig) -> Optional[str]:
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return "not convex"
    for i in range(4):
        v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return "degenerate corner"
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1, 1))))
        if not cfg.min_corner_angle_deg <= ang <= cfg.max_corner_angle_deg:
            return f"corner angle {ang:.0f} deg out of range"
    w, h = _rect_size(q)
    if h <= 0 or not cfg.min_aspect <= w / h <= cfg.max_aspect:
        return "implausible page aspect ratio"
    return None


def _edge_support(edges: np.ndarray, quad: np.ndarray) -> float:
    """Share of a 7 px band around the quad outline that coincides with detected edges."""
    band = np.zeros(edges.shape, np.uint8)
    cv2.polylines(band, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], True, 255, 7)
    near_edge = cv2.dilate(edges, np.ones((9, 9), np.uint8))   # tolerate a few px of polygon approximation error
    n = np.count_nonzero(band)
    return float(np.count_nonzero(cv2.bitwise_and(near_edge, band))) / n if n else 0.0


def _approx_quad(contour) -> Optional[np.ndarray]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
    return None


def detect_page_quad(img: np.ndarray, cfg: ProcessingConfig) -> QuadDetection:
    h, w = img.shape[:2]
    scale = min(1.0, cfg.analysis_max_side / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(20, 0.66 * med)), int(max(60, min(255, 1.33 * med))))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    density = float(np.count_nonzero(edges)) / edges.size
    if density > cfg.max_edge_density:
        return QuadDetection("skipped", f"edge map too dense to find a page outline ({density:.0%})")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = float(sw * sh)
    valid, invalid, frame_fill = [], [], False
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if area < cfg.min_page_area_ratio * frame:
            break
        quad = _approx_quad(c)
        if quad is None:
            continue
        if area / frame > cfg.max_page_area_ratio:
            frame_fill = True
            continue
        ordered = order_corners(quad)
        problem = "ambiguous corner order" if ordered is None else _quad_problem(ordered, cfg)
        if problem is None and _edge_support(edges, ordered) < cfg.min_edge_support:
            problem = "outline is not supported by image edges"
        (invalid if problem else valid).append((area, ordered, problem))
    if not valid:
        if invalid:
            return QuadDetection("rejected", f"candidate outline failed checks: {invalid[0][2]}")
        if frame_fill:
            return QuadDetection("skipped", "outline spans the whole frame; no separate page edge")
        return QuadDetection("skipped", "no large four-corner outline found")
    valid.sort(key=lambda t: -t[0])
    if len(valid) > 1 and valid[1][0] >= cfg.ambiguity_area_ratio * valid[0][0]:
        return QuadDetection("rejected", "ambiguous: two competing page outlines")
    quad = valid[0][1] * np.array([w / sw, h / sh])
    return QuadDetection("applied", "page outline passed geometry checks", quad)


def warp_to_rectangle(img: np.ndarray, quad: np.ndarray, cfg: ProcessingConfig) -> np.ndarray:
    wf, hf = _rect_size(quad)
    w, h = int(round(wf)), int(round(hf))
    if min(w, h) < cfg.min_output_side or w * h > cfg.max_output_pixels:
        raise ValueError("rectified size outside allowed range")
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _background_color(img: np.ndarray) -> tuple:
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).reshape(-1, img.shape[2])
    return tuple(float(v) for v in np.median(border, axis=0))


def rotate_expand(img: np.ndarray, ccw_deg: float, cfg: ProcessingConfig) -> np.ndarray:
    """Rotate by ccw_deg (positive = counter-clockwise) on an enlarged canvas so nothing is cropped."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ccw_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)), int(math.ceil(h * cos + w * sin))
    if nw * nh > cfg.max_output_pixels:
        raise ValueError("rotated size outside allowed range")
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=_background_color(img))


# ---------- skew estimation ----------

def _foreground(img: np.ndarray, cfg: ProcessingConfig):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, cfg.analysis_max_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    m = int(round(min(h, w) * cfg.border_margin_ratio))
    if m > 0 and h > 2 * m + 10 and w > 2 * m + 10:
        gray = gray[m:h - m, m:w - m]
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ratio = float(np.count_nonzero(mask)) / mask.size
    if ratio > 0.5:                      # content is the minority class
        mask, ratio = 255 - mask, 1.0 - ratio
    return mask, ratio


def _in_range(method: str, angle: float, cfg: ProcessingConfig) -> SkewEstimate:
    if abs(angle) > cfg.max_skew_deg:
        return SkewEstimate(method, None, f"estimate {angle:.1f} deg is outside +/-{cfg.max_skew_deg:g} deg")
    return SkewEstimate(method, round(angle, 2))


def _skew_min_area_rect(mask: np.ndarray, cfg: ProcessingConfig) -> SkewEstimate:
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 200:
        return SkewEstimate("min_area_rect", None, "too little foreground")
    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), _ = rect
    if rw * rh >= 0.98 * mask.shape[0] * mask.shape[1]:
        return SkewEstimate("min_area_rect", None, "content spans the whole frame; no distinct boundary")
    box = cv2.boxPoints(rect)
    e = box[1] - box[0]
    theta_cw = math.degrees(math.atan2(e[1], e[0]))      # y-down image coords: + is clockwise
    ccw = ((-theta_cw + 45.0) % 90.0) - 45.0             # fold into [-45, 45)
    return _in_range("min_area_rect", ccw, cfg)


def _skew_projection_profile(mask: np.ndarray, cfg: ProcessingConfig) -> SkewEstimate:
    h, w = mask.shape
    s = min(1.0, 600 / max(h, w))
    small = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else mask
    sh, sw = small.shape
    c = (sw / 2, sh / 2)

    def score(a: float) -> float:
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D(c, a, 1.0), (sw, sh), flags=cv2.INTER_LINEAR)
        return float(r.sum(axis=1, dtype=np.float64).var())

    coarse = np.arange(-cfg.max_skew_deg, cfg.max_skew_deg + 1e-9, 1.0)
    cs = np.array([score(a) for a in coarse])
    best_c = float(coarse[int(np.argmax(cs))])
    fine = np.arange(best_c - 1.0, best_c + 1.0 + 1e-9, 0.1)
    fs = np.array([score(a) for a in fine])
    best_rot, best = float(fine[int(np.argmax(fs))]), float(fs.max())
    if best <= 0:
        return SkewEstimate("projection_profile", None, "empty projection")
    prominence = (best - float(np.median(cs))) / best
    if prominence < cfg.min_peak_prominence:
        return SkewEstimate("projection_profile", None, f"no clear line structure (peak prominence {prominence:.2f})")
    if abs(best_rot) >= cfg.max_skew_deg - 0.05:
        return SkewEstimate("projection_profile", None, "peak at the edge of the search range")
    return _in_range("projection_profile", -best_rot, cfg)   # rotating by best_rot aligns => skew = -best_rot


def estimate_skew(img: np.ndarray, cfg: ProcessingConfig) -> SkewEstimate:
    mask, ratio = _foreground(img, cfg)
    if ratio < cfg.min_foreground_ratio:
        return SkewEstimate(cfg.skew_method, None, "near-empty page (almost no foreground)")
    if ratio > cfg.max_foreground_ratio:
        return SkewEstimate(cfg.skew_method, None, "no distinct foreground")
    if cfg.skew_method == "min_area_rect":
        return _skew_min_area_rect(mask, cfg)
    if cfg.skew_method == "projection_profile":
        return _skew_projection_profile(mask, cfg)
    a, b = _skew_min_area_rect(mask, cfg), _skew_projection_profile(mask, cfg)   # consensus
    if a.angle_deg is None or b.angle_deg is None:
        return SkewEstimate("consensus", None, f"estimator rejected ({a.reason or 'ok'}; {b.reason or 'ok'})")
    if abs(a.angle_deg - b.angle_deg) > cfg.consensus_tolerance_deg:
        return SkewEstimate("consensus", None, f"estimators disagree ({a.angle_deg:.2f} vs {b.angle_deg:.2f} deg)")
    return SkewEstimate("consensus", round((a.angle_deg + b.angle_deg) / 2, 2))


# ---------- full pipeline ----------

def align_image(data: bytes, cfg: ProcessingConfig) -> AlignmentOutcome:
    """Decode, optionally rectify, optionally de-skew, encode. Never raises."""
    try:
        return _align(data, cfg)
    except Exception as exc:  # noqa: BLE001 - one bad image must not abort a batch
        return AlignmentOutcome(FAILED, note=f"Unexpected processing error ({type(exc).__name__}).")


def _align(data: bytes, cfg: ProcessingConfig) -> AlignmentOutcome:
    fmt = sniff_format(data)
    if fmt is None:
        return AlignmentOutcome(FAILED, note="Not a valid JPEG or PNG (file signature mismatch).")
    dims = read_dimensions(data, fmt)
    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        return AlignmentOutcome(FAILED, note="Image header could not be read.")
    if dims[0] * dims[1] > cfg.max_input_pixels:
        return AlignmentOutcome(UNSUPPORTED, note=f"Image is {dims[0]}x{dims[1]} px, above the {cfg.max_input_pixels} pixel limit.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # EXIF orientation applied by decoder
    if img is None or img.size == 0:
        return AlignmentOutcome(FAILED, note="Image could not be decoded (corrupt or truncated).")

    ext = ".png" if fmt == "png" else ".jpg"
    quad = detect_page_quad(img, cfg)
    out = AlignmentOutcome(PROCESSED, output_ext=ext, perspective_action=quad.action,
                           perspective_reason=quad.reason, estimator=cfg.skew_method)
    if quad.action == "rejected":
        out.status = REVIEW
        out.note = "Page outline was ambiguous or implausible; original kept for review."
        return out

    work, perspective = img, False
    if quad.action == "applied":
        try:
            work, perspective = warp_to_rectangle(img, quad.quad, cfg), True
        except ValueError as exc:
            out.status, out.perspective_action, out.perspective_reason = REVIEW, "rejected", str(exc)
            out.note = "Rectification was not possible; original kept for review."
            return out

    est = estimate_skew(work, cfg)
    out.estimator, out.angle_deg = est.method, est.angle_deg
    rotated = False
    if est.angle_deg is None:
        if not perspective:
            out.status = REVIEW
            out.note = f"No usable perspective or skew evidence ({est.reason}); original kept for review."
            return out
        out.note = f"Perspective corrected; residual skew not corrected ({est.reason})."
    elif abs(est.angle_deg) >= cfg.min_correction_deg:
        work, rotated = rotate_expand(work, -est.angle_deg, cfg), True

    out.correction_applied = "+".join(n for n, on in (("perspective", perspective), ("rotation", rotated)) if on) or "none"
    if out.correction_applied == "none":
        out.unchanged_copy, out.output_bytes = True, data
        out.output_size = (img.shape[1], img.shape[0])
        out.note = f"Already aligned (|skew| < {cfg.min_correction_deg:g} deg); output is an unchanged copy of the input."
        return out

    if work.size == 0 or min(work.shape[:2]) < 1 or work.shape[0] * work.shape[1] > cfg.max_output_pixels:
        out.status, out.note = FAILED, "Output dimensions were invalid."
        return out
    params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, work, params)
    if not ok or buf.size == 0:
        out.status, out.note = FAILED, "Output image could not be encoded."
        return out
    out.output_bytes = buf.tobytes()
    out.output_size = (work.shape[1], work.shape[0])
    return out
