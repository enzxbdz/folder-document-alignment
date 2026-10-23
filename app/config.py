"""Limits, retention and processing defaults. Everything is overridable via DOCALIGN_* env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SKEW_METHODS = ("projection_profile", "min_area_rect", "consensus")
# PROVISIONAL: no development evidence has selected an estimator yet. Until it does,
# this default is a placeholder and the UI/README/audit label it that way.
DEFAULT_SKEW_METHOD = "projection_profile"
SKEW_DEFAULT_IS_PROVISIONAL = True


@dataclass(frozen=True)
class ProcessingConfig:
    """Deterministic parameters for the CV pipeline (no I/O, no HTTP)."""

    skew_method: str = DEFAULT_SKEW_METHOD
    analysis_max_side: int = 1000          # px; analysis runs on a downscaled copy
    max_skew_deg: float = 15.0             # estimates beyond this are out of range
    min_correction_deg: float = 0.3        # smaller estimates are treated as "already aligned"
    consensus_tolerance_deg: float = 1.0   # consensus mode: max disagreement between estimators
    min_page_area_ratio: float = 0.30      # page quad must cover at least this share of the frame
    max_page_area_ratio: float = 0.97      # a quad above this is treated as the frame itself
    min_corner_angle_deg: float = 50.0
    max_corner_angle_deg: float = 130.0
    min_aspect: float = 0.25               # rectified width/height bounds
    max_aspect: float = 4.0
    max_edge_density: float = 0.15         # above this the edge map is texture/noise, not an outline
    min_edge_support: float = 0.50         # share of the quad outline backed by detected edges
    ambiguity_area_ratio: float = 0.80     # a 2nd valid quad this close in area => ambiguous
    min_foreground_ratio: float = 0.002    # below: near-empty
    max_foreground_ratio: float = 0.50     # above: no distinct content
    border_margin_ratio: float = 0.03      # ignored when estimating skew (shadows, warp edges)
    min_peak_prominence: float = 0.10      # projection profile: (best - median) / best
    jpeg_quality: int = 95
    max_input_pixels: int = 60_000_000
    max_output_pixels: int = 120_000_000
    min_output_side: int = 100


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    work_dir: Path = Path(".work")
    max_files: int = 300                   # supported images per job (must be >= 50)
    max_client_entries: int = 2000         # total entries (incl. unsupported) listed per job
    max_total_bytes: int = 500 * 1024 * 1024
    max_file_bytes: int = 25 * 1024 * 1024
    max_job_seconds: int = 1800
    max_concurrent_jobs: int = 1
    max_live_jobs: int = 10
    workers: int = 2
    retention_minutes: int = 30
    zip_max_ratio: int = 200               # reject archives with a larger per-entry compression ratio
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        mb = 1024 * 1024
        method = os.environ.get("DOCALIGN_SKEW_METHOD", DEFAULT_SKEW_METHOD)
        if method not in SKEW_METHODS:
            method = DEFAULT_SKEW_METHOD
        proc = ProcessingConfig(
            skew_method=method,
            max_input_pixels=_env_int("DOCALIGN_MAX_PIXELS", ProcessingConfig.max_input_pixels),
        )
        return cls(
            work_dir=Path(os.environ.get("DOCALIGN_WORK_DIR", ".work")),
            max_files=max(1, _env_int("DOCALIGN_MAX_FILES", cls.max_files)),
            max_total_bytes=_env_int("DOCALIGN_MAX_TOTAL_MB", 500) * mb,
            max_file_bytes=_env_int("DOCALIGN_MAX_FILE_MB", 25) * mb,
            max_job_seconds=_env_int("DOCALIGN_MAX_JOB_SECONDS", cls.max_job_seconds),
            max_concurrent_jobs=max(1, _env_int("DOCALIGN_MAX_CONCURRENT_JOBS", 1)),
            max_live_jobs=max(1, _env_int("DOCALIGN_MAX_LIVE_JOBS", cls.max_live_jobs)),
            workers=max(1, _env_int("DOCALIGN_WORKERS", cls.workers)),
            retention_minutes=max(1, _env_int("DOCALIGN_RETENTION_MINUTES", cls.retention_minutes)),
            processing=proc,
        )
