"""Typed job and per-file result structures."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Status(str, Enum):
    PENDING = "pending"            # transient, UI only
    RUNNING = "running"            # transient, UI only
    PROCESSED = "processed"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"
    UNSUPPORTED_SKIPPED = "unsupported_skipped"


FINAL_STATUSES = (Status.PROCESSED, Status.REVIEW_REQUIRED, Status.FAILED, Status.UNSUPPORTED_SKIPPED)


class JobState(str, Enum):
    STAGED = "staged"        # uploaded and validated, waiting for the user to start
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"  # ZIP ready
    FAILED = "failed"        # job-level failure, no ZIP


@dataclass
class StagedInput:
    index: int
    rel_path: str                       # sanitized, collision-free relative path
    stored_path: Optional[Path]         # where the uploaded bytes live (None if not stored)
    size: int
    supported: bool
    note: str = ""


@dataclass
class FileResult:
    index: int
    source_path: str
    supported: bool
    status: Status = Status.PENDING
    output_path: str = ""               # path inside the ZIP, e.g. aligned/a/b.jpg
    output_kind: str = "none"           # aligned | original_copy | none
    estimator: str = ""
    angle_deg: Optional[float] = None   # estimated skew; + = content rotated counter-clockwise
    perspective_action: str = ""        # applied | skipped | rejected
    perspective_reason: str = ""
    correction_applied: str = ""        # none | rotation | perspective | perspective+rotation
    elapsed_s: float = 0.0
    note: str = ""
    sha256_input: str = ""

    def to_public(self) -> dict:
        return {
            "index": self.index, "path": self.source_path, "status": self.status.value,
            "output_path": self.output_path, "note": self.note,
            "correction": self.correction_applied, "angle_deg": self.angle_deg,
        }


@dataclass
class Job:
    id: str
    dir: Path
    mode: str                           # folder | zip
    source_name: str
    created_at: float
    expires_at: float
    state: JobState = JobState.STAGED
    settings: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    files: list = field(default_factory=list)
    zip_path: Optional[Path] = None
    error: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    deadline: float = 0.0               # time.monotonic() value
    lock: threading.RLock = field(default_factory=threading.RLock)
