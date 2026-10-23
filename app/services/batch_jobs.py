"""Job lifecycle, bounded concurrent processing, retention. In-memory store (single-process demo)."""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_SKEW_METHOD, SKEW_DEFAULT_IS_PROVISIONAL, SKEW_METHODS, ProcessingConfig, Settings
from ..models import FINAL_STATUSES, FileResult, Job, JobState, StagedInput, Status
from . import archive
from .image_alignment import FAILED, PROCESSED, REVIEW, UNSUPPORTED, align_image

log = logging.getLogger("docalign.jobs")


class JobError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class JobNotFound(JobError):
    status_code = 404


class JobStateError(JobError):
    status_code = 409


class JobBusy(JobError):
    status_code = 429


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, OSError):
        return "The output file could not be written."
    return f"Unexpected processing error ({type(exc).__name__})."


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs_root = settings.work_dir / "jobs"
        self._jobs: dict = {}
        self._lock = threading.RLock()

    # ----- lifecycle -----
    def startup(self) -> None:
        """Job state is in memory, so anything left on disk from an earlier run is orphaned: delete it."""
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        for child in self.jobs_root.iterdir():
            archive.remove_tree(child)

    def create_job(self, mode: str, source_name: str) -> Job:
        with self._lock:
            if len(self._jobs) >= self.settings.max_live_jobs:
                raise JobBusy("Too many stored jobs right now. Delete a finished job or try again after it expires.")
            job_id = secrets.token_urlsafe(24)
            jdir = self.jobs_root / job_id
            jdir.mkdir(parents=True, exist_ok=False)
            now = time.time()
            job = Job(job_id, jdir, mode, source_name, now, now + self.settings.retention_minutes * 60)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound("Job not found or expired.")
        return job

    def discard(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.id, None)
        archive.remove_tree(job.dir)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in (JobState.QUEUED, JobState.RUNNING):
            raise JobStateError("A running job cannot be deleted. Wait for it to finish.")
        self.discard(job)

    def sweep(self) -> int:
        now, removed = time.time(), 0
        with self._lock:
            stale = [j for j in self._jobs.values()
                     if j.state not in (JobState.QUEUED, JobState.RUNNING) and now > j.expires_at]
        for job in stale:
            self.discard(job)
            removed += 1
        return removed

    def finish_staging(self, job: Job, staged: list) -> None:
        job.inputs = list(staged)
        job.files = []
        for s in staged:
            fr = FileResult(s.index, s.rel_path, s.supported)
            if not s.supported:
                fr.status, fr.note = Status.UNSUPPORTED_SKIPPED, s.note
                fr.correction_applied = "none"
            job.files.append(fr)
        job.settings = self.settings_snapshot(self.settings.processing)
        job.state = JobState.STAGED

    def settings_snapshot(self, cfg: ProcessingConfig) -> dict:
        s = self.settings
        return {
            "skew_method": cfg.skew_method,
            "skew_method_is_provisional_default": cfg.skew_method == DEFAULT_SKEW_METHOD and SKEW_DEFAULT_IS_PROVISIONAL,
            "limits": {
                "max_files": s.max_files, "max_total_bytes": s.max_total_bytes, "max_file_bytes": s.max_file_bytes,
                "max_pixels": cfg.max_input_pixels, "max_job_seconds": s.max_job_seconds,
                "max_concurrent_jobs": s.max_concurrent_jobs, "workers": s.workers,
                "retention_minutes": s.retention_minutes,
            },
            "processing": asdict(cfg),
            "angle_convention": "angle_deg > 0: content rotated counter-clockwise; correction rotates clockwise",
        }

    # ----- running -----
    def start(self, job_id: str, skew_method: Optional[str] = None) -> Job:
        if skew_method is not None and skew_method not in SKEW_METHODS:
            raise JobError(f"Unknown skew method. Choose one of: {', '.join(SKEW_METHODS)}.")
        cfg = replace(self.settings.processing, skew_method=skew_method or self.settings.processing.skew_method)
        with self._lock:
            job = self.get(job_id)
            if job.state != JobState.STAGED:
                raise JobStateError("This job has already been started.")
            active = sum(1 for j in self._jobs.values() if j.state in (JobState.QUEUED, JobState.RUNNING))
            if active >= self.settings.max_concurrent_jobs:
                raise JobBusy("Another batch is processing. Try again when it finishes.")
            job.state = JobState.QUEUED
            job.settings = self.settings_snapshot(cfg)
        threading.Thread(target=self._run, args=(job, cfg), daemon=True, name=f"job-{job.id[:6]}").start()
        return job

    def _run(self, job: Job, cfg: ProcessingConfig) -> None:
        try:
            with job.lock:
                job.state, job.started_at = JobState.RUNNING, time.time()
            job.deadline = time.monotonic() + self.settings.max_job_seconds
            todo = [(fr, inp) for fr, inp in zip(job.files, job.inputs) if fr.status == Status.PENDING]
            workers = self.settings.workers
            it, pending = iter(todo), set()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="align") as ex:
                while True:
                    while len(pending) < workers * 2:        # bounded in-flight work => bounded memory
                        nxt = next(it, None)
                        if nxt is None:
                            break
                        pending.add(ex.submit(self._process_one, job, nxt[0], nxt[1], cfg))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
            zip_path = job.dir / "results.zip"
            archive.build_output_zip(job.dir / "out", job.files, job.settings, zip_path)
            for sub in ("input", "out", "upload.zip"):          # temporary working data is no longer needed
                p = job.dir / sub
                archive.remove_tree(p) if p.is_dir() else p.unlink(missing_ok=True)
            with job.lock:
                job.zip_path, job.state = zip_path, JobState.COMPLETED
                job.finished_at = time.time()
                job.expires_at = job.finished_at + self.settings.retention_minutes * 60
            c = self.counts(job)
            log.info("job %s finished: %s", job.id[:6], {k: c[k] for k in ("total", "processed", "review_required", "failed", "unsupported_skipped")})
        except Exception:  # noqa: BLE001
            log.exception("job %s failed", job.id[:6])
            with job.lock:
                job.state, job.error = JobState.FAILED, "The batch could not be completed because of an internal error."
                job.finished_at = time.time()
                job.expires_at = job.finished_at + self.settings.retention_minutes * 60
            archive.remove_tree(job.dir / "input")

    def _copy_original(self, out_root: Path, rel: str, data: bytes, sha: str) -> bool:
        """Write a verified copy of the original under review/. Returns success."""
        try:
            target = out_root / "review" / rel
            _atomic_write(target, data)
            return hashlib.sha256(target.read_bytes()).hexdigest() == sha
        except OSError:
            return False

    def _process_one(self, job: Job, fr: FileResult, inp: StagedInput, cfg: ProcessingConfig) -> None:
        t0 = time.perf_counter()
        out_root = job.dir / "out"
        with job.lock:
            fr.status = Status.RUNNING
        res = {"status": Status.FAILED, "output_path": "", "output_kind": "none", "note": "", "estimator": "",
               "angle_deg": None, "perspective_action": "", "perspective_reason": "", "correction_applied": ""}
        data, sha = b"", ""
        try:
            data = inp.stored_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if time.monotonic() > job.deadline:
                res["note"] = "Job time limit reached before this file was processed."
            else:
                out = align_image(data, cfg)
                res.update(estimator=out.estimator, angle_deg=out.angle_deg, perspective_action=out.perspective_action,
                           perspective_reason=out.perspective_reason, correction_applied=out.correction_applied, note=out.note)
                if out.status == PROCESSED:
                    rel = f"aligned/{fr.source_path}"
                    _atomic_write(out_root / rel, out.output_bytes)
                    if out.unchanged_copy and hashlib.sha256((out_root / rel).read_bytes()).hexdigest() != sha:
                        raise OSError("copy verification failed")
                    res.update(status=Status.PROCESSED, output_path=rel, output_kind="aligned")
                elif out.status == UNSUPPORTED:
                    res["status"] = Status.UNSUPPORTED_SKIPPED
                elif out.status == REVIEW:
                    res["status"] = Status.REVIEW_REQUIRED
                else:
                    res["status"] = Status.FAILED
        except Exception as exc:  # noqa: BLE001
            res.update(status=Status.FAILED, output_path="", output_kind="none", correction_applied="", note=_safe_error(exc))
        if res["status"] in (Status.REVIEW_REQUIRED, Status.FAILED) and data:
            if self._copy_original(out_root, fr.source_path, data, sha):
                res.update(output_path=f"review/{fr.source_path}", output_kind="original_copy")
            elif res["status"] == Status.FAILED:
                res["note"] = (res["note"] + " The original could not be copied.").strip()
            else:  # review needed but the retained original could not be written: report as a failure
                res.update(status=Status.FAILED, note="The original could not be copied to the review folder.")
        with job.lock:
            fr.sha256_input = sha
            fr.elapsed_s = time.perf_counter() - t0
            for k, v in res.items():
                setattr(fr, k, v)
            if fr.status == Status.FAILED:
                fr.correction_applied = fr.correction_applied or "none"

    # ----- reporting -----
    def counts(self, job: Job) -> dict:
        with job.lock:
            c = Counter(fr.status for fr in job.files)
            supported = sum(1 for fr in job.files if fr.supported)
            done_supported = sum(1 for fr in job.files if fr.supported and fr.status in FINAL_STATUSES)
        return {
            "total": len(job.files), "supported": supported, "unsupported": len(job.files) - supported,
            "processed": c[Status.PROCESSED], "review_required": c[Status.REVIEW_REQUIRED], "failed": c[Status.FAILED],
            "unsupported_skipped": c[Status.UNSUPPORTED_SKIPPED], "pending": c[Status.PENDING], "running": c[Status.RUNNING],
            "completed": done_supported,
        }

    def snapshot(self, job: Job) -> dict:
        with job.lock:
            files = [fr.to_public() for fr in job.files]
            state = job.state
        return {
            "id": job.id, "state": state.value, "mode": job.mode, "source_name": job.source_name,
            "created_at": job.created_at, "expires_at": job.expires_at,
            "retention_minutes": self.settings.retention_minutes,
            "counts": self.counts(job), "settings": job.settings, "error": job.error,
            "download_ready": state == JobState.COMPLETED and job.zip_path is not None,
            "files": files,
        }
