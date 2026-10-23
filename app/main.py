"""FastAPI routes. Image logic lives in services/image_alignment.py, never here."""
from __future__ import annotations

import logging
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from .config import DEFAULT_SKEW_METHOD, SKEW_DEFAULT_IS_PROVISIONAL, SKEW_METHODS, Settings
from .models import JobState
from .services import archive
from .services.batch_jobs import JobError, JobManager

BASE = Path(__file__).parent
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
log = logging.getLogger("docalign")


class StartRequest(BaseModel):
    skew_method: Optional[str] = None


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):g} MB"


def _sweeper(manager: JobManager, stop: threading.Event) -> None:
    while not stop.wait(30):
        try:
            manager.sweep()
        except Exception:  # noqa: BLE001
            log.exception("cleanup sweep failed")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    s = settings or Settings.from_env()
    manager = JobManager(s)
    limits = archive.StagingLimits(s.max_files, s.max_client_entries, s.max_total_bytes, s.max_file_bytes, s.zip_max_ratio)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.startup()
        stop = threading.Event()
        threading.Thread(target=_sweeper, args=(manager, stop), daemon=True, name="sweeper").start()
        yield
        stop.set()

    app = FastAPI(title="Document Alignment", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.manager, app.state.settings = manager, s
    app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
    templates = Jinja2Templates(directory=BASE / "templates")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'"
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.exception_handler(JobError)
    async def job_error_handler(_: Request, exc: JobError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    def public_config() -> dict:
        return {
            "max_files": s.max_files, "max_client_entries": s.max_client_entries,
            "max_total_bytes": s.max_total_bytes, "max_file_bytes": s.max_file_bytes,
            "max_pixels": s.processing.max_input_pixels, "max_job_seconds": s.max_job_seconds,
            "retention_minutes": s.retention_minutes, "workers": s.workers,
            "skew_methods": list(SKEW_METHODS), "default_skew_method": DEFAULT_SKEW_METHOD,
            "default_is_provisional": SKEW_DEFAULT_IS_PROVISIONAL,
        }

    @app.get("/")
    async def index(request: Request):
        ctx = {
            "cfg": public_config(), "max_total": _mb(s.max_total_bytes), "max_file": _mb(s.max_file_bytes),
            "max_mp": f"{s.processing.max_input_pixels / 1e6:g}",
        }
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/api/config")
    async def api_config():
        return public_config()

    def _job(job_id: str):
        if not JOB_ID.match(job_id):
            raise HTTPException(404, "Job not found or expired.")
        return manager.get(job_id)

    def _stage(form) -> dict:
        archive_part = form.get("archive")
        files = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
        has_zip = isinstance(archive_part, UploadFile)
        if has_zip and files:
            raise HTTPException(400, "Send either a folder or a ZIP file, not both.")
        if not has_zip and not files:
            raise HTTPException(400, "No files were received.")
        job = manager.create_job("zip" if has_zip else "folder", "")
        try:
            input_dir = job.dir / "input"
            if has_zip:
                zip_path = job.dir / "upload.zip"
                with open(zip_path, "wb") as out:
                    remaining = s.max_total_bytes
                    while chunk := archive_part.file.read(1024 * 1024):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise archive.StagingError(413, "The ZIP file is larger than the total size limit.")
                        out.write(chunk)
                staged = archive.stage_zip_upload(zip_path, input_dir, limits)
                zip_path.unlink(missing_ok=True)
                name = archive.sanitize_relative_path(archive_part.filename or "") or "archive.zip"
                job.source_name = name.split("/")[-1]
            else:
                paths = [p for p in form.getlist("paths") if isinstance(p, str)]
                entries = [((paths[i] if i < len(paths) else (uf.filename or "")), uf.file, "") for i, uf in enumerate(files)]
                entries += [(p, None, "Skipped by the browser: unsupported type.") for p in form.getlist("skipped") if isinstance(p, str)]
                staged = archive.stage_folder_upload(input_dir, entries, limits)
                first = next((x.rel_path for x in staged if x.supported), staged[0].rel_path if staged else "")
                job.source_name = first.split("/")[0] if "/" in first else "selected files"
            n_supported = sum(1 for x in staged if x.supported)
            if n_supported == 0:
                raise archive.StagingError(400, f"No supported images found ({len(staged)} entries checked). Use JPG, JPEG or PNG files.")
            manager.finish_staging(job, staged)
            return manager.snapshot(job)
        except archive.StagingError as exc:
            manager.discard(job)
            raise HTTPException(exc.status_code, exc.message)
        except HTTPException:
            manager.discard(job)
            raise
        except Exception:  # noqa: BLE001
            log.exception("staging failed")
            manager.discard(job)
            raise HTTPException(500, "The upload could not be processed.")

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request):
        cl = request.headers.get("content-length", "")
        if cl.isdigit() and int(cl) > s.max_total_bytes + 2 * 1024 * 1024:
            raise HTTPException(413, f"The upload is larger than the {_mb(s.max_total_bytes)} total limit.")
        try:
            form = await request.form(max_files=s.max_files + 1, max_fields=s.max_client_entries * 2 + 20)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "The upload could not be read.")
        try:
            return await run_in_threadpool(_stage, form)
        finally:
            await form.close()

    @app.post("/api/jobs/{job_id}/start")
    async def start_job(job_id: str, body: Optional[StartRequest] = None):
        job = _job(job_id)
        manager.start(job.id, body.skew_method if body else None)
        return manager.snapshot(job)

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        return manager.snapshot(_job(job_id))

    @app.get("/api/jobs/{job_id}/download")
    async def download(job_id: str):
        job = _job(job_id)
        if job.state != JobState.COMPLETED or job.zip_path is None or not job.zip_path.is_file():
            raise HTTPException(409, "The ZIP is not ready yet.")
        return FileResponse(job.zip_path, media_type="application/zip", filename="aligned_documents.zip",
                            headers={"Cache-Control": "no-store"})

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str):
        _job(job_id)
        manager.delete(job_id)

    return app


app = create_app()
