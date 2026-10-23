import dataclasses
import hashlib
import io
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from scripts.make_synthetic_samples import encode, make_text_page, rotate_ccw  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def page_jpg(seed=0, angle=0.0, w=600, h=800) -> bytes:
    return encode(rotate_ccw(make_text_page(w, h, seed), angle))


def make_client(tmp_path, **overrides):
    s = dataclasses.replace(Settings(work_dir=tmp_path / "work"), **overrides)
    return TestClient(create_app(s))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def post_folder(c, items, skipped=()):
    """items: list of (relative_path, bytes)."""
    parts = []
    for path, data in items:
        parts.append(("files", ("f", data, "application/octet-stream")))
        parts.append(("paths", (None, path)))
    parts += [("skipped", (None, p)) for p in skipped]
    return c.post("/api/jobs", files=parts)


def post_zip(c, data: bytes, name="batch.zip"):
    return c.post("/api/jobs", files={"archive": (name, data, "application/zip")})


def run_job(c, job_id, timeout=60, **start_body):
    r = c.post(f"/api/jobs/{job_id}/start", json=start_body)
    assert r.status_code == 200, r.text
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("completed", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def open_zip(c, job_id) -> zipfile.ZipFile:
    r = c.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200, r.text
    return zipfile.ZipFile(io.BytesIO(r.content))


def audit_rows(z: zipfile.ZipFile):
    import csv
    return list(csv.DictReader(io.StringIO(z.read("audit.csv").decode())))
