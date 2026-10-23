import io
import time
import zipfile

import pytest

from app.models import JobState
from app.services import batch_jobs
from scripts.make_synthetic_samples import blank_image, encode, make_text_page, two_pages_scene
from tests.conftest import (audit_rows, make_client, open_zip, page_jpg, post_folder, post_zip, run_job, sha)


def test_single_image_submit_process_download(client):
    data = page_jpg(angle=5.0)
    r = post_folder(client, [("scan/one.jpg", data)])
    assert r.status_code == 201
    job = r.json()
    assert job["state"] == "staged" and job["counts"]["supported"] == 1
    assert client.get(f"/api/jobs/{job['id']}/download").status_code == 409     # not before completion
    done = run_job(client, job["id"])
    assert done["state"] == "completed" and done["counts"]["processed"] == 1 and done["download_ready"]
    z = open_zip(client, job["id"])
    assert "aligned/scan/one.jpg" in z.namelist() and "audit.csv" in z.namelist()
    row = audit_rows(z)[0]
    assert row["status"] == "processed" and row["correction_applied"] == "rotation"
    assert abs(float(row["angle_deg"]) - 5.0) < 0.6 and row["sha256_input"] == sha(data)
    assert z.read("aligned/scan/one.jpg") != data


def test_batch_of_50_supported_images_in_one_job(client):
    items = [(f"batch/p{i:02d}.jpg", page_jpg(seed=i, angle=((i % 9) - 4) * 2.0)) for i in range(50)]
    r = post_folder(client, items)
    assert r.status_code == 201 and r.json()["counts"]["supported"] == 50
    done = run_job(client, r.json()["id"], timeout=180)
    c = done["counts"]
    assert done["state"] == "completed" and c["total"] == 50 and c["completed"] == 50
    assert c["processed"] + c["review_required"] + c["failed"] + c["unsupported_skipped"] == 50
    z = open_zip(client, done["id"])
    rows = audit_rows(z)
    assert len(rows) == 50
    out = [n for n in z.namelist() if n.startswith(("aligned/", "review/")) and not n.endswith("/")]
    assert len(out) == 50 and len(set(out)) == 50
    assert c["processed"] == 50      # observed for this synthetic set; says nothing about real photos


def test_mixed_valid_and_unsupported_do_not_abort_and_counts_reconcile(tmp_path):
    with make_client(tmp_path, max_file_bytes=300_000) as c:
        ok, blank = page_jpg(angle=4), encode(blank_image())
        items = [("m/ok.jpg", ok), ("m/blank.jpg", blank), ("m/notes.txt", b"hello"),
                 ("m/fake.jpg", b"this is not a jpeg"), ("../../escape.jpg", ok),
                 ("m/huge.png", b"\x89PNG\r\n\x1a\n" + b"0" * 400_000)]
        r = post_folder(c, items, skipped=["m/readme.pdf"])
        assert r.status_code == 201
        done = run_job(c, r.json()["id"])
        k = done["counts"]
        assert k["total"] == 7 and k["processed"] == 1 and k["review_required"] == 1 and k["failed"] == 1
        assert k["unsupported_skipped"] == 4        # txt, escaped path, oversize, client-skipped pdf
        assert sum(k[s] for s in ("processed", "review_required", "failed", "unsupported_skipped")) == k["total"]
        rows = {r["source_path"]: r for r in audit_rows(open_zip(c, done["id"]))}
        assert rows["m/fake.jpg"]["status"] == "failed" and "signature" in rows["m/fake.jpg"]["note"]
        assert any(p.startswith("rejected-entry-") for p in rows)
        assert not any(".." in p for p in rows)


def test_ambiguous_image_is_review_with_untouched_original(client):
    two = encode(two_pages_scene(make_text_page()))
    r = post_folder(client, [("amb/two.jpg", two)])
    done = run_job(client, r.json()["id"])
    assert done["counts"]["review_required"] == 1 and done["counts"]["processed"] == 0
    z = open_zip(client, done["id"])
    assert z.read("review/amb/two.jpg") == two
    assert not any(n.startswith("aligned/") and not n.endswith("/") for n in z.namelist())
    row = audit_rows(z)[0]
    assert row["status"] == "review_required" and row["output_kind"] == "original_copy"


def test_originals_unchanged_and_aligned_copy_is_exact_when_no_correction(client):
    straight, blank = page_jpg(seed=3, angle=0.0), encode(blank_image())
    r = post_folder(client, [("o/straight.jpg", straight), ("o/blank.jpg", blank)])
    done = run_job(client, r.json()["id"])
    z = open_zip(client, done["id"])
    rows = {r["source_path"]: r for r in audit_rows(z)}
    assert z.read("aligned/o/straight.jpg") == straight
    assert rows["o/straight.jpg"]["correction_applied"] == "none" and "unchanged copy" in rows["o/straight.jpg"]["note"]
    assert z.read("review/o/blank.jpg") == blank
    assert rows["o/blank.jpg"]["sha256_input"] == sha(blank)


def test_duplicate_names_and_nested_paths_never_overwrite(client):
    imgs = {k: page_jpg(seed=k, angle=3 + k) for k in range(5)}
    items = [("a/x.jpg", imgs[0]), ("b/x.jpg", imgs[1]), ("A/x.jpg", imgs[2]), ("a/x.jpg", imgs[3]), ("a/x.jpg/y.jpg", imgs[4])]
    done = run_job(client, post_folder(client, items).json()["id"])
    z = open_zip(client, done["id"])
    files = [n for n in z.namelist() if n.startswith("aligned/") and not n.endswith("/")]
    assert len(files) == 5 and len({n.casefold() for n in files}) == 5
    assert len({sha(z.read(n)) for n in files}) == 5


def test_zip_fallback_end_to_end(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            zf.writestr(f"docs/p{i}.jpg", page_jpg(seed=i, angle=6))
        zf.writestr("docs/skip.txt", "x")
    r = post_zip(client, buf.getvalue(), "my docs.zip")
    assert r.status_code == 201 and r.json()["mode"] == "zip" and r.json()["source_name"] == "my docs.zip"
    done = run_job(client, r.json()["id"])
    assert done["counts"]["processed"] == 3 and done["counts"]["unsupported_skipped"] == 1


def test_bad_uploads_return_safe_errors(tmp_path):
    with make_client(tmp_path, max_total_bytes=100_000, max_file_bytes=80_000, max_files=3) as c:
        assert post_zip(c, b"definitely not a zip").status_code == 400
        assert c.post("/api/jobs", files=[("other", ("x", b"1"))]).status_code == 400
        r = post_folder(c, [("only.txt", b"nope")])
        assert r.status_code == 400 and "No supported images" in r.json()["detail"]
        big = [(f"b/{i}.jpg", page_jpg(seed=i)) for i in range(3)]
        assert sum(len(d) for _, d in big) > 100_000 and all(len(d) < 80_000 for _, d in big)
        r = post_folder(c, big)
        assert r.status_code == 413
        r = post_folder(c, [(f"c/{i}.jpg", page_jpg(w=300, h=400)) for i in range(4)])
        assert r.status_code in (400, 413)
        assert "Traceback" not in r.text and "/tmp" not in r.text and str(tmp_path) not in r.text
        assert not any((tmp_path / "work" / "jobs").iterdir())       # failed staging leaves no job data behind


def test_write_failure_is_reported_per_file_and_batch_continues(client, monkeypatch):
    real = batch_jobs._atomic_write

    def flaky(path, data):
        if "aligned" in path.parts:
            raise OSError("disk full: /secret/location")
        return real(path, data)

    monkeypatch.setattr(batch_jobs, "_atomic_write", flaky)
    orig = page_jpg(angle=6)
    done = run_job(client, post_folder(client, [("w/a.jpg", orig), ("w/b.jpg", encode(blank_image()))]).json()["id"])
    assert done["state"] == "completed"
    assert done["counts"]["failed"] == 1 and done["counts"]["review_required"] == 1
    z = open_zip(client, done["id"])
    rows = {r["source_path"]: r for r in audit_rows(z)}
    assert rows["w/a.jpg"]["status"] == "failed" and "could not be written" in rows["w/a.jpg"]["note"]
    assert "secret" not in z.read("audit.csv").decode()
    assert z.read("review/w/a.jpg") == orig


def test_download_cannot_cross_jobs(client):
    a = post_folder(client, [("only_in_a/a.jpg", page_jpg(seed=1, angle=4))]).json()["id"]
    b = post_folder(client, [("only_in_b/b.jpg", page_jpg(seed=2, angle=7))]).json()["id"]
    assert a != b and len(a) >= 30
    run_job(client, a); run_job(client, b)
    za, zb = open_zip(client, a), open_zip(client, b)
    assert any("only_in_a" in n for n in za.namelist()) and not any("only_in_b" in n for n in za.namelist())
    assert any("only_in_b" in n for n in zb.namelist()) and not any("only_in_a" in n for n in zb.namelist())
    for bad in ("nope", "../" + a, a[:-1] + ("A" if a[-1] != "A" else "B"), "x" * 30):
        assert client.get(f"/api/jobs/{bad}/download").status_code == 404


def test_cannot_start_twice_and_concurrency_limit(tmp_path, monkeypatch):
    real = batch_jobs.align_image

    def slow(data, cfg):
        time.sleep(0.4)
        return real(data, cfg)

    monkeypatch.setattr(batch_jobs, "align_image", slow)
    with make_client(tmp_path, max_concurrent_jobs=1, workers=1) as c:
        a = post_folder(c, [("a.jpg", page_jpg(angle=5)), ("b.jpg", page_jpg(seed=2, angle=5))]).json()["id"]
        b = post_folder(c, [("c.jpg", page_jpg(seed=3, angle=5))]).json()["id"]
        assert c.post(f"/api/jobs/{a}/start", json={}).status_code == 200
        assert c.post(f"/api/jobs/{a}/start", json={}).status_code == 409
        assert c.post(f"/api/jobs/{b}/start", json={}).status_code == 429
        assert c.post(f"/api/jobs/{b}/start", json={"skew_method": "bogus"}).status_code in (400, 429)
        assert c.delete(f"/api/jobs/{a}").status_code == 409                # running jobs cannot be deleted


def test_retention_sweep_and_startup_cleanup(tmp_path):
    with make_client(tmp_path) as c:
        mgr = c.app.state.manager
        jid = post_folder(c, [("r.jpg", page_jpg(angle=5))]).json()["id"]
        run_job(c, jid)
        job = mgr.get(jid)
        assert job.dir.exists()
        job.expires_at = time.time() - 1
        assert mgr.sweep() == 1
        assert not job.dir.exists() and c.get(f"/api/jobs/{jid}").status_code == 404
        (mgr.jobs_root / "orphan").mkdir()
    with make_client(tmp_path) as c2:                                        # new "process": in-memory state is gone
        assert not (c2.app.state.manager.jobs_root / "orphan").exists()


def test_working_data_deleted_after_completion_and_settings_recorded(client):
    jid = post_folder(client, [("s.jpg", page_jpg(angle=5))]).json()["id"]
    done = run_job(client, jid, skew_method="consensus")
    job = client.app.state.manager.get(jid)
    assert sorted(p.name for p in job.dir.iterdir()) == ["results.zip"]
    assert done["settings"]["skew_method"] == "consensus" and not done["settings"]["skew_method_is_provisional_default"]
    z = open_zip(client, jid)
    assert '"skew_method": "consensus"' in z.read("settings.json").decode()


def test_index_page_states_limits(client):
    html = client.get("/").text
    assert "300 images per job" in html and "at least 50" in html and "does not read text" in html
    assert client.get("/api/config").json()["default_is_provisional"] is True
