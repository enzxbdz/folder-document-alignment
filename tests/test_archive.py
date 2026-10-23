import io
import zipfile

import pytest

from app.models import FileResult, Status
from app.services import archive
from app.services.archive import StagingError, StagingLimits, UniquePathAllocator, sanitize_relative_path
from tests.conftest import page_jpg

LIM = StagingLimits(max_files=50, max_entries=100, max_total_bytes=50_000_000, max_file_bytes=10_000_000, zip_max_ratio=200)


@pytest.mark.parametrize("raw", ["../x.jpg", "a/../../x.jpg", "/etc/passwd.jpg", "C:\\x.jpg", "a\\..\\..\\x.png", "", "a/\x00.jpg"])
def test_unsafe_paths_rejected(raw):
    assert sanitize_relative_path(raw) is None


def test_path_normalisation():
    assert sanitize_relative_path("a//b/./c.JPG") == "a/b/c.JPG"
    assert sanitize_relative_path("a\\b\\c.png") == "a/b/c.png"
    assert sanitize_relative_path("we<ir>d:name?.jpg") == "we_ir_d_name_.jpg"
    assert sanitize_relative_path("CON.jpg") == "_CON.jpg"
    assert len(sanitize_relative_path("x" * 300 + ".jpg").split("/")[-1]) <= 120
    assert sanitize_relative_path("/".join(["d"] * 40) + "/x.jpg") is None


def test_allocator_never_collides():
    a = UniquePathAllocator()
    got = [a.allocate(p) for p in ["a/x.jpg", "a/x.jpg", "A/X.JPG", "b/x.jpg", "a", "a/x.jpg/y.jpg"]]
    assert len({g.casefold() for g in got}) == len(got)
    assert got[0] == "a/x.jpg" and got[1] == "a/x__2.jpg"
    # a file name that is also used as a directory prefix must not clash
    assert not any(g.casefold() == "a" and any(o.casefold().startswith("a/") for o in got) for g in got)


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return buf.getvalue()


def test_zip_slip_and_junk_entries_are_not_extracted(tmp_path):
    img = page_jpg()
    zp = tmp_path / "u.zip"
    zp.write_bytes(_zip([("../evil.jpg", img), ("/abs.jpg", img), ("ok/a.jpg", img), ("ok/A.JPG", img),
                         ("__MACOSX/ok/._a.jpg", b"x"), ("notes.txt", b"hi")]))
    staged = archive.stage_zip_upload(zp, tmp_path / "in", LIM)
    by_note = {s.rel_path: s for s in staged}
    assert sum(s.supported for s in staged) == 2
    assert len({s.rel_path.casefold() for s in staged}) == len(staged)
    assert all(s.stored_path is None or s.stored_path.parent == tmp_path / "in" for s in staged)
    assert not (tmp_path.parent / "evil.jpg").exists()
    assert any("Unsafe" in s.note for s in staged) and "ok/a.jpg" in by_note


def test_zip_bomb_invalid_zip_and_size_limits(tmp_path):
    zp = tmp_path / "bomb.zip"
    zp.write_bytes(_zip([("big.png", b"\0" * 5_000_000)]))
    with pytest.raises(StagingError) as e:
        archive.stage_zip_upload(zp, tmp_path / "in", LIM)
    assert e.value.status_code == 400
    zp.write_bytes(b"this is not a zip")
    with pytest.raises(StagingError) as e:
        archive.stage_zip_upload(zp, tmp_path / "in2", LIM)
    assert e.value.status_code == 400
    zp.write_bytes(_zip([("a.jpg", page_jpg())]))
    small = StagingLimits(50, 100, 1000, 500, 200)
    with pytest.raises(StagingError) as e:
        archive.stage_zip_upload(zp, tmp_path / "in3", small)
    assert e.value.status_code == 413


def test_csv_formula_injection_is_neutralised():
    r = FileResult(0, "=cmd|' /C calc'!A0.jpg", True, Status.FAILED, note="+bad", angle_deg=-3.5)
    row = archive.audit_csv([r]).splitlines()[1]
    assert row.startswith("'=cmd") and ",'+bad," in row and ",-3.50," in row
