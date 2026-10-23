"""Safe handling of untrusted input (paths, uploads, ZIPs) and creation of the output ZIP."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from ..models import FileResult, StagedInput

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_BAD_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"|?*\\]')
_JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
MAX_COMPONENT = 120
MAX_DEPTH = 16
MAX_PATH = 400
_CHUNK = 1024 * 1024

AUDIT_COLUMNS = [
    "source_path", "status", "output_path", "output_kind", "estimator", "angle_deg",
    "perspective_action", "perspective_reason", "correction_applied", "elapsed_s", "note", "sha256_input",
]


class StagingError(Exception):
    """Whole-upload rejection with an HTTP-style status and a user-safe message."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class StagingLimits:
    max_files: int
    max_entries: int
    max_total_bytes: int
    max_file_bytes: int
    zip_max_ratio: int


# ---------- paths ----------

def sanitize_relative_path(raw: object) -> Optional[str]:
    """Return a normalized relative POSIX path, or None if the input is unsafe.

    Rejects absolute paths, drive letters, '..' components, NUL bytes, excessive depth/length.
    """
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None
    p = unicodedata.normalize("NFC", raw.replace("\\", "/"))
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        return None
    parts = []
    for comp in p.split("/"):
        if comp in ("", "."):
            continue
        if comp == "..":
            return None
        comp = _BAD_CHARS.sub("_", comp).strip().rstrip(". ")
        if not comp:
            comp = "_"
        if len(comp) > MAX_COMPONENT:
            stem, ext = os.path.splitext(comp)
            comp = stem[: MAX_COMPONENT - len(ext)] + ext
        if comp.split(".")[0].upper() in _RESERVED:
            comp = "_" + comp
        parts.append(comp)
    if not parts or len(parts) > MAX_DEPTH:
        return None
    out = "/".join(parts)
    return out if len(out) <= MAX_PATH else None


def is_supported_name(rel_path: str) -> bool:
    return os.path.splitext(rel_path)[1].lower() in SUPPORTED_EXTENSIONS


def _is_junk(rel_path: str) -> bool:
    parts = rel_path.split("/")
    base = parts[-1].lower()
    return parts[0] == "__MACOSX" or base.startswith("._") or base in _JUNK_NAMES


def _counter_name(name: str, n: int) -> str:
    stem, ext = os.path.splitext(name)
    return f"{stem}__{n}{ext}"


class UniquePathAllocator:
    """Gives every entry a distinct, case-insensitively unique path (no file/dir clashes either)."""

    def __init__(self) -> None:
        self._files: set = set()
        self._dirs: set = set()

    def allocate(self, path: str) -> str:
        parts = path.split("/")
        while True:  # an ancestor name already used by a file -> rename our directory component
            for i in range(1, len(parts)):
                if "/".join(parts[:i]).casefold() in self._files:
                    parts[i - 1] += "_dir"
                    break
            else:
                break
        original_last, n = parts[-1], 1
        cand = "/".join(parts)
        while cand.casefold() in self._files or cand.casefold() in self._dirs:
            n += 1
            parts[-1] = _counter_name(original_last, n)
            cand = "/".join(parts)
        self._files.add(cand.casefold())
        for i in range(1, len(parts)):
            self._dirs.add("/".join(parts[:i]).casefold())
        return cand


# ---------- staging uploads ----------

class _TooLarge(Exception):
    pass


def _copy_limited(src: BinaryIO, dst_path: Path, max_bytes: int, budget: list) -> int:
    """Stream src to dst_path. budget is a one-item list holding remaining total bytes."""
    written = 0
    try:
        with open(dst_path, "wb") as dst:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                budget[0] -= len(chunk)
                if budget[0] < 0:
                    raise StagingError(413, "The upload is larger than the total size limit.")
                if written > max_bytes:
                    raise _TooLarge()
                dst.write(chunk)
    except (_TooLarge, StagingError):
        dst_path.unlink(missing_ok=True)
        raise
    return written


def stage_folder_upload(input_dir: Path, entries: list, limits: StagingLimits) -> list:
    """entries: list of (raw_path, fileobj_or_None, preset_note). Returns StagedInput list."""
    if len(entries) > limits.max_entries:
        raise StagingError(413, f"Too many entries ({len(entries)}); the limit is {limits.max_entries}.")
    prepared = []
    for order, (raw, fobj, preset) in enumerate(entries):
        safe = sanitize_relative_path(raw)
        prepared.append((safe, order, fobj, preset))
    prepared.sort(key=lambda t: ((t[0] or "\uffff").casefold(), t[1]))

    n_supported = sum(1 for safe, _, f, _ in prepared if safe and f is not None and is_supported_name(safe) and not _is_junk(safe))
    if n_supported > limits.max_files:
        raise StagingError(413, f"Too many images: {n_supported} supported images exceeds the limit of {limits.max_files} per job.")

    alloc, budget, staged = UniquePathAllocator(), [limits.max_total_bytes], []
    input_dir.mkdir(parents=True, exist_ok=True)
    for idx, (safe, order, fobj, preset) in enumerate(prepared):
        if safe is None:
            staged.append(StagedInput(idx, alloc.allocate(f"rejected-entry-{order + 1}"), None, 0, False, "Unsafe path rejected."))
            continue
        rel = alloc.allocate(safe)
        if preset:
            staged.append(StagedInput(idx, rel, None, 0, False, preset))
        elif _is_junk(safe):
            staged.append(StagedInput(idx, rel, None, 0, False, "Hidden or system file."))
        elif not is_supported_name(safe):
            staged.append(StagedInput(idx, rel, None, 0, False, "Unsupported file type (only JPG, JPEG, PNG)."))
        elif fobj is None:
            staged.append(StagedInput(idx, rel, None, 0, False, "No file data received."))
        else:
            dst = input_dir / f"{idx:06d}.bin"
            try:
                size = _copy_limited(fobj, dst, limits.max_file_bytes, budget)
            except _TooLarge:
                mb = limits.max_file_bytes // (1024 * 1024)
                staged.append(StagedInput(idx, rel, None, 0, False, f"File exceeds the {mb} MB per-file limit."))
                continue
            staged.append(StagedInput(idx, rel, dst, size, True))
    return staged


def stage_zip_upload(zip_path: Path, input_dir: Path, limits: StagingLimits) -> list:
    """Safely extract supported images from an uploaded ZIP into input_dir (names are index-based)."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise StagingError(400, "The file is not a valid ZIP archive.")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > limits.max_entries:
            raise StagingError(413, f"The archive has {len(infos)} entries; the limit is {limits.max_entries}.")
        declared = sum(i.file_size for i in infos)
        if declared > limits.max_total_bytes:
            raise StagingError(413, "The archive would expand beyond the total size limit.")
        for i in infos:
            if i.compress_size > 0 and i.file_size > _CHUNK and i.file_size / i.compress_size > limits.zip_max_ratio:
                raise StagingError(400, "The archive has a suspicious compression ratio and was rejected.")

        prepared = []
        for order, info in enumerate(infos):
            prepared.append((sanitize_relative_path(info.filename), order, info))
        prepared.sort(key=lambda t: ((t[0] or "\uffff").casefold(), t[1]))
        n_supported = sum(1 for s, _, _ in prepared if s and is_supported_name(s) and not _is_junk(s))
        if n_supported > limits.max_files:
            raise StagingError(413, f"Too many images: {n_supported} supported images exceeds the limit of {limits.max_files} per job.")

        alloc, budget, staged = UniquePathAllocator(), [limits.max_total_bytes], []
        input_dir.mkdir(parents=True, exist_ok=True)
        for idx, (safe, order, info) in enumerate(prepared):
            if safe is None:
                staged.append(StagedInput(idx, alloc.allocate(f"rejected-entry-{order + 1}"), None, 0, False, "Unsafe path rejected."))
                continue
            rel = alloc.allocate(safe)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                staged.append(StagedInput(idx, rel, None, 0, False, "Symbolic link skipped."))
            elif info.flag_bits & 0x1:
                staged.append(StagedInput(idx, rel, None, 0, False, "Encrypted entry skipped."))
            elif _is_junk(safe):
                staged.append(StagedInput(idx, rel, None, 0, False, "Hidden or system file."))
            elif not is_supported_name(safe):
                staged.append(StagedInput(idx, rel, None, 0, False, "Unsupported file type (only JPG, JPEG, PNG)."))
            else:
                dst = input_dir / f"{idx:06d}.bin"
                try:
                    with zf.open(info) as src:
                        size = _copy_limited(src, dst, limits.max_file_bytes, budget)
                except _TooLarge:
                    mb = limits.max_file_bytes // (1024 * 1024)
                    staged.append(StagedInput(idx, rel, None, 0, False, f"File exceeds the {mb} MB per-file limit."))
                    continue
                except (zipfile.BadZipFile, RuntimeError, OSError, EOFError):
                    dst.unlink(missing_ok=True)
                    staged.append(StagedInput(idx, rel, None, 0, False, "Entry could not be read from the archive."))
                    continue
                staged.append(StagedInput(idx, rel, dst, size, True))
        return staged


# ---------- output ----------

def _safe_cell(value: str) -> str:
    """Neutralize spreadsheet formula injection in text cells."""
    return "'" + value if value and value[0] in "=+-@\t\r" else value


def audit_csv(results: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(AUDIT_COLUMNS)
    for r in sorted(results, key=lambda r: r.source_path.casefold()):
        w.writerow([
            _safe_cell(r.source_path), r.status.value, _safe_cell(r.output_path), r.output_kind, r.estimator,
            "" if r.angle_deg is None else f"{r.angle_deg:.2f}", r.perspective_action,
            _safe_cell(r.perspective_reason), r.correction_applied, f"{r.elapsed_s:.3f}",
            _safe_cell(r.note), r.sha256_input,
        ])
    return buf.getvalue()


def build_output_zip(out_root: Path, results: list, settings: dict, zip_path: Path) -> None:
    """Create the job ZIP from files under out_root only. Written atomically."""
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    root = out_root.resolve()
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("audit.csv", audit_csv(results))
            zf.writestr("settings.json", json.dumps(settings, indent=2, sort_keys=True))
            zf.writestr("aligned/", "")
            zf.writestr("review/", "")
            for r in sorted(results, key=lambda r: r.output_path):
                if not r.output_path:
                    continue
                src = (out_root / r.output_path).resolve()
                if not src.is_relative_to(root) or not src.is_file():
                    raise OSError("Output file missing or outside the job directory.")
                zf.write(src, r.output_path)
        os.replace(tmp, zip_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
