# Document Alignment App

Batch-align phone photos of documents. Pick a folder (or upload a ZIP), start one job, watch per-file progress, then download a ZIP with the aligned images, untouched originals for anything uncertain, and a CSV audit report.

Scope follows the project plan "A Folder-Based Document Alignment Engine for Phone-Captured Documents Using Classical Computer Vision": OpenCV + NumPy only, mostly flat single-page JPG/JPEG/PNG photos, conditional perspective rectification, residual skew correction, non-destructive outputs, conservative review/failure outcomes. **Not included:** OCR, ML/DL, PDF input, semantic understanding, curved-page dewarping.

**Deployment status:** local demo. Jobs live in process memory, so a restart loses job state (orphaned files on disk are deleted at startup). Job IDs are unguessable (192-bit) but act as capability URLs; there are no user accounts. Do not expose this to an untrusted network as-is.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
# open http://localhost:8000
```

Tests: `pip install -r requirements-dev.txt && python -m pytest -q`
Synthetic sample images (clearly synthetic, git-ignored): `python -m scripts.make_synthetic_samples sample_data/synthetic 50`

## Input

* **Folder:** `<input webkitdirectory>`. The browser sends each image plus its relative path (`paths` field). Unsupported files are listed by path only (not uploaded). Chromium/Edge/Firefox/Safari support folder pickers; where unavailable the page hides it.
* **ZIP fallback:** one `archive` upload. Entries are extracted by the server with per-entry and total byte budgets counted while streaming, a compression-ratio check, entry-count limit, and symlink/encrypted/junk (`__MACOSX`, `._*`, `.DS_Store`) skipping.
* The page shows file count, supported and unsupported counts, and validation errors **after upload and before processing**; processing starts only when you press Start.
* All names are untrusted: paths are normalized, `..`/absolute/drive paths are rejected (listed as `rejected-entry-N`), stored on disk under index-based names, and given collision-free case-insensitive output paths (`x.jpg`, `x__2.jpg`).

## Output ZIP

```
aligned/<relative path>   corrected images (same extension; or an unchanged byte copy, see below)
review/<relative path>    untouched originals for review_required and failed inputs
audit.csv                 one row per input
settings.json             estimator and every threshold used for the job
```

`audit.csv` columns: `source_path, status, output_path, output_kind, estimator, angle_deg, perspective_action, perspective_reason, correction_applied, elapsed_s, note, sha256_input`. Text cells are neutralized against spreadsheet formula injection.

## Status meanings

| status | meaning |
|---|---|
| `processed` | a permitted correction path completed and an output was written. Not a quality guarantee. |
| `review_required` | readable input, evidence ambiguous or insufficient; original kept in `review/`, never labelled corrected |
| `failed` | supported input could not be decoded, processed or saved; original kept in `review/` when it could be safely copied |
| `unsupported_skipped` | unsupported type, unsafe path, over per-file or pixel limit, hidden/system file |

`pending` and `running` appear only in the UI while a job is in progress.

**Already-aligned images:** if no perspective correction applies and |skew| < 0.3 deg, the output in `aligned/` is a byte-identical copy of the input, `correction_applied` is `none`, and the note says so. It is not presented as corrected.

## Pipeline (`app/services/image_alignment.py`)

1. Validate magic bytes and header dimensions (decompression-bomb guard), then decode. EXIF orientation is applied by the OpenCV decoder.
2. **Perspective:** Canny edges on a downscaled copy, external contours, convex hull, `approxPolyDP`. A candidate must be a convex quadrilateral covering 30-97% of the frame, interior angles 50-130 deg, plausible aspect ratio, and backed by detected edges; the edge map must not be noise-dense.
   * `applied`: one clear candidate, warped with a perspective transform to an axis-aligned rectangle sized by the longer opposite sides.
   * `skipped`: no four-corner outline, or the outline is the whole frame. Falls back to rotation only.
   * `rejected`: candidates exist but fail checks, or two similar-size outlines compete. Result is `review_required` (conservative, not a rotation fallback).
3. **Residual skew**, two estimators: `min_area_rect` (OpenCV `minAreaRect` over foreground pixels) and `projection_profile` (variance of row sums over rotation angles, 1 deg then 0.1 deg steps, with a peak-prominence check). A third mode, `consensus`, requires both to agree within 1 deg.
   * Rejected: empty/near-empty content, no distinct foreground, flat profile, |angle| > 15 deg, estimator disagreement.
   * If perspective was applied but skew is rejected, the perspective-only result is written and the note says residual skew was not corrected. If nothing usable exists, `review_required`.
4. Rotation uses an enlarged canvas so nothing is cropped; the fill colour is the median of the image border. Output dimensions and encoding are validated.

**Sign convention:** `angle_deg > 0` means page content is rotated counter-clockwise (as viewed); the correction rotates clockwise by that amount.

**Provisional default:** the default estimator (`projection_profile`) was **not** chosen from development evidence on real photos. It is configurable per job in the UI, via `DOCALIGN_SKEW_METHOD`, and labelled provisional in the UI and `settings.json`. Replace the default once the evaluation supports a choice. All thresholds in `ProcessingConfig` are likewise untuned starting values.

## Limits and retention (defaults; env vars in `.env.example`)

| setting | default |
|---|---|
| supported images per job | 300 (must be at least 50) |
| total upload / per file | 500 MB / 25 MB |
| pixels per image | 60 million |
| entries listed per job (incl. unsupported) | 2000 |
| job duration | 1800 s (remaining files become `failed`, originals preserved) |
| concurrent running jobs / stored jobs / workers | 1 / 10 / 2 |
| retention | 30 min |

The 50-image minimum is verified by tests. The 300 limit is a configured cap, **not a measured capacity**; no benchmark is claimed.

Memory: at most `workers x 2` images are in flight, each decoded once and released. Starlette spools multipart uploads to temporary files before the app reads them, so uploads are bounded by the request-size checks, not streamed end-to-end.

Retention: uploaded inputs and working files are deleted as soon as the ZIP is built. The ZIP is deleted `retention` minutes after the job finishes (staged jobs: after upload). A background sweeper runs every 30 s. Do not describe this as "fully private" or "offline"; images are processed on the server you run.

## API

`GET /api/config`, `POST /api/jobs` (multipart: `files`+`paths`[+`skipped`] or `archive`), `POST /api/jobs/{id}/start` (`{"skew_method": ...}` optional), `GET /api/jobs/{id}`, `GET /api/jobs/{id}/download` (409 until completed), `DELETE /api/jobs/{id}`.

## Layout

```
app/main.py              routes, security headers
app/config.py            limits, retention, ProcessingConfig
app/models.py            Status, JobState, FileResult, Job
app/services/image_alignment.py   pure CV pipeline (no HTTP/filesystem)
app/services/batch_jobs.py        job lifecycle, bounded pool, retention sweep
app/services/archive.py           path sanitizing, safe staging/extraction, output ZIP, audit CSV
scripts/make_synthetic_samples.py synthetic test pages with known transforms
tests/                            alignment, archive and API tests
```

## Verification scope

Automated checks use **synthetic** images (generated text-like bars, known rotations, a synthetic perspective "desk" scene). They verify implementation behaviour (sign convention, corner ordering, safety, reconciliation). They say nothing about accuracy on real phone photos; that needs a labelled real-photo evaluation, which is still future work.

## Implemented vs future work

Implemented: folder + ZIP intake, 50+ image batches, perspective with deterministic checks, two skew estimators + consensus, review/failed handling, audit CSV, retention, safe paths/ZIPs, per-job isolation.

Future work: real-photo evaluation and estimator selection, threshold tuning, comparison with other scanners (no superiority claim is made), persistent job store/queue and authentication for any multi-user deployment, cancel button, curved-page handling.
