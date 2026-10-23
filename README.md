# Folder-Based Document Alignment Engine

Batch-align phone-captured document images with classical computer vision. Select a folder of images, process the files together, review each file's outcome, and download the results as a ZIP.

## Features

- Batch processing of at least 50 supported image files in one job.
- Folder selection in the browser, with ZIP upload as a fallback.
- JPG, JPEG, and PNG input support.
- Conditional page-boundary detection and perspective rectification for suitable flat pages.
- Residual skew estimation and correction using classical computer-vision methods.
- Non-destructive processing: source images are preserved.
- Per-file outcomes: processed, review required, failed, or unsupported/skipped.
- Downloadable ZIP containing aligned results, review items, and a CSV audit report.

The app aligns images; it does not perform OCR or create searchable PDFs. It is intended for mostly flat, single-page document photos. Curved pages, direct PDF processing, and semantic document understanding are outside the current scope.

## Requirements

- Python 3.11 or newer
- A modern web browser

## Run locally

From the repository root, create and activate a virtual environment, install dependencies, and start the web app:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the local address printed by Uvicorn in your browser. If your repository uses a different application module or run command, follow the command in its project configuration or update this section to match it.

## Use the app

1. Choose a folder containing document images, or select a ZIP file if folder selection is unavailable.
2. Review the file counts and validation messages.
3. Start the batch and follow the per-file progress and status list.
4. When processing is complete, select **Download ZIP** to retrieve the results and audit report.

Inputs with uncertain page geometry are flagged for review instead of being presented as successfully corrected. Unsupported files and processing failures are listed in the audit report.

## Output contents

The downloaded archive contains the aligned images, a review folder for items that need manual attention, and `audit.csv`. The audit report records each input's status and relevant processing details. Review-required items are preserved as originals and are not labeled as corrected.

## Processing scope

The alignment pipeline uses OpenCV and NumPy operations. It is designed for in-plane rotation and planar perspective distortion in mostly flat documents. It does not address curled or folded pages, OCR, PDF ingestion, reading order, or document content interpretation.

Processing results depend on image quality and visible page geometry. A successful output write does not guarantee that every correction is visually accurate; inspect review-required files and verify important results.

## Privacy and file handling

Uploaded files are processed by the application server. Temporary-file retention and cleanup behavior depend on the app configuration and deployment. Before using sensitive documents, review the deployment's storage and retention settings. Do not expose this development server publicly without adding appropriate authentication, access controls, and operational safeguards.

## Development and verification

Run the checks included in the repository using the commands documented by the project. Include the observed test results in project reports; do not infer real-photo accuracy from successful synthetic tests alone.

## Project status

This is a student project exploring a focused, folder-oriented batch alignment workflow with auditable per-file outcomes. Its performance should be described only on the datasets and conditions actually evaluated. It is not a claim of superiority over CamScanner or other document-scanning applications.

## License

No license is specified yet. Add a license file and update this section if you choose to publish the project under a particular license.
