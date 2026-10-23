"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const SUPPORTED = [".jpg", ".jpeg", ".png"];
  const LABELS = { processed: "Processed", review_required: "Review required", failed: "Failed", unsupported_skipped: "Skipped (unsupported)", pending: "Waiting", running: "Working" };
  const METHOD_LABELS = { projection_profile: "Projection profile", min_area_rect: "Minimum-area rectangle", consensus: "Both must agree" };
  let cfg = null, job = null, pollTimer = null, xhr = null;

  const show = (id, on) => { $(id).hidden = !on; };
  const mb = (n) => (n / 1048576).toFixed(n < 10485760 ? 1 : 0) + " MB";
  const ext = (name) => { const i = name.lastIndexOf("."); return i < 0 ? "" : name.slice(i).toLowerCase(); };

  function showError(id, msg) { const el = $(id); el.textContent = msg || ""; el.hidden = !msg; }

  function countsList(el, items) {
    el.replaceChildren();
    for (const [key, label, value] of items) {
      const box = document.createElement("div");
      if (key) box.className = "c-" + key;
      const dt = document.createElement("dt"); dt.textContent = label;
      const dd = document.createElement("dd"); dd.textContent = String(value);
      box.append(dt, dd); el.append(box);
    }
  }

  async function init() {
    cfg = await (await fetch("/api/config")).json();
    if (!("webkitdirectory" in document.createElement("input"))) { show("folder-wrap", false); show("folder-unsupported", true); }
    const sel = $("skew-method");
    for (const m of cfg.skew_methods) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = (METHOD_LABELS[m] || m) + (m === cfg.default_skew_method ? " (default, provisional)" : "");
      sel.append(o);
    }
    sel.value = cfg.default_skew_method;
    $("skew-hint").textContent = "No development results have chosen an estimator yet, so the default is provisional. The choice is recorded in settings.json inside the ZIP.";
    $("folder-input").addEventListener("change", (e) => onFolder(Array.from(e.target.files)));
    $("zip-input").addEventListener("change", (e) => onZip(e.target.files[0]));
    $("start-btn").addEventListener("click", startJob);
    $("cancel-btn").addEventListener("click", resetAll);
    $("new-btn").addEventListener("click", resetAll);
  }

  function onFolder(files) {
    showError("choose-error", "");
    if (!files.length) return;
    const entries = files.map((f) => ({ file: f, path: f.webkitRelativePath || f.name }));
    const supported = entries.filter((e) => SUPPORTED.includes(ext(e.path)));
    const skipped = entries.filter((e) => !SUPPORTED.includes(ext(e.path)));
    const problems = [];
    if (!supported.length) problems.push("No JPG, JPEG or PNG files were found in this folder.");
    if (supported.length > cfg.max_files) problems.push(`This folder has ${supported.length} images; the limit is ${cfg.max_files} per job.`);
    if (entries.length > cfg.max_client_entries) problems.push(`This folder has ${entries.length} files; the limit is ${cfg.max_client_entries} entries.`);
    const bytes = supported.reduce((a, e) => a + e.file.size, 0);
    if (bytes > cfg.max_total_bytes) problems.push(`The images total ${mb(bytes)}; the limit is ${mb(cfg.max_total_bytes)}.`);
    if (problems.length) { showError("choose-error", problems.join(" ")); return; }
    const big = supported.filter((e) => e.file.size > cfg.max_file_bytes).length;
    const fd = new FormData();
    for (const e of supported) { fd.append("files", e.file, "f"); fd.append("paths", e.path); }
    for (const e of skipped) fd.append("skipped", e.path);
    upload(fd, entries[0].path.split("/")[0], big ? `${big} file(s) are over ${mb(cfg.max_file_bytes)} and will be skipped.` : "");
  }

  function onZip(file) {
    showError("choose-error", "");
    if (!file) return;
    if (file.size > cfg.max_total_bytes) { showError("choose-error", `This ZIP is ${mb(file.size)}; the limit is ${mb(cfg.max_total_bytes)}.`); return; }
    const fd = new FormData();
    fd.append("archive", file, file.name);
    upload(fd, file.name, "");
  }

  function upload(fd, name, warning) {
    show("upload-progress", true); $("upload-bar").value = 0;
    xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("upload-bar").value = (100 * e.loaded) / e.total; };
    xhr.onload = () => {
      show("upload-progress", false);
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
      if (xhr.status === 201) { job = body; renderStaged(name, warning); }
      else showError("choose-error", body.detail || "The upload failed. Please try again.");
    };
    xhr.onerror = () => { show("upload-progress", false); showError("choose-error", "Could not reach the server."); };
    xhr.send(fd);
  }

  function renderStaged(name, warning) {
    const c = job.counts;
    $("sel-name").textContent = (job.mode === "zip" ? "ZIP: " : "Folder: ") + (job.source_name || name);
    countsList($("sel-counts"), [["", "Files found", c.total], ["", "Supported images", c.supported], ["unsupported_skipped", "Unsupported / skipped", c.unsupported]]);
    const over = job.files.filter((f) => f.status === "unsupported_skipped" && /per-file limit/.test(f.note)).length;
    const msg = [warning, over ? `${over} file(s) exceeded the per-file limit and will be skipped.` : ""].filter(Boolean).join(" ");
    $("sel-warn").textContent = msg; $("sel-warn").hidden = !msg;
    show("step-check", true); show("step-choose", false);
    showError("start-error", "");
  }

  async function startJob() {
    $("start-btn").disabled = true;
    const res = await fetch(`/api/jobs/${job.id}/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ skew_method: $("skew-method").value }) });
    const body = await res.json().catch(() => ({}));
    $("start-btn").disabled = false;
    if (!res.ok) { showError("start-error", body.detail || "Could not start the job."); return; }
    job = body; show("step-check", false); show("step-run", true); show("step-done", false);
    poll();
  }

  function renderRows() {
    const tbody = $("file-rows"); tbody.replaceChildren();
    for (const f of job.files) {
      const tr = document.createElement("tr");
      const p = document.createElement("td"); p.className = "path"; p.textContent = f.path;
      const s = document.createElement("td"); const b = document.createElement("span");
      b.className = "badge s-" + f.status; b.textContent = LABELS[f.status] || f.status; s.append(b);
      const n = document.createElement("td");
      const bits = [];
      if (f.correction && f.correction !== "none") bits.push("Correction: " + f.correction.replace("+", " + "));
      if (f.angle_deg !== null && f.angle_deg !== undefined) bits.push(`skew ${f.angle_deg.toFixed(2)}\u00b0`);
      if (f.note) bits.push(f.note);
      n.textContent = bits.join(". ");
      tr.append(p, s, n); tbody.append(tr);
    }
  }

  function finalCounts(el, c) {
    countsList(el, [["processed", "Processed", c.processed], ["review_required", "Review required", c.review_required], ["failed", "Failed", c.failed], ["unsupported_skipped", "Unsupported / skipped", c.unsupported_skipped]]);
  }

  async function poll() {
    clearTimeout(pollTimer);
    let res;
    try { res = await fetch(`/api/jobs/${job.id}`); } catch (_) { pollTimer = setTimeout(poll, 2000); return; }
    if (!res.ok) { show("step-run", false); show("step-done", true); $("final-note").textContent = "This job is no longer available. It may have expired."; return; }
    job = await res.json();
    const c = job.counts;
    $("run-summary").textContent = `${c.completed} of ${c.supported} images finished (${c.unsupported} unsupported files skipped).`;
    $("run-bar").value = c.supported ? (100 * c.completed) / c.supported : 0;
    renderRows();
    if (job.state === "completed" || job.state === "failed") return finish();
    pollTimer = setTimeout(poll, 1000);
  }

  function finish() {
    show("step-done", true);
    finalCounts($("final-counts"), job.counts);
    const link = $("download-link");
    if (job.download_ready) {
      link.href = `/api/jobs/${job.id}/download`; link.hidden = false;
      $("final-note").textContent = `The ZIP will be deleted ${job.retention_minutes} minutes after this job finished.`;
    } else {
      link.hidden = true;
      $("final-note").textContent = job.error || "The job did not finish.";
    }
  }

  function resetAll() {
    clearTimeout(pollTimer);
    if (xhr) { xhr.abort(); xhr = null; }
    if (job && job.state !== "queued" && job.state !== "running") fetch(`/api/jobs/${job.id}`, { method: "DELETE" }).catch(() => {});
    job = null;
    $("folder-input").value = ""; $("zip-input").value = "";
    for (const id of ["step-check", "step-run", "step-done", "upload-progress"]) show(id, false);
    show("step-choose", true); showError("choose-error", "");
  }

  init();
})();
