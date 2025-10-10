// static/js/workspace.js
window.addEventListener("load", initWorkspace);

function initWorkspace() {
  console.log("workspace.js 2025-10-11d DPR-align, base-coords, dynamic-pageLayer");

  // ---- Session state ----
  const legacyWebRaw = sessionStorage.getItem("uploadedFileWithExtension") || "";
  const storedWebRaw = sessionStorage.getItem("uploadedWebPath") || legacyWebRaw || "";
  const storedWeb = normalizeToWebUrl(storedWebRaw);
  let storedDisk = sessionStorage.getItem("uploadedDiskPath") || "";
  const storedName = sessionStorage.getItem("uploadedFileName") || "";
  let currentFormId = sessionStorage.getItem("uploadedFormId") || null;

  // ---- DOM refs ----
  const $ = (id) => document.getElementById(id);
  const sidebarToggle = $("sidebarToggle");
  const sidebar = $("sidebar");
  const pageToggler = $("togglePages");
  const thumbSidebar = $("thumbnailSidebar");
  const analyzeBtn = $("analyzeTool");
  const summaryList = $("summaryList");
  const formTitle = $("formNameDisplay");
  const metricsRow = $("metricsRow");
  const pageInfo = $("pageInfo");
  const zoomInfo = $("zoomInfo");
  let pdfCanvas = $("pdfCanvas");
  let overlayCanvas = $("overlayCanvas");
  const eceBadge = $("eceScoreBadge");
  const downloadBtn = $("downloadPDF");
  const printBtn = $("printPDF");

  // Toolbar
  const enterEditBtn = $("enterEdit");        // optional; we auto-enter
  const undoBtn = $("btnUndo");
  const clearBtn = $("btnClear");
  const toolButtons = Array.from(document.querySelectorAll(".tool-btn"));

  if (!pdfCanvas || !overlayCanvas) { console.error("Canvas elements missing"); return; }
  const pdfCtx = pdfCanvas.getContext("2d");
  const overlayCtx = overlayCanvas.getContext("2d", { willReadFrequently: true });
  overlayCanvas.style.touchAction = "none";
  overlayCanvas.style.cursor = "crosshair";

  // Ensure a single wrapper with both canvases and an annotation layer
  const pageLayer = ensurePageLayer();
  const annotationLayer = ensureAnnotationLayer();

  // Save hooks
  eceBadge?.addEventListener("click", onSaveClick);
  downloadBtn?.addEventListener("click", onSaveClick);

  // Print
  printBtn?.addEventListener("click", async () => {
    try {
      if (!editMode) enterEdit();
      const bytes = await exportEditedPdf();
      const blob = new Blob([bytes], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const w = window.open(url, "_blank");
      setTimeout(() => { try { w?.print(); } catch {} }, 500);
    } catch (e) {
      console.error("Print failed", e);
      Swal.fire({ icon:"error", title:"Print failed", text: e?.message || "Could not generate PDF for printing." });
    }
  });

  // ---- Title ----
  if (formTitle) {
    const origName = (storedName || baseFromPath(storedWeb) || "Form").replace(/\.[^.]+$/, "");
    formTitle.textContent = origName || "Form";
  }

  // ---- Sidebar controls ----
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      sidebar.classList.toggle("open");
      sidebarToggle.style.display = sidebar.classList.contains("open") ? "none" : "flex";
    });
    document.addEventListener("click", (e) => {
      if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove("open");
        sidebarToggle.style.display = "flex";
      }
    });
  }
  pageToggler?.addEventListener("click", () => thumbSidebar?.classList.toggle("visible"));

  // ---- "Show boxes" toggle ----
  let toggleBoxes = null;
  if (metricsRow) {
    const boxesWrap = document.createElement("div");
    boxesWrap.style.cssText = "text-align:center;margin:6px 0 4px 0;";
    boxesWrap.innerHTML = `
      <label style="font-size:12px;opacity:.9;">
        <input type="checkbox" id="toggleBoxes"> Show boxes
      </label>`;
    metricsRow.insertAdjacentElement("afterend", boxesWrap);
    toggleBoxes = $("toggleBoxes");
  }

  // ---- PDF state ----
  let pdfDoc = null;
  let scale = 1.5;
  let currentPage = 1;

  // For proper scaling across zoom/DPR
  const pageBaseSize = {};    // pageNo -> { w, h } at scale 1 (CSS px)
  const pageCssSize  = {};    // pageNo -> { w, h } current viewport CSS size
  let cachedAnnotations = null;
  const pageHasFormFields = {}; // pageNo -> boolean

  if (typeof pdfjsLib === "undefined") { alert("PDF.js failed to load."); return; }

  // Boot viewer
  (async function boot() {
    try {
      if (!storedWeb) {
        const pick = await pickAndUploadFile();
        persistUpload(pick);
        await openPdf(pick.web_path);
      } else {
        try {
          await openPdf(storedWeb);
          if (!storedDisk) {
            const reup = await softReuploadForDisk(storedWeb);
            if (reup) persistUpload(reup);
          }
        } catch {
          const uploaded = await softReuploadForDisk(storedWeb);
          if (uploaded) {
            persistUpload(uploaded);
            await openPdf(uploaded.web_path);
          } else {
            const pick = await pickAndUploadFile();
            persistUpload(pick);
            await openPdf(pick.web_path);
          }
        }
      }
    } catch (e3) {
      console.error("[viewer] load failed:", e3);
      alert("Failed to load PDF.");
      return;
    }
    // start edit-ready
    enterEdit();
  })();

  // Re-render on resize / DPR change
  window.addEventListener("resize", () => renderPage(currentPage));

  function persistUpload(obj) {
    if (!obj) return;
    if (obj.web_path) sessionStorage.setItem("uploadedWebPath", normalizeToWebUrl(obj.web_path));
    if (obj.disk_path) sessionStorage.setItem("uploadedDiskPath", obj.disk_path);
    if (obj.form_id) { sessionStorage.setItem("uploadedFormId", obj.form_id); currentFormId = obj.form_id; }
    if (obj.file_name) {
      sessionStorage.setItem("uploadedFileName", obj.file_name);
      if (formTitle) formTitle.textContent = obj.file_name.replace(/\.[^.]+$/, "");
    }
    storedDisk = sessionStorage.getItem("uploadedDiskPath") || storedDisk;
  }

  async function openPdf(src) {
    const url = normalizeToWebUrl(src);
    try { pdfDoc = await pdfjsLib.getDocument(url).promise; }
    catch {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`PDF fetch failed (${r.status})`);
      const ab = await r.arrayBuffer();
      pdfDoc = await pdfjsLib.getDocument({ data: new Uint8Array(ab) }).promise;
    }
    renderPage(currentPage);
    buildThumbnails(pdfDoc);
  }

  function renderPage(pageNum) {
    pdfDoc.getPage(pageNum).then(async (page) => {
      const viewport = page.getViewport({ scale });
      const baseViewport = page.getViewport({ scale: 1 });

      if (!pageBaseSize[pageNum]) {
        pageBaseSize[pageNum] = { w: baseViewport.width, h: baseViewport.height };
      }

      // Record current CSS size for this page
      pageCssSize[pageNum] = { w: viewport.width, h: viewport.height };

       // Device Pixel Ratio
      const dpr = window.devicePixelRatio || 1;

      // Set wrapper CSS size
      pageLayer.style.width  = `${viewport.width}px`;
      pageLayer.style.height = `${viewport.height}px`;

      // Set canvas CSS size
      setCssSize(pdfCanvas, viewport.width, viewport.height);
      setCssSize(overlayCanvas, viewport.width, viewport.height);
      setCssSize(annotationLayer, viewport.width, viewport.height);

      // Internal resolution in device pixels
      setDeviceSize(pdfCanvas, viewport.width * dpr, viewport.height * dpr);
      setDeviceSize(overlayCanvas, viewport.width * dpr, viewport.height * dpr);

      /// ✅ Correct transform: just scale by DPR, no double-flip
      pdfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      try {
        const annots = await page.getAnnotations({ intent: "display" });
        pageHasFormFields[pageNum] = Array.isArray(annots) && annots.some(a => a && a.subtype === "Widget");
      } catch { pageHasFormFields[pageNum] = false; }

      page.render({ canvasContext: pdfCtx, viewport, renderInteractiveForms: true }).promise.then(() => {
        if (pageInfo) pageInfo.textContent = `Page ${pageNum} / ${pdfDoc.numPages}`;
        if (zoomInfo) zoomInfo.textContent = `${Math.round(scale * 100)}%`;

        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        if (toggleBoxes && toggleBoxes.checked && currentFormId) drawOverlay(currentFormId, pageNum);
        if (editMode) paintEdits(pageNum);

        applyPointerRouting();
      });
    });
  }

  function buildThumbnails(pdf) {
    const host = thumbSidebar; if (!host) return;
    host.innerHTML = "";
    for (let n = 1; n <= pdf.numPages; n++) {
      pdf.getPage(n).then((page) => {
        const viewport = page.getViewport({ scale: 0.3 });
        const c = document.createElement("canvas");
        c.width = viewport.width; c.height = viewport.height;
        page.render({ canvasContext: c.getContext("2d"), viewport }).promise.then(() => {
          const w = document.createElement("div");
          w.className = "thumbnail-wrapper";
          c.className = "thumbnail";
          c.title = `Page ${page.pageNumber}`;
          c.addEventListener("click", () => { currentPage = page.pageNumber; renderPage(currentPage); });
          const label = document.createElement("div");
          label.className = "thumbnail-label";
          label.textContent = `Page ${page.pageNumber}`;
          w.appendChild(c); w.appendChild(label);
          host.appendChild(w);
        });
      });
    }
  }

  // ========================
  // Run Analysis
  // ========================
  analyzeBtn?.addEventListener("click", runAnalysis);
  async function runAnalysis() {
    Swal.fire({ title: "Analyzing Form...", text: "Preparing summaries and overlays", allowOutsideClick: false, didOpen: () => Swal.showLoading() });
    try {
      let uploadInfo = await ensureUploadedToServer(storedWeb);
      if (uploadInfo) persistUpload(uploadInfo);
      const web_path  = sessionStorage.getItem("uploadedWebPath");
      const disk_path = sessionStorage.getItem("uploadedDiskPath");
      let   hashId    = sessionStorage.getItem("uploadedFormId") || currentFormId;
      if (!disk_path) throw new Error("Upload failed to provide a disk path.");

      const pre = await ensurePrelabelAndOverlays({ disk_path }, hashId);
      if (pre && pre.canonical_form_id) {
        hashId = pre.canonical_form_id; currentFormId = hashId; sessionStorage.setItem("uploadedFormId", hashId);
      }

      const guess = guessFromPath(baseFromPath(web_path) || "form.pdf");
      const reg = await GET_json("/panel");
      let explainer = await resolveExplainerByHash(reg, hashId);
      if (!explainer) {
        await POST_json("/api/explainer.ensure", {
          canonical_form_id: hashId, bucket: guess.bucket, human_title: guess.title,
          pdf_disk_path: disk_path, aliases: [guess.formId, baseFromPath(web_path)]
        });
        const reg2 = await GET_json("/panel");
        explainer = await resolveExplainerByHash(reg2, hashId);
      }
      if (!explainer) throw new Error("Failed to load explainer.");

      if (formTitle) formTitle.textContent = explainer.title || (storedName || "Form");
      renderSummaries(explainer);
      if (toggleBoxes && toggleBoxes.checked && currentFormId) await drawOverlay(currentFormId, currentPage);

      Swal.fire({ icon: "success", title: "Analysis Complete", timer: 1200, showConfirmButton: false });
    } catch (e) {
      console.error("runAnalysis error:", e);
      Swal.fire({ icon: "error", title: "Analysis Failed", text: e?.message || "Could not analyze this form." });
    }
  }

  function renderSummaries(explainer) {
    if (summaryList) summaryList.innerHTML = "";
    (explainer.sections || []).forEach((sec) => {
      const item = document.createElement("div");
      item.className = "accordion-item";
      const header = document.createElement("div");
      header.className = "accordion-header";
      header.textContent = sec.title || "";
      const content = document.createElement("div");
      content.className = "accordion-content active";
      (sec.fields || []).forEach((f) => {
        const row = document.createElement("p");
        row.className = "summary-line";
        row.innerHTML = `<span class="summary-label" data-label="${esc(f.label)}">${esc(f.label)}</span>: ${esc(f.summary)}`;
        content.appendChild(row);
      });
      header.addEventListener("click", () => content.classList.toggle("active"));
      item.appendChild(header); item.appendChild(content);
      summaryList?.appendChild(item);
    });
    if (metricsRow) metricsRow.textContent = "";
  }

  // click → jump
  summaryList?.addEventListener("click", async (ev) => {
    const lbl = ev.target.closest(".summary-label");
    if (!lbl || !currentFormId) return;
    try {
      if (!cachedAnnotations || cachedAnnotations.__formId !== currentFormId) {
        const res = await fetch(`/explanations/_annotations/${currentFormId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error("Annotations not found");
        cachedAnnotations = await res.json(); cachedAnnotations.__formId = currentFormId;
      }
      const anchor = findAnchorForLabel(lbl.textContent.trim(), cachedAnnotations);
      if (!anchor) return;
      currentPage = (anchor.page || 0) + 1;
      renderPage(currentPage);
      await new Promise((r) => setTimeout(r, 250));
      drawOverlay(currentFormId, currentPage, anchor);
    } catch (e) { console.warn("Label jump failed:", e); }
  });

  async function drawOverlay(formId, pageNumber, anchor = null) {
    try {
      if (!cachedAnnotations || cachedAnnotations.__formId !== formId) {
        const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) { overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); return; }
        cachedAnnotations = await res.json(); cachedAnnotations.__formId = formId;
      }
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      const pageIdx = pageNumber - 1;
      let rects = [];
      if (Array.isArray(cachedAnnotations.groups) && cachedAnnotations.groups.length) {
        rects = cachedAnnotations.groups.filter((g) => (g.page || 0) === pageIdx);
      } else if (Array.isArray(cachedAnnotations.tokens)) {
        rects = cachedAnnotations.tokens.filter((t) => (t.page || 0) === pageIdx);
      }

      const base = pageBaseSize[pageNumber];                  // scale=1 size from renderPage()
      const cssW = pageLayer.clientWidth;                     // wrapper’s CSS width
      const cssH = pageLayer.clientHeight;                    // wrapper’s CSS height
      const sx = cssW / base.w;                               // CSS scale only
      const sy = cssH / base.h;                               // CSS scale only

      overlayCtx.save();
      overlayCtx.lineWidth = 1;
      overlayCtx.strokeStyle = "rgba(20,20,20,0.25)";
      rects.forEach((r) => {
        const [x0,y0,x1,y1] = r.bbox;
        overlayCtx.strokeRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
      });

      if (anchor && Array.isArray(anchor.bbox)) {
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeStyle = "rgba(0,0,0,0.95)";
        overlayCtx.fillStyle   = "rgba(255,255,0,0.2)";
        const [x0,y0,x1,y1] = anchor.bbox;
        overlayCtx.fillRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
        overlayCtx.strokeRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
      }
      overlayCtx.restore();
    } catch {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  document.addEventListener("change", (ev) => {
    if (ev.target && ev.target.id === "toggleBoxes") {
      if (ev.target.checked && currentFormId) drawOverlay(currentFormId, currentPage);
      else overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  });

  // ---- Backend helpers ----
  async function softReuploadForDisk(webUrl) { try {
    const url = normalizeToWebUrl(webUrl);
    const r = await fetch(url, { cache: "no-store" }); if (!r.ok) return null;
    const blob = await r.blob(); const base = baseFromPath(url) || "form.pdf";
    const fd = new FormData(); fd.append("file", new File([blob], base, { type: "application/pdf" }));
    const up = await fetch("/api/upload", { method: "POST", body: fd }); if (!up.ok) return null;
    const out = await up.json();
    return { web_path: normalizeToWebUrl(out.web_path), disk_path: out.disk_path, form_id: out.canonical_form_id || out.form_id, file_name: base };
  } catch { return null; } }

  async function ensureUploadedToServer(webUrl) {
    const sessWeb = sessionStorage.getItem("uploadedWebPath");
    const sessDisk = sessionStorage.getItem("uploadedDiskPath");
    const sessForm = sessionStorage.getItem("uploadedFormId");
    if (sessWeb && sessDisk && sessForm) {
      return { web_path: normalizeToWebUrl(sessWeb), disk_path: sessDisk, form_id: sessForm, file_name: sessionStorage.getItem("uploadedFileName") };
    }
    if (typeof webUrl === "string" && webUrl) {
      const reup = await softReuploadForDisk(webUrl); if (reup) return reup;
    }
    return await pickAndUploadFile();
  }

  async function pickAndUploadFile() {
    const input = document.createElement("input");
    input.type = "file"; input.accept = "application/pdf"; input.style.display = "none";
    document.body.appendChild(input);
    const file = await new Promise((resolve, reject) => {
      input.addEventListener("change", () => { if (input.files && input.files[0]) resolve(input.files[0]); else reject(new Error("No file selected")); }, { once: true });
      input.click();
    }).finally(() => setTimeout(() => input.remove(), 0));
    const fd = new FormData(); fd.append("file", file);
    const up = await fetch("/api/upload", { method: "POST", body: fd });
    if (!up.ok) throw new Error("upload failed");
    const out = await up.json();
    return { web_path: normalizeToWebUrl(out.web_path), disk_path: out.disk_path, form_id: out.canonical_form_id || out.form_id, file_name: file.name };
  }

  async function ensurePrelabelAndOverlays(server, hashId) {
    const fd = new FormData();
    fd.append("pdf_disk_path", server.disk_path);
    fd.append("form_id", hashId);
    const r = await fetch("/api/prelabel", { method: "POST", body: fd });
    if (!r.ok) { const t = await r.text().catch(() => ""); throw new Error(t || "Prelabeling failed."); }
    return r.json();
  }

  async function resolveExplainerByHash(reg, hash) {
    const forms = Array.isArray(reg.forms) ? reg.forms : [];
    const hit = forms.find((f) => String(f.form_id || "") === String(hash || ""));
    if (!hit || !hit.path) return null;
    const url = "/" + String(hit.path).replace(/^\//, "");
    try { return await GET_json(url + (url.includes("?") ? "&" : "?") + "ts=" + Date.now()); } catch { return null; }
  }

  // ---- Utils ----
  async function GET_json(url) { const withTs = url.includes("?") ? url + "&ts=" + Date.now() : url + "?ts=" + Date.now(); const r = await fetch(withTs, { cache: "no-store" }); if (!r.ok) throw new Error(url); return r.json(); }
  async function POST_json(url, obj) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj || {}) });
    if (!r.ok) { const t = await r.text().catch(() => ""); throw new Error(t || `POST ${url} failed`); }
    return r.json();
  }
  function baseFromPath(p) { try { return String(p).split("/").pop().split("\\").pop().split("?")[0]; } catch { return p; } }
  function guessFromPath(stemLike) {
    const s = String(stemLike).toLowerCase();
    let bucket = "government";
    if (/(bdo|metrobank|slamci|fami|bank|account)/.test(s)) bucket = "banking";
    else if (/(bir|tax|2552|1604|1901|1902)/.test(s)) bucket = "tax";
    else if (/(manulife|sunlife|axa|fwd|claim|philhealth|allianz)/.test(s)) bucket = "healthcare";
    return { bucket, formId: s.replace(/\s+/g, "_").replace(/\.pdf$/i, ""), title: stemLike.replace(/\.pdf$/i, "") };
  }
  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] || c)); }
  function normalizeToWebUrl(p) {
    if (!p) return p;
    const s = String(p).replace(/\\/g, "/");
    if (s.startsWith("/uploads/")) return s;
    const m = s.match(/\/uploads\/([^\/?#]+)$/i); if (m) return "/uploads/" + m[1];
    const m2 = s.match(/\/?uploads\/([^\/?#]+)$/i); if (m2) return "/uploads/" + m2[1];
    return s;
  }
  function setCssSize(el, w, h) { el.style.width = `${w}px`; el.style.height = `${h}px`; }
  function setDeviceSize(canvas, w, h) { canvas.width = Math.max(1, Math.round(w)); canvas.height = Math.max(1, Math.round(h)); }

  // ===== Label helpers (robust fuzzy jump) =====
  function bboxUnion(a, b) {
    return [
      Math.min(a[0], b[0]),
      Math.min(a[1], b[1]),
      Math.max(a[2], b[2]),
      Math.max(a[3], b[3]),
    ];
  }

  // Tiny synonym map that helps common banking/ID terms align
  const SYN_MAP = {
    "tin": "tax identification number",
    "zipcode": "zip code",
    "zip": "zip code",
    "birthdate": "date of birth",
    "dob": "date of birth",
    "co owner": "co-owner",
    "coowner": "co-owner",
    "company tin": "tax identification number",
    "mobile": "mobile number",
    "telephone": "telephone number",
    "email": "email address",
  };

  function normalizeText(s) {
    if (!s) return "";
    // lowercase + strip diacritics
    s = s.toString().toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    // collapse punctuation to space, keep a-z0-9
    s = s.replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
    // synonym expansion
    for (const k of Object.keys(SYN_MAP)) {
      const re = new RegExp(`\\b${k}\\b`, "g");
      s = s.replace(re, SYN_MAP[k]);
    }
    return s;
  }

  function tokenize(s) {
    return normalizeText(s).split(" ").filter(Boolean);
  }

  function bigrams(tokens) {
    const out = [];
    for (let i = 0; i < tokens.length - 1; i++) out.push(tokens[i] + " " + tokens[i + 1]);
    return out;
  }

  // Weighted similarity: token overlap + bigram overlap + substring bonus + slight length preference
  function similarity(labelRaw, textRaw) {
    const L = tokenize(labelRaw);
    const T = tokenize(textRaw);
    if (!L.length || !T.length) return 0;

    const setT = new Set(T);
    const tokHits = L.filter(w => setT.has(w)).length;

    const LB = bigrams(L);
    const TB = new Set(bigrams(T));
    const biHits = LB.filter(bg => TB.has(bg)).length;

    const base = (tokHits / L.length) * 0.7 + (LB.length ? (biHits / LB.length) * 0.3 : 0);
    const lenPref = Math.min(1, L.length / Math.max(1, Math.log10(T.length + 2) + 1)) * 0.08;
    const substrBonus = normalizeText(textRaw).includes(normalizeText(labelRaw)) ? 0.12 : 0;

    return Math.min(1, base + lenPref + substrBonus);
  }

  function sameLine(tA, tB) {
    const midA = (tA.bbox[1] + tA.bbox[3]) / 2;
    const midB = (tB.bbox[1] + tB.bbox[3]) / 2;
    return Math.abs(midA - midB) < 8; // keep your tolerance
  }

  function unionLine(seed, pageTokens) {
    let box = seed.bbox.slice(0, 4);
    for (const t of pageTokens) if (sameLine(seed, t)) box = bboxUnion(box, t.bbox);
    return box;
  }

  // Main resolver with page prior + threshold
  function findAnchorForLabel(label, annotations) {
    if (!annotations) return null;
    const labelNorm = normalizeText(label);
    const MIN_SCORE = 0.60; // adjust 0.58–0.65 to taste

    // 1) Prefer exact normalized group label match
    if (Array.isArray(annotations.groups) && annotations.groups.length) {
      const exact = annotations.groups.find(g => normalizeText(g.label || "") === labelNorm);
      if (exact && Array.isArray(exact.bbox) && exact.bbox.length === 4) {
        return { page: (exact.page || 0), bbox: exact.bbox, source: "group-exact", score: 1 };
      }
    }

    const tokens = Array.isArray(annotations.tokens) ? annotations.tokens : [];
    if (!tokens.length && !(annotations.groups && annotations.groups.length)) return null;

    // 2) Page prior: if groups exist, use their pages; else rank pages by token similarity
    let pageCandidates = null;

    if (annotations.groups && annotations.groups.length) {
      const scoredGroups = annotations.groups
        .map(g => ({ g, s: similarity(label, g.label || "") }))
        .filter(x => x.s > 0);
      if (scoredGroups.length) {
        scoredGroups.sort((a, b) => b.s - a.s);
        pageCandidates = [...new Set(scoredGroups.slice(0, 2).map(x => x.g.page || 0))]; // top 1–2 pages
      }
    }

    if (!pageCandidates) {
      const pageScore = new Map(); // page -> max score
      for (const t of tokens) {
        const s = similarity(label, t.text || "");
        if (s <= 0) continue;
        const p = t.page || 0;
        pageScore.set(p, Math.max(pageScore.get(p) || 0, s));
      }
      const sorted = [...pageScore.entries()].sort((a, b) => b[1] - a[1]).map(([p]) => p);
      if (!sorted.length) return null;
      pageCandidates = sorted.slice(0, 1);
    }

    // 3) Within candidates, pick best token and expand to its whole line
    let best = null;
    for (const p of pageCandidates) {
      const pageTokens = tokens.filter(t => (t.page || 0) === p);
      for (const t of pageTokens) {
        const s = similarity(label, t.text || "");
        const leftBias = (t.bbox?.[0] ?? 1e9) / 1e6; // tiny preference for left-most labels
        const score = s - leftBias;
        if (!best || score > best.score) best = { token: t, score, page: p };
      }
    }

    if (!best || best.score < MIN_SCORE) return null;

    const lineBox = unionLine(best.token, tokens.filter(t => (t.page || 0) === best.page));
    return { page: best.page, bbox: lineBox, score: best.score, source: "token-fuzzy" };
  }

  // =========================
  // EDIT LAYER
  // =========================
  let editMode = false;
  let currentTool = null;     // << no default active
  let drawing = false;

  const pageEdits = {};
  const EDIT = (p) => (pageEdits[p] ??= { strokes: [], texts: [] });

  // Draw config (defaults; UI sliders can override)
  const PEN = { color: "#111111", width: 2.0,  alpha: 1.0 };
  const HL  = { color: "rgba(255,255,0,0.35)", width: 10.0, alpha: 0.35 };

  // Apply live pen/highlight settings from dropdowns if present
  const penWidthInput = document.getElementById("penWidth");
  const penColorInput = document.getElementById("penColor");
  const hlWidthInput  = document.getElementById("hlWidth");
  const hlColorInput  = document.getElementById("hlColor");
  function currentPen() {
    return {
      color: (penColorInput && penColorInput.value) || PEN.color,
      width: (penWidthInput && (+penWidthInput.value || PEN.width)) || PEN.width,
      alpha: PEN.alpha
    };
  }
  function currentHL() {
    const col = (hlColorInput && hlColorInput.value) || "#ffff00";
    const alpha = 0.35;
    return {
      color: rgbaFromHex(col, alpha),
      width: (hlWidthInput && (+hlWidthInput.value || HL.width)) || HL.width,
      alpha
    };
  }
  function rgbaFromHex(hex, a) {
    const h = hex.replace("#",""); const r = parseInt(h.slice(0,2),16); const g = parseInt(h.slice(2,4),16); const b = parseInt(h.slice(4,6),16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // Track active text node (for keyboard size tweaks)
  let activeTextNode = null;

  // tool selection — click again to deselect
  document.getElementById("editToolbar")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".tool-btn"); if (!btn || btn.disabled) return;
    const name = btn.dataset.tool;
    if (currentTool === name) {
      currentTool = null;
      toolButtons.forEach(b => b.classList.remove("active"));
      applyPointerRouting();
      return;
    }
    currentTool = name;
    toolButtons.forEach(b => b.classList.toggle("active", b === btn));
    applyPointerRouting();
  });

  // pointer routing
  function applyPointerRouting() {
    if (!editMode || !currentTool) { overlayCanvas.style.pointerEvents = "none"; return; }
    if (currentTool === "text" && pageHasFormFields[currentPage]) overlayCanvas.style.pointerEvents = "none";
    else overlayCanvas.style.pointerEvents = "auto";
    document.body.classList.toggle("text-mode", currentTool === "text");
    document.body.classList.toggle("pen-mode", currentTool === "pen");
    document.body.classList.toggle("hl-mode", currentTool === "highlight");
  }

  // scale helpers (base<->current CSS)
  function pageScaleFactors(pageNum) {
    const base = pageBaseSize[pageNum] || { w: overlayCanvas.width, h: overlayCanvas.height };
    const cssW = pageLayer?.clientWidth  || overlayCanvas.clientWidth  || base.w;
    const cssH = pageLayer?.clientHeight || overlayCanvas.clientHeight || base.h;
    return { sx: cssW / base.w, sy: cssH / base.h, invx: base.w / cssW, invy: base.h / cssH };
  }
  
  // CSS pixel coordinates under mouse
  function toCssXY(evt) {
    const rect = overlayCanvas.getBoundingClientRect();
    return { x: (evt.clientX - rect.left), y: (evt.clientY - rect.top) };
  }

  // paint strokes (convert base → current CSS)
  function paintEdits(pageNum) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const ed = pageEdits[pageNum]; if (!ed) return;
    const { sx, sy } = pageScaleFactors(pageNum);
    for (const s of ed.strokes) {
      overlayCtx.save();
      overlayCtx.lineJoin = "round"; overlayCtx.lineCap = "round";
      overlayCtx.lineWidth = s.width;
      overlayCtx.strokeStyle = s.color;
      overlayCtx.globalAlpha = s.alpha ?? 1.0;
      overlayCtx.beginPath();
      s.points.forEach((pt, i) => {
        const X = pt.x * sx, Y = pt.y * sy;
        if (i===0) overlayCtx.moveTo(X, Y); else overlayCtx.lineTo(X, Y);
      });
      overlayCtx.stroke();
      overlayCtx.restore();
    }
  }

  // --- Draggable + editable text nodes (Acrobat-like) ---
  function spawnTextNode(cssX, cssY) {
    const pageNum = currentPage;
    const { invx, invy, sx, sy } = pageScaleFactors(pageNum);

    const wrap = document.createElement("div");
    wrap.className = "text-annot";
    wrap.style.cssText = `
      position:absolute; left:${cssX}px; top:${cssY}px;
      outline:1.5px dashed #1e90ff; outline-offset:2px; border-radius:4px; padding:6px 8px 8px 28px;
      background:rgba(255,255,255,0.01); user-select:none;`;
    wrap.dataset.page = String(pageNum);

    // handle (six dots)
    const handle = document.createElement("div");
    handle.className = "ta-handle";
    handle.title = "Drag";
    handle.style.cssText = `
      position:absolute; left:4px; top:6px; width:16px; height:20px; cursor:grab; display:grid; gap:2px;
      grid-template-columns: repeat(2, 4px); grid-auto-rows: 4px;`;
    for (let i=0;i<6;i++){
      const dot=document.createElement("div");
      dot.style.cssText="width:4px;height:4px;border-radius:50%;background:#1e90ff;opacity:.9;";
      handle.appendChild(dot);
    }
    wrap.appendChild(handle);

    // content (editable)
    const content = document.createElement("div");
    content.className = "ta-content";
    content.contentEditable = "true";
    content.spellcheck = true;
    content.style.cssText = `
      min-width:30px; min-height:18px; line-height:1.25; color:#000;
      font:14px Inter, system-ui, sans-serif; user-select:text; cursor:text;`;
    content.textContent = "Type here";
    wrap.appendChild(content);

    // mini toolbar
    const bar = document.createElement("div");
    bar.className = "ta-toolbar";
    bar.style.cssText = `
      position:absolute; left:0; transform:translateY(100%); margin-top:6px; padding:6px 8px;
      border-radius:8px; box-shadow:0 6px 24px rgba(0,0,0,.12); background:#fff; display:none; gap:6px;`;
    bar.innerHTML = `
      <button data-act="smaller" title="Decrease font">A<span style="font-size:10px;vertical-align:super;">ˇ</span></button>
      <button data-act="larger" title="Increase font">A<span style="font-size:10px;vertical-align:super;">^</span></button>
      <button data-act="delete" title="Delete">🗑️</button>`;
    Array.from(bar.querySelectorAll("button")).forEach(b => {
      b.style.cssText = "border:1px solid #e5e7eb;border-radius:6px;padding:4px 8px;background:#fff;cursor:pointer;";
    });
    wrap.appendChild(bar);

    const model = {
      id: "t" + Date.now() + Math.random().toString(36).slice(2,6),
      x: cssX * invx, y: cssY * invy,                  // store in base coords
      text: "Type here", font: "Inter", size: 14, color: "#000000"
    };
    const ed = EDIT(pageNum); ed.texts.push(model);
    wrap.dataset.key = model.id;

    // focus/toolbar
    const showBar = () => { bar.style.display = "flex"; activeTextNode = wrap; };
    const hideBar = () => { bar.style.display = "none"; if (activeTextNode === wrap) activeTextNode = null; };

    content.addEventListener("focus", showBar);
    content.addEventListener("blur", () => setTimeout(() => { if (!wrap.contains(document.activeElement)) hideBar(); }, 0));

    // typing sync
    content.addEventListener("input", () => {
      model.text = content.textContent || "";
      model.size = parseInt(getComputedStyle(content).fontSize,10) || model.size;
    });

    // toolbar actions
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      if (btn.dataset.act === "smaller") {
        model.size = Math.max(8, (model.size || 14) - 1);
        content.style.fontSize = `${model.size}px`;
      } else if (btn.dataset.act === "larger") {
        model.size = Math.min(72, (model.size || 14) + 1);
        content.style.fontSize = `${model.size}px`;
      } else if (btn.dataset.act === "delete") {
        const idx = ed.texts.findIndex(t => t.id === model.id);
        if (idx >= 0) ed.texts.splice(idx,1);
        wrap.remove();
      }
      content.focus();
    });

    // drag only from handle or border
    let dragging = false, offX = 0, offY = 0;
    function startDrag(e) {
      dragging = true;
      wrap.setPointerCapture?.(e.pointerId);
      const rect = wrap.getBoundingClientRect();
      offX = e.clientX - rect.left; offY = e.clientY - rect.top;
      e.preventDefault();
    }
    function moveDrag(e) {
      if (!dragging) return;
      const parentRect = (pdfCanvas.parentElement || document.body).getBoundingClientRect();
      const nx = e.clientX - parentRect.left - offX;
      const ny = e.clientY - parentRect.top  - offY;
      wrap.style.left = `${nx}px`; wrap.style.top = `${ny}px`;
      model.x = nx * invx; model.y = ny * invy;       // keep base coords
    }
    function endDrag(e){ if(!dragging)return; dragging=false; wrap.releasePointerCapture?.(e.pointerId); }

    handle.addEventListener("pointerdown", startDrag);
    wrap.addEventListener("pointerdown", (e) => { if (e.target === wrap) startDrag(e); });
    wrap.addEventListener("pointermove", moveDrag);
    ["pointerup","pointercancel","pointerleave"].forEach(t => wrap.addEventListener(t, endDrag));

    // insert
    const parent = pdfCanvas.parentElement || document.body;
    if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
    parent.appendChild(wrap);
    content.focus();
    const sel = window.getSelection(); const range = document.createRange();
    range.selectNodeContents(content); range.collapse(false); sel.removeAllRanges(); sel.addRange(range);
  }

  // Edit mode toggle
  enterEditBtn?.addEventListener("click", () => { if (!editMode) enterEdit(); else exitEdit(); });
  function enterEdit() {
    editMode = true;
    toolButtons.forEach(b=> b.disabled = false);
    undoBtn.disabled = false; clearBtn.disabled = false;
    enterEditBtn?.classList.add("disabled"); if (enterEditBtn) enterEditBtn.textContent = "✎ Editing";
    paintEdits(currentPage);
    applyPointerRouting();
  }
  function exitEdit() {
    editMode = false;
    toolButtons.forEach(b=> { b.disabled = true; b.classList.remove("active"); });
    undoBtn.disabled = true; clearBtn.disabled = true;
    enterEditBtn?.classList.remove("disabled"); if (enterEditBtn) enterEditBtn.textContent = "✎ Edit";
    overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
    document.querySelectorAll(".text-annot").forEach(n => n.remove());
    activeTextNode = null;
    applyPointerRouting();
  }

  // Undo/Clear
  undoBtn?.addEventListener("click", () => {
    if (!editMode) return;
    const ed = EDIT(currentPage);
    if (ed.strokes.length) ed.strokes.pop();
    paintEdits(currentPage);
  });
  clearBtn?.addEventListener("click", async () => {
    if (!editMode) return;
    const ok = await Swal.fire({
      title: "Clear this page?", text: "All drawings and inserted text on the current page will be removed.",
      icon: "warning", showCancelButton: true, confirmButtonText: "Yes, clear page", cancelButtonText: "Cancel"
    }).then(r=>r.isConfirmed);
    if (!ok) return;
    pageEdits[currentPage] = { strokes: [], texts: [] };
    [...document.querySelectorAll(".text-annot")].forEach(n => n.remove());
    paintEdits(currentPage);
  });

  // Smooth drawing (pointer capture + rAF) + base-coordinate storage
  let rafPending = false;
  function schedulePaint() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; paintEdits(currentPage); });
  }

  overlayCanvas.addEventListener("pointerdown", (e) => {
    if (!editMode || !currentTool) return;

    if (currentTool === "text") {
      if (pageHasFormFields[currentPage]) return;  // yield to native fields
      const css = toCssXY(e);
      spawnTextNode(css.x, css.y);
      return;
    }

    if (currentTool === "pen" || currentTool === "highlight" || currentTool === "erase") {
      drawing = true;
      overlayCanvas.setPointerCapture?.(e.pointerId);
      e.preventDefault();

      const css = toCssXY(e);
      const { invx, invy } = pageScaleFactors(currentPage);
      const startBase = { x: css.x * invx, y: css.y * invy };

      if (currentTool !== "erase") {
        const conf = currentTool === "pen" ? currentPen() : currentHL();
        EDIT(currentPage).strokes.push({ tool: currentTool, color: conf.color, width: conf.width, alpha: conf.alpha, points: [ startBase ] });
      } else {
        const ed = EDIT(currentPage); const R = 10;
        // Compare in base space: convert radius to base units using inv factors (approx with x)
        const rBase = R * invx;
        ed.strokes = ed.strokes.filter(st => !st.points.some(p => Math.hypot(p.x-startBase.x, p.y-startBase.y) < rBase));
      }
      schedulePaint();
    }
  });

  overlayCanvas.addEventListener("pointermove", (e) => {
    if (!editMode || !drawing) return;
    const css = toCssXY(e);
    const { invx, invy } = pageScaleFactors(currentPage);
    const ptBase = { x: css.x * invx, y: css.y * invy };

    if (currentTool === "erase") {
      const ed = EDIT(currentPage); const R = 10;
      const rBase = R * invx;
      ed.strokes = ed.strokes.filter(st => !st.points.some(p => Math.hypot(p.x-ptBase.x, p.y-ptBase.y) < rBase));
    } else {
      const ed = EDIT(currentPage);
      const stroke = ed.strokes[ed.strokes.length-1];
      if (stroke) stroke.points.push(ptBase);
    }
    schedulePaint();
  });

  ["pointerup","pointerleave","pointercancel"].forEach(type=>{
    overlayCanvas.addEventListener(type, (e)=>{ 
      if (!drawing) return; drawing=false; overlayCanvas.releasePointerCapture?.(e.pointerId);
    });
  });

  // ===== Keyboard shortcuts =====
  document.addEventListener("keydown", (e) => {
    if (!editMode) return;
    if (e.ctrlKey && e.key.toLowerCase() === "z") { undoBtn?.click(); return; }

    if (!e.ctrlKey && !e.metaKey) {
      if (e.key.toLowerCase() === "t") { toggleTool("text"); return; }
      if (e.key.toLowerCase() === "h") { toggleTool("highlight"); return; }
      if (e.key.toLowerCase() === "d") { toggleTool("pen"); return; }
      if (e.key.toLowerCase() === "e") { toggleTool("erase"); return; }
    }
    function toggleTool(name){
      currentTool = currentTool === name ? null : name;
      toolButtons.forEach(b => b.classList.toggle("active", b.dataset.tool === currentTool));
      applyPointerRouting();
    }

    // Font size adjust for active text node: '[' smaller, ']' larger
    if (activeTextNode) {
      const content = activeTextNode.querySelector(".ta-content");
      const pageNum = Number(activeTextNode.dataset.page);
      const ed = EDIT(pageNum);
      const m = ed.texts.find(t => t.id === activeTextNode.dataset.key);
      if (!m) return;
      if (e.key === "[" || e.key === "{") {
        m.size = Math.max(8, (m.size || 14) - 1);
        content.style.fontSize = `${m.size}px`;
        e.preventDefault();
      } else if (e.key === "]" || e.key === "}") {
        m.size = Math.min(72, (m.size || 14) + 1);
        content.style.fontSize = `${m.size}px`;
        e.preventDefault();
      }
    }
  });

  // ---- Save / Download ----
  async function onSaveClick() {
    const ok = await Swal.fire({
      title: "Save edited PDF?",
      text: "Export and download a copy with your drawings and inserted text.",
      icon: "question",
      showCancelButton: true,
      confirmButtonText: "Save",
      cancelButtonText: "Cancel"
    }).then(r=>r.isConfirmed);
    if (!ok) return;

    if (!editMode) enterEdit();
    try {
      const bytes = await exportEditedPdf();
      const base = (sessionStorage.getItem("uploadedFileName") || "form.pdf").replace(/\.pdf$/i,"");
      const filename = `${base}_edited.pdf`;
      const blob = new Blob([bytes], { type: "application/pdf" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = filename;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(a.href);
      Swal.fire({ icon:"success", title:"Saved", timer:1000, showConfirmButton:false });
    } catch (e) {
      console.error("Save failed", e);
      Swal.fire({ icon:"error", title:"Save failed", text: e?.message || "Could not generate edited PDF." });
    }
  }

  // ---- Export edited PDF ----
  async function exportEditedPdf() {
    if (!pdfDoc) throw new Error("No document open");
    const { PDFDocument } = (window.PDFLib || {}); if (!PDFDocument) throw new Error("pdf-lib not loaded");

    const pngPages = [];
    for (let n=1; n<=pdfDoc.numPages; n++) {
      const page = await pdfDoc.getPage(n);
      // Use current scale for consistency with on-screen view
      const viewport = page.getViewport({ scale });
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(viewport.width));
      c.height = Math.max(1, Math.round(viewport.height));
      const ctx = c.getContext("2d");
      await page.render({ canvasContext: ctx, viewport }).promise;

      const ed = pageEdits[n];
      if (ed) {
        // Scale base -> export viewport
        const base = pageBaseSize[n] || { w: viewport.width, h: viewport.height };
        const sx = viewport.width / base.w;
        const sy = viewport.height / base.h;

        for (const s of ed.strokes) {
          ctx.save();
          ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.lineWidth = s.width;
          ctx.globalAlpha = s.alpha ?? 1.0; ctx.strokeStyle = s.color;
          ctx.beginPath(); s.points.forEach((pt,i)=>{
            const X = pt.x * sx, Y = pt.y * sy;
            if(i===0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          });
          ctx.stroke(); ctx.restore();
        }

        for (const t of ed.texts) {
          ctx.save(); ctx.globalAlpha = 1.0; ctx.fillStyle = t.color || "#000";
          const size = (t.size || 14);
          ctx.font = `${size}px ${t.font || "Inter"}, sans-serif`;
          ctx.fillText(t.text || "", t.x*sx, t.y*sy + size);
          ctx.restore();
        }
      }
      const dataUrl = c.toDataURL("image/png");
      pngPages.push({ dataUrl, w: c.width, h: c.height });
    }

    const outPdf = await PDFDocument.create();
    try {
      if (currentFormId) {
        outPdf.setSubject(`IntelliForm-FormId:${currentFormId}`);
        outPdf.setProducer("IntelliForm");
      }
      const baseTitle = formTitle?.textContent || "Edited Form";
      outPdf.setTitle(`${baseTitle} (edited)`);
      outPdf.setCreator("IntelliForm");
    } catch {}

    for (const p of pngPages) {
      const png = await outPdf.embedPng(p.dataUrl);
      const page = outPdf.addPage([p.w, p.h]);
      page.drawImage(png, { x: 0, y: 0, width: p.w, height: p.h });
    }
    const bytes = await outPdf.save();

    // optional: register checksum so backend can map edited files back to the form id
    try {
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const hex = Array.from(new Uint8Array(digest)).map(b=>b.toString(16).padStart(2,"0")).join("");
      await fetch("/api/edited.register", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({
          sha256: hex,
          form_id: currentFormId || null,
          source_disk_path: sessionStorage.getItem("uploadedDiskPath") || null,
          source_file_name: sessionStorage.getItem("uploadedFileName") || null
        })
      });
    } catch (e) { console.warn("edited.register failed", e); }

    return bytes;
  }

  // ---- Dynamic wrapper helpers ----
  function ensurePageLayer() {
    // If already present (from HTML update), reuse it
    let layer = document.querySelector("#pdfContainer .pageLayer");
    if (layer) return layer;

    // Otherwise create and wrap canvases
    const container = document.getElementById("pdfContainer");
    layer = document.createElement("div");
    layer.id = "pageLayer";
    layer.className = "pageLayer";
    layer.style.position = "relative";
    layer.style.marginTop = "60px";

    // Move canvases inside the layer
    const first = pdfCanvas; const second = overlayCanvas;
    // Remove any absolute on overlay for now; CSS will handle later
    second.style.left = "0"; second.style.top = "0";
    second.style.position = "absolute";
    second.style.pointerEvents = "none";

    // Clear container then append wrapper with canvases
    if (container) {
      container.innerHTML = "";
      container.appendChild(layer);
      layer.appendChild(first);
      layer.appendChild(second);
    }
    return layer;
  }

  function ensureAnnotationLayer() {
    let anno = document.getElementById("annotationLayer");
    if (anno) return anno;
    anno = document.createElement("div");
    anno.id = "annotationLayer";
    anno.className = "annotationLayer";
    anno.style.position = "absolute";
    anno.style.left = "0";
    anno.style.top = "0";
    anno.style.width = "100%";
    anno.style.height = "100%";
    anno.style.pointerEvents = "none"; // we’re not mounting native inputs yet
    pageLayer.insertBefore(anno, overlayCanvas); // between pdf and overlay
    return anno;
  }

  // ===== Keyboard-aware tool toggles helpers (exported within closure) =====
  function toggleTool(name){
    currentTool = currentTool === name ? null : name;
    toolButtons.forEach(b => b.classList.toggle("active", b.dataset.tool === currentTool));
    applyPointerRouting();
  }
}
