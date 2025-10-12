// static/js/workspace.js

window.addEventListener("load", initWorkspace);

function initWorkspace() {
  console.log("workspace.js 2025-10-12 overlay-split: boxesCanvas + drawCanvas, fixed z-index, label-jump");

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
  const searchTool     = $("searchTool");
  const searchInput    = $("searchInput");
  const searchNextBtn  = $("searchNext");
  const searchPrevBtn  = $("searchPrev");
  const searchClearBtn = $("searchClear");
  const searchStatus   = $("searchStatus");


  // Base canvases coming from HTML
  let pdfCanvas = $("pdfCanvas");
  let overlayCanvas = $("overlayCanvas"); // <- will become "draw canvas" only

  const eceBadge = $("eceScoreBadge");
  const downloadBtn = $("downloadPDF");
  const printBtn = $("printPDF");

    // Exit button → confirm, log, and start fresh
    const exitBtn = document.getElementById("btnExit");
    exitBtn?.addEventListener("click", async () => {
      const ok = await Swal.fire({
        title: "Exit and start over?",
        text: "This will discard the current workspace and return to the home screen.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "Exit",
        cancelButtonText: "Stay"
      }).then(r => r.isConfirmed);

      if (!ok) return;

      // best-effort session close log, then reset
      try { ws_sendAbandonBeacon("exit_clicked"); } catch {}
      resetAndGoHome();
    });

  // Toolbar
  const enterEditBtn = $("enterEdit");        // optional; we auto-enter
  const undoBtn = $("btnUndo");
  const clearBtn = $("btnClear");
  const toolButtons = Array.from(document.querySelectorAll(".tool-btn"));

  if (!pdfCanvas || !overlayCanvas) { console.error("Canvas elements missing"); return; }

  const pdfCtx = pdfCanvas.getContext("2d");
  const overlayCtx = overlayCanvas.getContext("2d", { willReadFrequently: true }); // drawing layer only
  overlayCanvas.style.touchAction = "none";
  overlayCanvas.style.cursor = "crosshair";

  // Ensure wrapper + annotation layer + BOXES canvas (new)
  const pageLayer = ensurePageLayer();
  const boxesCanvas = ensureBoxesCanvas();             // <— NEW canvas for boxes/highlights
  const boxesCtx = boxesCanvas.getContext("2d");
  const annotationLayer = ensureAnnotationLayer();

  // Z-order (bottom → top):
  // pdfCanvas (z:0) → boxesCanvas (z:1) → annotationLayer (z:2) → overlayCanvas (z:3) → .text-annot (z:4)
  pdfCanvas.style.zIndex = "0";
  boxesCanvas.style.zIndex = "1";
  annotationLayer.style.zIndex = "2";
  overlayCanvas.style.zIndex = "3";

  // Save hooks
  eceBadge?.addEventListener("click", onSaveClick);
  downloadBtn?.addEventListener("click", onSaveClick);

  // Print
  printBtn?.addEventListener("click", async () => {
    // confirmation moment is the print click
    const tConfirm = ws_now();

    try {
      if (!editMode) enterEdit();

      const t0 = ws_now();
      const bytes = await exportEditedPdf();
      const renderMs = ws_now() - t0;

      const blob = new Blob([bytes], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const w = window.open(url, "_blank");
      setTimeout(() => { try { w?.print(); } catch {} }, 500);

      // Log completion with finished_at = user click time
      await logWorkspaceEvent("printed", tConfirm, { render_ms: renderMs });

      // Optional follow-up: ask to start fresh after print
      const startFresh = await Swal.fire({
        title: "Start a new session?",
        text: "Return to the home screen to upload a new PDF.",
        icon: "question",
        showCancelButton: true,
        confirmButtonText: "Yes, start fresh",
        cancelButtonText: "Stay here"
      }).then(r => r.isConfirmed);

      if (startFresh) resetAndGoHome();

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
  // let toggleBoxes = null;
  // if (metricsRow) {
  //   const boxesWrap = document.createElement("div");
  //   boxesWrap.style.cssText = "text-align:center;margin:6px 0 4px 0;";
  //   boxesWrap.innerHTML = `
  //     <label style="font-size:12px;opacity:.9;">
  //       <input type="checkbox" id="toggleBoxes"> Show boxes
  //     </label>`;
  //   metricsRow.insertAdjacentElement("afterend", boxesWrap);
  //   toggleBoxes = $("toggleBoxes");
  // }

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
    // Reset workspace session when a new file is loaded
    workspaceShownAt = null;
    workspaceFinishedAt = null;
    workspaceDuration = null;
    workspaceLogged = false;
    ws_clearInflight();
    
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
      setCssSize(boxesCanvas, viewport.width, viewport.height);     // NEW
      setCssSize(overlayCanvas, viewport.width, viewport.height);
      setCssSize(annotationLayer, viewport.width, viewport.height);

      // Internal resolution in device pixels
      setDeviceSize(pdfCanvas, viewport.width * dpr, viewport.height * dpr);
      setDeviceSize(boxesCanvas, viewport.width * dpr, viewport.height * dpr); // NEW
      setDeviceSize(overlayCanvas, viewport.width * dpr, viewport.height * dpr);

      // Correct transforms: DPR scaling
      pdfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      boxesCtx.setTransform(dpr, 0, 0, dpr, 0, 0);      // NEW
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // try {
      //   const annots = await page.getAnnotations({ intent: "display" });
      //   pageHasFormFields[pageNum] = Array.isArray(annots) && annots.some(a => a && a.subtype === "Widget");
      // } catch { pageHasFormFields[pageNum] = false; }

      async function renderPdfAnnotations(page, viewport) {
        try {
          // Clear previous DOM widgets
          annotationLayer.innerHTML = "";

          // Get annotations
          const annotations = await page.getAnnotations({ intent: "display" });

          // Use a dontFlip viewport (PDF.js expects this for annotation DOM)
          const view = viewport.clone({ dontFlip: true });

          const params = {
            viewport: view,
            div: annotationLayer,
            annotations,
            page,
            renderForms: false,                    // render form widgets
            annotationStorage: pdfDoc?.annotationStorage || null,
            enableScripting: false                // keep JS off for safety
          };

          if (pdfjsLib?.AnnotationLayer?.render) {
            pdfjsLib.AnnotationLayer.render(params);
          } else if (pdfjsLib?.AnnotationLayerBuilder) {
            const builder = new pdfjsLib.AnnotationLayerBuilder({
              pageDiv: annotationLayer.parentElement,
              pdfPage: page,
              annotationStorage: pdfDoc?.annotationStorage || null
            });
            builder.render(view, "display");
          }
        } catch (e) {
          console.warn("[annotation render] failed:", e);
          annotationLayer.innerHTML = "";
        }
      }

      // page.render({ canvasContext: pdfCtx, viewport, renderInteractiveForms: true }).promise.then(() => {
        page.render({ canvasContext: pdfCtx, viewport }).promise.then(() => {
        renderPdfAnnotations(page, viewport);
        if (pageInfo) pageInfo.textContent = `Page ${pageNum} / ${pdfDoc.numPages}`;
        if (zoomInfo) zoomInfo.textContent = `${Math.round(scale * 100)}%`;

        // Clear both overlay layers appropriately
        boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        if (toggleBoxes && toggleBoxes.checked && currentFormId) drawOverlay(currentFormId, pageNum);
        if (editMode) paintEdits(pageNum);

        // Start workspace timer first render
        if (!workspaceShownAt) {
          workspaceShownAt = ws_now();
          workspaceLogged = false;
          ws_persistInflight();
        }

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

  // ---------- SweetAlert determinate progress helpers (force-centered) ----------
  function openProgress(title, subtitle){
    Swal.fire({
      title,
      html: `
        <div id="ap-wrap" style="max-width:360px; width:90vw; margin:0 auto;">
          <div id="ap-sub" style="margin:8px 0 12px; font-size:13px; opacity:.85;">
            ${subtitle || ""}
          </div>

          <div class="ap-track"
              style="height:10px; background:#eee; border-radius:6px; overflow:hidden;">
            <div id="ap-bar" style="height:100%; width:0%"></div>
          </div>

          <div id="ap-pct" style="margin-top:8px; font-size:12px; opacity:.75;">0%</div>
        </div>
      `,
      showConfirmButton: false,
      allowOutsideClick: false,
      didOpen: () => {
        // Center the whole HTML container (this beats any global CSS)
        const html = Swal.getHtmlContainer();
        if (html) {
          html.style.display = 'flex';
          html.style.flexDirection = 'column';
          html.style.alignItems = 'center';
          html.style.textAlign = 'center';
          // also make the inner track full width of the wrapper
          const track = html.querySelector('.ap-track');
          if (track) { track.style.width = '100%'; }
        }
        const bar = html.querySelector('#ap-bar');
        bar.style.background = 'linear-gradient(90deg, rgba(77,139,255,.95), rgba(77,139,255,.7))';
        bar.style.transition = 'width .25s ease';
      }
    });
  }

  function closeProgressSuccess(){
    Swal.update({
      icon: 'success',
      title: 'Analysis complete',
      html: `
        <div style="max-width:360px; margin:0 auto; text-align:center; font-size:14px; opacity:.9;">
          You can start filling or use the tools on the left.
        </div>
      `,
      showConfirmButton: true,
      confirmButtonText: 'Close'
    });
    // Ensure centering after update
    const html = Swal.getHtmlContainer();
    if (html) {
      html.style.display = 'flex';
      html.style.flexDirection = 'column';
      html.style.alignItems = 'center';
      html.style.textAlign = 'center';
    }
  }

  function updateProgress(pct, subtitle){
    const box = Swal.getHtmlContainer(); if (!box) return;
    const bar = box.querySelector('#ap-bar');
    const pctLbl = box.querySelector('#ap-pct');
    const sub = box.querySelector('#ap-sub');
    if (typeof pct === 'number') {
      const clamped = Math.max(0, Math.min(100, pct));
      bar.style.width = `${clamped}%`;
      pctLbl.textContent = `${Math.round(clamped)}%`;
    }
    if (subtitle != null) sub.textContent = subtitle;
  }

  function closeProgressSuccess(){
    // show an explicit Close button (no timer), keep centered layout
    Swal.update({
      icon: 'success',
      title: 'Analysis complete',
      html: `
        <div style="text-align:center; font-size:14px; opacity:.85;">
          You can start filling or use the tools on the left.
        </div>
      `,
      showConfirmButton: true,
      confirmButtonText: 'Close'
    });
  }

  function closeProgressError(msg){
    Swal.fire({ icon: 'error', title: 'Analysis failed', text: msg || 'Could not analyze this form.' });
  }

  // Utility to run a step with weight and auto progress update
  async function runStep(label, weight, fn, basePctRef){
    updateProgress(basePctRef.pct, label);
    try {
      const out = await fn();
      const target = basePctRef.pct + weight;
      const pctEl = Swal.getHtmlContainer()?.querySelector('#ap-pct');
      const nowPct = pctEl ? parseFloat(pctEl.textContent) || basePctRef.pct : basePctRef.pct;

      if (target - nowPct > 2) {
        updateProgress(nowPct + Math.min(5, (target - nowPct) * 0.5));
        await new Promise(r=>setTimeout(r, 120));
      }
      basePctRef.pct = target;
      updateProgress(basePctRef.pct, label);
      return out;
    } catch (e) {
      throw e;
    }
  }

  // ========================
  // Run Analysis
  // ========================
  analyzeBtn?.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    analysisStartAt = nowMs();
    lastFinishAt = null;
    lastDuration = null;

    // Open determinate progress
    openProgress("Analyzing form…", "Preparing your PDF viewer");
    const P = { pct: 0 }; // progress accumulator

    try {
      // 1) Ensure the PDF is uploaded / available (20%)
      const uploadInfo = await runStep("Processing your PDF…", 20, async () => {
        let up = await ensureUploadedToServer(storedWeb);
        if (up) persistUpload(up);
        return up;
      }, P);

      const web_path  = sessionStorage.getItem("uploadedWebPath");
      const disk_path = sessionStorage.getItem("uploadedDiskPath");
      let   hashId    = sessionStorage.getItem("uploadedFormId") || currentFormId;
      if (!disk_path) throw new Error("Upload failed to provide a disk path.");

      // 2) Prelabel + overlays (25%)
      const pre = await runStep("Identifying labels and positions…", 25, async () => {
        const r = await ensurePrelabelAndOverlays({ disk_path }, hashId);
        if (r && r.canonical_form_id) {
          hashId = r.canonical_form_id; currentFormId = hashId; sessionStorage.setItem("uploadedFormId", hashId);
        }
        return r;
      }, P);

      // 3) Resolve explainer registry (15%)
      const explainer = await runStep("Validating structure and sections…", 15, async () => {
        const guess = guessFromPath(baseFromPath(web_path) || "form.pdf");
        const reg = await GET_json("/panel");
        let ex = await resolveExplainerByHash(reg, hashId);
        if (!ex) {
          await POST_json("/api/explainer.ensure", {
            canonical_form_id: hashId, bucket: guess.bucket, human_title: guess.title,
            pdf_disk_path: disk_path, aliases: [guess.formId, baseFromPath(web_path)]
          });
          const reg2 = await GET_json("/panel");
          ex = await resolveExplainerByHash(reg2, hashId);
        }
        if (!ex) throw new Error("Failed to load explainer.");
        return ex;
      }, P);

      // 4) Render summaries (15%)
      await runStep("Generating summaries…", 15, async () => {
        // cache canonical id for user log and log tool metrics
        lastCanonicalId = explainer.canonical_id || hashId || currentFormId;
        logToolMetricsFromExplainer(explainer);

        if (formTitle) formTitle.textContent = explainer.title || (storedName || "Form");
        renderSummaries(explainer);
      }, P);

      // 5) Initial overlay draw (15%)
      await runStep("Overlaying navigation aids…", 15, async () => {
        if (toggleBoxes && toggleBoxes.checked && currentFormId) await drawOverlay(currentFormId, currentPage);
      }, P);

      // 6) Final polish (10%)
      await runStep("Finalizing… Please hold on!", 10, async () => {
        // any quick, non-blocking polish can go here later
      }, P);

      // Success
      lastFinishAt = nowMs();
      lastDuration = lastFinishAt - analysisStartAt;
      logUserSession({ status: "success" });
      closeProgressSuccess();

    } catch (e) {
      console.error("runAnalysis error:", e);
      lastFinishAt = nowMs();
      lastDuration = lastFinishAt - (analysisStartAt || lastFinishAt);
      logUserSession({ status: "error", message: e?.message || String(e) });
      closeProgressError(e?.message);
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
        if (!res.ok) { boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height); return; }
        cachedAnnotations = await res.json(); cachedAnnotations.__formId = formId;
      }

      // Only clear boxes layer
      boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);

      const pageIdx = pageNumber - 1;
      let rects = [];
      if (Array.isArray(cachedAnnotations.groups) && cachedAnnotations.groups.length) {
        rects = cachedAnnotations.groups.filter((g) => (g.page || 0) === pageIdx);
      } else if (Array.isArray(cachedAnnotations.tokens)) {
        rects = cachedAnnotations.tokens.filter((t) => (t.page || 0) === pageIdx);
      }

      const cssW = pageLayer.clientWidth;
      const cssH = pageLayer.clientHeight;
      // bboxes are normalized 0..1000 → convert directly to CSS pixels
      const sx = cssW / 1000.0;
      const sy = cssH / 1000.0;

      boxesCtx.save();
      boxesCtx.lineWidth = 1;
      boxesCtx.strokeStyle = "rgba(20,20,20,0.25)";
      rects.forEach((r) => {
        const [x0,y0,x1,y1] = r.bbox;
        boxesCtx.strokeRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
      });

      if (anchor && Array.isArray(anchor.bbox)) {
        boxesCtx.lineWidth = 2;
        boxesCtx.strokeStyle = "rgba(0,0,0,0.95)";
        boxesCtx.fillStyle   = "rgba(255,255,0,0.2)";
        const [x0,y0,x1,y1] = anchor.bbox;
        boxesCtx.fillRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
        boxesCtx.strokeRect(x0*sx, y0*sy, Math.max(1,(x1-x0)*sx), Math.max(1,(y1-y0)*sy));
      }
      boxesCtx.restore();
    } catch {
      boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
    }
  }

  document.addEventListener("change", (ev) => {
    if (ev.target && ev.target.id === "toggleBoxes") {
      if (ev.target.checked && currentFormId) drawOverlay(currentFormId, currentPage);
      else boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
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
  function bboxUnion(a, b) { return [ Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3]) ]; }

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
    s = s.toString().toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    s = s.replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ");
    for (const k of Object.keys(SYN_MAP)) {
      const re = new RegExp(`\\b${k}\\b`, "g");
      s = s.replace(re, SYN_MAP[k]);
    }
    return s;
  }

  function tokenize(s) { return normalizeText(s).split(" ").filter(Boolean); }
  function bigrams(tokens) { const out = []; for (let i = 0; i < tokens.length - 1; i++) out.push(tokens[i] + " " + tokens[i + 1]); return out; }

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
    return Math.abs(midA - midB) < 8;
  }

  function unionLine(seed, pageTokens) {
    let box = seed.bbox.slice(0, 4);
    for (const t of pageTokens) if (sameLine(seed, t)) box = bboxUnion(box, t.bbox);
    return box;
  }

  function findAnchorForLabel(label, annotations) {
    if (!annotations) return null;
    const labelNorm = normalizeText(label);
    const MIN_SCORE = 0.60;

    // exact group label match
    if (Array.isArray(annotations.groups) && annotations.groups.length) {
      const exact = annotations.groups.find(g => normalizeText(g.label || "") === labelNorm);
      if (exact && Array.isArray(exact.bbox) && exact.bbox.length === 4) {
        return { page: (exact.page || 0), bbox: exact.bbox, source: "group-exact", score: 1 };
      }
    }

    const tokens = Array.isArray(annotations.tokens) ? annotations.tokens : [];
    if (!tokens.length && !(annotations.groups && annotations.groups.length)) return null;

    let pageCandidates = null;

    if (annotations.groups && annotations.groups.length) {
      const scoredGroups = annotations.groups
        .map(g => ({ g, s: similarity(label, g.label || "") }))
        .filter(x => x.s > 0);
      if (scoredGroups.length) {
        scoredGroups.sort((a, b) => b.s - a.s);
        pageCandidates = [...new Set(scoredGroups.slice(0, 2).map(x => x.g.page || 0))];
      }
    }

    if (!pageCandidates) {
      const pageScore = new Map();
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

    let best = null;
    for (const p of pageCandidates) {
      const pageTokens = tokens.filter(t => (t.page || 0) === p);
      for (const t of pageTokens) {
        const s = similarity(label, t.text || "");
        const leftBias = (t.bbox?.[0] ?? 1e9) / 1e6;
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
  let currentTool = null;     // no default active
  let drawing = false;

  const pageEdits = {};
  const EDIT = (p) => (pageEdits[p] ??= { strokes: [], texts: [] });

  // Draw config
  const PEN = { color: "#111111", width: 2.0,  alpha: 1.0 };
  const HL  = { color: "rgba(255,255,0,0.35)", width: 10.0, alpha: 0.35 };

  // UI inputs (optional)
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
    return { color: rgbaFromHex(col, alpha), width: (hlWidthInput && (+hlWidthInput.value || HL.width)) || HL.width, alpha };
  }
  function rgbaFromHex(hex, a) {
    const h = hex.replace("#",""); const r = parseInt(h.slice(0,2),16); const g = parseInt(h.slice(2,4),16); const b = parseInt(h.slice(4,6),16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // Track active text node
  let activeTextNode = null;

  // tool selection
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

  // // pointer routing
  // function applyPointerRouting() {
  //   if (!editMode || !currentTool) { overlayCanvas.style.pointerEvents = "none"; return; }
  //   // if (currentTool === "text" && pageHasFormFields[currentPage]) overlayCanvas.style.pointerEvents = "none";
  //   // else overlayCanvas.style.pointerEvents = "auto";
  //   overlayCanvas.style.pointerEvents = currentTool ? "auto" : "none";
  //   document.body.classList.toggle("text-mode", currentTool === "text");
  //   document.body.classList.toggle("pen-mode", currentTool === "pen");
  //   document.body.classList.toggle("hl-mode", currentTool === "highlight");
  // }

  function applyPointerRouting() {
    overlayCanvas.style.pointerEvents = (editMode && currentTool) ? "auto" : "none";
    annotationLayer.style.pointerEvents = "none"; // we’ve disabled widgets globally
    document.body.classList.toggle("text-mode", currentTool === "text");
    document.body.classList.toggle("pen-mode",  currentTool === "pen");
    document.body.classList.toggle("hl-mode",   currentTool === "highlight");
  }

  // scale helpers
  function pageScaleFactors(pageNum) {
    const base = pageBaseSize[pageNum] || { w: overlayCanvas.width, h: overlayCanvas.height };
    const cssW = pageLayer?.clientWidth  || overlayCanvas.clientWidth  || base.w;
    const cssH = pageLayer?.clientHeight || overlayCanvas.clientHeight || base.h;
    return { sx: cssW / base.w, sy: cssH / base.h, invx: base.w / cssW, invy: base.h / cssH };
  }
  function toCssXY(evt) {
    const rect = overlayCanvas.getBoundingClientRect();
    return { x: (evt.clientX - rect.left), y: (evt.clientY - rect.top) };
  }

  // paint strokes (draw layer only)
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

  // --- Draggable + editable text nodes ---
  function spawnTextNode(cssX, cssY) {
    const pageNum = currentPage;
    const { invx, invy } = pageScaleFactors(pageNum);

    const wrap = document.createElement("div");
    wrap.className = "text-annot";
    wrap.style.cssText = `
      position:absolute; left:${cssX}px; top:${cssY}px; z-index:4;
      outline:1.5px dashed #1e90ff; outline-offset:2px; border-radius:4px; padding:6px 8px 8px 28px;
      background:rgba(255,255,255,0.01); user-select:none;`;
    wrap.dataset.page = String(pageNum);

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

    const content = document.createElement("div");
    content.className = "ta-content";
    content.contentEditable = "true";
    content.spellcheck = true;
    content.style.cssText = `
      min-width:30px; min-height:18px; line-height:1.25; color:#000;
      font:14px Inter, system-ui, sans-serif; user-select:text; cursor:text;`;
    content.textContent = "Type here";
    wrap.appendChild(content);

    const bar = document.createElement("div");
    bar.className = "ta-toolbar";
    bar.style.cssText = `
      position:absolute; left:0; transform:translateY(100%); margin-top:6px; padding:6px 8px;
      border-radius:8px; box-shadow:0 6px 24px rgba(0,0,0,.12); background:#fff; display:none; gap:6px; z-index:5;`;
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
      x: cssX * invx, y: cssY * invy,                  // base coords
      text: "Type here", font: "Inter", size: 14, color: "#000000"
    };
    const ed = EDIT(pageNum); ed.texts.push(model);
    wrap.dataset.key = model.id;

    const showBar = () => { bar.style.display = "flex"; activeTextNode = wrap; };
    const hideBar = () => { bar.style.display = "none"; if (activeTextNode === wrap) activeTextNode = null; };
    content.addEventListener("focus", showBar);
    content.addEventListener("blur", () => setTimeout(() => { if (!wrap.contains(document.activeElement)) hideBar(); }, 0));

    content.addEventListener("input", () => {
      model.text = content.textContent || "";
      model.size = parseInt(getComputedStyle(content).fontSize,10) || model.size;
    });

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
      model.x = nx * (pageScaleFactors(pageNum).invx); 
      model.y = ny * (pageScaleFactors(pageNum).invy);
    }
    function endDrag(e){ if(!dragging)return; dragging=false; wrap.releasePointerCapture?.(e.pointerId); }

    handle.addEventListener("pointerdown", startDrag);
    wrap.addEventListener("pointerdown", (e) => { if (e.target === wrap) startDrag(e); });
    wrap.addEventListener("pointermove", moveDrag);
    ["pointerup","pointercancel","pointerleave"].forEach(t => wrap.addEventListener(t, endDrag));

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

  // Smooth drawing
  let rafPending = false;
  function schedulePaint() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => { rafPending = false; paintEdits(currentPage); });
  }

  overlayCanvas.addEventListener("pointerdown", (e) => {
    if (!editMode || !currentTool) return;

    if (currentTool === "text") {
      if (pageHasFormFields[currentPage]) return;
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
        const rBase = R * invx;
        ed.strokes = ed.strokes.filter(st => !st.points.some(p => Math.hypot(p.x-startBase.x, p.y-startBase.y) < rBase));
      }
      schedulePaint();
    }
  });

  overlayCanvas.addEventListener("pointermove", (e) => {
    if (!editMode || !drawing) return;
    const css = toCssXY(e);
    const { invx } = pageScaleFactors(currentPage);
    const ptBase = { x: css.x * invx, y: css.y * (pageScaleFactors(currentPage).invy) };

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

        // ---- Workspace event logger: duration ends at the user's confirmation moment ----
        async function logWorkspaceEvent(status, finishedAt, extraMeta) {
          try {
            if (!workspaceShownAt || workspaceLogged) return;
            const cid = ws_currentCanonical();
            if (!cid) return;

            const payload = {
              user_id: getUserId(),
              canonical_id: cid,
              method: "intelliform",
              started_at: workspaceShownAt,
              finished_at: finishedAt,
              duration_ms: Math.max(0, finishedAt - workspaceShownAt),
              meta: Object.assign({ status }, extraMeta || {})
            };

            if (navigator.sendBeacon) {
              const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
              navigator.sendBeacon("/api/user.log", blob);
            } else {
              await POST_JSON("/api/user.log", payload);
            }
            workspaceLogged = true;
            ws_clearInflight();
          } catch (e) {
            console.warn("[workspace event log] failed:", e);
          }
        }

        // ---- Reset session storage and go back to index (fresh user upload) ----
        function resetAndGoHome() {
          try {
            // Do NOT clear the research user id – keep the identity across sessions.
            const KEEP = "research_user_id";

            [
              "uploadedWebPath",
              "uploadedDiskPath",
              "uploadedFormId",
              "uploadedFileName",
              "uploadedFileWithExtension",
              "WS_INFLIGHT_V1"
            ].forEach(k => { if (k !== KEEP) sessionStorage.removeItem(k); });

            // (Optional) also clear the same doc keys from localStorage if you store any there,
            // but do NOT touch 'research_user_id'.
          } catch {}
          window.location.href = "/";
        }

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

    // capture confirmation time BEFORE any rendering/export
    const tConfirm = ws_now();

    if (!editMode) enterEdit();

    let renderMs = 0;
    try {
      const t0 = ws_now();
      const bytes = await exportEditedPdf();
      renderMs = ws_now() - t0;

      const base = (sessionStorage.getItem("uploadedFileName") || "form.pdf").replace(/\.pdf$/i,"");
      const filename = `${base}_edited.pdf`;
      const blob = new Blob([bytes], { type: "application/pdf" });

      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = filename;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(a.href);

      // Success toast
      await Swal.fire({ icon:"success", title:"Saved", timer:800, showConfirmButton:false });

      // Ask if the user wants to start fresh
      const startFresh = await Swal.fire({
        title: "Start a new session?",
        text: "Return to the home screen to upload a new PDF.",
        icon: "question",
        showCancelButton: true,
        confirmButtonText: "Yes, start fresh",
        cancelButtonText: "Stay here"
      }).then(r => r.isConfirmed);

      // Log completion with finished_at = user confirmation time (not including export time)
      await logWorkspaceEvent("saved", tConfirm, { render_ms: renderMs });

      if (startFresh) {
        resetAndGoHome();
        return;
      }

    } catch (e) {
      console.error("Save failed", e);
      Swal.fire({ icon:"error", title:"Save failed", text: e?.message || "Could not generate edited PDF." });
      // do not log a completion if save failed
      return;
    }
  }

  // ---- Export edited PDF ----
  async function exportEditedPdf() {
    if (!pdfDoc) throw new Error("No document open");
    const { PDFDocument } = (window.PDFLib || {}); if (!PDFDocument) throw new Error("pdf-lib not loaded");

    const pngPages = [];
    for (let n=1; n<=pdfDoc.numPages; n++) {
      const page = await pdfDoc.getPage(n);
      const viewport = page.getViewport({ scale });
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(viewport.width));
      c.height = Math.max(1, Math.round(viewport.height));
      const ctx = c.getContext("2d");
      await page.render({ canvasContext: ctx, viewport }).promise;

      const ed = pageEdits[n];
      if (ed) {
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

    // optional: register checksum
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
    let layer = document.querySelector("#pdfContainer .pageLayer");
    if (layer) return layer;

    const container = document.getElementById("pdfContainer");
    layer = document.createElement("div");
    layer.id = "pageLayer";
    layer.className = "pageLayer";
    layer.style.position = "relative";
    layer.style.marginTop = "60px";

    // Move canvases inside the layer; overlay is positioned abs by CSS
    // We'll insert a separate boxesCanvas next.
    if (container) {
      container.innerHTML = "";
      container.appendChild(layer);
      layer.appendChild(pdfCanvas);
      overlayCanvas.style.left = "0"; overlayCanvas.style.top = "0";
      overlayCanvas.style.position = "absolute";
      overlayCanvas.style.pointerEvents = "none"; // enabled only when tools active
      layer.appendChild(overlayCanvas);
    }
    return layer;
  }

  function ensureBoxesCanvas() {
    let c = document.getElementById("boxesCanvas");
    if (c) return c;
    c = document.createElement("canvas");
    c.id = "boxesCanvas";
    c.style.position = "absolute";
    c.style.left = "0";
    c.style.top  = "0";
    c.style.pointerEvents = "none"; // overlays are not interactive
    // Insert between pdf and overlay
    pageLayer.insertBefore(c, overlayCanvas);
    return c;
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
      anno.style.pointerEvents = "none";         // <-- keep
      anno.style.zIndex = "2";                   // explicit, just in case

      // Inject minimal PDF.js annotation styles once
      if (!document.getElementById("pdfjs-annot-style")) {
        const s = document.createElement("style");
        s.id = "pdfjs-annot-style";
        s.textContent = `
          .annotationLayer { position:absolute; top:0; left:0; }
          .annotationLayer * { box-sizing: border-box; }
          /* Make the widgets visible and clickable */
          .annotationLayer .buttonWidgetAnnotation,
          .annotationLayer .textWidgetAnnotation input,
          .annotationLayer .choiceWidgetAnnotation select,
          .annotationLayer .checkboxWidgetAnnotation input {
            pointer-events: auto;
            font: 12px/1.2 Inter, system-ui, sans-serif;
          }
        `;
        document.head.appendChild(s);
      }

      // Place above boxesCanvas but below overlayCanvas
      pageLayer.insertBefore(anno, overlayCanvas);
      return anno;
  }

  // ===== Workspace session (viewing/annotating) timing =====
  const WS_SS_KEY = "WS_INFLIGHT_V1";
  let workspaceShownAt   = null;
  let workspaceFinishedAt = null;
  let workspaceDuration   = null;
  let workspaceLogged     = false;

  function ws_now() { return Date.now(); }
  function ws_currentCanonical() {
    return lastCanonicalId || sessionStorage.getItem("uploadedFormId") || "";
  }

  function ws_persistInflight() {
    try {
      const cid = ws_currentCanonical();
      if (!workspaceShownAt || !cid) return;
      sessionStorage.setItem(WS_SS_KEY, JSON.stringify({
        started_at: workspaceShownAt,
        canonical_id: cid,
        user_id: getUserId(),
        method: "intelliform",
        page_url: location.pathname + location.search
      }));
    } catch {}
  }

  function ws_clearInflight() { try { sessionStorage.removeItem(WS_SS_KEY); } catch {} }

  async function ws_tryRecoverAbandoned() {
    try {
      const raw = sessionStorage.getItem(WS_SS_KEY);
      if (!raw) return;
      const rec = JSON.parse(raw);
      if (!rec || !rec.started_at || !rec.canonical_id) { ws_clearInflight(); return; }

      const finished = ws_now();
      const duration = Math.max(0, finished - Number(rec.started_at));

      const body = JSON.stringify({
        user_id: rec.user_id || getUserId(),
        canonical_id: rec.canonical_id,
        method: rec.method || "intelliform",
        started_at: rec.started_at,
        finished_at: finished,
        duration_ms: duration,
        meta: { status: "abandoned_recovery", page_url: rec.page_url || null }
      });

      if (navigator.sendBeacon) {
        const blob = new Blob([body], { type: "application/json" });
        navigator.sendBeacon("/api/user.log", blob);
      } else {
        await POST_JSON("/api/user.log", JSON.parse(body));
      }
    } catch (e) {
      console.warn("[workspace recovery] failed:", e);
    } finally {
      ws_clearInflight();
    }
  }

  ws_tryRecoverAbandoned();

  // ===== Research logging (timers + endpoints) =====
  const LS_KEY_UID = "research_user_id";

  function getUserId() {
    const v =
      (window.getResearchUserId && window.getResearchUserId()) ||
      sessionStorage.getItem(LS_KEY_UID) ||
      localStorage.getItem(LS_KEY_UID);
    const s = (v ?? "").toString().trim();
    return s || "ANON";
  }

  // Optional helper so index page (or anywhere) can set it:
  window.setResearchUserId = function(name){
    const s = (name ?? "").toString().trim();
    if (!s) return;
    try {
      sessionStorage.setItem(LS_KEY_UID, s);
      localStorage.setItem(LS_KEY_UID, s);
    } catch {}
  };

  async function POST_JSON(url, obj) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obj || {})
    });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  }

  // --- timers & session state ---
  let analysisStartAt = null;
  let lastFinishAt    = null;
  let lastDuration    = null;
  let lastCanonicalId = null;

  function nowMs() { return Date.now(); }

  async function logToolMetricsFromExplainer(explainer) {
    try {
      const payload = {
        canonical_id: explainer.canonical_id || currentFormId || sessionStorage.getItem("uploadedFormId") || "",
        form_title:   explainer.title || (storedName || "Form"),
        bucket:       explainer.bucket || (guessFromPath(baseFromPath(storedWeb) || "form.pdf").bucket),
        metrics:      explainer.metrics || {},
        source:       "analysis",
        note:         "workspace.js post-analysis"
      };
      if (!payload.canonical_id) return;
      await POST_JSON("/api/metrics.log", payload);
    } catch (e) {
      console.warn("[metrics.log] failed:", e);
    }
  }

  async function logUserSession(finalMeta) {
    try {
      if (!lastCanonicalId) return;
      const user_id = getUserId();
      const body = {
        user_id,
        canonical_id: lastCanonicalId,
        method: "intelliform",
        started_at: analysisStartAt,
        finished_at: lastFinishAt,
        duration_ms: lastDuration,
        meta: finalMeta || {}
      };
      await POST_JSON("/api/user.log", body);
    } catch (e) {
      console.warn("[user.log] failed:", e);
    }
  }

  window.addEventListener("beforeunload", () => {
    try {
      if (!analysisStartAt || lastFinishAt) return;
      const cid = lastCanonicalId || sessionStorage.getItem("uploadedFormId") || "";
      if (!cid) return;
      const body = JSON.stringify({
        user_id: getUserId(),
        canonical_id: cid,
        method: "intelliform",
        started_at: analysisStartAt,
        finished_at: nowMs(),
        duration_ms: nowMs() - analysisStartAt,
        meta: { status: "abandoned" }
      });
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon && navigator.sendBeacon("/api/user.log", blob);
    } catch {}
  });

  function ws_sendAbandonBeacon(tag) {
    try {
      if (!workspaceShownAt || workspaceLogged) return;
      const cid = ws_currentCanonical();
      if (!cid) return;
      const finished = ws_now();
      const payload = {
        user_id: getUserId(),
        canonical_id: cid,
        method: "intelliform",
        started_at: workspaceShownAt,
        finished_at: finished,
        duration_ms: finished - workspaceShownAt,
        meta: { status: tag || "abandoned" }
      };
      const body = JSON.stringify(payload);
      if (navigator.sendBeacon) {
        const blob = new Blob([body], { type: "application/json" });
        navigator.sendBeacon("/api/user.log", blob);
        workspaceLogged = true;
        ws_clearInflight();
      } else {
        fetch("/api/user.log", { method: "POST", headers: { "Content-Type": "application/json" }, body }).catch(()=>{});
        workspaceLogged = true;
        ws_clearInflight();
      }
    } catch {}
  }

  window.addEventListener("pagehide", () => ws_sendAbandonBeacon("abandoned_pagehide"));
  let ws_visTimer = null;
  document.addEventListener("visibilitychange", () => {
    try {
      if (document.visibilityState === "hidden") {
        ws_visTimer = setTimeout(() => ws_sendAbandonBeacon("abandoned_hidden"), 2000);
      } else if (document.visibilityState === "visible" && ws_visTimer) {
        clearTimeout(ws_visTimer);
        ws_visTimer = null;
      }
    } catch {}
  });
  window.addEventListener("beforeunload", () => ws_sendAbandonBeacon("abandoned_beforeunload"));

    // ===== Ctrl+F-like search across tokens (uses annotations) =====
  let searchMatches = [];   // [{page, bbox, tokenIndex}]
  let searchIndex   = -1;

  function norm(s){
    return (s||"").toString().toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g,"");
  }

  async function ensureAnnotations(formId) {
    if (!cachedAnnotations || cachedAnnotations.__formId !== formId) {
      const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error("Annotations not found");
      cachedAnnotations = await res.json(); cachedAnnotations.__formId = formId;
    }
    return cachedAnnotations;
  }

    let lastQuery = "";

  async function runSearch(query){
    searchMatches = [];
    searchIndex = -1;
    if (searchStatus) searchStatus.textContent = "0/0";
    if (!query) return;

    const ok = await ensureAnnotationsReady();
    if (!ok) return;

    const ann = cachedAnnotations;
    const toks = Array.isArray(ann.tokens) ? ann.tokens : [];
    const q = norm(query);

    for (let i=0;i<toks.length;i++){
      const t = toks[i];
      if (!t || !t.text || !Array.isArray(t.bbox)) continue;
      if (norm(t.text).includes(q)) {
        searchMatches.push({ page: t.page||0, bbox: t.bbox, tokenIndex: i });
      }
    }

    if (!searchMatches.length) {
      try { boxesCtx.clearRect(0,0,boxesCanvas.width,boxesCanvas.height); } catch {}
      if (searchStatus) searchStatus.textContent = "0/0";
      return;
    }
    lastQuery = query;
    gotoMatch(0, /*isDelta*/false);
  }

  // Enter: new search if query changed; otherwise cycle forward
  searchInput?.addEventListener("keydown", (e)=>{
    if (e.key === "Enter") {
      const q = (searchInput.value || "").trim();
      if (!q) { e.preventDefault(); return; }
      if (q !== lastQuery) {
        runSearch(q);
      } else if (searchMatches.length) {
        gotoMatch(+1, true);
      } else {
        runSearch(q);
      }
      e.preventDefault();
    }
  });

  async function gotoMatch(idxOrDelta, isDelta=true){
    if (!searchMatches.length) return;
    if (isDelta) {
      searchIndex = (searchIndex + idxOrDelta + searchMatches.length) % searchMatches.length;
    } else {
      searchIndex = Math.min(Math.max(0, idxOrDelta), searchMatches.length-1);
    }
    const m = searchMatches[searchIndex];
    if (searchStatus) searchStatus.textContent = `${searchIndex+1}/${searchMatches.length}`;

    // jump to page & highlight
    currentPage = (m.page||0) + 1;
    renderPage(currentPage);
    setTimeout(()=> drawOverlay(currentFormId, currentPage, { page: m.page||0, bbox: m.bbox }), 160);
  }

  // UI bindings
  searchTool?.addEventListener("click", (e) => {
    // focus input when the tile is clicked (but let buttons work)
    if (!e.target.closest("button") && searchInput) {
      searchInput.focus(); searchInput.select?.();
    }
  });

  searchInput?.addEventListener("keydown", (e)=>{
    if (e.key === "Enter") {
      if (searchMatches.length) gotoMatch(+1, true); // Enter cycles forward
      else runSearch(searchInput.value);
      e.preventDefault();
    }
  });

  searchNextBtn?.addEventListener("click", ()=> gotoMatch(+1, true));
  searchPrevBtn?.addEventListener("click", ()=> gotoMatch(-1, true));
  searchClearBtn?.addEventListener("click", ()=>{
    if (searchInput) searchInput.value = "";
    searchMatches = []; searchIndex = -1;
    if (searchStatus) searchStatus.textContent = "0/0";
    try { boxesCtx.clearRect(0,0,boxesCanvas.width,boxesCanvas.height); } catch {}
  });

  // Optional: capture Ctrl/Cmd+F to focus our search box
  document.addEventListener("keydown", (e) => {
    const isCtrlF = (e.key.toLowerCase() === "f") && (e.ctrlKey || e.metaKey);
    if (isCtrlF) {
      e.preventDefault();
      searchInput?.focus();
      searchInput?.select?.();
    }
  });

  // --- make search work even before Analyze
  async function ensureAnnotationsReady() {
    try {
      if (currentFormId && cachedAnnotations && cachedAnnotations.__formId === currentFormId) return true;

      // try to fetch; if missing, force prelabel (no full analysis UI needed)
      const formId = sessionStorage.getItem("uploadedFormId") || currentFormId;
      const disk   = sessionStorage.getItem("uploadedDiskPath");
      if (!formId || !disk) return false;

      // try fetch first
      let ok = true;
      try {
        const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) ok = false; else { cachedAnnotations = await res.json(); cachedAnnotations.__formId = formId; currentFormId = formId; }
      } catch { ok = false; }

      if (!ok) {
        // run prelabel to generate annotations, then refetch
        await ensurePrelabelAndOverlays({ disk_path: disk }, formId);
        const res2 = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res2.ok) return false;
        cachedAnnotations = await res2.json(); cachedAnnotations.__formId = formId; currentFormId = formId;
      }
      return true;
    } catch { return false; }
  }

} // end initWorkspace
