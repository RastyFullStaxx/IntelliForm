// static/js/workspace.js

window.addEventListener("load", initWorkspace);

function initWorkspace() {
  console.log("workspace.js 2025-10-12 overlay-split: boxesCanvas + drawCanvas, fixed z-index, label-jump");
  const API_BASE = window.INTELLIFORM_API_BASE || "";
  const apiUrl = (path) => `${API_BASE}${path}`;
  const apiFetch = (path, options) => fetch(apiUrl(path), options);

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
  const floatingToggleWrapper = $("floatingToggleWrapper");
  const faqButton = $("faqButton");
  const faqPanel = $("faqPanel");
  const sidebarTitle = document.querySelector(".sidebar-title h5");
  const metricsRow = $("metricsRow");
  const pageInfo = $("pageInfo");
  const zoomInfo = $("zoomInfo");
  const searchTool     = $("searchTool");
  const searchInput    = $("searchInput");
  const searchNextBtn  = $("searchNext");
  const searchPrevBtn  = $("searchPrev");
  const searchClearBtn = $("searchClear");
  const searchStatus   = $("searchStatus");
  const toastHost = document.createElement("div");
  toastHost.id = "toolToast";
  toastHost.className = "tool-toast";
  document.body.appendChild(toastHost);

  const tourTrigger = document.createElement("button");
  tourTrigger.id = "tourTrigger";
  tourTrigger.type = "button";
  tourTrigger.innerHTML = `<span class="tour-trigger__icon">★</span> Walkthrough`;
  document.body.appendChild(tourTrigger);

  const setInfoValue = (chip, text) => {
    if (!chip) return;
    const v = chip.querySelector(".info-value, [data-role='value']");
    if (v) v.textContent = text;
    else chip.textContent = text;
  };

  const HAS_BRAND_DIALOG = typeof BrandDialog !== "undefined";
  const showAlert = (opts) => {
    if (HAS_BRAND_DIALOG) return BrandDialog.alert(opts);
    alert(`${opts.title || ""}\n\n${opts.text || ""}`);
  };
  const showConfirm = async (opts) => {
    if (HAS_BRAND_DIALOG) return await BrandDialog.confirm(opts);
    return window.confirm(`${opts.title || "Confirm"}\n\n${opts.text || ""}`);
  };

  let progressCtrl = null;
  function openProgress(title, subtitle){
    if (HAS_BRAND_DIALOG) {
      progressCtrl?.close();
      progressCtrl = BrandDialog.progress({ title, subtitle });
    } else {
      progressCtrl = {
        update(){},
        success(){},
        error(){},
        close(){ progressCtrl = null; }
      };
    }
  }
  function updateProgress(pct, subtitle){
    if (progressCtrl && progressCtrl.update) progressCtrl.update(pct, subtitle);
  }
  function closeProgressSuccess(){
    if (progressCtrl && progressCtrl.success) {
      progressCtrl.success("You can start filling or use the tools on the left.", { closeText: "Close" });
    }
  }
  function closeProgressError(msg){
    if (progressCtrl && progressCtrl.error) {
      progressCtrl.error(msg || "Could not analyze this form.");
    } else {
      showAlert({ variant: "danger", title: "Analysis failed", text: msg || "Could not analyze this form." });
    }
  }


  // Base canvases coming from HTML
  let pdfCanvas = $("pdfCanvas");
  let overlayCanvas = $("overlayCanvas"); // <- will become "draw canvas" only

  const eceBadge = $("eceScoreBadge");
  const downloadBtn = $("downloadPDF");
  const printBtn = $("printPDF");

    // Exit button → confirm, log, and start fresh
    const exitBtn = document.getElementById("btnExit");
    exitBtn?.addEventListener("click", async () => {
      const ok = await showConfirm({
        title: "Exit and start over?",
        text: "This will discard the current workspace and return to the home screen.",
        variant: "warning",
        confirmText: "Exit",
        cancelText: "Stay"
      });

      if (!ok) return;

      // best-effort session close log, then reset
      try { ws_sendAbandonBeacon("exit_clicked"); } catch {}
      await resetAndGoHome({ deleteUpload: true });
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
    try {
      if (!editMode) enterEdit();

      const t0 = ws_now();
      const bytes = await exportEditedPdf();
      const renderMs = ws_now() - t0;

      const blob = new Blob([bytes], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const w = window.open(url, "_blank");
      setTimeout(() => { try { w?.print(); } catch {} }, 500);

      // Optional follow-up: ask to start fresh after print
      const startFresh = await showConfirm({
        title: "Start a new session?",
        text: "Return to the home screen to upload a new PDF.",
        variant: "question",
        confirmText: "Yes, start fresh",
        cancelText: "Stay here"
      });

      if (startFresh) await resetAndGoHome({ deleteUpload: true });

      // Non-blocking log for print (does not end the session timer)
      if (isMetricsOptIn() && workspaceShownAt && !workspaceLogged) {
        const finished = ws_now();
        const payload = {
          user_id: getUserId(),
          canonical_id: ws_currentCanonical(),
          method: "intelliform",
          started_at: workspaceShownAt,
          finished_at: finished,
          duration_ms: finished - workspaceShownAt,
          meta: { status: "printed", render_ms: renderMs }
        };
        try {
          if (navigator.sendBeacon) {
            const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
            navigator.sendBeacon(apiUrl("/api/user.log"), blob);
          } else {
            await POST_JSON("/api/user.log", payload);
          }
        } catch (e) {
          console.warn("[print log] failed", e);
        }
      }

    } catch (e) {
      console.error("Print failed", e);
      showAlert({ variant: "danger", title: "Print failed", text: e?.message || "Could not generate PDF for printing." });
    }
  });

  // ---- Title ----
  if (formTitle) {
    const origName = (storedName || baseFromPath(storedWeb) || "Form").replace(/\.[^.]+$/, "");
    formTitle.textContent = origName || "Form";
  }

  // ---- Sidebar controls ----
  // Draggable toggle along the right edge (vertical only)
  if (floatingToggleWrapper) {
    let dragStartY = 0;
    let startTop = 0;
    let dragging = false;
    let moved = false;
    const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
    const stopDrag = () => {
      dragging = false;
      document.removeEventListener("pointermove", onDrag);
      document.removeEventListener("pointerup", stopDrag);
      document.removeEventListener("pointercancel", stopDrag);
    };
    const onDrag = (e) => {
      if (!dragging) return;
      const y = e.clientY ?? (e.touches && e.touches[0]?.clientY) ?? dragStartY;
      const delta = y - dragStartY;
      const minTop = 80;
      const maxTop = Math.max(minTop, window.innerHeight - 180);
      const nextTop = clamp(startTop + delta, minTop, maxTop);
      if (Math.abs(delta) > 2) moved = true;
      floatingToggleWrapper.style.top = `${nextTop}px`;
    };
    floatingToggleWrapper.addEventListener("pointerdown", (e) => {
      dragStartY = e.clientY;
      startTop = parseFloat(getComputedStyle(floatingToggleWrapper).top || "200");
      dragging = true;
      moved = false;
      document.addEventListener("pointermove", onDrag);
      document.addEventListener("pointerup", stopDrag);
      document.addEventListener("pointercancel", stopDrag);
    });
    sidebarToggle?.addEventListener("click", (e) => {
      if (moved) {
        e.stopImmediatePropagation();
        moved = false;
      }
    });
  }

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

  // ---- FAQ guide ----
  if (faqButton && faqPanel) {
    const clearGuideHighlights = () => {
      document.querySelectorAll(".guide-highlight").forEach((el) => el.classList.remove("guide-highlight"));
      faqPanel.querySelectorAll(".faq-row.active").forEach((r) => r.classList.remove("active"));
    };
    const highlightTargets = (selector) => {
      if (!selector) return;
      if (!sidebar.classList.contains("open") && sidebarToggle) {
        sidebar.classList.add("open");
        sidebarToggle.style.display = "none";
      }
      selector.split(",").forEach((sel) => {
        document.querySelectorAll(sel.trim()).forEach((el) => el.classList.add("guide-highlight"));
      });
    };

    const closeFaq = () => {
      faqPanel.classList.remove("open");
      clearGuideHighlights();
    };
    faqButton.addEventListener("click", (e) => {
      e.stopPropagation();
      faqPanel.classList.toggle("open");
    });
    faqPanel.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", (e) => {
      if (!faqPanel.contains(e.target) && !faqButton.contains(e.target)) closeFaq();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeFaq();
    });

    faqPanel.querySelectorAll(".faq-row").forEach((row) => {
      row.addEventListener("mouseenter", () => {
        clearGuideHighlights();
        const target = row.getAttribute("data-target");
        highlightTargets(target);
        row.classList.add("active");
      });
      row.addEventListener("mouseleave", () => clearGuideHighlights());
      row.addEventListener("focus", () => {
        clearGuideHighlights();
        highlightTargets(row.getAttribute("data-target"));
        row.classList.add("active");
      });
      row.addEventListener("blur", () => clearGuideHighlights());
      row.addEventListener("click", (e) => {
        e.stopPropagation();
        clearGuideHighlights();
        const targetSel = row.getAttribute("data-target");
        highlightTargets(targetSel);
        // Trigger the first matching control, if any
        if (targetSel) {
          const first = document.querySelector(targetSel.split(",")[0].trim());
          if (first) first.click();
        }
        closeFaq();
      });
    });
  }
  const syncThumbToggle = () => {
    if (!thumbSidebar || !pageToggler) return;
    const open = thumbSidebar.classList.contains("visible");
    pageToggler.classList.toggle("active", open);
  };
  pageToggler?.addEventListener("click", () => {
    thumbSidebar?.classList.toggle("visible");
    syncThumbToggle();
  });
  syncThumbToggle();

  // ---- "Show boxes" toggle ----
  const showBoxesBtn = $("btnShowBoxes");
  let showBoxes = false;
  const syncShowBoxesButton = () => {
    if (!showBoxesBtn) return;
    showBoxesBtn.classList.toggle("active", showBoxes);
    showBoxesBtn.setAttribute("aria-pressed", showBoxes ? "true" : "false");
  };
  syncShowBoxesButton();
  showBoxesBtn?.addEventListener("click", async () => {
    showBoxes = !showBoxes;
    syncShowBoxesButton();
    if (showBoxes && currentFormId) await drawOverlay(currentFormId, currentPage);
    else boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
  });

  // ---- PDF state ----
  let pdfDoc = null;
  let scale = 1.5;
  let currentPage = 1;
  const syncNavInfo = (pageNum = currentPage) => {
    if (pageInfo && pdfDoc) setInfoValue(pageInfo, `${pageNum} of ${pdfDoc.numPages || 1}`);
    if (zoomInfo) setInfoValue(zoomInfo, `${Math.round(scale * 100)}%`);
  };
  const markActiveThumbnail = (pageNum) => {
    document.querySelectorAll("#thumbnailSidebar .thumbnail").forEach((thumb) => {
      const match = (parseInt(thumb.dataset.page || "0", 10) === pageNum);
      thumb.classList.toggle("active", match);
      thumb.parentElement?.classList.toggle("active", match);
    });
  };

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
      try {
        const retry = await showConfirm({
          variant: "danger",
          title: "Failed to load PDF",
          text: "Would you like to re-upload the file?",
          confirmText: "Re-upload",
          cancelText: "Go home"
        });
        if (retry) {
          const pick = await pickAndUploadFile();
          persistUpload(pick);
          await openPdf(pick.web_path);
        } else {
          window.location.href = "/";
        }
      } catch {
        window.location.href = "/";
      }
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
    syncNavInfo(currentPage);
    // Mark workspace start once the PDF is ready (used for user/session timing)
    if (!workspaceShownAt && isMetricsOptIn()) {
      workspaceShownAt = ws_now();
      ws_persistInflight();
    }
  }

  // ---- Render gate ----
  let RENDER_SEQ = 0;        // increases every render request
  let LAST_COMMIT_SEQ = 0;   // the last seq that actually committed

  const setScaleAndRender = (nextScale) => {
    const clamped = Math.min(3, Math.max(0.5, nextScale));
    if (Math.abs(clamped - scale) < 0.001) return;
    scale = clamped;
    renderPage(currentPage);
    syncNavInfo(currentPage);
  };

  function renderPage(pageNum) {
    const mySeq = ++RENDER_SEQ;
    return new Promise((resolve, reject) => {
      if (!pdfDoc) { resolve(); return; }
      pdfDoc.getPage(pageNum).then(async (page) => {
        // If a newer render started, abort early
        if (mySeq !== RENDER_SEQ) { resolve(); return; }

        const viewport = page.getViewport({ scale });
        const baseViewport = page.getViewport({ scale: 1 });

        if (!pageBaseSize[pageNum]) {
          pageBaseSize[pageNum] = { w: baseViewport.width, h: baseViewport.height };
        }
        pageCssSize[pageNum] = { w: viewport.width, h: viewport.height };

        const dpr = window.devicePixelRatio || 1;
        pageLayer.style.width  = `${viewport.width}px`;
        pageLayer.style.height = `${viewport.height}px`;

        setCssSize(pdfCanvas, viewport.width, viewport.height);
        setCssSize(boxesCanvas, viewport.width, viewport.height);
        setCssSize(overlayCanvas, viewport.width, viewport.height);
        setCssSize(annotationLayer, viewport.width, viewport.height);

        setDeviceSize(pdfCanvas, viewport.width * dpr, viewport.height * dpr);
        setDeviceSize(boxesCanvas, viewport.width * dpr, viewport.height * dpr);
        setDeviceSize(overlayCanvas, viewport.width * dpr, viewport.height * dpr);

        pdfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        boxesCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

        // Render the page bitmap
        await page.render({ canvasContext: pdfCtx, viewport }).promise;
        if (mySeq !== RENDER_SEQ) { resolve(); return; }

        // Render annotations DOM
        await (async () => {
          try {
            annotationLayer.innerHTML = "";
            const annotations = await page.getAnnotations({ intent: "display" });
            const view = viewport.clone({ dontFlip: true });
            const params = {
              viewport: view, div: annotationLayer, annotations, page,
              renderForms: false, annotationStorage: pdfDoc?.annotationStorage || null, enableScripting: false
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
          } catch { annotationLayer.innerHTML = ""; }
        })();

        if (mySeq !== RENDER_SEQ) { resolve(); return; }

        // UI + clear overlays after a successful commit
        syncNavInfo(pageNum);
        markActiveThumbnail(pageNum);

        boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        applyPointerRouting();

        LAST_COMMIT_SEQ = mySeq;
        // Give layout a tick to settle, then resolve
        requestAnimationFrame(() => resolve());
      }).catch(reject);
    });
  }

  async function buildThumbnails(pdf) {
    const host = thumbSidebar; if (!host) return;
    host.innerHTML = "";

    // 1) Create ordered placeholders first (guarantees 1..N top-to-bottom)
    const frag = document.createDocumentFragment();
    const items = new Array(pdf.numPages);

    for (let n = 1; n <= pdf.numPages; n++) {
      const w = document.createElement("div");
      w.className = "thumbnail-wrapper";
      w.dataset.page = String(n); // 1-based

      const c = document.createElement("canvas");
      c.className = "thumbnail";
      c.title = `Page ${n}`;
      c.dataset.page = String(n);

      const label = document.createElement("div");
      label.className = "thumbnail-label";
      label.textContent = `Page ${n}`;

      // click always uses the 1-based dataset, never a closure over pdf.js pageNumber
      c.addEventListener("click", () => {
        const p = parseInt(c.dataset.page, 10) || 1;
        currentPage = p;
        renderPage(currentPage);
      });

      w.appendChild(c);
      w.appendChild(label);
      frag.appendChild(w);

      items[n - 1] = { n, canvas: c };
    }

    host.appendChild(frag);

    // 2) Render pages sequentially into the pre-ordered canvases
    for (const it of items) {
      const page = await pdf.getPage(it.n);
      const viewport = page.getViewport({ scale: 0.3 });
      it.canvas.width = viewport.width;
      it.canvas.height = viewport.height;
      await page.render({ canvasContext: it.canvas.getContext("2d"), viewport }).promise;
    }

    markActiveThumbnail(currentPage);
  }

  // Scroll-to-zoom (CTRL/⌘ + wheel, or two-finger trackpad with ctrl)
  const pdfContainer = document.getElementById("pdfContainer");
  pdfContainer?.addEventListener("wheel", (e) => {
    // Only zoom when modifier is held to avoid blocking natural scroll
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const delta = e.deltaY;
    const factor = delta > 0 ? 0.9 : 1.1;
    setScaleAndRender(scale * factor);
  }, { passive: false });

  // Utility to run a step with weight and auto progress update
  async function runStep(label, weight, fn, basePctRef){
    updateProgress(basePctRef.pct, label);
    const start = basePctRef.pct;
    const target = Math.min(100, start + weight);
    let active = true;
    let lastTick = performance.now();
    // gentle ticking toward the target while the task runs (slower, time-based)
    const ticker = () => {
      if (!active) return;
      const now = performance.now();
      const dt = now - lastTick;
      if (dt > 110) {
        const inc = Math.max(0.15, (target - basePctRef.pct) * 0.04);
        const next = Math.min(target - 1, basePctRef.pct + inc);
        if (next > basePctRef.pct) {
          basePctRef.pct = next;
          updateProgress(basePctRef.pct, label);
        }
        lastTick = now;
      }
      requestAnimationFrame(ticker);
    };
    requestAnimationFrame(ticker);

    const out = await fn();
    active = false;
    basePctRef.pct = target;
    updateProgress(basePctRef.pct, label);
    return out;
  }

  // ========================
  // Run Analysis
  // ========================
  analyzeBtn?.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    if (analyzeBtn) {
      analyzeBtn.classList.add("running");
      analyzeBtn.setAttribute("aria-busy", "true");
    }
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
        sidebar?.classList.remove("sidebar-collapsed");
        if (analyzeBtn) {
          analyzeBtn.classList.add("dismissed");
          analyzeBtn.setAttribute("aria-hidden", "true");
          analyzeBtn.setAttribute("tabindex", "-1");
          analyzeBtn.classList.remove("running");
          analyzeBtn.removeAttribute("aria-busy");
        }
        if (sidebarTitle) sidebarTitle.textContent = "Here are your summaries";
      }, P);

      // 5) Initial overlay draw (15%)
      await runStep("Overlaying navigation aids…", 15, async () => {
        if (showBoxes && currentFormId) await drawOverlay(currentFormId, currentPage);
      }, P);

      // 6) Final polish (10%)
      await runStep("Finalizing… Please hold on!", 10, async () => {
        // any quick, non-blocking polish can go here later
      }, P);

      // Success
      lastFinishAt = nowMs();
      lastDuration = lastFinishAt - analysisStartAt;
      closeProgressSuccess();

    } catch (e) {
      console.error("runAnalysis error:", e);
      lastFinishAt = nowMs();
      lastDuration = lastFinishAt - (analysisStartAt || lastFinishAt);
      closeProgressError(e?.message);
    } finally {
      if (analyzeBtn && !analyzeBtn.classList.contains("dismissed")) {
        analyzeBtn.classList.remove("running");
        analyzeBtn.removeAttribute("aria-busy");
      }
    }
  }

  function renderSummaries(explainer) {
    if (summaryList) summaryList.innerHTML = "";
    const sections = explainer.sections || [];
    sections.forEach((sec, idx) => {
      const item = document.createElement("div");
      item.className = "accordion-item";
      const header = document.createElement("div");
      header.className = "accordion-header";
      header.textContent = sec.title || "";
      const content = document.createElement("div");
      content.className = "accordion-content" + (idx === 0 ? " active" : "");
      (sec.fields || []).forEach((f) => {
        const row = document.createElement("p");
        row.className = "summary-line";
        row.innerHTML = `<span class="summary-label" data-label="${esc(f.label)}">${esc(f.label)}</span>: ${esc(f.summary)}`;
        content.appendChild(row);
      });
      header.addEventListener("click", () => {
        const isActive = content.classList.contains("active");
        // close all others
        summaryList?.querySelectorAll(".accordion-content.active").forEach((c) => {
          if (c !== content) {
            c.classList.remove("active");
            c.previousElementSibling?.classList.remove("open");
          }
        });
        if (isActive) {
          content.classList.remove("active");
          header.classList.remove("open");
        } else {
          content.classList.add("active");
          header.classList.add("open");
        }
      });
      header.classList.toggle("open", idx === 0);
      item.appendChild(header); item.appendChild(content);
      summaryList?.appendChild(item);
    });
    if (metricsRow) metricsRow.textContent = "";
  }

  // Debounce guard so clicks don't pile up
  let jumping = false;

  // click → jump
  summaryList?.addEventListener("click", async (ev) => {
    const lbl = ev.target.closest(".summary-label");
    if (!lbl || !currentFormId || jumping) return;

    jumping = true;
    try {
      const ok = await ensureAnnotationsReady();
      if (!ok) return;

      const anchor = findAnchorForLabel(lbl.textContent.trim(), cachedAnnotations);
      await jumpToAnchor(anchor); // assumes the cancellable, awaited renderPage version
    } catch (e) {
      console.warn("Label jump failed:", e);
    } finally {
      jumping = false;
    }
  });

  async function drawOverlay(formId, pageNumber, anchor = null) {
    try {
      if (!cachedAnnotations || cachedAnnotations.__formId !== formId) {
        const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) { boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height); return; }
        cachedAnnotations = await res.json();
        cachedAnnotations.__formId = formId;
      }

      // clear only the boxes layer
      boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);

      const pageIdx = pageNumber - 1;
      // Draw background rectangles only if showBoxes is ON
      if (showBoxes) {
        let rects = [];
        if (Array.isArray(cachedAnnotations.groups) && cachedAnnotations.groups.length) {
          rects = cachedAnnotations.groups.filter((g) => (g.page || 0) === pageIdx);
        } else if (Array.isArray(cachedAnnotations.tokens)) {
          rects = cachedAnnotations.tokens.filter((t) => (t.page || 0) === pageIdx);
        }

        boxesCtx.save();
        boxesCtx.lineWidth = 1;
        boxesCtx.strokeStyle = "rgba(20,20,20,0.25)";
        for (const r of rects) {
          const rect = pdfBBoxToCssRect(r.bbox, pageNumber);
          boxesCtx.strokeRect(rect.x, rect.y, rect.w, rect.h);
        }
        boxesCtx.restore();
      }

      // Anchor highlight (always draw if provided)
      if (anchor && Array.isArray(anchor.bbox)) {
        const rect = pdfBBoxToCssRect(anchor.bbox, pageNumber);
        boxesCtx.save();
        boxesCtx.lineWidth = 2;
        boxesCtx.strokeStyle = "rgba(0,0,0,0.95)";
        boxesCtx.fillStyle   = "rgba(255,255,0,0.2)";
        boxesCtx.fillRect(rect.x, rect.y, rect.w, rect.h);
        boxesCtx.strokeRect(rect.x, rect.y, rect.w, rect.h);
        boxesCtx.restore();
      }
    } catch (e) {
      // If anything fails, clear the layer but don't double-restore
      boxesCtx.clearRect(0, 0, boxesCanvas.width, boxesCanvas.height);
      // console.warn("drawOverlay failed:", e);
    }
  }

  function centerOnAnchor(pageNum, anchor){
    try {
      const scroller = document.getElementById("pdfContainer") || document.scrollingElement || document.documentElement;
      const rect = pdfBBoxToCssRect(anchor.bbox, pageNum);
      const midY = rect.y + rect.h * 0.5;

      // Get the page layer's top offset relative to the scroller
      const pageTop = (pageLayer.getBoundingClientRect().top + scroller.scrollTop) - (document.documentElement.getBoundingClientRect().top || 0);
      const target = Math.max(0, pageTop + midY - (scroller.clientHeight * 0.5));

      scroller.scrollTo({ top: target, behavior: "smooth" });
    } catch {}
  }

  // ---- Backend helpers ----
  async function softReuploadForDisk(webUrl) { try {
    const url = normalizeToWebUrl(webUrl);
    const r = await fetch(url, { cache: "no-store" }); if (!r.ok) return null;
    const blob = await r.blob(); const base = baseFromPath(url) || "form.pdf";
    const fd = new FormData(); fd.append("file", new File([blob], base, { type: "application/pdf" }));
    const up = await apiFetch("/api/upload", { method: "POST", body: fd }); if (!up.ok) return null;
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
    const up = await apiFetch("/api/upload", { method: "POST", body: fd });
    if (!up.ok) throw new Error("upload failed");
    const out = await up.json();
    return { web_path: normalizeToWebUrl(out.web_path), disk_path: out.disk_path, form_id: out.canonical_form_id || out.form_id, file_name: file.name };
  }

  async function ensurePrelabelAndOverlays(server, hashId) {
    const fd = new FormData();
    fd.append("pdf_disk_path", server.disk_path);
    fd.append("form_id", hashId);
    const r = await apiFetch("/api/prelabel", { method: "POST", body: fd });
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

  function cleanLabelRaw(s){
    // Trim trailing ":" or "*" and collapse spaces, case-insensitive compare
    return (s||"").toString().replace(/[:*]+$/,"").replace(/\s+/g," ").trim().toLowerCase();
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
    for (const t of pageTokens) if (sameLineLoose(seed, t)) box = bboxUnion(box, t.bbox);
    return box;
  }

  function stripPunctLite(s){
    // keep letters/numbers/spaces; drop common label punctuation (: * · – — . , / \ ( ) [ ])
    return (s||"").replace(/[:*·–—.,\/\\()[\]]/g, " ").replace(/\s+/g," ").trim();
  }

  function normalizeLabel(s){
    // your normalizeText + colon/asterisk handling, aggressive but safe for labels
    s = (s||"").toString().trim();
    // tolerate common UI suffixes like ":" or " *"
    s = s.replace(/[:*]+$/,"").trim();
    s = stripPunctLite(s);
    return normalizeText(s); // uses your SYN_MAP + lowercasing + space collapse
  }

  function sameLineLoose(a, b){
    // more tolerant than the old 8px; PDFs vary a lot
    const midA = (a.bbox[1] + a.bbox[3]) / 2;
    const midB = (b.bbox[1] + b.bbox[3]) / 2;
    return Math.abs(midA - midB) < 14;
  }

  function unionTokensBBox(tokens){
    let box = tokens[0].bbox.slice(0,4);
    for (let i=1;i<tokens.length;i++) box = bboxUnion(box, tokens[i].bbox);
    return box;
  }

  function buildLabelIndex(annotations) {
  const out = { map: new Map() };
  const tokens = Array.isArray(annotations.tokens) ? annotations.tokens : [];
  if (!tokens.length) return out;

  // group tokens by page + normalize
  const byPage = new Map();
  for (const t of tokens) {
    if (!t || !t.text || !Array.isArray(t.bbox)) continue;
    const p = t.page || 0;
    if (!byPage.has(p)) byPage.set(p, []);
    byPage.get(p).push({ ...t, _norm: normalizeLabel(t.text) });
  }

  for (const [pageIdx, arr] of byPage.entries()) {
    // sort by row then x
    arr.sort((a,b)=> (a.bbox[1]-b.bbox[1]) || (a.bbox[0]-b.bbox[0]));

    // segment to lines with loose tolerance
    const lines = [];
    for (const tok of arr) {
      let placed = false;
      for (const line of lines) {
        if (sameLineLoose(line[line.length-1], tok)) { line.push(tok); placed = true; break; }
      }
      if (!placed) lines.push([tok]);
    }

    // index windows up to length 8
    for (const line of lines) {
      line.sort((a,b)=> a.bbox[0]-b.bbox[0]);
      const norms = line.map(t => (t._norm||"").trim()).filter(Boolean);
      for (let i=0;i<line.length;i++){
        let joined = "";
        const acc = [];
        for (let j=i;j<line.length && (j-i)<8; j++){
          const piece = norms[j]; if (!piece) continue;
          joined = (joined ? (joined + " " + piece) : piece);
          acc.push(line[j]);

          const key = joined;
          if (!out.map.has(key)) out.map.set(key, { page: pageIdx, bbox: unionTokensBBox(acc) });

          const nospace = key.replace(/\s+/g,"");
          if (!out.map.has(nospace)) out.map.set(nospace, { page: pageIdx, bbox: unionTokensBBox(acc) });
        }
      }
    }
  }

  return out;
}

// Helper: keep punctuation that often appears in labels out of the comparison
function cleanLabelRaw(s){
  return (s || "")
    .toString()
    .replace(/[:*]+$/,"")       // trim trailing ":" or "*" (common UI suffixes)
    .replace(/\s+/g," ")        // collapse whitespace
    .trim()
    .toLowerCase();
}

/**
 * Robust label → anchor:
 * 0) EXACT phrase match on line-constrained token windows (index fast path)
 * 0a) RAW exact group-label match (literal wording, punctuation-trimmed)
 * 1) Exact group-label match (normalized)
 * 2) Fuzzy group (similarity)
 * 3) Fuzzy token (similarity) with small left-bias
 */
function findAnchorForLabel(labelRaw, annotations){
  if (!annotations) return null;

  const labelNorm    = normalizeLabel(labelRaw);
  const labelRawClean = cleanLabelRaw(labelRaw);
  if (!labelNorm) return null;

  // 0) phrase index (wins)
  const idx = annotations.__index;
  if (idx && idx.map) {
    const direct = idx.map.get(labelNorm) || idx.map.get(labelNorm.replace(/\s+/g,""));
    if (direct) return { page: direct.page, bbox: direct.bbox, score: 1.0, source: "index-exact" };
  }

  // 0a) RAW exact group label (strongest for “exact wording in PDF”)
  const groups0 = Array.isArray(annotations.groups) ? annotations.groups : [];
  if (groups0.length){
    const exactRaw = groups0.find(g => cleanLabelRaw(g.label || "") === labelRawClean);
    if (exactRaw && Array.isArray(exactRaw.bbox)) {
      return { page:(exactRaw.page||0), bbox: exactRaw.bbox, score: 1.0, source: "group-raw-exact" };
    }
  }

  // 1) exact group label (normalized)
  const groups = groups0;
  if (groups.length){
    const exact = groups.find(g => normalizeLabel(g.label || "") === labelNorm);
    if (exact && Array.isArray(exact.bbox)) {
      return { page:(exact.page||0), bbox:exact.bbox, score:.98, source:"group-exact" };
    }
  }

  // 2) fuzzy group
  let best = null;
  for (const g of groups){
    const s = similarity(labelRaw, g.label || "");
    if (s > 0 && (!best || s > best.score)) best = { page:(g.page||0), bbox:g.bbox, score:s, source:"group-fuzzy" };
  }
  if (best && best.score >= 0.70 && Array.isArray(best.bbox)) return best;

  // 3) fuzzy tokens (line union)
  const tokens = Array.isArray(annotations.tokens) ? annotations.tokens : [];
  if (tokens.length){
    let tBest = null;
    for (const t of tokens){
      const s = similarity(labelRaw, t.text || "");
      if (s <= 0) continue;
      const leftBias = (t.bbox?.[0] ?? 0) / 20000;
      const sc = s - leftBias;
      if (!tBest || sc > tBest.score) tBest = { token:t, score:sc, page:(t.page||0) };
    }
    if (tBest && tBest.score >= 0.48){
      const pageTokens = tokens.filter(x => (x.page||0) === tBest.page);
      const lineBox = unionLine(tBest.token, pageTokens);
      return { page: tBest.page, bbox: lineBox, score: tBest.score, source:"token-fuzzy" };
    }
  }

  return null;
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
  const CHECK_DEFAULT = { glyph: "✓", size: 16 };

  // UI inputs (optional)
  const penWidthInput = document.getElementById("penWidth");
  const penColorInput = document.getElementById("penColor");
  const hlWidthInput  = document.getElementById("hlWidth");
  const hlColorInput  = document.getElementById("hlColor");
  const hlAlphaInput  = document.getElementById("hlAlpha");
  const penResetBtn   = document.getElementById("penReset");
  const hlResetBtn    = document.getElementById("hlReset");
  const checkResetBtn = document.getElementById("checkReset");
  const checkGlyphSel = document.getElementById("checkGlyph");
  const checkSizeInput= document.getElementById("checkFontSize");
  function currentPen() {
    return {
      color: (penColorInput && penColorInput.value) || PEN.color,
      width: (penWidthInput && (+penWidthInput.value || PEN.width)) || PEN.width,
      alpha: PEN.alpha
    };
  }
  function currentHL() {
    const col = (hlColorInput && hlColorInput.value) || "#ffff00";
    const alphaRaw = hlAlphaInput ? (+hlAlphaInput.value || (HL.alpha * 100)) : (HL.alpha * 100);
    const alpha = Math.min(0.95, Math.max(0.05, alphaRaw / 100));
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
      showToolToast(`${(name || "Tool")} off`, { tone: "off" });
      return;
    }
    currentTool = name;
    toolButtons.forEach(b => b.classList.toggle("active", b === btn));
    applyPointerRouting();
    const toolName = {
      text: "Insert text",
      pen: "Draw (pen)",
      highlight: "Highlight",
      check: "Check placer",
      erase: "Eraser"
    }[name] || "Tool selected";
    showToolToast(`${toolName} activated`, { tone: "on" });
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
  function pdfBBoxToCssRect(bbox, pageNum) {
    // Convert PDF-space bbox (origin already top-left from our OCR) into CSS px
    const { sx, sy } = pageScaleFactors(pageNum);
    const x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
    const yTop = y0;
    const yBot = y1;
    return {
      x: x0 * sx,
      y: yTop * sy,
      w: Math.max(1, (x1 - x0) * sx),
      h: Math.max(1, (yBot - yTop) * sy)
    };
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
    const ok = await showConfirm({
      title: "Clear this page?",
      text: "All drawings and inserted text on the current page will be removed.",
      variant: "warning",
      confirmText: "Yes, clear page",
      cancelText: "Cancel"
    });
    if (!ok) return;
    pageEdits[currentPage] = { strokes: [], texts: [] };
    [...document.querySelectorAll(".text-annot")].forEach(n => n.remove());
    paintEdits(currentPage);
    playClearAnimation();
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
            if (!isMetricsOptIn() || !workspaceShownAt) return;
            // Allow final save to log even if we logged earlier events (e.g., print); other events respect the guard.
            if (status !== "saved" && workspaceLogged) return;
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
              navigator.sendBeacon(apiUrl("/api/user.log"), blob);
            } else {
              await POST_JSON("/api/user.log", payload);
            }
            if (status === "saved") {
              workspaceLogged = true;
              ws_clearInflight();
            }
          } catch (e) {
            console.warn("[workspace event log] failed:", e);
          }
        }

        // ---- Reset session storage and go back to index (fresh user upload) ----
        async function resetAndGoHome(options) {
          const opts = options || {};
          const shouldDelete = !!opts.deleteUpload;
          const diskPath = shouldDelete ? (sessionStorage.getItem("uploadedDiskPath") || "") : "";

          // Best-effort cleanup of the temporary upload, if requested.
          if (shouldDelete && diskPath) {
            try {
              await apiFetch("/api/upload.delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ disk_path: diskPath })
              });
            } catch (e) {
              console.warn("[workspace] upload delete failed:", e);
            }
          }

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
    const ok = await showConfirm({
      title: "Save edited PDF?",
      text: "Export and download a copy with your drawings and inserted text.",
      variant: "question",
      confirmText: "Save",
      cancelText: "Cancel"
    });
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
      const metricsNote = isMetricsOptIn() ? "<div style='font-size:12px;opacity:.9;'>Your beta metrics were recorded.</div>" : "";
      await showAlert({
        variant: "success",
        title: "Saved",
        html: metricsNote || undefined,
        autoCloseMs: 1200,
        confirmText: "Nice"
      });

      // Ask if the user wants to start fresh
      const startFresh = await showConfirm({
        title: "Start a new session?",
        text: "Return to the home screen to upload a new PDF.",
        variant: "question",
        confirmText: "Yes, start fresh",
        cancelText: "Stay here"
      });

      // Log completion with finished_at = user confirmation time (not including export time)
      await logWorkspaceEvent("saved", tConfirm, { render_ms: renderMs });

      if (startFresh) {
        await resetAndGoHome({ deleteUpload: false });
        return;
      }

    } catch (e) {
      console.error("Save failed", e);
      showAlert({ variant: "danger", title: "Save failed", text: e?.message || "Could not generate edited PDF." });
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
      await apiFetch("/api/edited.register", {
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
  let ws_silenceAbandon   = false;

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
      if (!raw || !isMetricsOptIn()) return;
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
        navigator.sendBeacon(apiUrl("/api/user.log"), blob);
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
  const METRICS_KEY = "if_metrics_opt_in";

  function isMetricsOptIn() {
    const flag = sessionStorage.getItem(METRICS_KEY) || localStorage.getItem(METRICS_KEY);
    return flag === "1";
  }

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
    const r = await apiFetch(url, {
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
      if (!payload.canonical_id || !isMetricsOptIn()) return;
      await POST_JSON("/api/metrics.log", payload);
    } catch (e) {
      console.warn("[metrics.log] failed:", e);
    }
  }

  async function logUserSession(finalMeta) {
    try {
      if (!lastCanonicalId || !isMetricsOptIn()) return;
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
      if (!analysisStartAt || lastFinishAt || !isMetricsOptIn()) return;
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
      navigator.sendBeacon && navigator.sendBeacon(apiUrl("/api/user.log"), blob);
    } catch {}
  });

  function ws_sendAbandonBeacon(tag) {
    try {
      if (!isMetricsOptIn() || !workspaceShownAt || workspaceLogged) return;
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
        navigator.sendBeacon(apiUrl("/api/user.log"), blob);
        workspaceLogged = true;
        ws_clearInflight();
      } else {
        apiFetch("/api/user.log", { method: "POST", headers: { "Content-Type": "application/json" }, body }).catch(()=>{});
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

  async function ensureAnnotationsReady() {
    try {
      const formId = sessionStorage.getItem("uploadedFormId") || currentFormId;
      const disk   = sessionStorage.getItem("uploadedDiskPath");
      if (!formId || !disk) return false;

      // If already loaded for this form (and index built), we're done
      if (cachedAnnotations && cachedAnnotations.__formId === formId && cachedAnnotations.__indexBuilt) {
        return true;
      }

      // Try fetch annotations
      let ok = true;
      try {
        const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) ok = false;
        else {
          cachedAnnotations = await res.json();
          cachedAnnotations.__formId = formId;
        }
      } catch { ok = false; }

      // If missing, run prelabel to generate them, then refetch
      if (!ok) {
        await ensurePrelabelAndOverlays({ disk_path: disk }, formId);
        const res2 = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res2.ok) return false;
        cachedAnnotations = await res2.json();
        cachedAnnotations.__formId = formId;
      }

      // Build exact-phrase index once (requires buildLabelIndex defined)
      if (!cachedAnnotations.__indexBuilt) {
        cachedAnnotations.__index = buildLabelIndex(cachedAnnotations);
        cachedAnnotations.__indexBuilt = true;
      }

      currentFormId = formId; // keep in sync
      return true;
    } catch {
      return false;
    }
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
    currentPage = (m.page || 0) + 1;
    await renderPage(currentPage);  // <-- await the commit
    if (RENDER_SEQ === LAST_COMMIT_SEQ) {
      await new Promise(r => setTimeout(r, 60)); // tiny settle, like jumpToAnchor
      drawOverlay(currentFormId, currentPage, { page: m.page || 0, bbox: m.bbox });
      centerOnAnchor(currentPage, { bbox: m.bbox }); 
    }
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

async function jumpToAnchor(anchor) {
  if (!anchor) return;
  currentPage = (anchor.page || 0) + 1;
  await renderPage(currentPage);                // <-- wait for the newest render to finish
  if (RENDER_SEQ !== LAST_COMMIT_SEQ) return;   // if another render started, skip
  await new Promise(r => setTimeout(r, 60));    // tiny settle
  drawOverlay(currentFormId, currentPage, anchor);
  centerOnAnchor(currentPage, anchor);
}

async function jumpToLabelText(text){
  if (jumping) return;
  jumping = true;
  try {
    const ok = await ensureAnnotationsReady();
    if (!ok) return;
    const anchor = findAnchorForLabel(text, cachedAnnotations);
    await jumpToAnchor(anchor);
  } finally {
    jumping = false;
  }
}

  /* -------------------------------
   * Check Placer integration
   * Paste ABOVE the closing "} // end initWorkspace"
   * ------------------------------- */

  // Live preset state (default to ✓, 16px)
  let CHECK_PRESET = {
    glyph: (document.body.dataset.textPreset || "✓"),
    size:  parseInt(document.body.dataset.textPresetSize || "16", 10) || 16,
  };

  // Listen for preset changes from the navbar dropdown script
  document.addEventListener("intelliform:textPreset", (e) => {
    try {
      const d = e?.detail || {};
      if (typeof d.glyph === "string" && d.glyph) CHECK_PRESET.glyph = d.glyph;
      if (d.size != null && !Number.isNaN(parseInt(d.size,10))) CHECK_PRESET.size = parseInt(d.size,10);
      // also mirror to dataset so other code can read it if needed
      document.body.dataset.textPreset = CHECK_PRESET.glyph;
      document.body.dataset.textPresetSize = String(CHECK_PRESET.size);
    } catch {}
  });

  // Reset handlers
  penResetBtn?.addEventListener("click", () => {
    if (penWidthInput) penWidthInput.value = String(PEN.width);
    if (penColorInput) penColorInput.value = "#111111";
    showToolToast("Pen reset to default", { tone: "on" });
  });
  hlResetBtn?.addEventListener("click", () => {
    if (hlWidthInput) hlWidthInput.value = String(HL.width);
    if (hlColorInput) hlColorInput.value = "#ffff00";
    if (hlAlphaInput) hlAlphaInput.value = String(Math.round(HL.alpha * 100));
    showToolToast("Highlight reset to default", { tone: "on" });
  });
  checkResetBtn?.addEventListener("click", () => {
    if (checkGlyphSel) checkGlyphSel.value = CHECK_DEFAULT.glyph;
    if (checkSizeInput) checkSizeInput.value = String(CHECK_DEFAULT.size);
    CHECK_PRESET.glyph = CHECK_DEFAULT.glyph;
    CHECK_PRESET.size = CHECK_DEFAULT.size;
    const evt = new CustomEvent("intelliform:textPreset", { detail: { glyph: CHECK_PRESET.glyph, size: CHECK_PRESET.size } });
    document.dispatchEvent(evt);
    showToolToast("Check placer reset to default", { tone: "on" });
  });

  // Persist check preset when fields change
  checkGlyphSel?.addEventListener("change", () => {
    CHECK_PRESET.glyph = checkGlyphSel.value || CHECK_DEFAULT.glyph;
    document.body.dataset.textPreset = CHECK_PRESET.glyph;
  });
  checkSizeInput?.addEventListener("input", () => {
    const val = parseInt(checkSizeInput.value || CHECK_DEFAULT.size, 10);
    if (!Number.isNaN(val)) {
      CHECK_PRESET.size = val;
      document.body.dataset.textPresetSize = String(val);
    }
  });

  // Helper: set active visual state in the toolbar
  function setToolbarActive(btnEl){
    const all = Array.from(document.querySelectorAll("#editToolbar .tool-btn"));
    all.forEach(b => b.classList.remove("active"));
    if (btnEl) btnEl.classList.add("active");
  }

  // Toast helper for tool selection
  let toastTimeout = null;
  function showToolToast(msg, opts){
    if (!toastHost) return;
    toastHost.textContent = msg;
    toastHost.classList.toggle("off", opts?.tone === "off");
    toastHost.classList.add("show");
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      toastHost.classList.remove("show");
    }, 2000);
  }

  // When the user clicks the Check Placer button:
  //  - route to TEXT tool under the hood
  //  - keep the check button visually "active"
  const checkBtn = document.querySelector('#editToolbar .tool-btn[data-tool="check"]');
  if (checkBtn) {
    checkBtn.addEventListener("click", (ev) => {
      const caret = ev.target.closest(".caret");
      const ddMenu = checkBtn.parentElement?.querySelector(".dropdown-menu");

      // caret click → toggle dropdown, do not change tool
      if (caret && ddMenu) {
        ev.preventDefault();
        ev.stopPropagation();
        const open = ddMenu.classList.contains("open");
        document.querySelectorAll("#editToolbar .dropdown-menu.open").forEach(m => m.classList.remove("open"));
        ddMenu.classList.toggle("open", !open);
        checkBtn.setAttribute("aria-expanded", String(!open));
        return;
      }

      ev.stopPropagation();

      // Toggle off if already active
      const alreadyActive = checkBtn.classList.contains("active");
      if (alreadyActive) {
        checkBtn.classList.remove("active");
        currentTool = null;
        applyPointerRouting();
        showToolToast("Check placer off", { tone: "off" });
        return;
      }

      // ensure we are in edit mode
      if (!editMode) enterEdit();

      // adopt the latest preset from body dataset if present
      CHECK_PRESET.glyph = (document.body.dataset.textPreset || CHECK_PRESET.glyph || "✓");
      CHECK_PRESET.size  = parseInt(document.body.dataset.textPresetSize || CHECK_PRESET.size || "16", 10) || 16;

      // Internally work as the TEXT tool
      currentTool = "text";
      applyPointerRouting();

      // visually mark the check button as active (not the Text button)
      setToolbarActive(checkBtn);
      showToolToast("Check placer activated");
    });
  }

  // After a text placement click, if Check Placer is the visually active tool,
  // replace the new text node content with the preset glyph and size.
  // We piggyback on the existing pointerdown flow that already spawns the node.
  overlayCanvas.addEventListener("pointerdown", () => {
    // Only when we are effectively in text mode AND the check button is the active one
    const isCheckActive = checkBtn && checkBtn.classList.contains("active");
    if (!editMode || currentTool !== "text" || !isCheckActive) return;

    // Let the built-in spawnTextNode run first, then patch the newest node
    setTimeout(() => {
      try {
        // find the most recently added .text-annot on the current page
        const nodes = Array.from(document.querySelectorAll('.text-annot'))
          .filter(n => (n.dataset.page|0) === (currentPage|0));
        if (!nodes.length) return;
        const node = nodes[nodes.length - 1];

        const content = node.querySelector('.ta-content');
        if (!content) return;

        // apply preset text + font size
        content.textContent = CHECK_PRESET.glyph || "✓";
        content.style.fontSize = `${CHECK_PRESET.size || 16}px`;

        // update the backing model so export/save uses the same values
        const pageNum = Number(node.dataset.page);
        const ed = EDIT(pageNum);
        const m = ed.texts.find(t => t.id === node.dataset.key);
        if (m) {
          m.text = content.textContent;
          m.size = CHECK_PRESET.size || 16;
        }
      } catch {}
    }, 0);
  });

  // Guided walkthrough (one-time per session) highlighting sidebar + toggle
  const tourSteps = [
    {
      targets: ["#togglePages", "#editToolbar"],
      title: "Left tools",
      body: "Toggle thumbnails, then draw, highlight, checks, text, or erase here."
    },
    {
      selector: ".navbar-center",
      title: "Page & Zoom",
      body: "See your current page and zoom level."
    },
    {
      targets: ["#btnUndo", "#btnClear"],
      title: "Undo & Clear",
      body: "Undo last action or clear the current page."
    },
    {
      selector: "#btnExit",
      title: "Exit",
      body: "Leave this workspace and return home."
    },
    {
      targets: ["#printPDF", "#downloadPDF"],
      title: "Print & Download",
      body: "Print your PDF or download it locally."
    },
    {
      selector: "#eceScoreBadge",
      title: "Save",
      body: "Save your filled form with detected labels."
    },
    {
      selector: "#sidebarToggle",
      title: "Sidebar toggle",
      body: "Open the toolbox for Analyze, Search, and summaries."
    },
    {
      selector: "#sidebar",
      title: "Toolbox",
      body: "Use this panel to run Analyze and review outputs.",
      sidebarOpen: true
    },
    {
      selector: "#analyzeTool",
      title: "Analyze",
      body: "Run AI to detect sections and fields.",
      sidebarOpen: true
    },
    {
      selector: "#searchTool",
      title: "Search",
      body: "Find text in the form and jump to matches.",
      sidebarOpen: true,
      sidebarCloseAfter: true
    },
    {
      selector: "#faqButton",
      title: "FAQ",
      body: "Open quick tips anytime."
    },
    {
      selector: "#tourTrigger",
      title: "Walkthrough",
      body: "Relaunch this tour any time from the bottom-left button."
    }
  ];

  function startWalkthrough() {
    let idx = 0;
    let overlay = null;
    let card = null;
    let arrow = null;
    let connector = null;
    const highlightClass = "tour-highlight";

    function cleanup() {
      overlay?.remove();
      card?.remove();
      arrow?.remove();
      connector?.remove();
      document.querySelectorAll(`.${highlightClass}`).forEach(el => el.classList.remove(highlightClass));
      document.body.classList.remove("tour-active");
    }

    function placeStep(step) {
      cleanup();
      if (!step) { return; }
      const targetEls = [
        ...(step.targets ? step.targets.map(sel => document.querySelector(sel)).filter(Boolean) : []),
        ...(step.selector ? [document.querySelector(step.selector)].filter(Boolean) : [])
      ];
      if (!targetEls.length) { next(); return; }

      const sidebar = document.getElementById("sidebar");
      const sidebarToggle = document.getElementById("sidebarToggle");
      if (step.sidebarOpen && sidebar) {
        sidebar.classList.add("open");
        if (sidebarToggle) sidebarToggle.style.display = "none";
      }

      targetEls.forEach(el => el.classList.add(highlightClass));

      const bounds = targetEls.reduce((acc, el, i) => {
        const r = el.getBoundingClientRect();
        if (i === 0) return { top:r.top, left:r.left, right:r.right, bottom:r.bottom };
        return {
          top: Math.min(acc.top, r.top),
          left: Math.min(acc.left, r.left),
          right: Math.max(acc.right, r.right),
          bottom: Math.max(acc.bottom, r.bottom)
        };
      }, {});
      bounds.width = bounds.right - bounds.left;
      bounds.height = bounds.bottom - bounds.top;

      overlay = document.createElement("div");
      overlay.className = "tour-overlay";
      overlay.addEventListener("click", () => next());
      const cx = bounds.left + bounds.width / 2;
      const cy = bounds.top + window.scrollY + bounds.height / 2;
      const r = Math.max(bounds.width, bounds.height) / 2 + 120; // larger cutout
      overlay.style.setProperty("--tour-hole-x", `${cx}px`);
      overlay.style.setProperty("--tour-hole-y", `${cy}px`);
      overlay.style.setProperty("--tour-hole-r", `${r}px`);
      document.body.appendChild(overlay);

      arrow = document.createElement("div");
      arrow.className = "tour-arrow";
      const arrowY = bounds.top + window.scrollY + bounds.height / 2 - 8;
      let arrowX = bounds.left + bounds.width + 14;
      if (arrowX > window.innerWidth - 40) arrowX = bounds.left - 20; // flip to left if near edge
      arrow.style.top = `${arrowY}px`;
      arrow.style.left = `${arrowX}px`;
      document.body.appendChild(arrow);

      card = document.createElement("div");
      card.className = "tour-card";
      card.innerHTML = `
        <div class="tour-title">${step.title}</div>
        <div class="tour-body">${step.body}</div>
        <div class="tour-actions">
          <button type="button" class="tour-btn tour-skip">Skip</button>
          <button type="button" class="tour-btn tour-next">${idx === tourSteps.length - 1 ? "Finish" : "Got it"}</button>
        </div>
      `;
      document.body.appendChild(card);

      // Connector from card center to arrow
      const cardRect = card.getBoundingClientRect();
      const cardCx = cardRect.left + cardRect.width / 2;
      const cardCy = cardRect.top + cardRect.height / 2;
      const arrowCx = arrowX + 8;
      const arrowCy = arrowY + 8;
      const dx = arrowCx - cardCx;
      const dy = arrowCy - cardCy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const angle = Math.atan2(dy, dx);
      connector = document.createElement("div");
      connector.className = "tour-connector";
      connector.style.width = `${dist}px`;
      connector.style.left = `${cardCx}px`;
      connector.style.top = `${cardCy}px`;
      connector.style.transform = `translate(0, -1.5px) rotate(${angle}rad)`;
      document.body.appendChild(connector);

      card.querySelector(".tour-next")?.addEventListener("click", (e) => { e.stopPropagation(); next(); });
      card.querySelector(".tour-skip")?.addEventListener("click", (e) => { e.stopPropagation(); endTour(); });
      // no inline trigger (use bottom-left button instead)

      document.body.classList.add("tour-active");
    }

    function next() {
      idx += 1;
      if (idx >= tourSteps.length) { endTour(); return; }
      const prev = tourSteps[idx - 1];
      if (prev && prev.sidebarCloseAfter) {
        const sidebar = document.getElementById("sidebar");
        const sidebarToggle = document.getElementById("sidebarToggle");
        if (sidebar) sidebar.classList.remove("open");
        if (sidebarToggle) sidebarToggle.style.display = "flex";
      }
      placeStep(tourSteps[idx]);
    }

    function endTour() {
      cleanup();
    }

    placeStep(tourSteps[idx]);
  }

  setTimeout(startWalkthrough, 800);
  tourTrigger.addEventListener("click", () => startWalkthrough());

  function playClearAnimation() {
    const host = document.getElementById("pageLayer") || document.getElementById("pdfContainer");
    if (!host) return;
    const overlay = document.createElement("div");
    overlay.className = "clear-overlay";
    host.appendChild(overlay);
    setTimeout(() => overlay.remove(), 900);
  }

} // end initWorkspace
