// static/js/workspace.js
window.addEventListener("load", initWorkspace);

function initWorkspace() {
  console.log("workspace.js 2025-10-10 template-hash flow");

  // ---- Session state (legacy-safe) ----
  const legacyWebRaw = sessionStorage.getItem("uploadedFileWithExtension") || "";
  const storedWebRaw = sessionStorage.getItem("uploadedWebPath") || legacyWebRaw || "";
  const storedWeb = normalizeToWebUrl(storedWebRaw);
  let storedDisk = sessionStorage.getItem("uploadedDiskPath") || "";
  const storedName = sessionStorage.getItem("uploadedFileName") || "";
  let currentFormId = sessionStorage.getItem("uploadedFormId") || null; // ← canonical hash

  // ---- DOM refs ----
  const byId = (id) => document.getElementById(id);
  const sidebarToggle = byId("sidebarToggle");
  const sidebar = byId("sidebar");
  const pageToggler = byId("togglePages");
  const thumbSidebar = byId("thumbnailSidebar");
  const analyzeBtn = byId("analyzeTool");
  const summaryList = byId("summaryList");
  const formTitle = byId("formNameDisplay");
  const metricsRow = byId("metricsRow");
  const pageInfo = byId("pageInfo");
  const zoomInfo = byId("zoomInfo");
  const pdfCanvas = byId("pdfCanvas");
  const overlayCanvas = byId("overlayCanvas");

  if (!pdfCanvas || !overlayCanvas) {
    console.error("Canvas elements missing");
    return;
  }

  const pdfCtx = pdfCanvas.getContext("2d");
  const overlayCtx = overlayCanvas.getContext("2d");

  // ---- Provisional title (replaced by explainer.title later) ----
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
  if (pageToggler && thumbSidebar) {
    pageToggler.addEventListener("click", () => thumbSidebar.classList.toggle("visible"));
  }

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
    toggleBoxes = byId("toggleBoxes");
  }

  // ---- PDF state ----
  let pdfDoc = null;
  let scale = 1.5;
  let currentPage = 1;
  const pageBaseSize = {};
  let cachedAnnotations = null; // the annotations JSON currently in memory

  // ---- PDF.js check ----
  if (typeof pdfjsLib === "undefined") {
    alert("PDF.js failed to load.");
    return;
  }

  // ---- Boot viewer ----
  (async function boot() {
    if (!storedWeb) {
      try {
        const pick = await pickAndUploadFile();
        persistUpload(pick);
        await openPdf(pick.web_path);
      } catch (e) {
        console.error("[viewer] picker/upload failed:", e);
        alert("Failed to load PDF.");
        return;
      }
      return;
    }

    try {
      await openPdf(storedWeb);
      if (!storedDisk) {
        const reup = await softReuploadForDisk(storedWeb);
        if (reup) persistUpload(reup);
      }
    } catch (e1) {
      console.warn("[viewer] direct open failed:", e1);
      try {
        const uploaded = await softReuploadForDisk(storedWeb);
        if (uploaded) {
          persistUpload(uploaded);
          await openPdf(uploaded.web_path);
          return;
        }
      } catch (e2) {
        console.warn("[viewer] soft reupload failed:", e2);
      }
      try {
        const pick = await pickAndUploadFile();
        persistUpload(pick);
        await openPdf(pick.web_path);
      } catch (e3) {
        console.error("[viewer] picker/upload failed:", e3);
        alert("Failed to load PDF.");
        return;
      }
    }
  })();

  function persistUpload(obj) {
    if (!obj) return;
    if (obj.web_path) sessionStorage.setItem("uploadedWebPath", normalizeToWebUrl(obj.web_path));
    if (obj.disk_path) sessionStorage.setItem("uploadedDiskPath", obj.disk_path);
    if (obj.form_id) {
      sessionStorage.setItem("uploadedFormId", obj.form_id);
      currentFormId = obj.form_id;
    }
    if (obj.file_name) {
      sessionStorage.setItem("uploadedFileName", obj.file_name);
      if (formTitle) formTitle.textContent = obj.file_name.replace(/\.[^.]+$/, "");
    }
    // refresh locals
    storedDisk = sessionStorage.getItem("uploadedDiskPath") || storedDisk;
  }

  async function openPdf(src) {
    const url = normalizeToWebUrl(src);
    try {
      const doc = await pdfjsLib.getDocument(url).promise;
      pdfDoc = doc;
    } catch (_) {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`PDF fetch failed (${r.status})`);
      const ab = await r.arrayBuffer();
      const doc = await pdfjsLib.getDocument({ data: new Uint8Array(ab) }).promise;
      pdfDoc = doc;
    }
    renderPage(currentPage);
    buildThumbnails(pdfDoc);
  }

  function renderPage(pageNum) {
    pdfDoc.getPage(pageNum).then((page) => {
      const viewport = page.getViewport({ scale });
      if (!pageBaseSize[pageNum]) {
        const baseViewport = page.getViewport({ scale: 1 });
        pageBaseSize[pageNum] = { w: baseViewport.width, h: baseViewport.height };
      }
      pdfCanvas.width = viewport.width;
      pdfCanvas.height = viewport.height;
      overlayCanvas.width = viewport.width;
      overlayCanvas.height = viewport.height;

      page.render({ canvasContext: pdfCtx, viewport }).promise.then(() => {
        if (pageInfo) pageInfo.textContent = `Page ${pageNum} / ${pdfDoc.numPages}`;
        if (zoomInfo) zoomInfo.textContent = `${Math.round(scale * 100)}%`;
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        if (toggleBoxes && toggleBoxes.checked && currentFormId) drawOverlay(currentFormId, pageNum);
      });
    });
  }

  function buildThumbnails(pdf) {
    if (!thumbSidebar) return;
    thumbSidebar.innerHTML = "";
    for (let n = 1; n <= pdf.numPages; n++) {
      pdf.getPage(n).then((page) => {
        const viewport = page.getViewport({ scale: 0.3 });
        const c = document.createElement("canvas");
        c.width = viewport.width;
        c.height = viewport.height;
        page.render({ canvasContext: c.getContext("2d"), viewport }).promise.then(() => {
          const w = document.createElement("div");
          w.className = "thumbnail-wrapper";
          c.className = "thumbnail";
          c.title = `Page ${page.pageNumber}`;
          c.addEventListener("click", () => {
            currentPage = page.pageNumber;
            renderPage(currentPage);
          });
          const label = document.createElement("div");
          label.className = "thumbnail-label";
          label.textContent = `Page ${page.pageNumber}`;
          w.appendChild(c);
          w.appendChild(label);
          thumbSidebar.appendChild(w);
        });
      });
    }
  }

  // ========================
  // 🔸 Run Analysis
  // ========================
  if (analyzeBtn) analyzeBtn.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    Swal.fire({
      title: "Analyzing Form...",
      text: "Preparing summaries and overlays",
      allowOutsideClick: false,
      didOpen: () => Swal.showLoading(),
    });

    try {
      // Ensure we have an upload with DISK PATH (for /api/prelabel)
      let uploadInfo = await ensureUploadedToServer(storedWeb);
      if (uploadInfo) persistUpload(uploadInfo);

      const web_path  = sessionStorage.getItem("uploadedWebPath");
      const disk_path = sessionStorage.getItem("uploadedDiskPath");
      let   hashId    = sessionStorage.getItem("uploadedFormId") || currentFormId;

      if (!disk_path) throw new Error("Upload failed to provide a disk path.");

      // 1) PRELABEL (server returns canonical hash; we enforce it)
      const prelabelResp = await ensurePrelabelAndOverlays({ disk_path }, hashId);
      if (prelabelResp && prelabelResp.canonical_form_id) {
        hashId = prelabelResp.canonical_form_id;
        currentFormId = hashId;
        sessionStorage.setItem("uploadedFormId", hashId);
      }

      // 2) EXPLAINER: resolve by exact hash in registry; fallback to ensure
      const guess = guessFromPath(baseFromPath(web_path) || "form.pdf");
      const reg = await GET_json("/panel");
      let explainer = await resolveExplainerByHash(reg, hashId);
      if (!explainer) {
        // not found → ensure via facade (LLM fallback); include pdf_disk_path for better context
        await POST_json("/api/explainer.ensure", {
          canonical_form_id: hashId,
          bucket: guess.bucket,
          human_title: guess.title,
          pdf_disk_path: disk_path,
          aliases: [guess.formId, baseFromPath(web_path)]
        });
        // reload registry, then fetch
        const reg2 = await GET_json("/panel");
        explainer = await resolveExplainerByHash(reg2, hashId);
      }
      if (!explainer) throw new Error("Failed to load explainer.");

      // 3) UI: title + summaries
      if (formTitle) formTitle.textContent = explainer.title || (storedName || "Form");
      renderSummaries(explainer);

      // 4) Overlays if toggled
      if (toggleBoxes && toggleBoxes.checked && currentFormId) {
        await drawOverlay(currentFormId, currentPage);
      }

      Swal.fire({
        icon: "success",
        title: "Analysis Complete",
        text: "Summaries and overlays are ready",
        timer: 1400,
        showConfirmButton: false,
      });
    } catch (e) {
      console.error("runAnalysis error:", e);
      Swal.fire({
        icon: "error",
        title: "Analysis Failed",
        text: e?.message || "Could not analyze this form.",
      });
    }
  }

  // ---- Render summaries ----
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
      item.appendChild(header);
      item.appendChild(content);
      if (summaryList) summaryList.appendChild(item);
    });
    if (metricsRow) metricsRow.textContent = "";
  }

  // ---- Click-to-jump from summaries to overlay ----
  summaryList?.addEventListener("click", async (ev) => {
    const lbl = ev.target.closest(".summary-label");
    if (!lbl || !currentFormId || !cachedAnnotations) return;
    const targetLabel = lbl.textContent.trim().toLowerCase();

    const match = (cachedAnnotations.groups || []).find(
      (g) => g.label && g.label.toLowerCase() === targetLabel
    );
    if (match) {
      currentPage = (match.page || 0) + 1;
      renderPage(currentPage);
      await new Promise((r) => setTimeout(r, 300));
      drawOverlay(currentFormId, currentPage);
    }
  });

  // ---- Overlay drawing (groups → tokens fallback) ----
  async function drawOverlay(formId, pageNumber) {
    try {
      if (!cachedAnnotations || cachedAnnotations.__formId !== formId) {
        const res = await fetch(`/explanations/_annotations/${formId}.json?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) {
          overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          return;
        }
        cachedAnnotations = await res.json();
        cachedAnnotations.__formId = formId;
      }
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      const pageIdx = pageNumber - 1;
      let rects = [];

      if (Array.isArray(cachedAnnotations.groups) && cachedAnnotations.groups.length) {
        rects = cachedAnnotations.groups.filter((g) => (g.page || 0) === pageIdx);
      } else if (Array.isArray(cachedAnnotations.tokens)) {
        rects = cachedAnnotations.tokens.filter((t) => (t.page || 0) === pageIdx);
      }

      const base = pageBaseSize[pageNumber] || { w: overlayCanvas.width, h: overlayCanvas.height };
      const sx = overlayCanvas.width / base.w;
      const sy = overlayCanvas.height / base.h;

      overlayCtx.lineWidth = 1.5;
      overlayCtx.strokeStyle = "rgba(20,20,20,0.85)";
      rects.forEach((r) => {
        const [x0, y0, x1, y1] = r.bbox;
        overlayCtx.strokeRect(
          x0 * sx,
          y0 * sy,
          Math.max(1, (x1 - x0) * sx),
          Math.max(1, (y1 - y0) * sy)
        );
      });
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
  async function softReuploadForDisk(webUrl) {
    try {
      const url = normalizeToWebUrl(webUrl);
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return null;
      const blob = await r.blob();
      const base = baseFromPath(url) || "form.pdf";
      const fd = new FormData();
      fd.append("file", new File([blob], base, { type: "application/pdf" }));
      const up = await fetch("/api/upload", { method: "POST", body: fd });
      if (!up.ok) return null;
      const out = await up.json();
      return {
        web_path: normalizeToWebUrl(out.web_path),
        disk_path: out.disk_path,
        form_id: out.canonical_form_id || out.form_id, // prefer hash
        file_name: base,
      };
    } catch {
      return null;
    }
  }

  async function ensureUploadedToServer(webUrl) {
    const sessWeb = sessionStorage.getItem("uploadedWebPath");
    const sessDisk = sessionStorage.getItem("uploadedDiskPath");
    const sessForm = sessionStorage.getItem("uploadedFormId");
    if (sessWeb && sessDisk && sessForm) {
      return {
        web_path: normalizeToWebUrl(sessWeb),
        disk_path: sessDisk,
        form_id: sessForm,
        file_name: sessionStorage.getItem("uploadedFileName"),
      };
    }
    if (typeof webUrl === "string" && webUrl) {
      const reup = await softReuploadForDisk(webUrl);
      if (reup) return reup;
    }
    return await pickAndUploadFile();
  }

  async function pickAndUploadFile() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/pdf";
    input.style.display = "none";
    document.body.appendChild(input);

    const file = await new Promise((resolve, reject) => {
      input.addEventListener(
        "change",
        () => {
          if (input.files && input.files[0]) resolve(input.files[0]);
          else reject(new Error("No file selected"));
        },
        { once: true }
      );
      input.click();
    }).finally(() => setTimeout(() => input.remove(), 0));

    const fd = new FormData();
    fd.append("file", file);
    const up = await fetch("/api/upload", { method: "POST", body: fd });
    if (!up.ok) throw new Error("upload failed");
    const out = await up.json();
    return {
      web_path: normalizeToWebUrl(out.web_path),
      disk_path: out.disk_path,
      form_id: out.canonical_form_id || out.form_id, // prefer hash
      file_name: file.name,
    };
  }

  async function ensurePrelabelAndOverlays(server, hashId) {
    try {
      const fd = new FormData();
      fd.append("pdf_disk_path", server.disk_path);
      fd.append("form_id", hashId);
      const r = await fetch("/api/prelabel", { method: "POST", body: fd });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(t || "Prelabeling failed.");
      }
      const resp = await r.json();
      return resp;
    } catch (e) {
      console.error("ensurePrelabelAndOverlays error:", e);
      throw e;
    }
  }

  // ---- Explainer resolving by exact hash (preferred) ----
  async function resolveExplainerByHash(reg, hash) {
    const forms = Array.isArray(reg.forms) ? reg.forms : [];
    const hit = forms.find((f) => String(f.form_id || "") === String(hash || ""));
    if (!hit || !hit.path) return null;
    const url = "/" + String(hit.path).replace(/^\//, "");
    try {
      return await GET_json(url + (url.includes("?") ? "&" : "?") + "ts=" + Date.now());
    } catch {
      return null;
    }
  }

  // ---- Utils ----
  async function GET_json(url) {
    const withTs = url.includes("?") ? url + "&ts=" + Date.now() : url + "?ts=" + Date.now();
    const r = await fetch(withTs, { cache: "no-store" });
    if (!r.ok) throw new Error(url);
    return r.json();
  }

  async function POST_json(url, obj) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(obj || {}),
    });
    if (!r.ok) {
      const t = await r.text().catch(() => "");
      throw new Error(t || `POST ${url} failed`);
    }
    return r.json();
  }

  function baseFromPath(p) {
    try { return String(p).split("/").pop().split("\\").pop().split("?")[0]; }
    catch { return p; }
  }

  function guessFromPath(stemLike) {
    const s = String(stemLike).toLowerCase();
    let bucket = "government";
    if (/(bdo|metrobank|slamci|fami|bank|account)/.test(s)) bucket = "banking";
    else if (/(bir|tax|2552|1604|1901|1902)/.test(s)) bucket = "tax";
    else if (/(manulife|sunlife|axa|fwd|claim|philhealth|allianz)/.test(s)) bucket = "healthcare";
    return { bucket, formId: s.replace(/\s+/g, "_").replace(/\.pdf$/i, ""), title: stemLike.replace(/\.pdf$/i, "") };
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] || c));
  }

  function normalizeToWebUrl(p) {
    if (!p) return p;
    const s = String(p).replace(/\\/g, "/");
    if (s.startsWith("/uploads/")) return s;
    const m = s.match(/\/uploads\/([^\/?#]+)$/i);
    if (m) return "/uploads/" + m[1];
    const m2 = s.match(/\/?uploads\/([^\/?#]+)$/i);
    if (m2) return "/uploads/" + m2[1];
    return s;
  }
}
