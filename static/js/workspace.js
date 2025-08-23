document.addEventListener('DOMContentLoaded', () => {
  // ---- DOM refs
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebar       = document.getElementById('sidebar');
  const burgerToggle  = document.getElementById('togglePages');
  const thumbnailSidebar = document.getElementById('thumbnailSidebar');
  const analyzeBtn    = document.getElementById('analyzeTool');
  const summaryList   = document.getElementById('summaryList');
  const badgeNode     = document.getElementById('eceScoreBadge'); // reused badge
  const formNameNode  = document.getElementById('formNameDisplay');
  const metricsRow    = document.getElementById('metricsRow');

  const pageInfoEl = document.getElementById('pageInfo');
  const zoomInfoEl = document.getElementById('zoomInfo');

  const canvas       = document.getElementById('pdfCanvas');
  const ctx          = canvas.getContext('2d');
  const overlayCanvas= document.getElementById('overlayCanvas');
  const overlayCtx   = overlayCanvas.getContext('2d');

  // ---- UI toggles
  sidebarToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    sidebar.classList.toggle('open');
    sidebarToggle.style.display = sidebar.classList.contains('open') ? 'none' : 'flex';
  });
  document.addEventListener('click', (ev) => {
    if (!sidebar.contains(ev.target) && !sidebarToggle.contains(ev.target)) {
      sidebar.classList.remove('open');
      sidebarToggle.style.display = 'flex';
    }
  });
  burgerToggle.addEventListener('click', () => {
    thumbnailSidebar.classList.toggle('visible');
  });

  // ---- Query param: src=<server path to PDF>
  const params = new URLSearchParams(location.search);
  const pdfSrc = params.get('src');
  if (!pdfSrc) {
    Swal.fire({icon:'warning', title:'No PDF', text:'Missing ?src=<pdf path> from upload step.'});
    return;
  }
  formNameNode.textContent = decodeURIComponent(pdfSrc.split('/').pop());

  // ---- PDF.js render state
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  let pdfDoc = null;
  let scale = 1.5;
  let currentPage = 1;

  // analysis state
  let predictions = []; // [{label, summary, page, bbox:[x0,y0,x1,y1], group?}]
  let groups = {};      // groupName -> fields[]

  // ---- Load & render PDF
  pdfjsLib.getDocument(pdfSrc).promise
    .then((doc) => {
      pdfDoc = doc;
      renderPage(currentPage);
      buildThumbnails();
    })
    .catch((err) => {
      console.error(err);
      Swal.fire({icon:'error', title:'Load failed', text:'Unable to load the PDF.'});
    });

  function renderPage(pageNum) {
    pdfDoc.getPage(pageNum).then((page) => {
      const viewport = page.getViewport({ scale });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      overlayCanvas.width = viewport.width;
      overlayCanvas.height = viewport.height;

      const renderContext = { canvasContext: ctx, viewport };
      page.render(renderContext).promise.then(() => {
        pageInfoEl.textContent = `Page ${pageNum} / ${pdfDoc.numPages}`;
        zoomInfoEl.textContent = `${Math.round(scale * 100)}%`;
        drawOverlayForPage(pageNum);
      });
    });
  }

  function buildThumbnails() {
    thumbnailSidebar.innerHTML = '';
    for (let p = 1; p <= pdfDoc.numPages; p++) {
      pdfDoc.getPage(p).then((page) => {
        const vp = page.getViewport({ scale: 0.3 });
        const c = document.createElement('canvas');
        c.width = vp.width; c.height = vp.height;
        page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise.then(() => {
          const wrap = document.createElement('div'); wrap.className = 'thumbnail-wrapper';
          c.className = 'thumbnail'; c.title = `Page ${page.pageNumber}`;
          c.addEventListener('click', () => { currentPage = page.pageNumber; renderPage(currentPage); });
          const label = document.createElement('div'); label.className = 'thumbnail-label'; label.textContent = `Page ${page.pageNumber}`;
          wrap.appendChild(c); wrap.appendChild(label);
          thumbnailSidebar.appendChild(wrap);
        });
      });
    }
  }

  // ---- Overlay drawing (field boxes)
  function drawOverlayForPage(pageNumber) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!predictions || !predictions.length) return;

    const fieldsOnPage = predictions.filter(f => (f.page || 1) === pageNumber);
    fieldsOnPage.forEach(f => {
      const { x, y, w, h } = bboxToCanvasRect(f.bbox, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeStyle = 'rgba(62,216,255,0.9)';
      overlayCtx.fillStyle = 'rgba(62,216,255,0.15)';
      overlayCtx.strokeRect(x, y, w, h);
      overlayCtx.fillRect(x, y, w, h);
    });
  }

  /**
   * Convert a bbox to canvas-space rect.
   * - If values in [0..1], treat as normalized (top-left origin).
   * - Else proportionally map (assumes top-left). Adjust if your coords differ.
   */
  function bboxToCanvasRect(bbox, canvasW, canvasH) {
    if (!bbox || bbox.length !== 4) return { x: 0, y: 0, w: 0, h: 0 };
    let [x0, y0, x1, y1] = bbox.map(Number);
    const isNormalized = [x0,y0,x1,y1].every(v => v >= 0 && v <= 1);
    if (isNormalized) {
      const x = x0 * canvasW;
      const y = y0 * canvasH;
      const w = Math.max(0, (x1 - x0) * canvasW);
      const h = Math.max(0, (y1 - y0) * canvasH);
      return { x, y, w, h };
    } else {
      // naive proportional fallback; replace with page-size-aware mapping if needed
      const x = (x0 / 1000) * canvasW;
      const y = (y0 / 1000) * canvasH;
      const w = Math.max(0, ((x1 - x0) / 1000) * canvasW);
      const h = Math.max(0, ((y1 - y0) / 1000) * canvasH);
      return { x, y, w, h };
    }
  }

  // ---- Analyze: call backend, populate UI
  analyzeBtn.addEventListener('click', async () => {
    try {
      Swal.fire({title:'Analyzing...', text:'Running IntelliForm pipeline', allowOutsideClick:false, didOpen:() => Swal.showLoading()});
      // IMPORTANT: query param file_path to match backend
      const res = await fetch(`/api/analyze?file_path=${encodeURIComponent(pdfSrc)}`, { method: 'POST' });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `Analyze failed (${res.status})`);
      }
      const data = await res.json();
      applyResults(data);
      Swal.close();
      Swal.fire({icon:'success', title:'Analysis Complete', timer:1200, showConfirmButton:false});
    } catch (err) {
      console.error(err);
      Swal.fire({icon:'error', title:'Analyze Error', text: err.message || 'Failed to analyze.'});
    }
  });

  function applyResults(data) {
    formNameNode.textContent = data.title || formNameNode.textContent;

    // Badge: GT-free diagnostics
    const m = data.metrics || {};
    const count = (m.fields_count != null) ? `Fields: ${m.fields_count}` : null;
    const secs  = (m.processing_sec != null) ? `Time: ${Number(m.processing_sec).toFixed(2)}s` : null;
    const pages = (m.pages != null) ? `Pages: ${m.pages}` : null;
    const badgeText = [count, secs, pages].filter(Boolean).join(' • ') || '—';
    badgeNode.textContent = badgeText;

    // Metrics row (only shows what’s present)
    const pr  = (m.precision != null) ? `Precision: ${(Number(m.precision)*100).toFixed(1)}%` : null;
    const rc  = (m.recall    != null) ? `Recall: ${(Number(m.recall)*100).toFixed(1)}%`    : null;
    const f1  = (m.f1        != null) ? `F1: ${(Number(m.f1)*100).toFixed(1)}%`            : null;
    const rl  = (m.rougeL    != null) ? `ROUGE‑L: ${(Number(m.rougeL)*100).toFixed(1)}%`   : null;
    const mt  = (m.meteor    != null) ? `METEOR: ${(Number(m.meteor)*100).toFixed(1)}%`    : null;
    metricsRow.innerHTML = [pr, rc, f1, rl, mt].filter(Boolean).join(' · ');

    // Fields → group → accordion
    predictions = Array.isArray(data.fields) ? data.fields : [];
    groups = {};
    predictions.forEach(f => {
      const g = f.group || 'Fields';
      if (!groups[g]) groups[g] = [];
      groups[g].push(f);
    });
    renderAccordion(groups);

    drawOverlayForPage(currentPage);
  }

  function renderAccordion(groupMap) {
    summaryList.innerHTML = '';
    Object.entries(groupMap).forEach(([groupName, items]) => {
      const item = document.createElement('div');
      item.className = 'accordion-item';

      const header = document.createElement('div');
      header.className = 'accordion-header';
      header.textContent = groupName;

      const content = document.createElement('div');
      content.className = 'accordion-content active';

      items.forEach((f, idx) => {
        const row = document.createElement('p');
        const label = f.label ?? `Field ${idx+1}`;
        const summary = f.summary ?? '';
        row.innerHTML = `<span class="summary-label">${escapeHtml(label)}</span>: ${escapeHtml(summary)}`;
        row.style.cursor = 'pointer';
        row.addEventListener('click', () => {
          if (f.page) {
            currentPage = f.page;
            renderPage(currentPage);
            setTimeout(() => flashBox(f), 200);
          } else {
            flashBox(f);
          }
        });
        content.appendChild(row);
      });

      header.addEventListener('click', () => content.classList.toggle('active'));

      item.appendChild(header);
      item.appendChild(content);
      summaryList.appendChild(item);
    });
  }

  function flashBox(field) {
    const rect = bboxToCanvasRect(field.bbox, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.save();
    overlayCtx.lineWidth = 4;
    overlayCtx.strokeStyle = 'rgba(255, 230, 0, 0.95)';
    overlayCtx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    setTimeout(() => { drawOverlayForPage(currentPage); overlayCtx.restore(); }, 700);
  }

  function escapeHtml(str) {
    return (str ?? '').toString()
      .replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  }

  // ---- Print & Download
  document.getElementById('downloadPDF').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = pdfSrc;
    a.download = decodeURIComponent(pdfSrc.split('/').pop());
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  });

  document.getElementById('printPDF').addEventListener('click', () => {
    if (!canvas) { Swal.fire({icon:'info', title:'No page rendered'}); return; }
    const dataUrl = canvas.toDataURL();
    const w = window.open('', '_blank');
    w.document.write(`
      <html><head><title>Print PDF</title></head>
      <body style="margin:0;">
        <img src="${dataUrl}" style="width:100%;"/>
        <script>
          window.onload = function () {
            window.print();
            window.onafterprint = function () { window.close(); };
          };
        </script>
      </body></html>
    `);
  });
});
