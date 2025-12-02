// /static/js/researcher-dashboard.js  (Option A: single container, with CRUD + summary)
const API_BASE = window.INTELLIFORM_API_BASE || "";
const UI_BASE  = window.INTELLIFORM_UI_BASE  || API_BASE;
const apiUrl = (p) => `${API_BASE}${p}`;
const uiUrl  = (p) => `${UI_BASE}${p}`;

document.addEventListener("DOMContentLoaded", () => {
  // Tab buttons (data-tab = live|trained|funsd|user)
  document.querySelectorAll(".rd-tab[data-tab]").forEach(btn => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });

  // Download current view as JSON
  document.getElementById("downloadJsonBtn")?.addEventListener("click", onDownloadClick);

  // Edit toggle
  const editBtn = document.getElementById("editToggleBtn");
  const cancelBtn = document.getElementById("btnCancelEdit");
  editBtn?.addEventListener("click", toggleEditMode);
  cancelBtn?.addEventListener("click", disableEditMode);

  // CRUD actions
  document.getElementById("btnDeleteSelected")?.addEventListener("click", onDeleteSelected);
  document.getElementById("btnUndoLast")?.addEventListener("click", onUndoLast);

  // Default: Live Tool Metrics
  setActiveTab("live");
});

/* ============== Global state ============== */
const lastData = { live: [], trained: [], funsd: [], user: [] };  // newest-first per tab
let editMode = false;
let lastUndoToken = { tool: null, user: null }; // remember per kind
const pageState = { live: 1, trained: 1, funsd: 1, user: 1 };
const PAGE_SIZE = 10;

/* Selection model: row_ids of the rendered (newest-first) rows */
let selectedIds = new Set();
let lastUndoAvailable = { tool: false, user: false }; // becomes true after a delete

/* ============== Tab controls ============== */
function getTargetEl() {
  return document.getElementById("recordData");
}
function getActiveTab() {
  const active = document.querySelector(".rd-tab.active[data-tab]");
  return active ? active.dataset.tab : "live";
}
function setActiveTab(tab) {
  // Toggle button states
  document.querySelectorAll(".rd-tab[data-tab]").forEach(b => {
    const isActive = b.dataset.tab === tab;
    b.classList.toggle("active", isActive);
    b.setAttribute("aria-selected", String(isActive));
  });
  // Reset selection when switching tabs
  clearSelection();
  // Reset pagination for this tab
  pageState[tab] = 1;
  // Re-render current tab
  renderTab(tab);
}

/* ============== Render ============== */
async function renderTab(tab, forceRefetch = false) {
  const box = getTargetEl();
  box.innerHTML = loaderHTML();
  box.setAttribute("aria-busy", "true");

  try {
    if (tab === "user") {
      const rows = (!forceRefetch && lastData.user.length) ? lastData.user : await fetchLogs("user", 1000);
      lastData.user = rows;
      const summary = renderUserSummary(rows);
      const paged = slicePage(rows, tab);
      box.innerHTML = summary + renderUserTable(paged, { withCheckboxes: editMode }) + renderPagination(tab, rows.length);
      bindCheckboxHandlers();
      bindPaginationHandlers(tab, rows.length);
      refreshCrudToolbar();
      return;
    }

    // Tool logs
    const all = (!forceRefetch && lastData.live.length && tab === "live")
      ? lastData.live
      : await fetchLogs("tool", 2000);
    let filtered = all;

    if (tab === "live") {
      filtered = all.filter(r => (r.source || "analysis").toLowerCase() === "analysis");
      lastData.live = filtered;
    } else if (tab === "trained") {
      const rows = (!forceRefetch && lastData.trained.length) ? lastData.trained : await fetchStaticRows("ph_trained");
      lastData.trained = rows;
      const summary = renderToolSummary(rows, { label: "PH Forms", includeTextMetrics: true });
      const paged = slicePage(rows, tab);
      box.innerHTML = summary + renderToolTable(paged, { withCheckboxes: editMode, includeTextMetrics: true }) + renderPagination(tab, rows.length);
      bindCheckboxHandlers();
      bindPaginationHandlers(tab, rows.length);
      refreshCrudToolbar();
      return;
    } else if (tab === "funsd") {
      const funsdRows = (!forceRefetch && lastData.funsd.length) ? lastData.funsd : await fetchFunsdRowsStatic();
      lastData.funsd = funsdRows;
      const summary = renderToolSummary(funsdRows, { label: "FUNSD Benchmarks" });
      const paged = slicePage(funsdRows, tab);
      box.innerHTML = summary + renderToolTable(paged, { withCheckboxes: editMode }) + renderPagination(tab, funsdRows.length);
      bindCheckboxHandlers();
      bindPaginationHandlers(tab, funsdRows.length);
      refreshCrudToolbar();
      return;
    }

    filtered.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
    lastData[tab] = filtered;
    const summary = renderToolSummary(filtered, { label: "Live (beta)" });
    const paged = slicePage(filtered, tab);
    box.innerHTML = summary + renderToolTable(paged, { withCheckboxes: editMode }) + renderPagination(tab, filtered.length);
    bindCheckboxHandlers();
    bindPaginationHandlers(tab, filtered.length);
    refreshCrudToolbar();
  } catch (e) {
    displayError(box, e?.message || "Failed to load logs.");
  } finally {
    box.setAttribute("aria-busy", "false");
  }
}

function slicePage(rows, tab) {
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const page = Math.max(1, Math.min(pageState[tab] || 1, totalPages));
  pageState[tab] = page;
  const start = (page - 1) * PAGE_SIZE;
  return rows.slice(start, start + PAGE_SIZE);
}

function renderPagination(tab, total) {
  if (total <= PAGE_SIZE) return "";
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const page = Math.max(1, Math.min(pageState[tab] || 1, totalPages));
  const prevDisabled = page <= 1 ? "disabled" : "";
  const nextDisabled = page >= totalPages ? "disabled" : "";
  return `
    <div class="rd-pagination">
      <button class="rd-btn ghost" data-page="${page-1}" ${prevDisabled}>Prev</button>
      <div class="rd-pagination__info">Page ${page} / ${totalPages}</div>
      <button class="rd-btn ghost" data-page="${page+1}" ${nextDisabled}>Next</button>
    </div>
  `;
}

function bindPaginationHandlers(tab, total) {
  const pag = getTargetEl().querySelector(".rd-pagination");
  if (!pag) return;
  pag.querySelectorAll("button[data-page]").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = parseInt(btn.getAttribute("data-page"), 10);
      if (!Number.isFinite(target)) return;
      const maxPage = Math.max(1, Math.ceil(total / PAGE_SIZE));
      pageState[tab] = Math.max(1, Math.min(target, maxPage));
      renderTab(tab, false);
    });
  });
}

/* ============== Fetch ============== */
async function fetchLogs(kind = "tool", limit = 200) {
  const url = apiUrl(`/api/research/logs?kind=${encodeURIComponent(kind)}&limit=${limit}&ts=${Date.now()}`);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) {
    // Treat missing endpoint/file as "no data yet" to avoid hard errors in UI
    return [];
  }
  if (!res.ok) throw new Error(`Failed to load ${kind} logs (${res.status})`);
  const data = await res.json();
  const rows = Array.isArray(data.rows) ? data.rows : [];
  // newest first by ts or ts_utc
  rows.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
  return rows;
}

/* ============== Download ============== */
async function onDownloadClick() {
  const tab = getActiveTab();
  try {
    if (!lastData[tab] || lastData[tab].length === 0) {
      await renderTab(tab, /*force*/true);
    }
    const rows = lastData[tab] || [];
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download =
      tab === "user"    ? "user-metrics.json" :
      tab === "trained" ? "trained-tool-metrics.json" :
      tab === "funsd"   ? "funsd-tool-metrics.json" :
                          "live-tool-metrics.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    displayError(getTargetEl(), e?.message || "Download failed");
  }
}

/* ============== Edit mode ============== */
function toggleEditMode() {
  if (editMode) { disableEditMode(); return; }
  enableEditMode();
}
function enableEditMode() {
  editMode = true;
  clearSelection();
  // Show toolbar
  document.getElementById("editToolbar").hidden = false;
  // Update button UI
  const btn = document.getElementById("editToggleBtn");
  if (btn) {
    btn.textContent = "Cancel Edit";
    btn.setAttribute("aria-pressed", "true");
    btn.classList.add("editing");
  }
  // Re-render current tab with checkboxes
  renderTab(getActiveTab(), true);
}
function disableEditMode() {
  editMode = false;
  clearSelection();
  document.getElementById("editToolbar").hidden = true;
  const btn = document.getElementById("editToggleBtn");
  if (btn) {
    btn.textContent = "Edit";
    btn.setAttribute("aria-pressed", "false");
    btn.classList.remove("editing");
  }
  renderTab(getActiveTab(), true);
}

function clearSelection() {
  selectedIds.clear();
  refreshCrudToolbar();
}

/* ============== CRUD toolbar behavior ============== */
function refreshCrudToolbar() {
  const count = selectedIds.size;
  const lbl = document.getElementById("selectedCount");
  const delBtn = document.getElementById("btnDeleteSelected");
  const undoBtn = document.getElementById("btnUndoLast");
  const tab = getActiveTab();
  const kind = (tab === "user") ? "user" : "tool";

  if (lbl) lbl.textContent = `${count} selected`;
  if (delBtn) delBtn.disabled = (count === 0);
  if (undoBtn) undoBtn.disabled = !lastUndoAvailable[kind];
}

/* inject checkbox event handlers after table render */
function bindCheckboxHandlers() {
  if (!editMode) return;

  const table = getTargetEl().querySelector("table");
  if (!table) return;

  const master = table.querySelector('thead input[type="checkbox"][data-master="1"]');
  const rowChecksSelector = 'tbody input[type="checkbox"][data-id]';

  if (master) {
    master.addEventListener("change", () => {
      selectedIds.clear();
      const checks = table.querySelectorAll(rowChecksSelector);
      checks.forEach(cb => {
        cb.checked = master.checked;
        const id = cb.getAttribute("data-id");
        if (cb.checked && id) selectedIds.add(id);
      });
      refreshCrudToolbar();
    });
  }

  table.querySelectorAll(rowChecksSelector).forEach(cb => {
    cb.addEventListener("change", () => {
      const id = cb.getAttribute("data-id");
      if (!id) return;

      if (cb.checked) selectedIds.add(id);
      else selectedIds.delete(id);

      if (master) {
        const total = table.querySelectorAll(rowChecksSelector).length;
        const checked = table.querySelectorAll(`${rowChecksSelector}:checked`).length;
        master.checked = (total > 0 && total === checked);
        master.indeterminate = (checked > 0 && checked < total);
      }
      refreshCrudToolbar();
    });
  });
}

/* ============== CRUD actions ============== */
async function onDeleteSelected() {
  const tab = getActiveTab();
  const kind = (tab === "user") ? "user" : "tool";
  const ids = [...selectedIds];

  if (ids.length === 0) return;

  const confirm = await swalConfirm(
    `Delete ${ids.length} selected row${ids.length>1?'s':''}?`,
    "This removes them from the JSONL log. You can click Undo right after to restore from the last backup."
  );
  if (!confirm) return;

  try {
    const resp = await fetch("/api/research/logs.delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, row_ids: ids })     // <-- send row_ids
    });
    if (!resp.ok) throw new Error(await resp.text());
    const out = await resp.json();

    lastUndoAvailable[kind] = true;                    // enable Undo
    await swalOk("Deleted", `${ids.length} row(s) removed.`);
    clearSelection();
    await renderTab(tab, true);
  } catch (e) {
    await swalError("Delete failed", e?.message || "Request error.");
  }
}

async function onUndoLast() {
  const tab = getActiveTab();
  const kind = (tab === "user") ? "user" : "tool";

  try {
    const resp = await fetch("/api/research/logs.undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind })                   // <-- no token needed
    });
    if (!resp.ok) throw new Error(await resp.text());
    const out = await resp.json();

    await swalOk("Undo complete", "Restored from the latest backup.");
    lastUndoAvailable[kind] = false;                   // single-shot UX (optional)
    clearSelection();
    await renderTab(tab, true);
  } catch (e) {
    await swalError("Undo failed", e?.message || "Request error.");
  }
}

/* ============== Renderers ============== */
function renderToolTable(rows, opts = {}) {
  const withCk = !!opts.withCheckboxes;
  const includeTextMetrics = !!opts.includeTextMetrics;

  const safe = rows.map((r) => {
    const m = r.metrics || {};
    let { tp, fp, fn, precision, recall, f1 } = m;
    let rouge = toNum(m.rouge_l ?? m.rouge ?? m.rougeL);
    let meteor = toNum(m.meteor);
    tp = toInt(tp); fp = toInt(fp); fn = toInt(fn);
    precision = toNum(precision); recall = toNum(recall); f1 = toNum(f1);
    if (!isFinite(f1) && isFinite(precision) && isFinite(recall) && (precision + recall) > 0) {
      f1 = (2 * precision * recall) / (precision + recall);
    }
    return {
      id: r.row_id || "",                                // <-- keep row_id
      when: fmtWhen(r.ts_utc || r.ts),
      title: r.form_title || r.title || "(untitled)",
      tp, fp, fn, precision, recall, f1,
      rouge, meteor
    };
  });

  if (!safe.length) return emptyState("No tool metrics logged yet.");

  const headLeft = withCk ? `<th class="center" rowspan="${includeTextMetrics ? 2 : 1}" style="width:28px">
      <input type="checkbox" data-master="1" aria-label="Select all"/>
    </th>` : "";

  const body = safe.map((r, i) => `
    <tr>
      ${withCk ? `<td class="num center">
        <input type="checkbox" data-id="${esc(r.id)}" aria-label="Select row ${i+1}"/>
      </td>` : ""}
      <td class="nowrap center">${esc(r.when)}</td>
      <td class="center">${esc(r.title)}</td>
      <td class="num center">${r.tp}</td>
      <td class="num center">${r.fp}</td>
      <td class="num center">${r.fn}</td>
      <td class="num center">${pct(r.precision)}</td>
      <td class="num center">${pct(r.recall)}</td>
      <td class="num center">${pct(r.f1)}</td>
      ${includeTextMetrics ? `<td class="num center">${pct(r.rouge)}</td>
      <td class="num center">${pct(r.meteor)}</td>` : ""}
    </tr>
  `).join("");

  if (includeTextMetrics) {
    return tableWrap(`
      <thead>
        <tr>
          ${headLeft}
          <th class="center" rowspan="2">When</th>
          <th class="center" rowspan="2">Title</th>
          <th class="center" colspan="6">Extraction</th>
          <th class="center" colspan="2">Text</th>
        </tr>
        <tr>
          <th class="center">TP</th>
          <th class="center">FP</th>
          <th class="center">FN</th>
          <th class="center">Precision</th>
          <th class="center">Recall</th>
          <th class="center">F1</th>
          <th class="center">ROUGE-L</th>
          <th class="center">METEOR</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    `);
  }

  return tableWrap(`
      <thead>
        <tr>
          ${headLeft}
          <th class="center">When</th>
          <th class="center">Title</th>
          <th class="center">TP</th>
          <th class="center">FP</th>
          <th class="center">FN</th>
          <th class="center">Precision</th>
          <th class="center">Recall</th>
          <th class="center">F1</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    `);
}

function summarizeTool(rows) {
  if (!rows || !rows.length) return null;
  const metrics = rows.map(r => r.metrics || {});
  const nums = (key) => metrics.map(m => toNum(m[key])).filter((x) => Number.isFinite(x));
  const avg = (arr) => arr.length ? (arr.reduce((a,b)=>a+b,0) / arr.length) : NaN;
  const precision = avg(nums("precision"));
  const recall    = avg(nums("recall"));
  const f1        = avg(nums("f1"));
  const rouge     = avg(nums("rouge_l") || nums("rouge") || nums("rougeL"));
  const meteor    = avg(nums("meteor"));
  const latestTs  = Math.max(...rows.map(r => toTs(r.ts_utc || r.ts || 0)));

  // bucket rollup (avg f1 per bucket)
  const bucketMap = {};
  rows.forEach(r => {
    const b = (r.bucket || "Unlabeled").toString();
    const f = toNum((r.metrics || {}).f1);
    if (!bucketMap[b]) bucketMap[b] = { sum:0, n:0 };
    if (Number.isFinite(f)) { bucketMap[b].sum += f; bucketMap[b].n += 1; }
  });
  const bucketRows = Object.entries(bucketMap)
    .map(([name, {sum,n}]) => ({ name, f1: n ? sum/n : NaN }))
    .filter(x => Number.isFinite(x.f1))
    .sort((a,b) => b.f1 - a.f1)
    .slice(0,3);

  return { precision, recall, f1, rouge, meteor, latestTs, total: rows.length, bucketRows };
}

function renderToolSummary(rows, opts = {}) {
  const stats = summarizeTool(rows);
  const label = opts.label || "Live Metrics";
  if (!stats) return emptyState("No tool metrics logged yet.");

  const { precision, recall, f1, rouge, meteor, latestTs, total, bucketRows } = stats;
  const bucketHtml = bucketRows && bucketRows.length ? bucketRows.map(b => `
    <div class="rd-chip">
      <span>${esc(b.name)}</span>
      <strong>${pct(b.f1)}</strong>
    </div>`).join("") : `<div class="rd-chip muted">No buckets</div>`;

  return `
    <div class="rd-summary">
      <div class="rd-summary__head">
        <div class="rd-summary__title">${esc(label)}</div>
        <div class="rd-summary__meta">Latest: ${fmtWhen(latestTs) || "—"} · ${total} runs</div>
      </div>
      <div class="rd-summary__grid">
        <div class="rd-card metric">
          <div class="kpi">${pct(precision)}</div>
          <div class="kpi-label">Avg Precision</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${pct(recall)}</div>
          <div class="kpi-label">Avg Recall</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${pct(f1)}</div>
          <div class="kpi-label">Avg F1</div>
        </div>
        ${opts.includeTextMetrics ? `
        <div class="rd-card metric">
          <div class="kpi">${pct(rouge)}</div>
          <div class="kpi-label">Avg ROUGE-L</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${pct(meteor)}</div>
          <div class="kpi-label">Avg METEOR</div>
        </div>` : ""}
        <div class="rd-card metric">
          <div class="kpi">${total}</div>
          <div class="kpi-label">Runs</div>
        </div>
      </div>
      <div class="rd-summary__buckets">
        <div class="bucket-label">Top buckets</div>
        <div class="bucket-chips">${bucketHtml}</div>
      </div>
    </div>
  `;
}

function renderUserTable(rows, opts = {}) {
  const withCk = !!opts.withCheckboxes;

  const safe = rows.map((r) => {
    const dur = toInt(r.duration_ms);
    return {
      id: r.row_id || "",                                  // <-- keep row_id
      when: fmtWhen(r.ts_utc || r.ts),
      user: r.user_id || "ANON",
      method: (r.method || "intelliform").toLowerCase(),
      started_at: fmtWhen(r.started_at),
      finished_at: fmtWhen(r.finished_at),
      duration_ms: Number.isFinite(dur) ? dur : null
    };
  });

  if (!safe.length) return emptyState("No user sessions logged yet.");

  const headLeft = withCk ? `<th style="width:28px;text-align:center">
      <input type="checkbox" data-master="1" aria-label="Select all"/>
    </th>` : "";

  const body = safe.map((r, i) => `
    <tr>
      ${withCk ? `<td class="num" style="text-align:center">
        <input type="checkbox" data-id="${esc(r.id)}" aria-label="Select row ${i+1}"/>
      </td>` : ""}
      <td class="nowrap center">${esc(r.when)}</td>
      <td class="nowrap center">${esc(r.user)}</td>
      <td class="nowrap center">${esc(r.method)}</td>
      <td class="nowrap center">${esc(r.started_at)}</td>
      <td class="nowrap center">${esc(r.finished_at)}</td>
      <td class="num">${r.duration_ms != null ? ms(r.duration_ms) : "—"}</td>
    </tr>
  `).join("");

  return tableWrap(`
    <thead>
      <tr>
        ${headLeft}
        <th>Logged</th>
        <th>User</th>
        <th>Method</th>
        <th>Started</th>
        <th>Finished</th>
        <th>Duration</th>
      </tr>
    </thead>
    <tbody>${body}</tbody>
  `);
}

function summarizeUser(rows) {
  if (!rows || !rows.length) return null;
  const durations = rows.map(r => toInt(r.duration_ms)).filter((d) => Number.isFinite(d));
  const avgDur = durations.length ? (durations.reduce((a,b)=>a+b,0) / durations.length) : NaN;
  const totalDur = durations.reduce((a,b)=>a+b,0);
  const latestTs = Math.max(...rows.map(r => toTs(r.ts_utc || r.ts || 0)));
  const uniqueUsers = new Set(rows.map(r => (r.user_id || "").toString().trim() || "ANON")).size;
  return { avgDur, totalDur, latestTs, total: rows.length, uniqueUsers };
}

function renderUserSummary(rows) {
  const stats = summarizeUser(rows);
  if (!stats) return emptyState("No user sessions logged yet.");
  const { avgDur, totalDur, latestTs, total, uniqueUsers } = stats;
  return `
    <div class="rd-summary">
      <div class="rd-summary__head">
        <div class="rd-summary__title">User Sessions</div>
        <div class="rd-summary__meta">Latest: ${fmtWhen(latestTs) || "—"} · ${total} sessions</div>
      </div>
      <div class="rd-summary__grid">
        <div class="rd-card metric">
          <div class="kpi">${uniqueUsers}</div>
          <div class="kpi-label">Unique Users</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${avgDur ? ms(Math.round(avgDur)) : "—"}</div>
          <div class="kpi-label">Avg Duration</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${ms(totalDur)}</div>
          <div class="kpi-label">Total Time</div>
        </div>
        <div class="rd-card metric">
          <div class="kpi">${total}</div>
          <div class="kpi-label">Sessions</div>
        </div>
      </div>
    </div>
  `;
}

/* ============== SweetAlert helpers (fallback to native) ============== */
async function swalConfirm(title, text) {
  if (window.Swal && Swal.fire) {
    const r = await Swal.fire({
      icon: "warning",
      title, text,
      showCancelButton: true,
      confirmButtonText: "Delete",
      cancelButtonText: "Cancel",
    });
    return r.isConfirmed;
  }
  return window.confirm(`${title}\n\n${text}`);
}
async function swalOk(title, text) {
  if (window.Swal && Swal.fire) {
    await Swal.fire({ icon: "success", title, text, timer: 1200, showConfirmButton: false });
  } else {
    alert(`${title}\n\n${text}`);
  }
}
async function swalError(title, text) {
  if (window.Swal && Swal.fire) {
    await Swal.fire({ icon: "error", title, text });
  } else {
    alert(`${title}\n\n${text}`);
  }
}

/* ============== DOM helpers & formatters ============== */
function tableWrap(inner){ return `<table>${inner}</table>`; }
function loaderHTML(){
  return `<div class="error-container" style="opacity:.6">
    <p class="error-message">Loading…</p>
  </div>`;
}
function emptyState(msg){
  return `<div class="error-container" style="opacity:.8">
    <p class="error-message">${esc(msg)}</p>
  </div>`;
}
function displayError(container, message) {
  container.innerHTML = `<div class="error-container">
    <p class="error-message">Error loading metrics: ${esc(message)}</p>
  </div>`;
}

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c] || c)); }
function pct(x) { return Number.isFinite(x) ? (x * 100).toFixed(1) + "%" : "—"; }
function ms(n){ if(!Number.isFinite(n)||n<0)return"—"; if(n<1000)return`${n} ms`; const s=n/1000; if(s<60)return`${s.toFixed(1)} s`; const m=Math.floor(s/60),sec=Math.round(s%60); return`${m}m ${sec}s`; }
function toNum(x){ const n=Number(x); return Number.isFinite(n)?n:NaN; }
function toInt(x){ const n=parseInt(x,10); return Number.isFinite(n)?n:0; }
function toTs(v){ if(!v)return 0; if(typeof v==="number")return v; const t=Date.parse(v); return Number.isFinite(t)?t:0; }
function fmtWhen(v){ const t=toTs(v); if(!t)return"—"; try{ return new Date(t).toLocaleString(); }catch{ return "—"; } }

async function fetchAggregate(datasetKey) {
  // datasetKey: "ph_trained" or "funsd"
  const url = uiUrl(`/static/research_dashboard/${datasetKey}/${datasetKey}_aggregate.json?v=${Date.now()}`);
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Aggregate not found: ${datasetKey} (${res.status})`);
  return await res.json();
}

function aggregateToRows(agg, label = "Summary") {
  // Produce ONE summary row that your existing table can render
  const m = agg?.macro || {};
  return [{
    row_id: `${label}-${agg?.generated_at || ""}`,
    ts_utc: Date.now(),
    form_title: `${label} — ${agg?.count ?? 0} forms`,
    metrics: {
      tp: agg?.spans?.TP ?? undefined,   // only if you later add spans here
      fp: agg?.spans?.FP ?? undefined,
      fn: agg?.spans?.FN ?? undefined,
      precision: m.precision ?? NaN,
      recall: m.recall ?? NaN,
      f1: m.f1 ?? NaN
    }
  }];
}

async function fetchStaticRows(datasetKey) {
  // datasetKey: "ph_trained"
  const url = uiUrl(`/static/research_dashboard/${datasetKey}/${datasetKey}_rows.json?v=${Date.now()}`);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`Rows not found: ${datasetKey} (${res.status})`);
  const rows = await res.json();
  // Ensure newest-first
  rows.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
  return rows;
}

async function fetchFunsdRowsStatic() {
  const url = uiUrl(`/static/research_dashboard/funsd/funsd_rows.json?ts=${Date.now()}`);
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`Failed to load FUNSD rows (${res.status})`);
  const data = await res.json();
  const rows = Array.isArray(data.rows) ? data.rows : [];
  // newest first by ts_utc
  rows.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
  return rows;
}
