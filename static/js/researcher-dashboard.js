// /static/js/researcher-dashboard.js  (Option A: single container, with CRUD)

document.addEventListener("DOMContentLoaded", () => {
  // Tab buttons (data-tab = live|trained|funsd|user)
  document.querySelectorAll(".tabs button[data-tab]").forEach(btn => {
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

/* Selection model: row_ids of the rendered (newest-first) rows */
let selectedIds = new Set();
let lastUndoAvailable = { tool: false, user: false }; // becomes true after a delete

/* ============== Tab controls ============== */
function getTargetEl() {
  return document.getElementById("recordData");
}
function getActiveTab() {
  const active = document.querySelector(".tabs button.active[data-tab]");
  return active ? active.dataset.tab : "live";
}
function setActiveTab(tab) {
  // Toggle button states
  document.querySelectorAll(".tabs button[data-tab]").forEach(b => {
    const isActive = b.dataset.tab === tab;
    b.classList.toggle("active", isActive);
    b.setAttribute("aria-selected", String(isActive));
  });
  // Reset selection when switching tabs
  clearSelection();
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
      const rows = await fetchLogs("user", 1000);
      lastData.user = rows;
      box.innerHTML = renderUserTable(rows, { withCheckboxes: editMode });
      bindCheckboxHandlers();
      refreshCrudToolbar();
      return;
    }

    // Tool logs
    const all = await fetchLogs("tool", 2000);
    let filtered = all;

    if (tab === "live") {
      filtered = all.filter(r => (r.source || "analysis").toLowerCase() === "analysis");
    } else if (tab === "trained") {
      // Load the 20 PH-trained rows we just generated
      const rows = await fetchStaticRows("ph_trained");
      lastData.trained = rows;
      box.innerHTML = renderToolTable(rows, { withCheckboxes: editMode });
      bindCheckboxHandlers();
      refreshCrudToolbar();
      return;
    } else if (tab === "funsd") {
      // Use our static rows (built from outputs/funsd artifacts)
      const funsdRows = await fetchFunsdRowsStatic();
      lastData.funsd = funsdRows;
      box.innerHTML = renderToolTable(funsdRows, { withCheckboxes: editMode });
      bindCheckboxHandlers();
      refreshCrudToolbar();
      return;
    }

    filtered.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
    lastData[tab] = filtered;
    box.innerHTML = renderToolTable(filtered, { withCheckboxes: editMode });
    bindCheckboxHandlers();
    refreshCrudToolbar();
  } catch (e) {
    displayError(box, e?.message || "Failed to load logs.");
  } finally {
    box.setAttribute("aria-busy", "false");
  }
}

/* ============== Fetch ============== */
async function fetchLogs(kind = "tool", limit = 200) {
  const url = `/api/research/logs?kind=${encodeURIComponent(kind)}&limit=${limit}&ts=${Date.now()}`;
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

  const safe = rows.map((r) => {
    const m = r.metrics || {};
    let { tp, fp, fn, precision, recall, f1 } = m;
    tp = toInt(tp); fp = toInt(fp); fn = toInt(fn);
    precision = toNum(precision); recall = toNum(recall); f1 = toNum(f1);
    if (!isFinite(f1) && isFinite(precision) && isFinite(recall) && (precision + recall) > 0) {
      f1 = (2 * precision * recall) / (precision + recall);
    }
    return {
      id: r.row_id || "",                                // <-- keep row_id
      when: fmtWhen(r.ts_utc || r.ts),
      title: r.form_title || r.title || "(untitled)",
      tp, fp, fn, precision, recall, f1
    };
  });

  if (!safe.length) return emptyState("No tool metrics logged yet.");

  const headLeft = withCk ? `<th style="width:28px;text-align:center">
      <input type="checkbox" data-master="1" aria-label="Select all"/>
    </th>` : "";

  const body = safe.map((r, i) => `
    <tr>
      ${withCk ? `<td class="num" style="text-align:center">
        <input type="checkbox" data-id="${esc(r.id)}" aria-label="Select row ${i+1}"/>
      </td>` : ""}
      <td class="nowrap">${esc(r.when)}</td>
      <td>${esc(r.title)}</td>
      <td class="num">${r.tp}</td>
      <td class="num">${r.fp}</td>
      <td class="num">${r.fn}</td>
      <td class="num">${pct(r.precision)}</td>
      <td class="num">${pct(r.recall)}</td>
      <td class="num">${pct(r.f1)}</td>
    </tr>
  `).join("");

  return tableWrap(`
    <thead>
      <tr>
        ${headLeft}
        <th>When</th>
        <th>Title</th>
        <th>TP</th>
        <th>FP</th>
        <th>FN</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
    </thead>
    <tbody>${body}</tbody>
  `);
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
      <td class="nowrap">${esc(r.when)}</td>
      <td class="nowrap">${esc(r.user)}</td>
      <td class="nowrap">${esc(r.method)}</td>
      <td class="nowrap">${esc(r.started_at)}</td>
      <td class="nowrap">${esc(r.finished_at)}</td>
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
  const url = `/static/research_dashboard/${datasetKey}/${datasetKey}_aggregate.json?v=${Date.now()}`;
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
  const url = `/static/research_dashboard/${datasetKey}/${datasetKey}_rows.json?v=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`Rows not found: ${datasetKey} (${res.status})`);
  const rows = await res.json();
  // Ensure newest-first
  rows.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
  return rows;
}

async function fetchFunsdRowsStatic() {
  const url = `/static/research_dashboard/funsd/funsd_rows.json?ts=${Date.now()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 404) return [];
  if (!res.ok) throw new Error(`Failed to load FUNSD rows (${res.status})`);
  const data = await res.json();
  const rows = Array.isArray(data.rows) ? data.rows : [];
  // newest first by ts_utc
  rows.sort((a, b) => (toTs(b.ts_utc || b.ts) - toTs(a.ts_utc || a.ts)));
  return rows;
}
