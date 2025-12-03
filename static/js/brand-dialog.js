// static/js/brand-dialog.js
// Lightweight branded modal, prompt, loading, and progress UI (no external deps)
(function () {
  const VARIANTS = {
    success: { label: "Success", accent: "var(--bd-accent-2)", glow: "rgba(62,216,255,0.35)", icon: "✓" },
    warning: { label: "Heads up", accent: "var(--bd-accent)", glow: "rgba(255,213,74,0.35)", icon: "!" },
    danger:  { label: "Action needed", accent: "#ff7a91", glow: "rgba(255,122,145,0.42)", icon: "⨉" },
    info:    { label: "Notice", accent: "var(--bd-navy)", glow: "rgba(3,26,70,0.32)", icon: "ℹ" },
    question:{ label: "Confirm", accent: "var(--bd-accent-2)", glow: "rgba(62,216,255,0.3)", icon: "?" }
  };

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c] || c));
  }

  function ensureRoot() {
    let root = document.getElementById("bd-dialog-root");
    if (!root) {
      root = document.createElement("div");
      root.id = "bd-dialog-root";
      document.body.appendChild(root);
    }
    return root;
  }

  function lockBody(on) {
    document.body.classList.toggle("bd-dialog-open", !!on);
  }

  function cleanup(root) {
    if (!root) return;
    root.innerHTML = "";
    lockBody(false);
  }

  function buttonsMarkup(opts) {
    const order = opts.reverseButtons ? ["cancel", "deny", "confirm"] : ["confirm", "deny", "cancel"];
    const map = {
      confirm: opts.confirmText ? `<button class="bd-btn primary" data-role="confirm">${esc(opts.confirmText)}</button>` : "",
      cancel: opts.cancelText ? `<button class="bd-btn ghost" data-role="cancel">${esc(opts.cancelText)}</button>` : "",
      deny: opts.denyText ? `<button class="bd-btn ghost" data-role="deny">${esc(opts.denyText)}</button>` : ""
    };
    return order.map((k) => map[k]).join("");
  }

  function dialogMarkup(opts) {
    const cfg = VARIANTS[opts.variant] || VARIANTS.info;
    const vibe =
      opts.variant === "success" ? "bd-panel--success" :
      opts.variant === "warning" ? "bd-panel--warning" :
      opts.variant === "danger"  ? "bd-panel--danger"  :
      "bd-panel--info";

    const input = opts.input ? `
      <label class="bd-input__label">${esc(opts.input.label || opts.inputLabel || "")}</label>
      <input class="bd-input" type="${esc(opts.input.type || "text")}" value="${esc(opts.input.value || "")}" placeholder="${esc(opts.input.placeholder || "")}" autocomplete="${esc(opts.input.autocomplete || "off")}"/>
      <div class="bd-input__hint">${esc(opts.input.hint || "")}</div>
      <div class="bd-input__error" aria-live="polite"></div>
    ` : "";

    const textBlock = opts.html
      ? `<div class="bd-body">${opts.html}</div>`
      : (opts.text ? `<div class="bd-body">${esc(opts.text)}</div>` : "");

    return `
      <div class="bd-layer" role="dialog" aria-modal="true" aria-label="${esc(opts.title || cfg.label)}">
        <div class="bd-panel ${vibe}" data-variant="${esc(opts.variant || "info")}">
          <div class="bd-halo" style="background:${cfg.glow};"></div>
          <div class="bd-badge" style="color:${cfg.accent}">
            <span class="bd-icon" aria-hidden="true">${cfg.icon}</span>
            <span class="bd-eyebrow">${cfg.label}</span>
          </div>
          <div class="bd-title">${esc(opts.title || "")}</div>
          ${textBlock}
          ${input}
          <div class="bd-actions ${opts.reverseButtons ? "reverse" : ""}">
            ${buttonsMarkup(opts)}
          </div>
          ${opts.autoCloseMs ? `<div class="bd-timer" style="--bd-timer:${opts.autoCloseMs}ms;"></div>` : ""}
        </div>
      </div>
    `;
  }

  function attachInputAttributes(inputEl, attrs) {
    if (!inputEl || !attrs) return;
    Object.entries(attrs).forEach(([k, v]) => {
      try { inputEl.setAttribute(k, v); } catch {}
    });
  }

  function openDialog(opts = {}) {
    return new Promise((resolve) => {
      const root = ensureRoot();
      lockBody(true);
      root.innerHTML = dialogMarkup(opts);

      const overlay = root.querySelector(".bd-layer");
      const panel = root.querySelector(".bd-panel");
      const confirmBtn = root.querySelector('[data-role="confirm"]');
      const cancelBtn = root.querySelector('[data-role="cancel"]');
      const denyBtn = root.querySelector('[data-role="deny"]');
      const inputEl = root.querySelector(".bd-input");
      const errEl = root.querySelector(".bd-input__error");

      attachInputAttributes(inputEl, opts.input?.attributes || opts.inputAttributes);

      let closed = false;
      const finish = (kind, value) => {
        if (closed) return;
        closed = true;
        cleanup(root);
        resolve({
          isConfirmed: kind === "confirm",
          isDenied: kind === "deny",
          isDismissed: kind === "dismiss",
          value
        });
      };

      const validate = async () => {
        if (!opts.input) return { ok: true, value: undefined };
        const raw = inputEl.value;
        if (typeof opts.validate === "function") {
          const msg = await opts.validate(raw);
          if (msg) {
            if (errEl) errEl.textContent = msg;
            inputEl?.focus();
            return { ok: false };
          }
        }
        return { ok: true, value: raw };
      };

      confirmBtn?.addEventListener("click", async () => {
        const { ok, value } = await validate();
        if (!ok) return;
        finish("confirm", value);
      });
      cancelBtn?.addEventListener("click", () => finish("dismiss"));
      denyBtn?.addEventListener("click", () => finish("deny", inputEl ? inputEl.value : undefined));

      const allowOutside = opts.allowOutsideClick !== false;
      overlay?.addEventListener("click", (e) => {
        if (!allowOutside) return;
        if (e.target === overlay) finish("dismiss");
      });

      const onKey = async (e) => {
        if (e.key === "Escape") { e.preventDefault(); finish("dismiss"); }
        if (e.key === "Enter" && document.activeElement === inputEl) {
          e.preventDefault();
          const { ok, value } = await validate();
          if (ok) finish("confirm", value);
        }
      };
      document.addEventListener("keydown", onKey, { once: true });

      if (inputEl) inputEl.focus({ preventScroll: true });
      else confirmBtn?.focus({ preventScroll: true });

      if (opts.autoCloseMs && Number.isFinite(opts.autoCloseMs)) {
        setTimeout(() => finish("confirm", inputEl ? inputEl.value : undefined), opts.autoCloseMs);
      }

      requestAnimationFrame(() => panel?.classList.add("bd-panel--in"));
    });
  }

  function alert(opts = {}) {
    return openDialog(Object.assign({
      variant: opts.variant || "info",
      confirmText: opts.confirmText || "OK",
      cancelText: null,
      denyText: null
    }, opts));
  }

  async function confirm(opts = {}) {
    const res = await openDialog(Object.assign({
      variant: opts.variant || "warning",
      confirmText: opts.confirmText || "Confirm",
      cancelText: opts.cancelText || "Cancel",
      denyText: null
    }, opts));
    return res.isConfirmed;
  }

  async function prompt(opts = {}) {
    const res = await openDialog(Object.assign({
      variant: opts.variant || "info",
      confirmText: opts.confirmText || "Save",
      cancelText: opts.cancelText || "Cancel",
      input: Object.assign({ type: "text" }, opts.input || {}),
      allowOutsideClick: false
    }, opts));
    return res;
  }

  function loading(opts = {}) {
    const root = ensureRoot();
    lockBody(true);
    const cfg = VARIANTS.info;
    root.innerHTML = `
      <div class="bd-layer" role="alert" aria-live="polite">
        <div class="bd-panel bd-panel--info bd-panel--loading" data-variant="info">
          <div class="bd-halo" style="background:${cfg.glow};"></div>
          <div class="bd-spinner"></div>
          <div class="bd-title">${esc(opts.title || "Working…")}</div>
          ${opts.text ? `<div class="bd-body">${esc(opts.text)}</div>` : ""}
        </div>
      </div>
    `;
    requestAnimationFrame(() => root.querySelector(".bd-panel")?.classList.add("bd-panel--in"));
    return { close: () => cleanup(root) };
  }

  function progress(opts = {}) {
    const root = ensureRoot();
    lockBody(true);
    const cfg = VARIANTS.info;
    root.innerHTML = `
      <div class="bd-layer" role="alert" aria-live="polite">
        <div class="bd-panel bd-panel--info bd-panel--progress" data-variant="info">
          <div class="bd-halo" style="background:${cfg.glow};"></div>
          <div class="bd-badge" style="color:${cfg.accent}">
            <span class="bd-icon" aria-hidden="true">↻</span>
            <span class="bd-eyebrow">In progress</span>
          </div>
          <div class="bd-title">${esc(opts.title || "Processing…")}</div>
          <div class="bd-body bd-body--muted" id="bd-progress-sub">${esc(opts.subtitle || opts.text || "")}</div>
          <div class="bd-progress">
            <div class="bd-progress__track">
              <div class="bd-progress__bar" style="width:0%;"></div>
            </div>
            <div class="bd-progress__pct" id="bd-progress-pct">0%</div>
          </div>
          <div class="bd-actions"></div>
        </div>
      </div>
    `;
    const panel = root.querySelector(".bd-panel");
    const bar = root.querySelector(".bd-progress__bar");
    const pctEl = root.querySelector("#bd-progress-pct");
    const subEl = root.querySelector("#bd-progress-sub");
    const actions = root.querySelector(".bd-actions");
    requestAnimationFrame(() => panel?.classList.add("bd-panel--in"));

    const ctrl = {
      update(pct, subtitle) {
        if (Number.isFinite(pct)) {
          const c = Math.max(0, Math.min(100, pct));
          if (bar) bar.style.width = `${c}%`;
          if (pctEl) pctEl.textContent = `${Math.round(c)}%`;
        }
        if (subtitle != null && subEl) subEl.textContent = subtitle;
      },
      success(text, opts2 = {}) {
        panel?.classList.remove("bd-panel--danger", "bd-panel--warning");
        panel?.classList.add("bd-panel--success");
        if (subEl) subEl.textContent = text || "Done. You can continue.";
        if (actions && !actions.children.length) {
          const btn = document.createElement("button");
          btn.className = "bd-btn primary";
          btn.textContent = opts2.closeText || "Close";
          btn.addEventListener("click", () => ctrl.close());
          actions.appendChild(btn);
        }
      },
      error(text, opts2 = {}) {
        panel?.classList.remove("bd-panel--success");
        panel?.classList.add("bd-panel--danger");
        if (subEl) subEl.textContent = text || "Something went wrong.";
        if (actions && !actions.children.length) {
          const btn = document.createElement("button");
          btn.className = "bd-btn primary";
          btn.textContent = opts2.closeText || "Close";
          btn.addEventListener("click", () => ctrl.close());
          actions.appendChild(btn);
        }
      },
      close() {
        cleanup(root);
      }
    };
    return ctrl;
  }

  window.BrandDialog = { open: openDialog, alert, confirm, prompt, loading, progress, close: () => cleanup(ensureRoot()) };
})();
