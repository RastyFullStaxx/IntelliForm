// static/js/index.js
document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('fileUpload');
  const dropArea = document.getElementById('dropArea');
  const fileDisplayContainer = document.getElementById('fileDisplayContainer');
  const uploadedFileName = document.getElementById('uploadedFileName');
  const deleteBtn = document.getElementById('deleteFileBtn');
  const startBtn = document.getElementById('startFillingBtn');
  const betaBtn = document.getElementById('betaTesterBtn');
  const METRICS_KEY = 'if_metrics_opt_in';
  const LS_UID = 'research_user_id';

  let selectedFile = null;

  const getTesterName = () => {
    const name = (sessionStorage.getItem(LS_UID) || localStorage.getItem(LS_UID) || "").trim();
    return name;
  };

  const isTesterModeOn = () => {
    const flag = sessionStorage.getItem(METRICS_KEY) || localStorage.getItem(METRICS_KEY);
    return flag === '1';
  };

  function updateBetaButtonUI() {
    if (!betaBtn) return;
    const active = isTesterModeOn();
    const icon = active ? '<i class="bi bi-check-circle-fill"></i>' : '<i class="bi bi-stars"></i>';
    const label = active ? 'Tester Mode Enabled' : 'Beta Tester';
    betaBtn.innerHTML = `${icon} ${label}`;
    betaBtn.classList.toggle('active', active);
    betaBtn.setAttribute('aria-pressed', active ? 'true' : 'false');
  }

  function setMetricsOptIn(on, name) {
    try {
      sessionStorage.setItem(METRICS_KEY, on ? '1' : '0');
      localStorage.setItem(METRICS_KEY, on ? '1' : '0');
      if (on && name) {
        sessionStorage.setItem(LS_UID, name);
        localStorage.setItem(LS_UID, name);
      } else if (!on) {
        sessionStorage.removeItem(LS_UID);
        localStorage.removeItem(LS_UID);
      }
    } catch {}
    updateBetaButtonUI();
  }

  function ensureMetricsDefault() {
    const hasFlag = sessionStorage.getItem(METRICS_KEY) || localStorage.getItem(METRICS_KEY);
    if (!hasFlag) setMetricsOptIn(false);
    else updateBetaButtonUI();
  }

  ensureMetricsDefault();

  async function promptForTesterName(existingName = "") {
    const promptFn = window.promptForResearchUserId || (async () => {
      const res = await BrandDialog.prompt({
        title: "Add your name",
        text: "We tag tester feedback to your name so we can follow up if needed.",
        variant: "info",
        confirmText: "Save",
        cancelText: "Cancel",
        input: {
          label: "Name for tester mode",
          placeholder: "TYPE YOUR NAME",
          value: existingName || "",
          autocomplete: "off",
          attributes: { autocapitalize: "words", autocorrect: "off", spellcheck: "false" }
        },
        validate: (val) => {
          const v = String(val || "").trim();
          if (!v || v.length < 2) return "Please enter at least 2 characters";
          return "";
        }
      });
      if (res.isDismissed) return null;
      return res.value;
    });
    return await promptFn(existingName);
  }

  betaBtn?.addEventListener('click', async () => {
    try {
      const currentlyOn = isTesterModeOn();
      const existingName = getTesterName();

      if (currentlyOn) {
        const leave = await BrandDialog.confirm({
          title: "Disable tester mode?",
          text: `Tester mode is enabled${existingName ? ` for ${existingName}` : ""}. Turn it off and stop sharing usage data?`,
          variant: "warning",
          confirmText: "Disable",
          cancelText: "Stay enrolled",
          reverseButtons: true
        });
        if (!leave) return;
        setMetricsOptIn(false);
        BrandDialog.alert({
          variant: "info",
          title: "Tester mode disabled",
          text: "Thanks for considering being a beta tester — we hope you'll change your mind soon.",
          confirmText: "Got it"
        });
        return;
      }

      const consent = await BrandDialog.confirm({
        title: "Enroll as a beta tester?",
        text: "If you turn on tester mode, IntelliForm will record your usage to help evaluate and improve the system. Do you consent?",
        variant: "info",
        confirmText: "I consent",
        cancelText: "No thanks",
        reverseButtons: true
      });
      if (!consent) return;

      const id = await promptForTesterName(existingName);
      if (id && id.trim()) {
        setMetricsOptIn(true, id.trim());
        BrandDialog.alert({
          variant: "success",
          title: "Tester mode enabled",
          text: "Thanks for enrolling as a beta tester to help us improve IntelliForm.",
          confirmText: "Great!"
        });
      }
    } catch (e) {
      console.warn('Beta opt-in failed', e);
    }
  });

  function showUploadedFile(name) {
    uploadedFileName.textContent = name;
    fileDisplayContainer.classList.remove('d-none');
    startBtn.classList.remove('d-none');
    dropArea?.classList.add('d-none');   // hide drop zone once a file is chosen
  }

  function resetUI() {
    fileDisplayContainer.classList.add('d-none');
    startBtn.classList.add('d-none');
    dropArea?.classList.remove('d-none'); // show drop zone again
    fileInput.value = '';
    selectedFile = null;
  }

  function handleFile(file) {
    if (file && file.type === 'application/pdf') {
      selectedFile = file;
      BrandDialog.alert({
        variant: "success",
        title: "File ready",
        text: `"${file.name}" selected.`,
        confirmText: "Nice",
        autoCloseMs: 1200
      });
      showUploadedFile(file.name);
    } else {
      BrandDialog.alert({
        variant: "danger",
        title: "Invalid File",
        text: "Please upload a valid PDF file."
      });
    }
  }

  // File input
  fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

  // Drag events
  ['dragenter', 'dragover'].forEach(type => {
    dropArea.addEventListener(type, e => {
      e.preventDefault();
      dropArea.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(type => {
    dropArea.addEventListener(type, e => {
      e.preventDefault();
      dropArea.classList.remove('drag-over');
    });
  });

  // Drop handler
  dropArea.addEventListener('drop', (e) => {
    const droppedFile = e.dataTransfer.files && e.dataTransfer.files[0];
    if (droppedFile) {
      const dt = new DataTransfer();
      dt.items.add(droppedFile);
      fileInput.files = dt.files;
      handleFile(droppedFile);
    }
  });

  // Delete uploaded file (client-side reset only)
  deleteBtn.addEventListener('click', () => {
    BrandDialog.confirm({
      title: "Remove uploaded file?",
      text: "This will cancel the selection and allow you to choose another file.",
      variant: "warning",
      confirmText: "Yes, remove it",
      cancelText: "Cancel",
      reverseButtons: true
    }).then((ok) => {
      if (!ok) return;
      resetUI();
      BrandDialog.alert({
        variant: "info",
        title: "Selection cleared",
        text: "You may now upload another PDF file.",
        autoCloseMs: 1400,
        confirmText: "Got it"
      });
    });
  });

  // Upload to backend, then redirect to workspace with server path
  async function uploadAndNavigate(file) {
    const formData = new FormData();
    formData.append('file', file);

    let uploading;
    try {
      uploading = BrandDialog.loading({
        title: "Uploading…",
        text: "Please wait while we upload your document."
      });

      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      if (!res.ok) {
        let msg = `Upload failed (${res.status})`;
        try {
          const t = await res.text();
          if (t) msg += `: ${t}`;
        } catch {}
        throw new Error(msg);
      }

      const data = await res.json();

      // Prefer server's web_path; fall back to file_id → /uploads/<file_id>
      const webPath = data.web_path || (data.file_id ? `/uploads/${data.file_id}` : null);
      if (!webPath) throw new Error('Upload succeeded but server did not return web_path or file_id.');

      const diskPath = data.disk_path || '';
      // HASH FIRST: prefer canonical_form_id; keep form_id for backward compatibility
      const canonicalId = data.canonical_form_id || data.form_id || null;

      // Persist for workspace.js (new keys)
      sessionStorage.setItem('uploadedWebPath', webPath);
      if (diskPath) sessionStorage.setItem('uploadedDiskPath', diskPath);
      if (canonicalId) sessionStorage.setItem('uploadedFormId', canonicalId);

      // Legacy keys (display/compat only)
      const cleanName = (file.name || 'document.pdf');
      sessionStorage.setItem('uploadedFileName', cleanName);        // for title display
      sessionStorage.setItem('uploadedFileWithExtension', webPath); // legacy path

      uploading?.close();

      // Navigate to workspace
      window.location.href = '/workspace';
    } catch (err) {
      console.error(err);
      BrandDialog.alert({
        variant: "danger",
        title: "Upload Error",
        text: err.message || "Something went wrong while uploading."
      });
    } finally {
      uploading?.close();
    }
  }

  // Start Filling Button → ensure we have a user id, then upload
  startBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      BrandDialog.alert({
        variant: "warning",
        title: "No file selected",
        text: "Please choose a PDF file first."
      });
      return;
    }

    // If the user never opted in, force metrics off and anonymize id
    const opted = sessionStorage.getItem(METRICS_KEY) === '1' || localStorage.getItem(METRICS_KEY) === '1';
    if (!opted) {
      setMetricsOptIn(false, 'ANON');
    }

    uploadAndNavigate(selectedFile);
  });
});


// static/js/index.userid.js — user_id capture (research participant)
(function () {
  const LS_KEY = "research_user_id";
  const ALWAYS_PROMPT = true; // always prompt when Beta Tester opts in

  // normalize: trim, collapse spaces, UPPERCASE for consistent display
  function normalizeId(s) {
    return String(s || "").trim().replace(/\s+/g, " ").toUpperCase();
  }

  async function promptForUserId() {
    if (!ALWAYS_PROMPT) {
      const existing = localStorage.getItem(LS_KEY);
      if (existing && existing.trim()) return existing;
    }

    const existing = (localStorage.getItem(LS_KEY) || sessionStorage.getItem(LS_KEY) || "").trim();
    const res = await BrandDialog.prompt({
      title: "Enter Your Name (or ID)",
      text: "Example: JUAN DELA CRUZ",
      variant: "info",
      confirmText: "Save",
      cancelText: "Cancel",
      denyText: existing ? "Clear name" : null,
      reverseButtons: true,
      input: {
        label: "Identify yourself so we can log beta metrics",
        placeholder: "TYPE YOUR NAME",
        value: existing,
        autocomplete: "off",
        attributes: { autocapitalize: "characters", autocorrect: "off", spellcheck: "false" }
      },
      validate: (val) => {
        const v = normalizeId(val);
        if (!v || v.length < 2) return "Please enter at least 2 characters";
        return "";
      }
    });

    if (res.isDismissed) return null;
    if (res.isDenied) {
      try {
        localStorage.removeItem(LS_KEY);
        sessionStorage.removeItem(LS_KEY);
      } catch {}
      return "__TURN_OFF__";
    }

    const normalized = normalizeId(res.value);
    try {
      localStorage.setItem(LS_KEY, normalized);
      sessionStorage.setItem(LS_KEY, normalized); // mirror for this tab
    } catch {}
    return normalized;
  }

  // Expose helpers for other scripts (used by workspace.js and the Start button)
  window.getResearchUserId = function () {
    return localStorage.getItem(LS_KEY) || sessionStorage.getItem(LS_KEY) || null;
  };
  window.promptForResearchUserId = async function () {
    return await promptForUserId();
  };
})();
