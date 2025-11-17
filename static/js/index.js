// static/js/index.js
document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('fileUpload');
  const dropArea = document.getElementById('dropArea');
  const fileDisplayContainer = document.getElementById('fileDisplayContainer');
  const uploadedFileName = document.getElementById('uploadedFileName');
  const deleteBtn = document.getElementById('deleteFileBtn');
  const startBtn = document.getElementById('startFillingBtn');
  const betaBtn = document.getElementById('betaTesterBtn');
  const betaStatus = document.getElementById('betaStatus');
  const METRICS_KEY = 'if_metrics_opt_in';
  const LS_UID = 'research_user_id';

  let selectedFile = null;

  // Always start the landing page fresh: clear any previously saved tester name/flag
  try {
    sessionStorage.removeItem(METRICS_KEY);
    localStorage.removeItem(METRICS_KEY);
    sessionStorage.removeItem(LS_UID);
    localStorage.removeItem(LS_UID);
  } catch {}

  function setMetricsOptIn(on, name) {
    try {
      sessionStorage.setItem(METRICS_KEY, on ? '1' : '0');
      localStorage.setItem(METRICS_KEY, on ? '1' : '0');
      if (on && name) {
        sessionStorage.setItem(LS_UID, name);
        localStorage.setItem(LS_UID, name);
      }
    } catch {}
    if (betaStatus) {
      betaStatus.textContent = on && name ? `Metrics on — ${name}` : 'Metrics: off (guest)';
      betaStatus.classList.toggle('active', !!(on && name));
    }
  }

  function ensureMetricsDefault() {
    const hasFlag = sessionStorage.getItem(METRICS_KEY) || localStorage.getItem(METRICS_KEY);
    if (!hasFlag) setMetricsOptIn(false);
  }

  ensureMetricsDefault();

  betaBtn?.addEventListener('click', async () => {
    try {
      const promptFn = window.promptForResearchUserId || (async () => {
        const { value } = await Swal.fire({
          title: "Enter Your Name",
          input: "text",
          inputLabel: "For beta metrics",
          inputPlaceholder: "TYPE YOUR NAME",
          inputAttributes: { autocapitalize: "characters", autocorrect: "off", spellcheck: "false" },
          allowOutsideClick: false,
          confirmButtonText: "Save",
          showCancelButton: true,
          preConfirm: (val) => {
            const v = String(val || "").trim();
            if (!v || v.length < 2) {
              Swal.showValidationMessage("Please enter at least 2 characters");
              return false;
            }
            return v;
          }
        });
        return value;
      });
      const id = await promptFn();
      if (id && id.trim()) {
        setMetricsOptIn(true, id.trim());
      }
    } catch (e) {
      console.warn('Beta opt-in failed', e);
    }
  });

  function showUploadedFile(name) {
    uploadedFileName.textContent = name;
    fileDisplayContainer.classList.remove('d-none');
    startBtn.classList.remove('d-none');
  }

  function resetUI() {
    fileDisplayContainer.classList.add('d-none');
    startBtn.classList.add('d-none');
    fileInput.value = '';
    selectedFile = null;
  }

  function handleFile(file) {
    if (file && file.type === 'application/pdf') {
      selectedFile = file;
      Swal.fire({
        icon: 'success',
        title: 'File ready',
        text: `“${file.name}” selected.`,
        timer: 1200,
        showConfirmButton: false
      });
      showUploadedFile(file.name);
    } else {
      Swal.fire({
        icon: 'error',
        title: 'Invalid File',
        text: 'Please upload a valid PDF file.'
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
    Swal.fire({
      title: 'Remove uploaded file?',
      text: 'This will cancel the selection and allow you to choose another file.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'Yes, remove it',
      cancelButtonText: 'Cancel',
      reverseButtons: true
    }).then(result => {
      if (result.isConfirmed) {
        resetUI();
        Swal.fire({
          icon: 'info',
          title: 'Selection cleared',
          text: 'You may now upload another PDF file.',
          timer: 1500,
          showConfirmButton: false
        });
      }
    });
  });

  // Upload to backend, then redirect to workspace with server path
  async function uploadAndNavigate(file) {
    const formData = new FormData();
    formData.append('file', file);

    let uploadingAlert;
    try {
      uploadingAlert = Swal.fire({
        title: 'Uploading...',
        text: 'Please wait while we upload your document.',
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading()
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

      Swal.close();

      // Navigate to workspace
      window.location.href = '/workspace';
    } catch (err) {
      console.error(err);
      Swal.fire({ icon: 'error', title: 'Upload Error', text: err.message || 'Something went wrong while uploading.' });
    } finally {
      if (uploadingAlert) Swal.close();
    }
  }

  // Start Filling Button → ensure we have a user id, then upload
  startBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      Swal.fire({
        icon: 'warning',
        title: 'No file selected',
        text: 'Please choose a PDF file first.'
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

    const { value } = await Swal.fire({
      title: "Enter Your Name (or ID)",
      input: "text",
      inputLabel: "Example: JUAN DELA CRUZ",
      inputPlaceholder: "TYPE YOUR NAME",
      inputAttributes: { autocapitalize: "characters", autocorrect: "off", spellcheck: "false" },
      allowOutsideClick: false,
      allowEscapeKey: false,
      confirmButtonText: "Save",
      showCancelButton: false,
      preConfirm: (val) => {
        const v = normalizeId(val);
        if (!v || v.length < 2) {
          Swal.showValidationMessage("Please enter at least 2 characters");
          return false;
        }
        return v;
      }
    });

    const normalized = normalizeId(value);
    localStorage.setItem(LS_KEY, normalized);
    sessionStorage.setItem(LS_KEY, normalized); // mirror for this tab
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
