// static/js/index.js
document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('fileUpload');
  const dropArea = document.getElementById('dropArea');
  const fileDisplayContainer = document.getElementById('fileDisplayContainer');
  const uploadedFileName = document.getElementById('uploadedFileName');
  const deleteBtn = document.getElementById('deleteFileBtn');
  const startBtn = document.getElementById('startFillingBtn');
  const scanText = document.getElementById('scanText');
  const uploadBox = document.querySelector('.upload-btn');

  let selectedFile = null;

  function showUploadedFile(name) {
    uploadedFileName.textContent = name;
    fileDisplayContainer.classList.remove('d-none');
    startBtn.classList.remove('d-none');
    scanText.classList.add('d-none');
    uploadBox.classList.add('d-none');
  }

  function resetUI() {
    fileDisplayContainer.classList.add('d-none');
    startBtn.classList.add('d-none');
    scanText.classList.remove('d-none');
    uploadBox.classList.remove('d-none');
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

    try {
      // Last-chance: ensure a non-empty research_user_id
      const currentId = (window.getResearchUserId && window.getResearchUserId()) || '';
      if (!currentId || currentId.trim() === '' || currentId.trim().toUpperCase() === 'ANON') {
        if (window.promptForResearchUserId) {
          const id = await window.promptForResearchUserId();
          if (id && id.trim()) {
            sessionStorage.setItem('research_user_id', id.trim());
            localStorage.setItem('research_user_id', id.trim());
          }
        }
      }
    } catch (e) {
      console.warn('User ID ensure step failed:', e);
    }

    uploadAndNavigate(selectedFile);
  });
});


// static/js/index.userid.js — user_id capture (research participant)
(function () {
  const LS_KEY = "research_user_id";
  const ALWAYS_PROMPT = true; // ask on every index load/refresh

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

  // Prompt immediately on index load
  document.addEventListener("DOMContentLoaded", async () => {
    try {
      const id = await promptForUserId();
      sessionStorage.setItem(LS_KEY, id); // ensure present even if storage is blocked later
    } catch (e) {
      console.warn("User ID prompt failed:", e);
      sessionStorage.setItem(LS_KEY, "ANON");
    }
  });
})();
