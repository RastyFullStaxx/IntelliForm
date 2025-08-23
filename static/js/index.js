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
      // Reflect dropped file into the input for consistency
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
    // Match your FastAPI param name—commonly "file"
    formData.append('file', file);

    let uploadingAlert;
    try {
      uploadingAlert = Swal.fire({
        title: 'Uploading...',
        text: 'Please wait while we upload your document.',
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading()
      });

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
        // No need to set Content-Type; browser sets the multipart boundary
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();

      // Resolve a usable path from the server response
      // Prefer explicit path keys; fallback to /uploads/<filename>
      const serverPath =
        data.path ||
        data.file_path ||
        data.url ||
        (data.filename ? `/uploads/${data.filename}` : null);

      if (!serverPath) {
        throw new Error('Upload succeeded but server did not return a file path.');
      }

      Swal.close();
      // Pass the PDF location to workspace via query param
      const nextUrl = `/workspace.html?src=${encodeURIComponent(serverPath)}`;
      window.location.href = nextUrl;

    } catch (err) {
      console.error(err);
      Swal.fire({
        icon: 'error',
        title: 'Upload Error',
        text: err.message || 'Something went wrong while uploading.'
      });
    } finally {
      if (uploadingAlert) Swal.close();
    }
  }

  // Start Filling Button → uploads to server, then navigates
  startBtn.addEventListener('click', () => {
    if (!selectedFile) {
      Swal.fire({
        icon: 'warning',
        title: 'No file selected',
        text: 'Please choose a PDF file first.'
      });
      return;
    }
    uploadAndNavigate(selectedFile);
  });
});
