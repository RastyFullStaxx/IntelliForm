// workspace.js (Static Mode - No API Calls)

document.addEventListener('DOMContentLoaded', function () {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    const burgerToggle = document.getElementById('togglePages');
    const thumbnailSidebar = document.getElementById('thumbnailSidebar');
    const canvas = document.getElementById('pdfCanvas');
    const ctx = canvas.getContext('2d');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const overlayCtx = overlayCanvas.getContext('2d');
    const pageInfo = document.getElementById('pageInfo');
    const zoomInfo = document.getElementById('zoomInfo');
    const analyzeTool = document.getElementById('analyzeTool');
  
    let pdf = null;
    let currentPage = 1;
    let scale = 1.5;
    let inferenceResults = [];
  
    // Sidebar Toggle
    sidebarToggle.addEventListener('click', function (e) {
      e.stopPropagation();
      sidebar.classList.toggle('open');
      sidebarToggle.style.display = sidebar.classList.contains('open') ? 'none' : 'flex';
    });
  
    // Burger Toggle Thumbnails
    burgerToggle.addEventListener('click', function () {
      thumbnailSidebar.classList.toggle('visible');
    });
  
    // Close Sidebar Outside Click
    document.addEventListener('click', function (event) {
      if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
        sidebar.classList.remove('open');
        sidebarToggle.style.display = 'flex';
      }
    });
  
    // Load File from Session Storage
    const fileURL = sessionStorage.getItem('uploadedPDF');
    const fileName = sessionStorage.getItem('uploadedPDFName');
    if (!fileURL || !fileName) {
      alert("No PDF specified. Please upload via the homepage.");
      return;
    }
  
    const loadingTask = pdfjsLib.getDocument(fileURL);
    loadingTask.promise.then(function (loadedPdf) {
      pdf = loadedPdf;
      renderPage(currentPage);
      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        renderThumbnail(pageNum);
      }
    }).catch(function (error) {
      console.error("Error loading PDF:", error);
      alert("Failed to load PDF.");
    });
  
    // Inference Mocks
    const summaryMap = {
      "form_a.pdf": {
        ece: 0.2025,
        results: [
          { page_num: 1, label: "Name", summary: "Enter your full name", bbox: [100, 200, 300, 240] },
          { page_num: 1, label: "Date", summary: "Input today's date", bbox: [100, 250, 300, 290] }
        ]
      },
      "form_b.pdf": {
        ece: 0.3271,
        results: [
          { page_num: 1, label: "Student ID", summary: "Enter your university ID", bbox: [80, 180, 280, 220] },
          { page_num: 1, label: "Course", summary: "Specify your current course", bbox: [80, 230, 280, 270] }
        ]
      },
      "form_c.pdf": {
        ece: 0.1548,
        results: [
          { page_num: 1, label: "Phone Number", summary: "Provide a valid contact number", bbox: [90, 160, 290, 200] }
        ]
      },
      "form_d.pdf": {
        ece: 0.4123,
        results: [
          { page_num: 1, label: "Position", summary: "Job title of applicant", bbox: [70, 190, 270, 230] },
          { page_num: 1, label: "Signature", summary: "Sign this field", bbox: [70, 260, 270, 300] }
        ]
      },
      "form_e.pdf": {
        ece: 0.2314,
        results: [
          { page_num: 1, label: "Recipient", summary: "To whom the letter is addressed", bbox: [60, 170, 260, 210] }
        ]
      }
    };
  
    // Analyze Button Trigger
    analyzeTool.addEventListener("click", function () {
      const key = fileName.toLowerCase();
      if (!summaryMap[key]) {
        alert("This file is not recognized for static inference.");
        return;
      }
  
      const { results, ece } = summaryMap[key];
      inferenceResults = results;
      document.getElementById("eceScoreBadge").textContent = `📊 ECE Score: ${ece.toFixed(4)}`;
  
      const summaryList = document.getElementById("summaryList");
      summaryList.innerHTML = "";
  
      inferenceResults.forEach(field => {
        const item = document.createElement("div");
        item.className = "list-group-item list-group-item-action";
        item.innerHTML = `
          <div class="d-flex w-100 justify-content-between">
            <h6 class="mb-1 text-success">${field.label}</h6>
            <small class="text-muted">Page ${field.page_num}</small>
          </div>
          <p class="mb-1">${field.summary}</p>
        `;
        summaryList.appendChild(item);
      });
  
      renderPage(currentPage);
  
      fetch("/write-metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: key, ece: ece.toFixed(4), fields: results.length })
      });
    });
  
    function renderPage(pageNumber) {
      pdf.getPage(pageNumber).then(function (page) {
        const viewport = page.getViewport({ scale: scale });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        overlayCanvas.width = viewport.width;
        overlayCanvas.height = viewport.height;
  
        const renderContext = { canvasContext: ctx, viewport: viewport };
        page.render(renderContext).promise.then(() => {
          pageInfo.textContent = `Page ${pageNumber} / ${pdf.numPages}`;
          zoomInfo.textContent = `${Math.round(scale * 100)}%`;
          renderBoundingBoxes(pageNumber);
        });
      });
    }
  
    function renderThumbnail(pageNumber) {
      pdf.getPage(pageNumber).then(function (page) {
        const viewport = page.getViewport({ scale: 0.3 });
        const thumbCanvas = document.createElement('canvas');
        const thumbCtx = thumbCanvas.getContext('2d');
        thumbCanvas.width = viewport.width;
        thumbCanvas.height = viewport.height;
  
        page.render({ canvasContext: thumbCtx, viewport: viewport }).promise.then(() => {
          const thumbWrapper = document.createElement('div');
          thumbWrapper.className = 'thumbnail';
          thumbWrapper.appendChild(thumbCanvas);
          thumbnailSidebar.appendChild(thumbWrapper);
  
          thumbCanvas.addEventListener('click', () => {
            currentPage = pageNumber;
            renderPage(currentPage);
          });
        });
      });
    }
  
    function renderBoundingBoxes(pageNumber) {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeStyle = "red";
      overlayCtx.font = "12px Arial";
      overlayCtx.fillStyle = "red";
  
      const pageResults = inferenceResults.filter(item => item.page_num === pageNumber);
      pageResults.forEach(field => {
        const [x1, y1, x2, y2] = field.bbox;
        const scaleX = overlayCanvas.width / 1000;
        const scaleY = overlayCanvas.height / 1000;
  
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;
  
        overlayCtx.strokeRect(left, top, width, height);
        overlayCtx.fillText(`${field.label}: ${field.summary}`, left, top - 5);
      });
    }
  });
  