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

  // === Sidebar Toggle ===
  sidebarToggle.addEventListener('click', function (e) {
    e.stopPropagation();
    sidebar.classList.toggle('open');
    sidebarToggle.style.display = sidebar.classList.contains('open') ? 'none' : 'flex';
  });

  // === Burger Toggle Thumbnails ===
  burgerToggle.addEventListener('click', function () {
    thumbnailSidebar.classList.toggle('visible');
  });

  // === Close Sidebar Outside Click ===
  document.addEventListener('click', function (event) {
    if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
      sidebar.classList.remove('open');
      sidebarToggle.style.display = 'flex';
    }
  });

  // === Get File Name from URL Query ===
  function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
  }

  const file = getQueryParam("file");
  if (!file) {
    alert("No PDF specified");
    return;
  }

  const url = `/static/uploads/${file}`;
  const loadingTask = pdfjsLib.getDocument(url);

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

  // === Analyze Button Trigger ===
  analyzeTool.addEventListener("click", function () {
    fetch(`/analyze-saved?file=${encodeURIComponent(file)}`)
      .then(res => res.json())
      .then(data => {
        console.log("Inference Results:", data);
        inferenceResults = data.results || [];
        document.getElementById("eceScoreBadge").textContent = `📊 ECE Score: ${data.ece.toFixed(4)}`;
  
        const summaryList = document.getElementById("summaryList");
        summaryList.innerHTML = ""; // Clear previous entries
  
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
      })
      .catch(err => console.error("Inference error:", err));
  });  

  // === Render Page and Overlay ===
  function renderPage(pageNumber) {
    pdf.getPage(pageNumber).then(function (page) {
      const viewport = page.getViewport({ scale: scale });
  
      // Set both canvases to the same size
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      overlayCanvas.width = viewport.width;
      overlayCanvas.height = viewport.height;
  
      const renderContext = {
        canvasContext: ctx,
        viewport: viewport
      };
  
      page.render(renderContext).promise.then(() => {
        pageInfo.textContent = `Page ${pageNumber} / ${pdf.numPages}`;
        zoomInfo.textContent = `${Math.round(scale * 100)}%`;
        renderBoundingBoxes(pageNumber);
      });
    });
  }
  

  // === Display Page on Thumbnail Click ===
  function displayPage(pageNumber) {
    currentPage = pageNumber;
    renderPage(currentPage);
  }

  // === Render Thumbnails ===
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
          displayPage(pageNumber);
        });
      });
    });
  }

  // === Render Bounding Boxes + Summaries ===
  function renderBoundingBoxes(pageNumber) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeStyle = "red";
    overlayCtx.font = "12px Arial";
    overlayCtx.fillStyle = "red";
  
    const pageResults = inferenceResults.filter(item => item.page_num === pageNumber);
    console.log(`🖼️ Drawing ${pageResults.length} bounding box(es) on Page ${pageNumber}`);
  
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
  
  
