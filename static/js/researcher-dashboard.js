document.addEventListener("DOMContentLoaded", () => {
    renderModelMetrics();

    const tabs = document.querySelectorAll(".tabs button");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
        });
    });
});

function renderModelMetrics() {
    const metricsData = document.getElementById("recordData");

    fetch("/static/sample data/model-metrics.json")  // Choose file to open
        .then(res => { 
            if (!res.ok) {
                throw new Error(`File not found: ${res.status}`);
            }
            return res.json();
        })
        .then(models => {

        let html = `
            <table>
                <thead>
                    <tr>
                        <th>Form</th>
                        <th>TP</th>
                        <th>FP</th>
                        <th>FN</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                    </tr>
                </thead>
                <tbody>
        `;

        models.forEach((model, index) => {
            const m = model.metrics;
            html += `
                <tr>
                    <td>${model.name}</td>
                    <td>${m.tp}</td>
                    <td>${m.fp}</td>
                    <td>${m.fn}</td>
                    <td>${(m.precision * 100).toFixed(1)}%</td>
                    <td>${(m.recall * 100).toFixed(1)}%</td>
                    <td>${(m.f1 * 100).toFixed(1)}%</td>
                </tr>
            `;
        });

        html += `</tbody></table>`;
        metricsData.innerHTML = html;
    })
    .catch(err => {
        displayError(metricsData, err.message);
    });
}

function renderUserMetrics() {
    const metricsData = document.getElementById("recordData");

    fetch("/static/sample data/user-metrics.json")  // Choose file to open
        .then(res => { 
            if (!res.ok) {
                throw new Error(`File not found: ${res.status}`);
            }
            return res.json();
        })
        .then(models => {

        let html = `
            <table>
                <thead>
                    <tr>
                        <th>Participant</th>
                        <th>IntelliForm</th>
                        <th>Vanilla LayoutLMv3</th>
                        <th>Manual Method</th>
                    </tr>
                </thead>
                <tbody>
        `;

        models.forEach((model, index) => {
            const m = model.metrics;
            html += `
                <tr>
                    <td>${model.name}</td>
                    <td>${m.intelliform}s</td>
                    <td>${m.layoutlmv3}s</td>
                    <td>${m.manual}s</td>
                </tr>
            `;
        });

        html += `</tbody></table>`;
        metricsData.innerHTML = html;
    })
    .catch(err => {
        displayError(metricsData, err.message);
    });
}


function displayError(container, message) {
    container.innerHTML = `
            <div class="error-container">
                <p class="error-message">Error loading metrics: ${message}</p>
            </div>
    `;
}