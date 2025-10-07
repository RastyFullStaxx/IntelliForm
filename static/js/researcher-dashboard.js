document.addEventListener("DOMContentLoaded", () => {
    renderModelMetrics();
});

function renderModelMetrics() {
    const metricsData = document.getElementById("recordData");

    fetch("/static/sample data/model-metrics.json")  // Choose file to open
        .then(res => res.json())
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
    });
}

function renderUserMetrics() {
    const metricsData = document.getElementById("recordData");

    fetch("/static/sample data/user-metrics.json")  // Choose file to open
        .then(res => res.json())
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
    });
}