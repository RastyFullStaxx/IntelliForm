document.addEventListener("DOMContentLoaded", () => {
    renderModelMetrics();
});

// document.addEventListener("DOMContentLoaded", () => {
//     const userList = document.getElementById("users");
//     const userData = document.getElementById("data");

//     fetch("data.json")
//         .then(res => res.json())
//         .then(users => {
     
//         users.forEach(user => {
//             const li = document.createElement("li");
//             li.classList.add("user-item");
//             li.dataset.id = user.id;

//             const bullet = document.createElement("span");
//             bullet.classList.add("bullet");

//             const nameSpan = document.createElement("span");
//             nameSpan.textContent = user.name;

//             li.appendChild(bullet);
//             li.appendChild(nameSpan);

//             li.addEventListener("click", () => {
//                 document.querySelectorAll(".user-item").forEach(el => el.classList.remove("active"));
//                 li.classList.add("active");
//                 renderRecordData(user);
//             });

//             userList.appendChild(li);
//         });
//     });

//     function renderRecordData(user) {
//     let html = `
//         <h2>${user.name}</h2>
//         <table>
//         <thead>
//             <tr>
//                 <th class="process-col">Process</th>
//                 <th class="time-col">Time</th>
//             </tr>
//         </thead>
//         <tbody class="bar-graph bar-graph-horizontal bar-graph-one">
//     `;

//     user.records.forEach(record => {
//         html += `
//         <tr>
//             <td class="process">${record.process}</td>
//             <td>
//                 <span class="bar" data-value="${record.time}"></span>
//             </td>
//         </tr>
//         `;
//     });

//     html += `</tbody></table>`;
//     userData.innerHTML = html;
//     displayBarGraph();
//     }

//     function displayBarGraph() {
//         const bars = document.querySelectorAll(".bar-graph-one .bar");
//         let max = 0;

//         // find max seconds
//         bars.forEach(bar => {
//             let val = parseFloat(bar.getAttribute("data-value"));
//             if (val > max) max = val;
//         });

//         // set widths and text
//         bars.forEach(bar => {
//             let val = parseFloat(bar.getAttribute("data-value"));
//             let percent = (val / max) * 100;
//             bar.style.width = percent + "%";
//         });
//     }
// });


// Table renderer for model metrics
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