const STATE = {
    scripts: [],
    selectedScriptId: null,
    refreshHandle: null,
    logHandle: null,
    ws: null,
    wsKeepalive: null,
};

const STATUS_LABELS = {
    idle: "Idle",
    running: "Running",
    stopping: "Stopping",
    stopped: "Stopped",
    completed: "Completed",
    failed: "Failed",
};

const STEP_LABELS = {
    action: "Action Needed",
    in_progress: "In Progress",
    blocked: "Blocked",
    complete: "Complete",
};

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    return payload;
}

function formatTimestamp(value) {
    if (!value) {
        return "Never";
    }
    return new Date(value).toLocaleString();
}

function sanitize(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}

function setChip(elementId, text, tone) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    element.textContent = text;
    element.classList.remove("warning", "danger", "live");
    if (tone) {
        element.classList.add(tone);
    }
}

async function refreshOverview() {
    try {
        const data = await fetchJson("/ui/api/overview");
        document.getElementById("overviewHost").textContent = data.hostname || "-";
        document.getElementById("overviewPython").textContent = data.python || "-";
        document.getElementById("overviewRunning").textContent = String(data.running_scripts ?? 0);
        document.getElementById("overviewDb").textContent =
            data.db_status === "up" ? "Connected" : "Disconnected";
        setChip("onboardingBadge", data.local_only ? "Local Only" : "Remote Access", data.local_only ? null : "warning");
    } catch (error) {
        document.getElementById("overviewDb").textContent = "Unavailable";
        console.error("Overview refresh failed", error);
    }
}

function renderScripts() {
    const container = document.getElementById("scriptGrid");
    container.innerHTML = STATE.scripts
        .map((script) => {
            const selected = STATE.selectedScriptId === script.id ? "selected" : "";
            const running = script.status === "running" || script.status === "stopping";
            return `
                <article class="script-card ${selected}" data-script-id="${sanitize(script.id)}">
                    <div class="script-head">
                        <p class="script-category">${sanitize(script.category)}</p>
                        <span class="status-pill ${sanitize(script.status)}">${sanitize(STATUS_LABELS[script.status] || script.status)}</span>
                    </div>
                    <h3>${sanitize(script.title)}</h3>
                    <p>${sanitize(script.description)}</p>
                    <p class="script-meta">Last start: ${sanitize(formatTimestamp(script.started_at))}</p>
                    <div class="script-actions">
                        <button class="primary-btn" data-action="start" data-script-id="${sanitize(script.id)}" ${running ? "disabled" : ""}>Run</button>
                        <button class="danger-btn" data-action="stop" data-script-id="${sanitize(script.id)}" ${running ? "" : "disabled"}>Stop</button>
                        <button class="ghost-btn" data-action="select" data-script-id="${sanitize(script.id)}">Logs</button>
                    </div>
                </article>
            `;
        })
        .join("");
}

async function refreshScripts() {
    try {
        const data = await fetchJson("/ui/api/scripts");
        STATE.scripts = data.scripts || [];
        renderScripts();
        if (!STATE.selectedScriptId && STATE.scripts.length > 0) {
            STATE.selectedScriptId = STATE.scripts[0].id;
        }
    } catch (error) {
        console.error("Scripts refresh failed", error);
    }
}

async function refreshLogs() {
    if (!STATE.selectedScriptId) {
        return;
    }
    try {
        const data = await fetchJson(`/ui/api/scripts/${encodeURIComponent(STATE.selectedScriptId)}/logs?tail=220`);
        const output = document.getElementById("logOutput");
        const logTitle = document.getElementById("logTitle");
        logTitle.textContent = `${data.script.title} Logs`;
        output.textContent = data.logs.length > 0 ? data.logs.join("\n") : "No logs yet.";
        output.scrollTop = output.scrollHeight;
    } catch (error) {
        console.error("Log refresh failed", error);
    }
}

async function refreshOnboarding() {
    try {
        const data = await fetchJson("/ui/api/onboarding");
        const list = document.getElementById("onboardingList");
        const dbStatus = data.db_status === "up" ? "DB Healthy" : "DB Needs Setup";
        setChip("onboardingBadge", dbStatus, data.db_status === "up" ? "live" : "warning");

        list.innerHTML = (data.steps || [])
            .map(
                (step) => `
                <article class="step-card">
                    <span class="step-dot ${sanitize(step.status)}"></span>
                    <div class="step-copy">
                        <h3>${sanitize(step.title)}</h3>
                        <p>${sanitize(step.description)}</p>
                    </div>
                    <span class="step-state">${sanitize(STEP_LABELS[step.status] || step.status)}</span>
                </article>
            `
            )
            .join("");
    } catch (error) {
        console.error("Onboarding refresh failed", error);
    }
}

async function refreshAttendance() {
    try {
        const data = await fetchJson("/ui/api/attendance/recent?limit=10");
        const rows = data.records || [];
        const body = document.getElementById("attendanceRows");
        if (rows.length === 0) {
            body.innerHTML = `<tr><td colspan="5">No records yet.</td></tr>`;
            return;
        }
        body.innerHTML = rows
            .map(
                (row) => `
                <tr>
                    <td>${sanitize(row.person_name)}</td>
                    <td>${sanitize(row.employee_id || "-")}</td>
                    <td>${sanitize(row.date)}</td>
                    <td>${sanitize(row.time)}</td>
                    <td>${sanitize(row.method)}</td>
                </tr>
            `
            )
            .join("");
    } catch (error) {
        console.error("Attendance refresh failed", error);
    }
}

async function startScript(scriptId, payload = {}) {
    await fetchJson(`/ui/api/scripts/${encodeURIComponent(scriptId)}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    STATE.selectedScriptId = scriptId;
    await refreshScripts();
    await refreshLogs();
    await refreshOnboarding();
}

async function stopScript(scriptId) {
    await fetchJson(`/ui/api/scripts/${encodeURIComponent(scriptId)}/stop`, {
        method: "POST",
    });
    await refreshScripts();
    await refreshLogs();
    await refreshOnboarding();
}

function wireScriptActions() {
    const container = document.getElementById("scriptGrid");
    const dialog = document.getElementById("enrollDialog");
    const enrollForm = document.getElementById("enrollForm");
    const cancelEnroll = document.getElementById("cancelEnroll");
    let pendingEnrollScript = null;

    container.addEventListener("click", async (event) => {
        const button = event.target.closest("button[data-action]");
        if (!button) {
            return;
        }

        const action = button.getAttribute("data-action");
        const scriptId = button.getAttribute("data-script-id");
        if (!scriptId) {
            return;
        }

        if (action === "select") {
            STATE.selectedScriptId = scriptId;
            await refreshScripts();
            await refreshLogs();
            return;
        }

        try {
            if (action === "start") {
                if (scriptId === "register_face") {
                    pendingEnrollScript = scriptId;
                    dialog.showModal();
                    return;
                }
                await startScript(scriptId);
            } else if (action === "stop") {
                await stopScript(scriptId);
            }
        } catch (error) {
            alert(error.message);
        }
    });

    cancelEnroll.addEventListener("click", () => dialog.close());

    enrollForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(enrollForm);
        const name = String(formData.get("name") || "").trim();
        const employeeId = String(formData.get("employee_id") || "").trim();
        if (!name || !pendingEnrollScript) {
            return;
        }

        try {
            await startScript(pendingEnrollScript, {
                name,
                employee_id: employeeId || null,
            });
            dialog.close();
            enrollForm.reset();
        } catch (error) {
            alert(error.message);
        }
    });
}

function connectStream() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${window.location.host}/ws/stream`;
    const image = document.getElementById("streamFrame");
    const placeholder = document.getElementById("streamPlaceholder");

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "blob";
    STATE.ws = ws;

    ws.onopen = () => {
        setChip("streamStatus", "Live", "live");
        STATE.wsKeepalive = window.setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send("ping");
            }
        }, 4000);
    };

    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            return;
        }
        const blob = event.data;
        const blobUrl = URL.createObjectURL(blob);
        const previous = image.dataset.lastBlob;
        image.src = blobUrl;
        image.dataset.lastBlob = blobUrl;
        placeholder.style.display = "none";
        if (previous) {
            URL.revokeObjectURL(previous);
        }
    };

    ws.onerror = () => {
        setChip("streamStatus", "Stream Error", "danger");
    };

    ws.onclose = () => {
        if (STATE.wsKeepalive) {
            window.clearInterval(STATE.wsKeepalive);
            STATE.wsKeepalive = null;
        }
        setChip("streamStatus", "Reconnecting...", "warning");
        window.setTimeout(connectStream, 2500);
    };
}

function wireGlobalActions() {
    document.getElementById("clearLogSelection").addEventListener("click", () => {
        STATE.selectedScriptId = null;
        document.getElementById("logTitle").textContent = "Logs";
        document.getElementById("logOutput").textContent = "Select a script to view logs.";
        refreshScripts();
    });
}

async function initialLoad() {
    await Promise.all([refreshOverview(), refreshScripts(), refreshOnboarding(), refreshAttendance()]);
    await refreshLogs();
}

function startPolling() {
    if (!STATE.refreshHandle) {
        STATE.refreshHandle = window.setInterval(async () => {
            await Promise.all([refreshOverview(), refreshScripts(), refreshOnboarding(), refreshAttendance()]);
        }, 5000);
    }

    if (!STATE.logHandle) {
        STATE.logHandle = window.setInterval(refreshLogs, 1800);
    }
}

window.addEventListener("DOMContentLoaded", async () => {
    wireScriptActions();
    wireGlobalActions();
    connectStream();
    await initialLoad();
    startPolling();
});
