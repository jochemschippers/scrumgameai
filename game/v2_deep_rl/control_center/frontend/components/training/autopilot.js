/** Implement autopilot user-interface behavior. */

import { state } from '../../state/store.js';
import { $, showMessage, runLabelFromPath } from '../../utils/helpers.js';
import { escapeHtml, formatNumber } from '../../utils/formatting.js';
import { apiRequest, isGuest } from '../../api/client.js';

const _GUEST_ATTR = 'disabled title="Guests cannot perform this action"';

/** Handle action tag. */
export function actionTag(action) {
  const tones = {
    continue: "good",
    lower_lr: "warn",
    extend_epsilon_decay: "info",
    fine_tune: "ai",
    stop: "bad",
  };
  return `<span class="tag ${tones[action] || ""}">${escapeHtml(action || "-")}</span>`;
}

/** Fetch autopilot data. */
export async function fetchAutopilotData(runId) {
  if (!runId) return;
  try {
    const [settings, historyResult, stopStatus] = await Promise.all([
      apiRequest("/autopilot/settings"),
      apiRequest(`/autopilot/history/${encodeURIComponent(runId)}`),
      apiRequest("/autopilot/status"),
    ]);
    state.autopilotSettings = settings;
    state.autopilotHistory = historyResult.items || [];
    state.autopilotStopRequested = stopStatus.stop_requested || false;
  } catch (_error) {
    // Leave previous state intact on error.
  }
  renderAutopilotPanel();
}

/** Render autopilot panel. */
export function renderAutopilotPanel() {
  const settings = state.autopilotSettings;
  const history = state.autopilotHistory;
  const runDetail = state.runDetail;

  // --- Controls card ---
  const controlCard = $("autopilotControlCard");
  const statusLabel = $("autopilotStatusLabel");

  if (!settings) {
    controlCard.className = "empty-state";
    controlCard.textContent = "Open a training run to manage autopilot.";
    statusLabel.textContent = "-";
  } else {
    const logicOn = settings.logic_enabled;
    const aiOn = settings.ai_enabled;
    const stopPending = state.autopilotStopRequested;
    const lastDecision = history.length ? history[history.length - 1] : null;
    const aiUsed = history.filter((d) => d.advisor === "ai").length;

    statusLabel.textContent = logicOn ? (aiOn ? "logic + AI" : "logic only") : "disabled";

    controlCard.className = "list-card";
    const guestAttr = isGuest() ? _GUEST_ATTR : "";
    controlCard.innerHTML = `
      <div class="autopilot-toggles">
        <div class="autopilot-toggle-row">
          <span>Logic Autopilot</span>
          <button class="button ${logicOn ? "primary" : "secondary"} autopilot-toggle-btn" data-key="logic_enabled" data-value="${!logicOn}" type="button" ${guestAttr}>
            ${logicOn ? "Enabled" : "Disabled"}
          </button>
        </div>
        <div class="autopilot-toggle-row">
          <span>AI Advisor <em style="font-size:11px;color:var(--muted)">(on plateau)</em></span>
          <button class="button ${aiOn ? "primary" : "secondary"} autopilot-toggle-btn" data-key="ai_enabled" data-value="${!aiOn}" type="button" ${guestAttr}>
            ${aiOn ? "Enabled" : "Disabled"}
          </button>
        </div>
      </div>
      <div class="card-meta">
        <span class="tag ${stopPending ? "bad" : ""}">Stop after cycle: ${stopPending ? "pending" : "off"}</span>
        <span class="tag">AI used: ${aiUsed} / 3</span>
        ${lastDecision ? `<span class="tag">Last: ${actionTag(lastDecision.action)} by ${escapeHtml(lastDecision.advisor)}</span>` : ""}
      </div>
      <div class="inline-actions">
        <button class="button primary" id="startAutopilotLoopButton" type="button" ${guestAttr}>Start Autopilot Loop</button>
        ${stopPending
          ? `<button class="button secondary" id="clearStopButton" type="button" ${guestAttr}>Resume Auto-Chain</button>`
          : `<button class="button secondary" id="requestStopButton" type="button" ${guestAttr}>Stop After Cycle</button>`
        }
      </div>
    `;

    const startLoopBtn = controlCard.querySelector("#startAutopilotLoopButton");
    if (startLoopBtn && !isGuest()) {
      startLoopBtn.addEventListener("click", async () => {
        const runId = state.activeRunId;
        if (!runId) { showMessage("No run selected.", "error"); return; }
        try {
          startLoopBtn.disabled = true;
          startLoopBtn.textContent = "Running...";
          const result = await apiRequest(`/autopilot/run/${runId}`, { method: "POST", body: JSON.stringify({}) });
          showMessage(`Autopilot loop started: ${result.action} - ${result.reason}`);
          await fetchAutopilotData(runId);
        } catch (error) {
          showMessage(error.message, "error");
          startLoopBtn.disabled = false;
          startLoopBtn.textContent = "Start Autopilot Loop";
        }
      });
    }

    const stopBtn = controlCard.querySelector("#requestStopButton");
    if (stopBtn && !isGuest()) {
      stopBtn.addEventListener("click", async () => {
        try {
          await apiRequest("/autopilot/stop-after-cycle", { method: "POST" });
          showMessage("Autopilot will stop after the current training block.");
          state.autopilotStopRequested = true;
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    }

    const resumeBtn = controlCard.querySelector("#clearStopButton");
    if (resumeBtn && !isGuest()) {
      resumeBtn.addEventListener("click", async () => {
        try {
          await apiRequest("/autopilot/stop-after-cycle", { method: "DELETE" });
          showMessage("Stop request cleared. Autopilot will resume after next block.");
          state.autopilotStopRequested = false;
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    }

    controlCard.querySelectorAll(".autopilot-toggle-btn").forEach((btn) => {
      if (isGuest()) return;
      btn.addEventListener("click", async () => {
        try {
          state.autopilotSettings = await apiRequest("/autopilot/settings", {
            method: "POST",
            body: JSON.stringify({ [btn.dataset.key]: btn.dataset.value === "true" }),
          });
          renderAutopilotPanel();
          renderAutopilotTrainingPanel();
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
    });
  }

  // --- Run settings card ---
  const settingsCard = $("autopilotRunSettingsCard");
  const tc = runDetail?.training_config;
  if (!tc) {
    settingsCard.className = "empty-state";
    settingsCard.textContent = "Open a training run to see active settings.";
  } else {
    settingsCard.className = "list-card";
    settingsCard.innerHTML = `
      <div class="metric-grid">
        <div class="metric-card"><span>Learning Rate</span><strong>${formatNumber(tc.learning_rate, 6)}</strong></div>
        <div class="metric-card"><span>Epsilon Decay Ep.</span><strong>${tc.epsilon_decay_episodes?.toLocaleString() ?? "-"}</strong></div>
        <div class="metric-card"><span>Episodes</span><strong>${tc.episodes?.toLocaleString() ?? "-"}</strong></div>
        <div class="metric-card"><span>Gamma</span><strong>${formatNumber(tc.gamma, 4)}</strong></div>
        <div class="metric-card"><span>Batch Size</span><strong>${tc.batch_size ?? "-"}</strong></div>
        <div class="metric-card"><span>Epsilon Min</span><strong>${formatNumber(tc.epsilon_min, 4)}</strong></div>
      </div>
    `;
  }

  // --- Decision history ---
  const decisionsCard = $("autopilotDecisionsCard");
  const countLabel = $("autopilotDecisionCount");
  countLabel.textContent = `${history.length} decision${history.length !== 1 ? "s" : ""}`;

  if (!history.length) {
    decisionsCard.className = "empty-state";
    decisionsCard.textContent = "No autopilot decisions recorded for this run.";
    return;
  }

  decisionsCard.className = "decision-list";
  decisionsCard.innerHTML = [...history].reverse().map((d) => {
    const ts = d.decided_at ? new Date(d.decided_at).toLocaleString() : "-";
    const m = d.metrics || {};
    return `
      <div class="decision-row">
        <div class="decision-row-head">
          ${actionTag(d.action)}
          <span class="tag ${d.advisor === "ai" ? "ai" : ""}">${escapeHtml(d.advisor || "logic")}</span>
          <span class="tag">${escapeHtml(ts)}</span>
          ${d.job_enqueued ? `<span class="tag good">job #${d.job_id} queued</span>` : ""}
        </div>
        <p class="decision-reason">${escapeHtml(d.reason || "")}</p>
        <div class="card-meta">
          ${m.latest_epsilon != null ? `<span class="tag">eps ${formatNumber(m.latest_epsilon, 3)}</span>` : ""}
          ${m.latest_reward != null ? `<span class="tag">reward ${formatNumber(m.latest_reward)}</span>` : ""}
          ${m.bankruptcy_rate != null ? `<span class="tag">bankruptcy ${formatNumber(m.bankruptcy_rate, 3)}</span>` : ""}
          ${m.invalid_action_rate != null ? `<span class="tag">invalid ${formatNumber(m.invalid_action_rate, 3)}</span>` : ""}
          ${m.reward_improvement_ratio != null ? `<span class="tag">improvement ${formatNumber(m.reward_improvement_ratio * 100, 1)}%</span>` : ""}
        </div>
        ${d.next_payload ? `<div class="card-meta">
          <span class="tag info">lr ${formatNumber(d.next_payload.learning_rate, 6)}</span>
          <span class="tag info">eps decay ${d.next_payload.epsilon_decay_episodes?.toLocaleString()}</span>
          <span class="tag info">${d.next_payload.episodes?.toLocaleString()} ep</span>
        </div>` : ""}
      </div>
    `;
  }).join("");
}

/** Render autopilot training panel. */
export function renderAutopilotTrainingPanel() {
  const card = $("autopilotTrainingCard");
  const label = $("autopilotTrainingStatusLabel");
  const settings = state.autopilotSettings;

  if (!settings) {
    card.className = "empty-state";
    card.textContent = "Connect to the backend to manage autopilot settings.";
    label.textContent = "-";
    return;
  }

  const logicOn = settings.logic_enabled;
  const aiOn = settings.ai_enabled;
  const stopPending = state.autopilotStopRequested;
  label.textContent = logicOn ? (aiOn ? "logic + AI" : "logic only") : "disabled";

  card.className = "list-card";
  const guestAttr = isGuest() ? _GUEST_ATTR : "";
  card.innerHTML = `
    <div class="autopilot-toggles">
      <div class="autopilot-toggle-row">
        <span>Logic Autopilot</span>
        <button class="button ${logicOn ? "primary" : "secondary"} ap-toggle-btn" data-key="logic_enabled" data-value="${!logicOn}" type="button" ${guestAttr}>
          ${logicOn ? "Enabled" : "Disabled"}
        </button>
      </div>
      <div class="autopilot-toggle-row">
        <span>AI Advisor <em style="font-size:11px;color:var(--muted)">(on plateau)</em></span>
        <button class="button ${aiOn ? "primary" : "secondary"} ap-toggle-btn" data-key="ai_enabled" data-value="${!aiOn}" type="button" ${guestAttr}>
          ${aiOn ? "Enabled" : "Disabled"}
        </button>
      </div>
    </div>
    <div class="card-meta">
      <span class="tag ${stopPending ? "bad" : ""}">Stop after cycle: ${stopPending ? "pending" : "off"}</span>
    </div>
    <div class="inline-actions">
      <button class="button primary" id="startAutopilotTrainingBtn" type="button"
        ${isGuest() ? _GUEST_ATTR : (state.trainingProgress?.status === "completed" ? "" : "disabled")}>
        Start Autopilot Loop
      </button>
      ${stopPending
        ? `<button class="button secondary" id="clearStopTrainingBtn" type="button" ${guestAttr}>Resume Auto-Chain</button>`
        : `<button class="button secondary" id="requestStopTrainingBtn" type="button" ${guestAttr}>Stop After Cycle</button>`
      }
    </div>
  `;

  card.querySelectorAll(".ap-toggle-btn").forEach((btn) => {
    if (isGuest()) return;
    btn.addEventListener("click", async () => {
      try {
        state.autopilotSettings = await apiRequest("/autopilot/settings", {
          method: "POST",
          body: JSON.stringify({ [btn.dataset.key]: btn.dataset.value === "true" }),
        });
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  const startLoopTrainingBtn = card.querySelector("#startAutopilotTrainingBtn");
  if (startLoopTrainingBtn && !isGuest()) {
    startLoopTrainingBtn.addEventListener("click", async () => {
      const progress = state.trainingProgress;
      const runId = progress?.run_dir ? runLabelFromPath(progress.run_dir) : state.activeProgressRunId;
      if (!runId) { showMessage("No completed run to analyze.", "error"); return; }
      try {
        startLoopTrainingBtn.disabled = true;
        startLoopTrainingBtn.textContent = "Running...";
        const result = await apiRequest(`/autopilot/run/${runId}`, { method: "POST", body: JSON.stringify({}) });
        showMessage(`Autopilot loop started: ${result.action} - ${result.reason}`);
        await fetchAutopilotData(runId);
      } catch (error) {
        showMessage(error.message, "error");
        renderAutopilotTrainingPanel();
      }
    });
  }

  const stopBtn = card.querySelector("#requestStopTrainingBtn");
  if (stopBtn && !isGuest()) {
    stopBtn.addEventListener("click", async () => {
      try {
        await apiRequest("/autopilot/stop-after-cycle", { method: "POST" });
        state.autopilotStopRequested = true;
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) { showMessage(error.message, "error"); }
    });
  }

  const resumeBtn = card.querySelector("#clearStopTrainingBtn");
  if (resumeBtn && !isGuest()) {
    resumeBtn.addEventListener("click", async () => {
      try {
        await apiRequest("/autopilot/stop-after-cycle", { method: "DELETE" });
        state.autopilotStopRequested = false;
        renderAutopilotTrainingPanel();
        renderAutopilotPanel();
      } catch (error) { showMessage(error.message, "error"); }
    });
  }
}

/** Refresh autopilot settings. */
export async function refreshAutopilotSettings() {
  try {
    const [settings, stopStatus] = await Promise.all([
      apiRequest("/autopilot/settings"),
      apiRequest("/autopilot/status"),
    ]);
    state.autopilotSettings = settings;
    state.autopilotStopRequested = stopStatus.stop_requested || false;
  } catch (_error) {
    // Non-fatal. Leave previous state.
  }
  renderAutopilotTrainingPanel();
  renderAutopilotPanel();
}
