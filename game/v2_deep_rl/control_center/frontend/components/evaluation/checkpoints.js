/** Implement checkpoints user-interface behavior. */

import { state } from '../../state/store.js';
import { $, showMessage, downloadCheckpointFile } from '../../utils/helpers.js';
import { escapeHtml, checkpointUiLabel, checkpointCompatibilityTone } from '../../utils/formatting.js';
import { apiRequest } from '../../api/client.js';
import { renderContextCard, updateSummaryPills } from '../navigation.js';
import { renderTrainingSelectionSummary, renderTrainingPreflight } from '../training/jobs.js';

/** Render compatibility. */
export function renderCompatibility() {
  const container = $("compatibilityCard");
  if (!state.compatibility) {
    container.className = "empty-state";
    container.textContent = "Select a blueprint and brain, then run compatibility.";
    return;
  }

  container.className = "list-card";
  container.innerHTML = `
    <h4>Compatibility Result</h4>
    <p>${escapeHtml(state.compatibility.message)}</p>
    <div class="card-meta">
      <span class="tag ${checkpointCompatibilityTone(state.compatibility.strict_resume_status)}">Strict: ${escapeHtml(state.compatibility.strict_resume_status)}</span>
      <span class="tag ${checkpointCompatibilityTone(state.compatibility.fine_tune_status)}">Fine-Tune: ${escapeHtml(state.compatibility.fine_tune_status)}</span>
    </div>
  `;
}

/** Clear compatibility result. */
export function clearCompatibilityResult() {
  state.compatibility = null;
  renderCompatibility();
  renderContextCard();
}

/** Render checkpoint detail. */
export function renderCheckpointDetail() {
  const container = $("checkpointDetailCard");
  const checkpoint = state.checkpoints.find((item) => item.id === state.activeCheckpointId);
  if (!checkpoint) {
    container.className = "empty-state";
    container.textContent = "Select a brain from the library or workspace selector to inspect it.";
    return;
  }

  container.className = "list-card";
  container.innerHTML = `
    <h4>${escapeHtml(checkpoint.label)}</h4>
    <p>${escapeHtml(checkpointUiLabel(checkpoint))}</p>
    <div class="card-meta">
      <span class="tag">${escapeHtml(checkpoint.checkpoint_type)}</span>
      ${checkpoint.episode != null ? `<span class="tag">ep ${Number(checkpoint.episode).toLocaleString()}</span>` : ""}
      <span class="tag ${checkpointCompatibilityTone(checkpoint.compatibility_status)}">${escapeHtml(checkpoint.compatibility_status)}</span>
    </div>
    <div class="card-meta">
      <span class="tag">state ${checkpoint.state_dim || "-"}</span>
      <span class="tag">actions ${checkpoint.num_actions || "-"}</span>
    </div>
    <div class="inline-actions">
      <button class="button secondary download-brain-button" type="button">Download Brain</button>
      ${checkpoint.source_type === "run" && checkpoint.source_run ? `<button class="button secondary open-brain-inspect-button" data-run-id="${escapeHtml(checkpoint.source_run)}" type="button">Open Inspect</button>` : ""}
    </div>
  `;

  container.querySelector(".download-brain-button")?.addEventListener("click", () => {
    try {
      downloadCheckpointFile(checkpoint);
      showMessage(`Downloading ${checkpoint.label}.`);
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  container.querySelector(".open-brain-inspect-button")?.addEventListener("click", (event) => {
    document.dispatchEvent(new CustomEvent("openInspectForRun", {
      detail: { runId: event.target.dataset.runId, announce: true },
    }));
  });
}

/** Refresh checkpoints. */
export async function refreshCheckpoints() {
  try {
    const checkpoints = await apiRequest("/checkpoints", {}, 120000);
    state.checkpoints = checkpoints.items || [];
    document.dispatchEvent(new CustomEvent("checkpointsLoaded"));
    renderCheckpointDetail();
    renderTrainingSelectionSummary();
    renderTrainingPreflight();
    renderCompatibility();
  } catch (_err) {
    // Non-fatal.
  }
}

/** Run compatibility. */
export async function runCompatibility() {
  if (!state.activeCheckpointId || !state.activeGameConfigId) {
    showMessage("Select both a game config and a checkpoint first.", "error");
    return;
  }
  const checkpointId = encodeURIComponent(state.activeCheckpointId);
  const gameConfigId = encodeURIComponent(state.activeGameConfigId);
  state.compatibility = await apiRequest(`/checkpoints/${checkpointId}/compatibility?game_config_id=${gameConfigId}`);
  renderCompatibility();
}
