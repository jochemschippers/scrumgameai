import { state } from '../../state/store.js';
import { $, showMessage, parseJsonEditor, clone } from '../../utils/helpers.js';
import { apiRequest } from '../../api/client.js';
import { escapeHtml, formatJson } from '../../utils/formatting.js';

export function renderTrainingConfigs() {
  $("trainingConfigCount").textContent = `${state.trainingConfigs.length}`;
  const container = $("trainingConfigsList");
  container.innerHTML = "";
  state.trainingConfigs.forEach((config) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${escapeHtml(config.label)}</h4>
      <div class="card-meta">
        <span class="tag">${escapeHtml(config.source)} profile</span>
        <span class="tag">${escapeHtml(String(config.episodes))} episodes</span>
        <span class="tag">lr ${escapeHtml(String(config.learning_rate))}</span>
        <span class="tag">gamma ${escapeHtml(String(config.gamma))}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

export function renderTrainingConfigValidation() {
  const container = $("trainingConfigValidationCard");
  if (!state.trainingConfigValidation) {
    container.className = "empty-state";
    container.textContent = "Validate the current training config to see structural errors.";
    return;
  }
  if (!state.trainingConfigValidation.valid) {
    container.className = "list-card";
    container.innerHTML = `<h4>Validation Error</h4><p>${escapeHtml(state.trainingConfigValidation.error || "Unknown validation error.")}</p>`;
    return;
  }
  container.className = "list-card";
  container.innerHTML = `
    <h4>Validation OK</h4>
    <div class="card-meta">
      <span class="tag good">valid</span>
      <span class="tag">${escapeHtml(String(state.trainingConfigValidation.episodes))} episodes</span>
      <span class="tag">lr ${escapeHtml(String(state.trainingConfigValidation.learning_rate))}</span>
      <span class="tag">gamma ${escapeHtml(String(state.trainingConfigValidation.gamma))}</span>
      <span class="tag">batch ${escapeHtml(String(state.trainingConfigValidation.batch_size))}</span>
    </div>
  `;
}

export async function loadActiveTrainingConfigIntoEditor() {
  if (!state.activeTrainingConfigId) {
    showMessage("Select a training config first.", "error");
    return;
  }
  const payload = await apiRequest(`/configs/training/${encodeURIComponent(state.activeTrainingConfigId)}`);
  state.activeTrainingConfigPayload = payload;
  $("trainingConfigEditor").value = formatJson(payload.config);
  $("trainingConfigFileNameInput").value = payload.id === "default_training_config" ? "my_training_config" : payload.id;
}

export async function saveTrainingConfig(overwrite = false) {
  const config = parseJsonEditor("trainingConfigEditor");
  const body = {
    config,
    file_name: $("trainingConfigFileNameInput").value.trim(),
  };
  if (overwrite) {
    body.id = state.activeTrainingConfigId;
  }
  const payload = await apiRequest("/configs/training", {
    method: "POST",
    body: JSON.stringify(body),
  });
  showMessage(`Saved training config ${payload.label}.`);
  // refreshAll needs to be called from main.js
  state.activeTrainingConfigId = payload.id;
  await loadActiveTrainingConfigIntoEditor();
  await validateTrainingConfigDraft(false).catch(() => {});
}

export async function validateTrainingConfigDraft(showSuccess = false) {
  try {
    const config = parseJsonEditor("trainingConfigEditor");
    state.trainingConfigValidation = await apiRequest("/configs/training/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    renderTrainingConfigValidation();
    if (showSuccess) {
      showMessage("Training config validation passed.");
    }
    return state.trainingConfigValidation;
  } catch (error) {
    state.trainingConfigValidation = { valid: false, error: error.message };
    renderTrainingConfigValidation();
    if (showSuccess) {
      showMessage(error.message, "error");
    }
    throw error;
  }
}
