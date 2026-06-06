/** Implement game config user-interface behavior. */

import { state } from '../../state/store.js';
import { $, clone, showMessage, parseJsonEditor } from '../../utils/helpers.js';
import { apiRequest } from '../../api/client.js';
import { escapeHtml, formatJson } from '../../utils/formatting.js';
import { readVisualEditorIntoState, renderVisualEditor } from './visualEditor.js';

/** Render game configs. */
export function renderGameConfigs() {
  $("gameConfigCount").textContent = `${state.gameConfigs.length}`;
  const container = $("gameConfigsList");
  container.innerHTML = "";
  state.gameConfigs.forEach((config) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${escapeHtml(config.config_name || config.label)}</h4>
      <div class="card-meta">
        <span class="tag">${escapeHtml(config.source)}</span>
        <span class="tag">${escapeHtml(String(config.products_count))} products</span>
        <span class="tag">${escapeHtml(String(config.sprints_per_product))} sprints</span>
      </div>
    `;
    container.appendChild(card);
  });
}

/** Render game config validation. */
export function renderGameConfigValidation() {
  const container = $("gameConfigValidationCard");
  if (!state.gameConfigValidation) {
    container.className = "empty-state";
    container.textContent = "Validate the current game config to see structural errors.";
    return;
  }
  if (!state.gameConfigValidation.valid) {
    container.className = "list-card";
    container.innerHTML = `<h4>Validation Error</h4><p>${escapeHtml(state.gameConfigValidation.error || "Unknown validation error.")}</p>`;
    return;
  }
  container.className = "list-card";
  container.innerHTML = `
    <h4>Validation OK</h4>
    <div class="card-meta">
      <span class="tag good">valid</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.products_count))} products</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.sprints_per_product))} sprints</span>
      <span class="tag">${escapeHtml(String(state.gameConfigValidation.actions_count))} actions</span>
    </div>
  `;
}

/** Load active game config into editor. */
export async function loadActiveGameConfigIntoEditor() {
  if (!state.activeGameConfigId) {
    showMessage("Select a game config first.", "error");
    return;
  }
  const payload = await apiRequest(`/configs/game/${encodeURIComponent(state.activeGameConfigId)}`);
  state.activeGameConfigPayload = payload;
  state.visualGameConfig = clone(payload.config);
  $("gameConfigEditor").value = formatJson(payload.config);
  $("gameConfigFileNameInput").value = payload.id === "default_game_config" ? "my_game_config" : payload.id;
  renderVisualEditor();
}

/** Save game config. */
export async function saveGameConfig(overwrite = false) {
  let config;
  try {
    config = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(config);
  } catch (_error) {
    config = parseJsonEditor("gameConfigEditor");
  }
  const body = {
    config,
    file_name: $("gameConfigFileNameInput").value.trim(),
  };
  if (overwrite) {
    body.id = state.activeGameConfigId;
  }
  const payload = await apiRequest("/configs/game", {
    method: "POST",
    body: JSON.stringify(body),
  });
  showMessage(`Saved game config ${payload.label}.`);
  // refreshAll needs to be called from main.js
  state.activeGameConfigId = payload.id;
  await loadActiveGameConfigIntoEditor();
  await validateGameConfigDraft(false).catch(() => {});
}

/** Validate game config draft. */
export async function validateGameConfigDraft(showSuccess = false) {
  try {
    const config = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(config);
    state.gameConfigValidation = await apiRequest("/configs/game/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    });
    renderGameConfigValidation();
    if (showSuccess) {
      showMessage("Game config validation passed.");
    }
    return state.gameConfigValidation;
  } catch (error) {
    state.gameConfigValidation = { valid: false, error: error.message };
    renderGameConfigValidation();
    if (showSuccess) {
      showMessage(error.message, "error");
    }
    throw error;
  }
}
