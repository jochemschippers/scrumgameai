/**
 * Configuration Editor Action Handlers.
 * 
 * This module defines top-level user actions for the configuration editor:
 *   - Downloading the active config as a JSON file.
 *   - Copying the config JSON directly to the clipboard.
 *   - Importing a local configuration file into the editor.
 *   - Resetting the editor state back to baseline defaults.
 * 
 * Connections:
 *   - Imports: `state` from `state.js`, `DEFAULT_CONFIG` from `constants.js`, `$` from `utils.js`,
 *              `canonicalConfig` from `form.js`, `ensureShapeConsistencyFromState` from `board.js`,
 *              and `renderAll` from `render.js`.
 *   - Invoked by click event listeners in `main.js`.
 */

import { state } from './state.js';
import { DEFAULT_CONFIG } from './constants.js';
import { $ } from './utils.js';
import { canonicalConfig } from './form.js';
import { ensureShapeConsistencyFromState } from './board.js';
import { renderAll } from './render.js';

/**
 * Serializes the current form state into JSON and triggers a browser download.
 */
export function downloadJson() {
  const fileName = $("downloadFileNameInput").value.trim() || "custom_game_config.json";
  const blob = new Blob([JSON.stringify(canonicalConfig(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName.endsWith(".json") ? fileName : `${fileName}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/**
 * Copies the current canonical configuration JSON text to the user's system clipboard.
 */
export async function copyJson() {
  const text = JSON.stringify(canonicalConfig(), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    window.alert("Config JSON copied to clipboard.");
  } catch (error) {
    window.alert("Clipboard copy failed. Use Download JSON instead.");
  }
}

/**
 * Reads a JSON file uploaded by the user, validates/hydrates the shared mutable `state`,
 * ensures structural shapes are aligned, and triggers a full UI render.
 * 
 * @param {File} file - The uploaded file object.
 */
export function importJsonFile(file) {
  const reader = new FileReader();
  reader.onload = function () {
    try {
      const parsed = JSON.parse(String(reader.result || "{}"));
      Object.keys(state).forEach((key) => delete state[key]);
      Object.assign(state, structuredClone(parsed));
      delete $("downloadFileNameInput").dataset.touched;
      ensureShapeConsistencyFromState();
      renderAll();
    } catch (error) {
      window.alert("Could not parse that JSON file.");
    }
  };
  reader.readAsText(file);
}

/**
 * Resets the shared mutable `state` back to the default configuration values.
 */
export function resetDefaults() {
  Object.keys(state).forEach((key) => delete state[key]);
  Object.assign(state, structuredClone(DEFAULT_CONFIG));
  delete $("downloadFileNameInput").dataset.touched;
  renderAll();
}
