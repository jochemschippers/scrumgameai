import { state } from './state.js';
import { DEFAULT_CONFIG } from './constants.js';
import { $ } from './utils.js';
import { canonicalConfig } from './form.js';
import { ensureShapeConsistencyFromState } from './board.js';
import { renderAll } from './render.js';

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

export async function copyJson() {
  const text = JSON.stringify(canonicalConfig(), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    window.alert("Config JSON copied to clipboard.");
  } catch (error) {
    window.alert("Clipboard copy failed. Use Download JSON instead.");
  }
}

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

export function resetDefaults() {
  Object.keys(state).forEach((key) => delete state[key]);
  Object.assign(state, structuredClone(DEFAULT_CONFIG));
  delete $("downloadFileNameInput").dataset.touched;
  renderAll();
}
