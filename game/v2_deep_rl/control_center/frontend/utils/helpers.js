import { state } from '../state/store.js';
import { escapeHtml } from './formatting.js';

export function $(id) {
  return document.getElementById(id);
}

export function showMessage(text, type = "success") {
  const box = $("globalMessage");
  if (state.messageTimer) {
    clearTimeout(state.messageTimer);
    state.messageTimer = null;
  }
  box.innerHTML = `
    <span>${escapeHtml(text)}</span>
    <button class="message-close" type="button" aria-label="Dismiss message">x</button>
  `;
  box.className = `message ${type}`;
  box.querySelector(".message-close")?.addEventListener("click", clearMessage);
  if (type !== "error") {
    state.messageTimer = setTimeout(clearMessage, 2600);
  }
}

export function clearMessage() {
  const box = $("globalMessage");
  if (state.messageTimer) {
    clearTimeout(state.messageTimer);
    state.messageTimer = null;
  }
  box.innerHTML = "";
  box.className = "message hidden";
}

export function clone(value) {
  return structuredClone(value);
}

export function parseJsonEditor(id) {
  return JSON.parse($(id).value);
}

export function normalizeProductKey(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function numberValue(inputId, fallback = 0) {
  const value = Number($(inputId).value);
  return Number.isFinite(value) ? value : fallback;
}

export function parseNumberList(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

export function parseSeedList(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

export function selectedGameConfig() {
  return state.gameConfigs.find((item) => item.id === state.activeGameConfigId) || null;
}

export function selectedTrainingConfig() {
  return state.trainingConfigs.find((item) => item.id === state.activeTrainingConfigId) || null;
}

export function selectedCheckpoint() {
  return state.checkpoints.find((item) => item.id === state.activeCheckpointId) || null;
}

export function checkpointByPath(pathValue) {
  return state.checkpoints.find((item) => item.path === pathValue) || null;
}

export function selectedProgressJob() {
  return state.jobs.find((item) => item.id === state.activeProgressJobId) || null;
}

export function selectedProgressRun() {
  return state.runs.find((item) => item.id === state.activeProgressRunId) || null;
}

export function jobForRunId(runId) {
  return state.jobs.find((item) => runLabelFromPath(item.run_dir) === runId) || null;
}

export function currentTrainingMode() {
  return $("trainModeSelect")?.value || "train";
}

export function runLabelFromPath(pathValue) {
  if (!pathValue) return "-";
  const normalized = String(pathValue).replaceAll("\\", "/").split("/");
  return normalized[normalized.length - 1] || pathValue;
}

export function downloadJsonFile(fileName, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadCsvFile(fileName, headers, rows) {
  const escape = (value) => {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = [headers.map(escape).join(",")]
    .concat(rows.map((row) => row.map(escape).join(",")))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadCheckpointFile(checkpoint) {
  if (!checkpoint?.id) {
    throw new Error("Select a brain before downloading.");
  }
  const anchor = document.createElement("a");
  anchor.href = `${state.apiBaseUrl}/checkpoints/${encodeURIComponent(checkpoint.id)}/download`;
  anchor.download = checkpoint.label || "checkpoint.pth";
  anchor.click();
}

export function buildOptions(selectId, items, valueKey = "id", labelKey = "label", emptyLabel = "None") {
  const select = $(selectId);
  const currentValue = select.value;
  select.innerHTML = "";
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    select.appendChild(option);
    return;
  }

  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item[valueKey];
    option.textContent = item[labelKey] || item.label || item.id || emptyLabel;
    select.appendChild(option);
  });

  if (items.some((item) => item[valueKey] === currentValue)) {
    select.value = currentValue;
  }
}
