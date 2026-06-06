/** Implement formatting behavior for the utils package. */

import { state } from '../state/store.js';

/** Format json. */
export function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

/** Handle escape html. */
export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/** Handle checkpoint ui label. */
export function checkpointUiLabel(checkpoint) {
  if (!checkpoint) return "";
  let source = checkpoint.source_type || "checkpoint";
  if (checkpoint.source_type === "run" && checkpoint.source_run) {
    source = formatRunSourceLabel(checkpoint.source_run);
  } else if (checkpoint.source_type === "current_artifacts") {
    source = "Current Artifacts";
  } else if (checkpoint.source_type === "reference_v1") {
    source = "Reference V1";
  } else if (checkpoint.source_type === "playable_model_v1") {
    source = "Playable Model V1";
  }

  const typeLabel = checkpoint.checkpoint_type || "checkpoint";
  const episodeLabel = checkpoint.episode != null ? ` ep${Number(checkpoint.episode).toLocaleString()}` : "";
  const legacySuffix = checkpoint.checkpoint_format === "legacy" ? " [legacy]" : "";
  return `${source} | ${typeLabel}${episodeLabel}${legacySuffix}`;
}

/** Format run source label. */
export function formatRunSourceLabel(runName) {
  const raw = String(runName || "").trim();
  const match = raw.match(/^run_(\d{4}-\d{2}-\d{2})_(\d{4})(?:_(.+))?$/);
  if (!match) return raw || "Run";

  const [, datePart, timePart, remainder] = match;
  const timeLabel = `${timePart.slice(0, 2)}:${timePart.slice(2)}`;
  const suffixParts = String(remainder || "")
    .split("_")
    .filter(Boolean);

  let iteration = "";
  if (suffixParts.length && /^\d{2}$/.test(suffixParts[suffixParts.length - 1])) {
    iteration = ` (${Number(suffixParts.pop()) + 1})`;
  }

  const customName = suffixParts.join(" ").trim();
  return customName
    ? `${datePart} ${timeLabel}${iteration} | ${customName}`
    : `${datePart} ${timeLabel}${iteration}`;
}

/** Handle checkpoint compatibility tone. */
export function checkpointCompatibilityTone(status) {
  const value = String(status || "").toLowerCase();
  if (value.includes("mismatch") || value.includes("incompatible")) return "bad";
  if (value.includes("compatible")) return "good";
  if (value.includes("unknown")) return "warn";
  return "";
}

/** Handle checkpoint category. */
export function checkpointCategory(checkpoint) {
  const type = String(checkpoint?.checkpoint_type || "").toLowerCase();
  const label = String(checkpoint?.label || "").toLowerCase();
  if (type.includes("best") || label.includes("best")) return "best";
  if (type.includes("final") || label.includes("final")) return "final";
  if (type.includes("episode") || label.includes("checkpoint_episode")) return "intermediate";
  return "other";
}

/** Handle checkpoint group label. */
export function checkpointGroupLabel(checkpoint) {
  if (checkpoint.source_type === "run") {
    return checkpoint.source_run || "run";
  }
  if (checkpoint.source_type === "current_artifacts") {
    return "Current Artifacts";
  }
  if (checkpoint.source_type === "reference_v1") {
    return "Reference V1";
  }
  if (checkpoint.source_type === "playable_model_v1") {
    return "PlayableModelV1";
  }
  return checkpoint.source_type || "Other";
}

/** Return whether model selectable. */
export function isModelSelectable(checkpoint) {
  return checkpointCategory(checkpoint) !== "intermediate";
}

/** Handle sidebar checkpoint options. */
export function sidebarCheckpointOptions() {
  const includeAll = Boolean(state.includeCheckpointSelections);
  return state.checkpoints
    .filter((item) => includeAll || isModelSelectable(item))
    .map((item) => ({ ...item, ui_label: checkpointUiLabel(item) }))
    .sort((left, right) => left.ui_label.localeCompare(right.ui_label));
}

/** Format number. */
export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

/** Format currency. */
export function formatCurrency(value) {
  const amount = Number(value || 0);
  return `${amount < 0 ? "-" : ""}€${Math.abs(Math.round(amount)).toLocaleString()}`;
}
