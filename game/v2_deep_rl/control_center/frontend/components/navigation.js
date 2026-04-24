import { state } from '../state/store.js';
import { $, selectedGameConfig, selectedCheckpoint, selectedTrainingConfig } from '../utils/helpers.js';
import { escapeHtml, checkpointUiLabel } from '../utils/formatting.js';
import { PAGES } from '../constants/defaults.js';

export const pages = PAGES;

export function setPage(pageId) {
  state.activePage = pageId;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.page === pageId);
  });
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("is-active", page.id === `page-${pageId}`);
  });
  $("pageTitle").textContent = pages[pageId].title;
  $("pageSubtitle").textContent = pages[pageId].subtitle;
  $("contextPageUsage").textContent = pages[pageId].usage;
  renderContextCard();
}

export function updateStatusCard() {
  const card = $("backendStatusCard");
  if (!state.health) {
    card.className = "status-card status-muted";
    card.innerHTML = "<span>Not connected</span>";
    return;
  }
  card.className = "status-card status-ok";
  card.innerHTML = `<span>${state.health.status} | v${state.health.api_version}</span>`;
}

export function updateSummaryPills() {
  const activeGameConfig = state.gameConfigs.find((item) => item.id === state.activeGameConfigId);
  const activeCheckpoint = state.checkpoints.find((item) => item.id === state.activeCheckpointId);
  const visibleJobs = state.jobs.filter((job) => ["queued", "completed", "failed", "stopped"].includes(job.status));
  $("summaryRuleSignature").textContent = `Blueprint: ${activeGameConfig?.rule_signature || "-"}`;
  $("summaryCheckpointStatus").textContent = `Brain: ${activeCheckpoint?.checkpoint_format || "-"}`;
  $("summaryJobCount").textContent = `Jobs: ${visibleJobs.length}`;
}

export function renderContextCard() {
  const body = $("contextCardBody");
  const gameConfig = selectedGameConfig();
  const trainingConfig = selectedTrainingConfig();
  const checkpoint = selectedCheckpoint();
  const compatibilityText = state.compatibility
    ? `${state.compatibility.strict_resume_status} / ${state.compatibility.fine_tune_status}`
    : "not checked";
  body.innerHTML = `
    <div class="context-item">
      <span>Active Blueprint</span>
      <strong>${escapeHtml(gameConfig?.label || "-")}</strong>
    </div>
    <div class="context-item">
      <span>Active Training Profile</span>
      <strong>${escapeHtml(trainingConfig?.label || "-")}</strong>
    </div>
    <div class="context-item">
      <span>Active Brain</span>
      <strong>${escapeHtml(checkpoint ? checkpointUiLabel(checkpoint) : "-")}</strong>
    </div>
    <div class="context-item">
      <span>Compatibility Status</span>
      <strong>${escapeHtml(compatibilityText)}</strong>
    </div>
  `;
}
