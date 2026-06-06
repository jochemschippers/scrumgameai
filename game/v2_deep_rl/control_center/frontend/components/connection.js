/**
 * Backend Connection and Progress Polling Handler.
 * 
 * This module manages the connection lifecycle between the frontend interface and the FastAPI server.
 * It tries auto-connecting to candidate URLs, toggles manual input connection screens, and schedules
 * throttled background polling cycles (`_runPollCycle`) to fetch jobs, training progress, and autopilot metrics.
 * 
 * Connections:
 *   - Imports: Shared state, API clients, UI notifications, and component render functions.
 *   - Exported functions: `autoConnect`, `startProgressPolling`, and UI selectors. Called by `main.js`.
 */

import { state } from '../state/store.js';
import { apiRequest } from '../api/client.js';
import { showLoginScreen } from './auth.js';
import { $, showMessage, runLabelFromPath } from '../utils/helpers.js';
import { AUTO_CONNECT_URLS } from '../constants/defaults.js';
import { updateStatusCard, updateSummaryPills, renderContextCard } from './navigation.js';
import { renderJobs } from './training/jobs.js';
import { fetchTrainingProgress } from './training/progress.js';
import { fetchAutopilotData, renderAutopilotTrainingPanel } from './training/autopilot.js';
import { refreshCampaigns } from './training/campaigns.js';

let _pollInFlight = false;

export { AUTO_CONNECT_URLS };

/** Show connected ui. */
export function _showConnectedUi() {
  const manual = $("backendManualConnect");
  const actions = $("backendConnectedActions");
  if (manual) manual.style.display = "none";
  if (actions) actions.style.display = "";
}

/** Show manual connect ui. */
export function _showManualConnectUi() {
  const manual = $("backendManualConnect");
  const actions = $("backendConnectedActions");
  if (manual) manual.style.display = "";
  if (actions) actions.style.display = "none";
}

/** Handle try connect. */
export async function _tryConnect(url) {
  state.apiBaseUrl = url.replace(/\/$/, "");
  // Note: refreshAll will be called from main.js
}

/** Handle auto connect. */
export async function autoConnect() {
  for (const url of AUTO_CONNECT_URLS) {
    try {
      state.apiBaseUrl = url.replace(/\/$/, "");
      const health = await apiRequest("/health");
      state.health = health;
      _showConnectedUi();
      showMessage(`Connected to ${url}`);
      return true;
    } catch (_err) {
      // try next
    }
  }
  // All candidates failed. Show manual fields so user can enter a custom URL.
  _showManualConnectUi();
  state.health = null;
  updateStatusCard();
  showMessage("Could not auto-connect. Enter the backend URL manually.", "error");
  return false;
}

/** Run poll cycle. */
export async function _runPollCycle() {
  if (_pollInFlight) return;
  _pollInFlight = true;
  try {
    // Refresh the jobs list first. Subsequent fetches depend on the updated job state.
    const jobsPayload = await apiRequest("/jobs").catch(() => null);
    if (jobsPayload) {
      state.jobs = jobsPayload.items || [];
      renderJobs();
      updateSummaryPills();
    }
    await refreshCampaigns().catch(() => {});

    // Auto-advance to a new running training job when the tracked job is completed.
    // Skip if the user is actively inspecting a specific job. Replacing activeProgressJobId
    // would swap out that job's training charts for the running job's, causing a visible jump.
    const trackedJob = state.jobs.find((j) => j.id === state.activeProgressJobId);
    const userInspectingJob = state.activePage === "inspect" && state.activeJobDetailId != null;
    if (!userInspectingJob && (!state.activeProgressJobId || trackedJob?.status === "completed")) {
      const runningJob = state.jobs.find(
        (j) => j.status === "running" && ["train", "fine_tune"].includes(j.job_type)
      );
      if (runningJob) {
        state.activeProgressJobId = runningJob.id;
      }
    }

    // Determine autopilot run ID from existing state (stale-ok for this cycle).
    const runId =
      state.activeRunId ||
      (state.trainingProgress?.run_dir ? runLabelFromPath(state.trainingProgress.run_dir) : null) ||
      (state.jobDetail?.run_dir ? runLabelFromPath(state.jobDetail.run_dir) : null);

    // Fire progress + autopilot requests in parallel. They are independent of each other.
    await Promise.all([
      state.activeProgressJobId
        ? fetchTrainingProgress(state.activeProgressJobId, false).catch(() => {})
        : Promise.resolve(),
      runId
        ? fetchAutopilotData(runId).catch(() => {})
        : Promise.resolve(),
    ]);

    if (runId) renderAutopilotTrainingPanel();
  } finally {
    _pollInFlight = false;
  }
}

/** Start progress polling. */
export function startProgressPolling() {
  if (state.progressPollHandle) {
    clearTimeout(state.progressPollHandle);
  }
  /** Schedule schedule. */
  const schedule = async () => {
    await _runPollCycle();
    state.progressPollHandle = setTimeout(schedule, 5000);
  };
  state.progressPollHandle = setTimeout(schedule, 5000);
}
