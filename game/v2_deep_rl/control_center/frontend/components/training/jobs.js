/** Implement jobs user-interface behavior. */

import { state } from '../../state/store.js';
import { $, showMessage, selectedGameConfig, selectedTrainingConfig, selectedCheckpoint, currentTrainingMode, runLabelFromPath, buildOptions } from '../../utils/helpers.js';
import { escapeHtml, checkpointUiLabel, checkpointCompatibilityTone as compatTone } from '../../utils/formatting.js';
import { apiRequest, isGuest } from '../../api/client.js';
import { renderTrainingProgress, fetchTrainingProgress } from './progress.js';
import { refreshCampaigns } from './campaigns.js';
import { updateSummaryPills, setPage } from '../navigation.js';

const JOBS_PER_PAGE = 5;

/** Render jobs. */
export function renderJobs() {
  const container = $("jobsList");
  const paginationContainer = $("jobsPagination");
  container.innerHTML = "";
  if (paginationContainer) paginationContainer.innerHTML = "";

  const visibleJobs = state.jobs.filter((job) => ["queued", "running", "completed", "failed", "stopped"].includes(job.status));

  if (
    state.activeProgressJobId &&
    !state.jobs.some((job) => job.id === state.activeProgressJobId)
  ) {
    state.activeProgressJobId = null;
    state.trainingProgress = null;
  }

  if (!visibleJobs.length) {
    state.jobsPage = 0;
    container.innerHTML = `<div class="empty-state">No jobs yet.</div>`;
    if (!state.activeProgressJobId) {
      renderTrainingProgress();
    }
    return;
  }

  if (!state.activeProgressJobId) {
    const userInspectingJob = state.activePage === "inspect" && state.activeJobDetailId != null;
    if (!userInspectingJob) {
      const preferredJob = state.jobs.find((job) => ["running", "queued"].includes(job.status) && ["train", "fine_tune"].includes(job.job_type))
        || state.jobs.find((job) => ["train", "fine_tune"].includes(job.job_type));
      if (preferredJob) {
        state.activeProgressJobId = preferredJob.id;
      }
    }
  }

  // Clamp page to valid range after any job dismissals
  const totalPages = Math.ceil(visibleJobs.length / JOBS_PER_PAGE);
  if (state.jobsPage >= totalPages) state.jobsPage = totalPages - 1;
  if (state.jobsPage < 0) state.jobsPage = 0;

  const pageStart = state.jobsPage * JOBS_PER_PAGE;
  const pageJobs = visibleJobs.slice(pageStart, pageStart + JOBS_PER_PAGE);

  pageJobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "list-card";
    const queuedTrainingJobs = visibleJobs
      .filter((item) => ["train", "fine_tune"].includes(item.job_type) && item.status === "queued")
      .slice()
      .reverse();
    const queueIndex = queuedTrainingJobs.findIndex((item) => item.id === job.id);
    const queueTag = queueIndex >= 0 ? `<span class="tag">queue #${queueIndex + 1}</span>` : "";
    const statusTone = job.status === "completed"
      ? "good"
      : job.status === "failed" || job.status === "stopped"
        ? "bad"
        : job.status === "running"
          ? "warn"
          : "";
    const guestAttr = isGuest() ? 'disabled title="Guests cannot perform this action"' : "";
    const stopButton = ["queued", "running"].includes(job.status)
      ? `<button class="button secondary stop-job-button" data-job-id="${job.id}" type="button" ${guestAttr}>Stop</button>`
      : "";
    const dismissButton = ["completed", "failed", "stopped"].includes(job.status)
      ? `<button class="button secondary dismiss-job-button" data-job-id="${job.id}" type="button" ${guestAttr}>Dismiss</button>`
      : "";
    const inspectButton = `<button class="button secondary open-inspect-job-button" data-job-id="${job.id}" type="button">Open Inspect</button>`;
    card.innerHTML = `
      <h4>Job #${job.id} - ${escapeHtml(job.job_type)}</h4>
      <div class="card-meta">
        <span class="tag ${statusTone}">${job.status}</span>
        ${queueTag}
        <span class="tag">${job.created_at}</span>
        <span class="tag">${runLabelFromPath(job.run_dir)}</span>
      </div>
      <div class="inline-actions">
        ${inspectButton}
        ${stopButton}
        ${dismissButton}
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".stop-job-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await apiRequest(`/jobs/${button.dataset.jobId}/stop`, { method: "POST" });
        showMessage(`Stopped job ${button.dataset.jobId}.`);
        await refreshJobs();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".open-inspect-job-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.dispatchEvent(new CustomEvent("openInspectForJob", {
        detail: { jobId: Number(button.dataset.jobId), announce: true },
      }));
    });
  });

  container.querySelectorAll(".dismiss-job-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const jobId = Number(button.dataset.jobId);
      try {
        await apiRequest(`/jobs/${jobId}`, { method: "DELETE" });
        if (state.activeJobDetailId === jobId) {
          state.activeJobDetailId = null;
          state.jobDetail = null;
          state.jobLog = null;
          renderJobDetail();
          renderJobLog();
        }
        showMessage(`Dismissed job ${jobId}.`);
        await refreshJobs();
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  // Render pagination controls
  if (paginationContainer && totalPages > 1) {
    const prevDisabled = state.jobsPage === 0 ? "disabled" : "";
    const nextDisabled = state.jobsPage >= totalPages - 1 ? "disabled" : "";
    paginationContainer.innerHTML = `
      <button class="button secondary jobs-page-prev" type="button" ${prevDisabled}>&#8592; Prev</button>
      <span class="jobs-page-label">Page ${state.jobsPage + 1} of ${totalPages}</span>
      <button class="button secondary jobs-page-next" type="button" ${nextDisabled}>Next &#8594;</button>
    `;
    paginationContainer.querySelector(".jobs-page-prev")?.addEventListener("click", () => {
      if (state.jobsPage > 0) {
        state.jobsPage -= 1;
        renderJobs();
      }
    });
    paginationContainer.querySelector(".jobs-page-next")?.addEventListener("click", () => {
      if (state.jobsPage < totalPages - 1) {
        state.jobsPage += 1;
        renderJobs();
      }
    });
  }

  if (state.activeProgressJobId) {
    fetchTrainingProgress(state.activeProgressJobId, false).catch(() => {});
  } else {
    renderTrainingProgress();
  }
}

/** Render job detail. */
export function renderJobDetail() {
  const label = $("jobDetailLabel");
  const container = $("jobDetailCard");
  if (!state.jobDetail) {
    label.textContent = "No job selected";
    container.className = "empty-state";
    container.textContent = "Select a job to inspect its details.";
    return;
  }
  label.textContent = `Job #${state.jobDetail.id}`;
  const payload = state.jobDetail.payload || {};
  container.className = "list-card";
  const runId = runLabelFromPath(state.jobDetail.run_dir);
  const resumeFrom = payload.resume_from || "";
  const resumeCheckpoint = resumeFrom
    ? state.checkpoints.find((c) => c.path === resumeFrom || c.id === resumeFrom) || null
    : null;
  const resumeLabel = resumeCheckpoint
    ? checkpointUiLabel(resumeCheckpoint)
    : resumeFrom
      ? resumeFrom.replace(/\\/g, "/").split("/").slice(-3).join("/")
      : "";
  container.innerHTML = `
    <h4>Job #${state.jobDetail.id} - ${escapeHtml(state.jobDetail.job_type)}</h4>
    <div class="card-meta">
      <span class="tag">${escapeHtml(state.jobDetail.status)}</span>
      <span class="tag">${escapeHtml(state.jobDetail.created_at || "-")}</span>
      <span class="tag">${escapeHtml(runLabelFromPath(state.jobDetail.run_dir))}</span>
    </div>
    <div class="card-meta">
      ${payload.resume_mode ? `<span class="tag">resume ${escapeHtml(payload.resume_mode)}</span>` : "<span class='tag'>new run</span>"}
      ${payload.episodes ? `<span class="tag">${escapeHtml(String(payload.episodes))} episodes</span>` : ""}
      ${payload.evaluation_episodes ? `<span class="tag">${escapeHtml(String(payload.evaluation_episodes))} eval eps</span>` : ""}
      ${payload.autopilot_after_completion ? "<span class='tag'>autopilot</span>" : ""}
    </div>
    ${resumeLabel ? `<div class="checkpoint-subtitle path-wrap">From: ${escapeHtml(resumeLabel)}</div>` : ""}
    ${state.jobDetail.error_message ? `<p>${escapeHtml(state.jobDetail.error_message)}</p>` : ""}
    <div class="inline-actions">
      ${state.jobDetail.run_dir ? `<button class="button secondary open-job-run-button" data-run-id="${escapeHtml(runId)}" type="button">Open Run</button>` : ""}
      <button class="button secondary refresh-job-log-button" data-job-id="${state.jobDetail.id}" type="button">Refresh Log</button>
    </div>
  `;

  container.querySelector(".open-job-run-button")?.addEventListener("click", async (event) => {
    try {
      const { fetchRunDetail } = await import('./progress.js');
      await fetchRunDetail(event.target.dataset.runId, true);
      setPage("evaluate");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });

  container.querySelector(".refresh-job-log-button")?.addEventListener("click", async (event) => {
    try {
      state.jobLog = await apiRequest(`/jobs/${Number(event.target.dataset.jobId)}/log`);
      renderJobLog();
      showMessage("Refreshed job log.");
    } catch (error) {
      showMessage(error.message, "error");
    }
  });
}

/** Render job log. */
export function renderJobLog() {
  const container = $("jobLogCard");
  if (!state.jobLog) {
    container.className = "empty-state";
    container.textContent = "Select a job to inspect the latest log lines.";
    return;
  }
  container.className = "log-card";
  container.textContent = (state.jobLog.lines || []).join("\n") || "(no log yet)";
}

/** Render training selection summary. */
export function renderTrainingSelectionSummary() {
  const container = $("trainingSelectionSummary");
  const gameConfig = selectedGameConfig();
  const trainingConfig = selectedTrainingConfig();
  const checkpoint = selectedCheckpoint();
  const mode = currentTrainingMode();
  const resumeText =
    mode === "train"
      ? "New training ignores the active checkpoint."
        : checkpoint
          ? `Resume source: ${checkpointUiLabel(checkpoint)}`
        : "Select an active checkpoint for resume or fine-tune.";

  container.innerHTML = `
    <h4>Current Selection</h4>
    <div class="card-meta">
      <span class="tag">Game: ${escapeHtml(gameConfig?.label || "-")}</span>
      <span class="tag">Training: ${escapeHtml(trainingConfig?.label || "-")}</span>
      <span class="tag">Mode: ${escapeHtml(mode === "train" ? "new training" : mode === "resume" ? "strict resume" : "fine-tune")}</span>
    </div>
    <p>${escapeHtml(resumeText)}</p>
  `;
}

/** Render training preflight. */
export function renderTrainingPreflight() {
  const container = $("trainingPreflightCard");
  const mode = currentTrainingMode();
  const checkpoint = selectedCheckpoint();
  const gameConfig = selectedGameConfig();
  if (mode === "train") {
    state.trainingPreflight = null;
    container.className = "list-card";
    container.innerHTML = `
      <h4>Launch Check</h4>
      <p>New training will start from random weights.</p>
      <div class="card-meta">
        <span class="tag good">safe to launch</span>
        <span class="tag">game ${escapeHtml(gameConfig?.label || "-")}</span>
      </div>
    `;
    return;
  }

  if (!gameConfig || !checkpoint) {
    container.className = "empty-state";
    container.textContent = "Select both a game config and a checkpoint to validate resume or fine-tune.";
    return;
  }

  if (!state.trainingPreflight) {
    container.className = "empty-state";
    container.textContent = "Checking compatibility for the selected mode...";
    return;
  }

  const strictOkay = String(state.trainingPreflight.strict_resume_status || "").includes("compatible");
  const fineTuneOkay = String(state.trainingPreflight.fine_tune_status || "").includes("compatible");
  const activeOkay = mode === "resume" ? strictOkay : fineTuneOkay;
  const launchTone = activeOkay ? "good" : "bad";
  const activeLabel = mode === "resume" ? "strict resume" : "fine-tune";
  const guidance = activeOkay
    ? `The current ${activeLabel} pair looks usable.`
    : `The current ${activeLabel} pair is not safe to launch.`;

  container.className = "list-card";
  container.innerHTML = `
    <h4>Launch Check</h4>
    <p>${escapeHtml(guidance)}</p>
    <div class="card-meta">
      <span class="tag ${launchTone}">${escapeHtml(activeLabel)} ${activeOkay ? "ready" : "blocked"}</span>
      <span class="tag ${compatTone(state.trainingPreflight.strict_resume_status)}">strict ${escapeHtml(state.trainingPreflight.strict_resume_status)}</span>
      <span class="tag ${compatTone(state.trainingPreflight.fine_tune_status)}">fine-tune ${escapeHtml(state.trainingPreflight.fine_tune_status)}</span>
    </div>
    <div class="card-meta">
      <span class="tag">brain ${escapeHtml(checkpointUiLabel(checkpoint))}</span>
    </div>
  `;
}

/** Refresh jobs. */
export async function refreshJobs() {
  const payload = await apiRequest("/jobs");
  state.jobs = payload.items || [];
  renderJobs();
  updateSummaryPills();
  renderTrainingSelectionSummary();
  await refreshCampaigns().catch(() => {});
  if (state.activeJobDetailId) {
    await fetchJobDetail(state.activeJobDetailId, false).catch(() => {});
  }
}

/** Fetch job detail. */
export async function fetchJobDetail(jobId, announce = false) {
  state.activeJobDetailId = Number(jobId);
  state.jobDetail = await apiRequest(`/jobs/${state.activeJobDetailId}`);
  renderJobDetail();
  try {
    state.jobLog = await apiRequest(`/jobs/${state.activeJobDetailId}/log`);
  } catch (_error) {
    state.jobLog = null;
  }
  renderJobLog();
  if (announce) {
    showMessage(`Loaded job ${jobId}.`);
  }
}

/** Refresh training preflight. */
export async function refreshTrainingPreflight() {
  const mode = currentTrainingMode();
  const checkpoint = selectedCheckpoint();
  const gameConfig = selectedGameConfig();
  if (mode === "train") {
    state.trainingPreflight = null;
    renderTrainingPreflight();
    return;
  }
  if (!checkpoint || !gameConfig) {
    state.trainingPreflight = null;
    renderTrainingPreflight();
    return;
  }
  try {
    state.trainingPreflight = await apiRequest(
      `/checkpoints/${encodeURIComponent(checkpoint.id)}/compatibility?game_config_id=${encodeURIComponent(gameConfig.id)}`
    );
  } catch (_error) {
    state.trainingPreflight = null;
  }
  renderTrainingPreflight();
}

/** Queue training job. */
export async function queueTrainingJob(event) {
  event.preventDefault();
  const mode = currentTrainingMode();
  const gameConfig = selectedGameConfig();
  const trainingConfig = selectedTrainingConfig();
  const checkpoint = selectedCheckpoint();

  if (!gameConfig || !trainingConfig) {
    showMessage("Select both a game config and training config first.", "error");
    return;
  }
  if (mode !== "train" && !checkpoint) {
    showMessage("Select an active checkpoint for resume or fine-tune.", "error");
    return;
  }

  const autopilot = $("autopilotAfterCompletionInput")?.checked || false;
  const campaignEnabled = $("campaignEnabledInput")?.checked || false;
  const campaignId = $("campaignIdInput")?.value.trim() || null;
  const campaignVariations = Number($("campaignVariationsInput")?.value || 5);

  const jobPayload = {
    job_type: mode === "fine_tune" ? "fine_tune" : "train",
    game_config_id: gameConfig.id,
    training_config_id: trainingConfig.id,
    resume_mode: mode !== "train" ? mode : undefined,
    resume_from: mode !== "train" && checkpoint ? checkpoint.path : undefined,
    autopilot_after_completion: autopilot,
  };

  const job = await apiRequest("/jobs/train", {
    method: "POST",
    body: JSON.stringify(jobPayload),
  });
  showMessage(`Queued ${job.job_type} job #${job.id}.`);

  if (campaignEnabled && campaignId) {
    try {
      await apiRequest("/campaigns", {
        method: "POST",
        body: JSON.stringify({
          campaign_id: campaignId,
          base_job_id: job.id,
          max_variations: campaignVariations,
        }),
      });
      showMessage(`Campaign "${campaignId}" started.`);
    } catch (error) {
      showMessage(`Job queued but campaign failed: ${error.message}`, "error");
    }
  }

  await refreshJobs();
}

/** Queue robustness job. */
export async function queueRobustnessJob(event) {
  event.preventDefault();
  const runId = $("robustnessRunSelect").value;
  const selectedRun = state.runs.find((item) => item.id === runId);
  if (!selectedRun) {
    showMessage("Select a run for robustness evaluation.", "error");
    return;
  }

  const job = await apiRequest("/jobs/evaluate", {
    method: "POST",
    body: JSON.stringify({
      job_type: "robustness",
      run_dir: selectedRun.path,
    }),
  });
  showMessage(`Queued robustness job #${job.id}.`);
  await refreshJobs();
}
