import { state } from '../../state/store.js';
import { $, showMessage, selectedProgressJob, selectedProgressRun, runLabelFromPath, checkpointByPath, buildOptions } from '../../utils/helpers.js';
import { escapeHtml, formatNumber } from '../../utils/formatting.js';
import { renderLineChart } from '../../utils/charts.js';
import { apiRequest } from '../../api/client.js';

export function renderRuns() {
  $("runCount").textContent = `${state.runs.length}`;
  buildOptions("robustnessRunSelect", state.runs);
  const container = $("runsList");
  container.innerHTML = "";
  state.runs.forEach((run) => {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <h4>${run.label}</h4>
      <p>${run.run_notes || "No notes"}</p>
      <div class="card-meta">
        <span class="tag">${run.created_at || "unknown date"}</span>
        <span class="tag">${run.average_reward_per_episode ?? "-"} avg reward</span>
        <span class="tag">${run.bankruptcy_rate ?? "-"} bankruptcy</span>
      </div>
      <div class="inline-actions">
        <button class="button secondary view-run-button" data-run-id="${run.id}" type="button">View Run</button>
        <button class="button secondary open-inspect-run-button" data-run-id="${run.id}" type="button">Open Inspect</button>
        ${run.best_checkpoint_path ? `<button class="button secondary use-run-best-button" data-run-id="${run.id}" type="button">Use Best Brain</button>` : ""}
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".view-run-button").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await fetchRunDetail(button.dataset.runId, true);
      } catch (error) {
        showMessage(error.message, "error");
      }
    });
  });

  container.querySelectorAll(".open-inspect-run-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.dispatchEvent(new CustomEvent("openInspectForRun", {
        detail: { runId: button.dataset.runId, announce: true },
      }));
    });
  });

  container.querySelectorAll(".use-run-best-button").forEach((button) => {
    button.addEventListener("click", () => {
      const run = state.runs.find((item) => item.id === button.dataset.runId);
      const checkpoint = checkpointByPath(run?.best_checkpoint_path);
      if (!checkpoint) {
        showMessage("Best checkpoint for this run is not available in the current catalog.", "error");
        return;
      }
      document.dispatchEvent(new CustomEvent("activateCheckpoint", {
        detail: { checkpointId: checkpoint.id, runLabel: run.label },
      }));
    });
  });
}

export function renderTrainingProgress() {
  const progressLabel = $("trainingProgressJobLabel");
  const container = $("trainingProgressCard");
  const progress = state.trainingProgress;
  const job = selectedProgressJob();
  const run = selectedProgressRun();

  if (state.activeProgressRunId && !run && !job) {
    state.activeProgressRunId = null;
    state.trainingProgress = null;
  }

  if (!progress) {
    progressLabel.textContent = "No job selected";
    container.className = "empty-state";
    container.textContent = "Open a training job or finished run to see progress.";
    renderLineChart("trainingRewardChart", [], "rolling_average_reward", "#4cb782", "");
    renderLineChart("trainingEvalChart", [], "average_reward", "#7fb7ff", "");
    return;
  }

  const percent = Math.max(0, Math.min(100, Math.round((progress.progress_ratio || 0) * 100)));
  const latest = progress.latest_training_row || {};
  const latestEval = progress.latest_evaluation_row || {};
  const progressStatus = job?.status || progress.status || "completed";
  progressLabel.textContent = job
    ? `Job #${job.id} | ${job.job_type} | ${job.status}`
    : run
      ? `${run.id} | ${progressStatus}`
      : `${runLabelFromPath(progress.run_dir)} | ${progressStatus}`;
  const runPath = job?.run_dir || run?.path || progress.run_dir || "";
  const runName = runLabelFromPath(runPath);
  const completedEpisodes = Number.isFinite(progress.completed_episodes)
    ? progress.completed_episodes
    : (progress.start_episode ? Math.max(0, (progress.latest_episode || 0) - progress.start_episode + 1) : (progress.latest_episode || 0));

  container.className = "progress-stack";
  container.innerHTML = `
    <div class="list-card">
      <h4>${escapeHtml(runName || "Training run")}</h4>
      <p>${progress.total_episodes ? `${completedEpisodes} / ${progress.total_episodes} episodes this run` : `${progress.latest_episode} episodes logged`}</p>
      <div class="progress-track">
        <div class="progress-fill" style="width: ${percent}%"></div>
      </div>
      <div class="card-meta">
        <span class="tag">${percent}%</span>
        <span class="tag">epsilon ${formatNumber(latest.epsilon, 4)}</span>
        <span class="tag">status ${escapeHtml(progressStatus)}</span>
      </div>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><span>Rolling Reward</span><strong>${formatNumber(latest.rolling_average_reward)}</strong></div>
      <div class="metric-card"><span>Episode Reward</span><strong>${formatNumber(latest.episode_reward)}</strong></div>
      <div class="metric-card"><span>Recent Loss</span><strong>${formatNumber(latest.mean_recent_loss)}</strong></div>
      <div class="metric-card"><span>Average Ending Money</span><strong>${formatNumber(latest.average_ending_money)}</strong></div>
      <div class="metric-card"><span>Eval Reward</span><strong>${formatNumber(latestEval.average_reward)}</strong></div>
      <div class="metric-card"><span>Eval Bankruptcy Rate</span><strong>${formatNumber(latestEval.bankruptcy_rate)}</strong></div>
    </div>
  `;

  renderLineChart(
    "trainingRewardChart",
    progress.training_series || [],
    "rolling_average_reward",
    "#4cb782",
    "rolling average reward over logged training points"
  );
  renderLineChart(
    "trainingEvalChart",
    progress.evaluation_series || [],
    "average_reward",
    "#7fb7ff",
    "evaluation reward over periodic evaluation points"
  );
}

export async function fetchTrainingProgress(jobId, announce = false) {
  state.activeProgressJobId = Number(jobId);
  state.activeProgressRunId = null;
  state.trainingProgress = await apiRequest(`/jobs/${state.activeProgressJobId}/progress`);
  renderTrainingProgress();
  if (announce) {
    showMessage(`Loaded progress for job ${jobId}.`);
  }
}

export async function fetchRunProgress(runId, announce = false) {
  state.activeProgressRunId = runId;
  state.activeProgressJobId = null;
  state.trainingProgress = await apiRequest(`/runs/${encodeURIComponent(runId)}/progress`);
  renderTrainingProgress();
  if (announce) {
    showMessage(`Loaded progress for run ${runId}.`);
  }
}

export async function fetchRunDetail(runId, announce = false) {
  state.activeRunId = runId;
  state.runRating = null;
  state.runDetail = await apiRequest(`/runs/${encodeURIComponent(runId)}`);
  // renderRunDetail stays in main.js — dispatch event so it can be called there
  document.dispatchEvent(new CustomEvent("runDetailLoaded"));
  // Fetch rating in background
  apiRequest(`/runs/${encodeURIComponent(runId)}/rating`).then((rating) => {
    state.runRating = rating;
    document.dispatchEvent(new CustomEvent("runDetailLoaded"));
  }).catch(() => {});
  if (announce) {
    showMessage(`Loaded run ${runId}.`);
  }
}
