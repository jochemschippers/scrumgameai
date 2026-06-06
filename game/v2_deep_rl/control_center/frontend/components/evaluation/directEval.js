/** Implement direct eval user-interface behavior. */

import { state } from '../../state/store.js';
import { $, showMessage, downloadJsonFile, downloadCsvFile, parseSeedList } from '../../utils/helpers.js';
import { escapeHtml } from '../../utils/formatting.js';
import { renderBarChart, renderTable } from '../../utils/charts.js';
import { apiRequest } from '../../api/client.js';
import { renderCheckpointComparison } from './comparison.js';

/** Render direct evaluation. */
export function renderDirectEvaluation() {
  const host = $("directEvaluationResult");
  const metricsHost = $("directEvaluationMetrics");
  if (!state.directEvaluation) {
    host.className = "empty-state";
    host.textContent = "Run a direct evaluation to see results here.";
    if (metricsHost) metricsHost.innerHTML = "";
    renderBarChart("directEvaluationChart", [], "episode_reward", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("directEvaluationMoneyChart", [], "ending_money", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("directEvaluationTurnsChart", [], "turns_played", "seed", "#5a9e7a", "#5a9e7a", "");
    renderTable("directEvaluationTable", [], []);
    return;
  }
  const summary = state.directEvaluation.summary || {};
  const winRate = ((1 - summary.bankruptcies / summary.episodes) * 100).toFixed(1);
  host.className = "list-card";
  host.innerHTML = `
    <h4>${escapeHtml(state.directEvaluation.checkpoint?.label || "Evaluation")} — ${summary.episodes} episodes</h4>
    <p>${escapeHtml(state.directEvaluation.game_config?.label || "-")}</p>
  `;
  if (metricsHost) {
    metricsHost.innerHTML = `
      <div class="metric-card">
        <span>Avg Reward</span>
        <strong>${Number(summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Win Rate</span>
        <strong>${winRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Bankruptcies</span>
        <strong>${summary.bankruptcies} / ${summary.episodes}</strong>
      </div>
      <div class="metric-card">
        <span>Avg Turns</span>
        <strong>${Number(summary.mean_turns_played).toFixed(1)}</strong>
      </div>
      <div class="metric-card">
        <span>Avg Loans Taken</span>
        <strong>${Number(summary.mean_loans_taken).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Reward Range</span>
        <strong>${Number(summary.min_reward).toFixed(1)} - ${Number(summary.max_reward).toFixed(1)}</strong>
      </div>
      <div class="metric-card">
        <span>Invalid Actions</span>
        <strong>${summary.invalid_actions}</strong>
      </div>
    `;
  }
  renderBarChart(
    "directEvaluationChart",
    state.directEvaluation.results || [],
    "episode_reward",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "episode reward by seed"
  );
  renderBarChart(
    "directEvaluationMoneyChart",
    state.directEvaluation.results || [],
    "ending_money",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "ending money by seed"
  );
  renderBarChart(
    "directEvaluationTurnsChart",
    state.directEvaluation.results || [],
    "turns_played",
    "seed",
    "#5a9e7a",
    "#5a9e7a",
    "turns played by seed"
  );
  renderTable(
    "directEvaluationTable",
    [
      { key: "seed", label: "Seed" },
      { key: "episode_reward", label: "Reward" },
      { key: "ending_money", label: "Bank" },
      { key: "turns_played", label: "Turns" },
      { key: "loan_turns", label: "Loan Turns" },
      { key: "loans_taken", label: "Loans" },
      { key: "invalid_action_count", label: "Invalid" },
      { key: "terminal_reason", label: "End" },
    ],
    state.directEvaluation.results || []
  );
}

/** Clear evaluation results. */
export function clearEvaluationResults() {
  state.directEvaluation = null;
  state.comparisonEvaluation = null;
  renderDirectEvaluation();
  renderCheckpointComparison();
}

/** Run direct evaluation. */
export async function runDirectEvaluation(event) {
  event.preventDefault();
  if (!state.activeCheckpointId || !state.activeGameConfigId) {
    showMessage("Select an active checkpoint and game config first.", "error");
    return;
  }
  state.directEvaluation = await apiRequest("/testing/evaluate", {
    method: "POST",
    body: JSON.stringify({
      checkpoint_id: state.activeCheckpointId,
      game_config_id: state.activeGameConfigId,
      seeds: parseSeedList($("testingSeedsInput").value),
    }),
  }, 120000);
  renderDirectEvaluation();
}

/** Export direct evaluation json. */
export function exportDirectEvaluationJson() {
  if (!state.directEvaluation) {
    showMessage("Run a direct evaluation first.", "error");
    return;
  }
  downloadJsonFile("direct_evaluation.json", state.directEvaluation);
  showMessage("Exported direct evaluation JSON.");
}

/** Export direct evaluation csv. */
export function exportDirectEvaluationCsv() {
  if (!state.directEvaluation) {
    showMessage("Run a direct evaluation first.", "error");
    return;
  }
  const rows = state.directEvaluation.results || [];
  downloadCsvFile(
    "direct_evaluation.csv",
    ["seed", "episode_reward", "ending_money", "loan_turns", "loans_taken", "invalid_action_count", "terminal_reason"],
    rows.map((row) => [
      row.seed,
      row.episode_reward,
      row.ending_money,
      row.loan_turns,
      row.loans_taken,
      row.invalid_action_count,
      row.terminal_reason || "completed",
    ])
  );
  showMessage("Exported direct evaluation CSV.");
}
