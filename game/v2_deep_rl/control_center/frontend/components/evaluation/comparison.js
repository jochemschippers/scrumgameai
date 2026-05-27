import { state } from '../../state/store.js';
import { $, showMessage, downloadJsonFile, downloadCsvFile, parseSeedList, buildOptions } from '../../utils/helpers.js';
import { escapeHtml, sidebarCheckpointOptions } from '../../utils/formatting.js';
import { renderBarChart, renderTable } from '../../utils/charts.js';
import { apiRequest } from '../../api/client.js';

export function renderCheckpointComparison() {
  const host = $("checkpointCompareResult");
  const metricsHost = $("comparisonMetrics");
  const compareOptions = sidebarCheckpointOptions()
    .filter((item) => item.id !== state.activeCheckpointId);
  buildOptions("compareCheckpointSelect", compareOptions, "id", "ui_label", "No comparison brains");
  if (!state.comparisonEvaluation) {
    host.className = "empty-state";
    host.textContent = "Compare the selected right-side brain against the active brain from the top bar.";
    if (metricsHost) metricsHost.innerHTML = "";
    renderBarChart("comparisonChart", [], "reward_delta", "seed", "#6b8aa3", "#cc5f5f", "");
    renderBarChart("comparisonMoneyChart", [], "bank_delta", "seed", "#6b8aa3", "#cc5f5f", "");
    renderTable("comparisonTable", [], []);
    return;
  }
  const left = state.comparisonEvaluation.left;
  const right = state.comparisonEvaluation.right;
  const comparisonRows = (left.results || []).map((leftRow, index) => {
    const rightRow = right.results?.[index] || {};
    return {
      seed: leftRow.seed,
      left_reward: leftRow.episode_reward,
      right_reward: rightRow.episode_reward,
      reward_delta: Number(leftRow.episode_reward || 0) - Number(rightRow.episode_reward || 0),
      left_bank: leftRow.ending_money,
      right_bank: rightRow.ending_money,
      bank_delta: Number(leftRow.ending_money || 0) - Number(rightRow.ending_money || 0),
      left_turns: leftRow.turns_played,
      right_turns: rightRow.turns_played,
      left_terminal: leftRow.terminal_reason || "completed",
      right_terminal: rightRow.terminal_reason || "completed",
    };
  });
  host.className = "list-card";
  host.innerHTML = `
    <h4>${escapeHtml(left.checkpoint.label)} <span style="color:var(--muted)">vs</span> ${escapeHtml(right.checkpoint.label)}</h4>
    <p>Left is the active brain. Positive delta means left wins.</p>
  `;
  if (metricsHost) {
    const leftWinRate = ((1 - left.summary.bankruptcies / left.summary.episodes) * 100).toFixed(1);
    const rightWinRate = ((1 - right.summary.bankruptcies / right.summary.episodes) * 100).toFixed(1);
    metricsHost.innerHTML = `
      <div class="metric-card">
        <span>Delta Reward</span>
        <strong>${Number(state.comparisonEvaluation.delta_mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Delta Bank</span>
        <strong>${Number(state.comparisonEvaluation.delta_mean_ending_money).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Left Reward</span>
        <strong>${Number(left.summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Right Reward</span>
        <strong>${Number(right.summary.mean_reward).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Left Win Rate</span>
        <strong>${leftWinRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Right Win Rate</span>
        <strong>${rightWinRate}%</strong>
      </div>
      <div class="metric-card">
        <span>Left Bank</span>
        <strong>${Number(left.summary.mean_ending_money).toFixed(2)}</strong>
      </div>
      <div class="metric-card">
        <span>Right Bank</span>
        <strong>${Number(right.summary.mean_ending_money).toFixed(2)}</strong>
      </div>
    `;
  }
  renderBarChart(
    "comparisonChart",
    comparisonRows,
    "reward_delta",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "left minus right reward by seed"
  );
  renderBarChart(
    "comparisonMoneyChart",
    comparisonRows,
    "bank_delta",
    "seed",
    "#6b8aa3",
    "#cc5f5f",
    "left minus right bank by seed"
  );
  renderTable(
    "comparisonTable",
    [
      { key: "seed", label: "Seed" },
      { key: "left_reward", label: "L Reward" },
      { key: "right_reward", label: "R Reward" },
      { key: "reward_delta", label: "Delta Reward" },
      { key: "left_bank", label: "L Bank" },
      { key: "right_bank", label: "R Bank" },
      { key: "bank_delta", label: "Delta Bank" },
      { key: "left_turns", label: "L Turns" },
      { key: "right_turns", label: "R Turns" },
      { key: "left_terminal", label: "L End" },
      { key: "right_terminal", label: "R End" },
    ],
    comparisonRows
  );
}

export async function runCheckpointComparison(event) {
  event.preventDefault();
  const rightCheckpointId = $("compareCheckpointSelect").value;
  if (!state.activeCheckpointId || !rightCheckpointId || !state.activeGameConfigId) {
    showMessage("Select an active brain, a comparison brain, and a blueprint first.", "error");
    return;
  }
  state.comparisonEvaluation = await apiRequest("/testing/compare", {
    method: "POST",
    body: JSON.stringify({
      left_checkpoint_id: state.activeCheckpointId,
      right_checkpoint_id: rightCheckpointId,
      game_config_id: state.activeGameConfigId,
      seeds: parseSeedList($("compareSeedsInput").value),
    }),
  }, 120000);
  renderCheckpointComparison();
}

export function exportComparisonJson() {
  if (!state.comparisonEvaluation) {
    showMessage("Run a checkpoint comparison first.", "error");
    return;
  }
  downloadJsonFile("checkpoint_comparison.json", state.comparisonEvaluation);
  showMessage("Exported checkpoint comparison JSON.");
}

export function exportComparisonCsv() {
  if (!state.comparisonEvaluation) {
    showMessage("Run a checkpoint comparison first.", "error");
    return;
  }
  const leftRows = state.comparisonEvaluation.left?.results || [];
  const rightRows = state.comparisonEvaluation.right?.results || [];
  downloadCsvFile(
    "checkpoint_comparison.csv",
    ["seed", "left_reward", "right_reward", "reward_delta", "left_bank", "right_bank", "bank_delta", "left_terminal", "right_terminal"],
    leftRows.map((leftRow, index) => {
      const rightRow = rightRows[index] || {};
      return [
        leftRow.seed,
        leftRow.episode_reward,
        rightRow.episode_reward,
        Number(leftRow.episode_reward || 0) - Number(rightRow.episode_reward || 0),
        leftRow.ending_money,
        rightRow.ending_money,
        Number(leftRow.ending_money || 0) - Number(rightRow.ending_money || 0),
        leftRow.terminal_reason || "completed",
        rightRow.terminal_reason || "completed",
      ];
    })
  );
  showMessage("Exported checkpoint comparison CSV.");
}
