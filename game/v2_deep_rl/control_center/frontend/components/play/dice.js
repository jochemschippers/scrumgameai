import { $ } from '../../utils/helpers.js';
import { escapeHtml, formatCurrency } from '../../utils/formatting.js';
import { latestPlayTurn, productNameById } from './session.js';

export function showPlayDiceOverlay() {
  $("playDiceOverlay")?.classList.remove("hidden");
}

export function hidePlayDiceOverlay() {
  $("playDiceOverlay")?.classList.add("hidden");
}

export function renderPlayDicePreview(humanSeat, actionId) {
  const host = $("playDiceCard");
  if (!host || !humanSeat) return;
  const action = (humanSeat.valid_actions || []).find((item) => Number(item.action_id) === Number(actionId));
  host.innerHTML = `
    <div class="dice-summary">
      <div class="dice-stat"><span>Queued</span><strong>${escapeHtml(action?.label || "Action")}</strong></div>
      <div class="dice-stat"><span>Product</span><strong>${escapeHtml(productNameById(actionId || humanSeat.state?.current_product))}</strong></div>
    </div>
    <p>Submit action to roll the five Daily Scrums.</p>
  `;
}

export function renderPlayDiceZone() {
  const host = $("playDiceCard");
  if (!host) return;
  const row = latestPlayTurn();
  const dice = row?.dice;
  if (!dice?.daily_scrums?.length) {
    host.innerHTML = `<p>Select an action to reveal the daily scrum results.</p>`;
    return;
  }

  const netResult = Number(dice.variance || 0);
  const payout = Number(dice.payout || 0);
  const resultClass = netResult <= 0 ? "play-number-good" : "play-number-bad";
  host.innerHTML = `
    <div class="dice-summary">
      <div class="dice-stat"><span>Rolling</span><strong>${escapeHtml(dice.dice_label)}</strong></div>
      <div class="dice-stat"><span>Total / Target</span><strong>${dice.total_rolled} / ${dice.target_total}</strong></div>
      <div class="dice-stat"><span>Net Result</span><strong class="${resultClass}">${netResult}</strong></div>
    </div>
    <div class="dice-roll-list">
      ${dice.daily_scrums.map((scrum) => `
        <div class="dice-roll-row">
          <strong>D${escapeHtml(String(scrum.scrum_number))}</strong>
          <span class="dice-roll-values">[${(scrum.rolls || []).map((roll) => escapeHtml(String(roll))).join(", ")}]</span>
          <span class="dice-roll-total">${escapeHtml(String(scrum.roll_total))}</span>
        </div>
      `).join("")}
    </div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Penalty</span><strong class="play-number-bad">-${formatCurrency(dice.planning_penalty)}</strong></div>
      <div class="dice-stat"><span>Payout</span><strong class="${payout >= 0 ? "play-number-good" : "play-number-bad"}">${formatCurrency(payout)}</strong></div>
      <div class="dice-stat"><span>Result</span><strong class="${row.outcome === "Success" ? "play-number-good" : "play-number-bad"}">${escapeHtml(row.outcome || "-")}</strong></div>
    </div>
  `;
}
