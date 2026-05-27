import { state } from '../../state/store.js';
import { $ } from '../../utils/helpers.js';
import { escapeHtml, formatCurrency } from '../../utils/formatting.js';
import { DICE_BOX_MODULE_URL, DICE_BOX_ASSET_PATH } from '../../constants/defaults.js';
import { latestPlayTurn, productNameById } from './session.js';

export function showPlayDiceOverlay() {
  $("playDiceOverlay")?.classList.remove("hidden");
}

export function hidePlayDiceOverlay() {
  $("playDiceOverlay")?.classList.add("hidden");
}

export function diceNotationFromTurnDice(dice) {
  const firstScrum = dice?.daily_scrums?.[0];
  const diceCount = Number(firstScrum?.dice_count || 0);
  const diceSides = Number(firstScrum?.dice_sides || 0);
  const scrumCount = Number(dice?.daily_scrums?.length || 0);
  if (!diceCount || !diceSides || !scrumCount) return "1d6";
  return `${diceCount * scrumCount}d${diceSides}`;
}

export async function ensurePlayDiceBox() {
  if (state.playDiceBoxReady && state.playDiceBox) return state.playDiceBox;
  if (state.playDiceBoxInitPromise) return state.playDiceBoxInitPromise;

  state.playDiceBoxInitPromise = (async () => {
    const host = $("playDiceBox");
    if (!host) return null;
    const module = await import(DICE_BOX_MODULE_URL);
    const DiceBox = module.default;
    const box = new DiceBox("#playDiceBox", {
      assetPath: DICE_BOX_ASSET_PATH,
      theme: "default",
      scale: 6,
      gravity: 1,
      mass: 1,
      friction: 0.8,
      restitution: 0.25,
    });
    await box.init();
    state.playDiceBox = box;
    state.playDiceBoxReady = true;
    return box;
  })().catch((error) => {
    state.playDiceBoxReady = false;
    state.playDiceBox = null;
    state.playDiceBoxInitPromise = null;
    console.warn("3D dice failed to initialize; using fallback dice.", error);
    return null;
  });

  return state.playDiceBoxInitPromise;
}

export function renderFallbackDice(dice) {
  const host = $("playDiceBox");
  if (!host) return;
  const firstScrum = dice?.daily_scrums?.[0];
  const sides = Number(firstScrum?.dice_sides || 6);
  const rolls = (dice?.daily_scrums || [])
    .flatMap((scrum) => scrum.rolls || [])
    .slice(0, 12);
  host.innerHTML = `
    <div class="fallback-dice-stage">
      ${rolls.map((roll, index) => `
        <div class="fallback-die" style="animation-delay:${index * 35}ms">
          ${escapeHtml(String(Math.max(1, Math.min(Number(roll || 1), sides))))}
        </div>
      `).join("")}
    </div>
  `;
}

export async function rollPlayDice(dice) {
  showPlayDiceOverlay();
  const slot = $("playDiceCard")?.querySelector(".dice-animation-slot");
  slot?.classList.add("is-rolling");
  const boxHost = $("playDiceBox");
  if (boxHost) {
    boxHost.innerHTML = `<div class="fallback-dice-stage"><div class="fallback-die">?</div></div>`;
  }

  const diceBox = await ensurePlayDiceBox();
  if (diceBox) {
    try {
      await diceBox.roll(diceNotationFromTurnDice(dice));
      slot?.classList.remove("is-rolling");
      return;
    } catch (error) {
      console.warn("3D dice roll failed; using fallback dice.", error);
    }
  }

  renderFallbackDice(dice);
  window.setTimeout(() => slot?.classList.remove("is-rolling"), 650);
}

export function renderPlayDicePreview(humanSeat, actionId) {
  const host = $("playDiceCard");
  if (!host || !humanSeat) return;
  const action = (humanSeat.valid_actions || []).find((item) => Number(item.action_id) === Number(actionId));
  host.innerHTML = `
    <div class="dice-animation-slot">Dice animation slot</div>
    <div id="playDiceBox"></div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Queued</span><strong>${escapeHtml(action?.label || "Action")}</strong></div>
      <div class="dice-stat"><span>Product</span><strong>${escapeHtml(productNameById(actionId || humanSeat.state?.current_product))}</strong></div>
      <div class="dice-stat"><span>Target</span><strong>60</strong></div>
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
    host.innerHTML = `
      <div class="dice-animation-slot">Dice animation slot</div>
      <div id="playDiceBox"></div>
      <p>Select an action to reveal the daily scrum math.</p>
    `;
    return;
  }

  const variance = Number(dice.variance || 0);
  const payout = Number(dice.payout || 0);
  host.innerHTML = `
    <div class="dice-animation-slot">3D dice stage</div>
    <div id="playDiceBox"></div>
    <div class="dice-summary">
      <div class="dice-stat"><span>Rolling</span><strong>${escapeHtml(dice.dice_label)}</strong></div>
      <div class="dice-stat"><span>Total / Target</span><strong>${dice.total_rolled} / ${dice.target_total}</strong></div>
      <div class="dice-stat"><span>Variance</span><strong class="${variance <= 0 ? "play-number-good" : "play-number-bad"}">${variance}</strong></div>
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
      <div class="dice-stat"><span>Result</span><strong>${escapeHtml(row.outcome || "-")}</strong></div>
    </div>
  `;
}
