import { state } from './state.js';
import { $, normalizeProductKey, numberValue, parseNumberList } from './utils.js';
import { syncShapeFromInputs } from './board.js';

export function readFormIntoState() {
  state.config_name = $("configNameInput").value.trim();
  state.schema_version = $("schemaVersionInput").value.trim();
  state.config_description = $("configDescriptionInput").value.trim();
  state.players_count = Math.max(1, numberValue("playersCountInput", 1));
  state.max_turns = Math.max(1, numberValue("maxTurnsInput", 1));
  state.starting_money = numberValue("startingMoneyInput");
  state.ring_value = numberValue("ringValueInput");
  state.cost_continue = numberValue("costContinueInput");
  state.cost_switch_mid = numberValue("costSwitchMidInput");
  state.cost_switch_after = numberValue("costSwitchAfterInput");
  state.mandatory_loan_amount = numberValue("mandatoryLoanInput");
  state.loan_interest = numberValue("loanInterestInput");
  state.penalty_negative = numberValue("penaltyNegativeInput");
  state.penalty_positive = numberValue("penaltyPositiveInput");
  state.daily_scrums_per_sprint = Math.max(1, numberValue("dailyScrumsInput", 1));
  state.daily_scrum_target = Math.max(1, numberValue("dailyScrumTargetInput", 1));

  syncShapeFromInputs();

  state.product_names = state.product_names.map((_, index) => {
    const input = $(`productNameInput_${index}`);
    return input ? input.value.trim() || `Product ${index + 1}` : `Product ${index + 1}`;
  });

  state.board_ring_values = state.board_ring_values.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`ringValue_${productIndex}_${sprintIndex}`, 1))
  );
  state.board_features = state.board_features.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`featureValue_${productIndex}_${sprintIndex}`, 1))
  );

  state.dice_rules = state.dice_rules.map((rule, index) => ({
    min_features: Math.max(1, numberValue(`diceMin_${index}`, rule.min_features)),
    max_features: (() => {
      const raw = $(`diceMax_${index}`).value.trim();
      return raw === "" ? null : Math.max(1, Number(raw));
    })(),
    dice_count: Math.max(1, numberValue(`diceCount_${index}`, rule.dice_count)),
    dice_sides: Math.max(2, numberValue(`diceSides_${index}`, rule.dice_sides)),
  }));

  state.refinement.active = $("refinementActiveInput").checked;
  state.refinement.model_name = $("refinementModelInput").value.trim();
  state.refinement.die_sides = Math.max(2, numberValue("refinementDieSidesInput", 20));
  state.refinement.product_rules = state.refinement.product_rules.map((rule, index) => ({
    product_key: normalizeProductKey($(`refinementKey_${index}`).value) || normalizeProductKey(state.product_names[index]),
    increase_rolls: parseNumberList($(`refinementIncrease_${index}`).value),
    decrease_rolls: parseNumberList($(`refinementDecrease_${index}`).value),
  }));

  state.incident.active = $("incidentActiveInput").checked;
  state.incident.allow_player_specific_incidents = $("playerSpecificIncidentsInput").checked;
  state.incident.draw_probability = Number($("incidentDrawProbabilityInput").value);
  state.incident.severity_multiplier = Number($("incidentSeverityMultiplierInput").value);
  state.incident.cards = state.incident.cards.map((card, index) => ({
    card_id: numberValue(`incidentId_${index}`, card.card_id),
    name: $(`incidentName_${index}`).value.trim(),
    description: $(`incidentDescription_${index}`).value.trim(),
    effect_type: $(`incidentEffect_${index}`).value.trim(),
    target_products: $(`incidentTargets_${index}`).value
      .split(",")
      .map((value) => normalizeProductKey(value))
      .filter(Boolean),
    delta_money: numberValue(`incidentDelta_${index}`, 0),
    target_sprint: (() => {
      const raw = $(`incidentSprint_${index}`).value.trim();
      if (raw === "") return null;
      const sprintCount = state.board_ring_values[0]?.length || 1;
      return Math.max(1, Math.min(sprintCount, Number(raw)));
    })(),
    set_value_money: (() => {
      const raw = $(`incidentExactValue_${index}`).value.trim();
      return raw === "" ? null : Number(raw);
    })(),
    future_only: $(`incidentFutureOnly_${index}`).checked,
    weight: Number($(`incidentWeight_${index}`).value),
  }));
}

export function canonicalConfig() {
  readFormIntoState();
  return structuredClone(state);
}
