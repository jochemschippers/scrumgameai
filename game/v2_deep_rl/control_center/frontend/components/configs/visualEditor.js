/** Implement visual editor user-interface behavior. */

import { state } from '../../state/store.js';
import { $, clone, numberValue, normalizeProductKey, parseNumberList, parseJsonEditor } from '../../utils/helpers.js';
import { formatJson, escapeHtml } from '../../utils/formatting.js';
import { DEFAULT_GAME_CONFIG } from '../../constants/defaults.js';
import { isGuest } from '../../api/client.js';

/** Handle ensure visual game config. */
export function ensureVisualGameConfig() {
  if (!state.visualGameConfig) {
    state.visualGameConfig = clone(DEFAULT_GAME_CONFIG);
  }
}

/** Handle rebuild visual board. */
export function rebuildVisualBoard(productCount, sprintCount) {
  const config = state.visualGameConfig;
  const nextRingValues = [];
  const nextFeatures = [];
  for (let productIndex = 0; productIndex < productCount; productIndex += 1) {
    const ringRow = [];
    const featureRow = [];
    for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
      ringRow.push(config.board_ring_values?.[productIndex]?.[sprintIndex] ?? 1);
      featureRow.push(config.board_features?.[productIndex]?.[sprintIndex] ?? 1);
    }
    nextRingValues.push(ringRow);
    nextFeatures.push(featureRow);
  }
  config.board_ring_values = nextRingValues;
  config.board_features = nextFeatures;
}

/** Handle rebuild visual product names. */
export function rebuildVisualProductNames(productCount) {
  const config = state.visualGameConfig;
  const nextNames = [];
  for (let index = 0; index < productCount; index += 1) {
    nextNames.push(config.product_names?.[index] ?? `Product ${index + 1}`);
  }
  config.product_names = nextNames;
}

/** Handle rebuild visual refinement rules. */
export function rebuildVisualRefinementRules() {
  state.visualGameConfig.refinement.product_rules = state.visualGameConfig.product_names.map((name) => ({
    product_key: normalizeProductKey(name),
    increase_rolls: [1, 2],
    decrease_rolls: [19, 20],
  }));
}

/** Handle ensure visual shape consistency. */
export function ensureVisualShapeConsistency() {
  ensureVisualGameConfig();
  const config = state.visualGameConfig;
  const productCount = Math.max(1, config.product_names?.length || config.board_ring_values?.length || 1);
  const sprintCount = Math.max(1, config.board_ring_values?.[0]?.length || config.board_features?.[0]?.length || 1);
  rebuildVisualProductNames(productCount);
  rebuildVisualBoard(productCount, sprintCount);
  if (!Array.isArray(config.refinement?.product_rules) || config.refinement.product_rules.length !== productCount) {
    rebuildVisualRefinementRules();
  }
}

/** Synchronize visual shape from inputs. */
export function syncVisualShapeFromInputs() {
  const productCount = Math.max(1, numberValue("productsCountInput", state.visualGameConfig.product_names.length));
  const sprintCount = Math.max(1, numberValue("sprintsPerProductInput", state.visualGameConfig.board_ring_values[0]?.length || 1));
  rebuildVisualProductNames(productCount);
  rebuildVisualBoard(productCount, sprintCount);
  if (!Array.isArray(state.visualGameConfig.refinement.product_rules) || state.visualGameConfig.refinement.product_rules.length !== productCount) {
    rebuildVisualRefinementRules();
  }
}

/** Read visual editor into state. */
export function readVisualEditorIntoState() {
  ensureVisualGameConfig();
  const config = state.visualGameConfig;
  config.config_name = $("configNameInput").value.trim();
  config.schema_version = $("schemaVersionInput").value.trim();
  config.config_description = $("configDescriptionInput").value.trim();
  config.players_count = Math.max(1, numberValue("playersCountInput", 1));
  config.max_turns = Math.max(1, numberValue("maxTurnsInput", 1));
  config.starting_money = numberValue("startingMoneyInput");
  config.ring_value = numberValue("ringValueInput");
  config.cost_continue = numberValue("costContinueInput");
  config.cost_switch_mid = numberValue("costSwitchMidInput");
  config.cost_switch_after = numberValue("costSwitchAfterInput");
  config.mandatory_loan_amount = numberValue("mandatoryLoanInput");
  config.loan_interest = numberValue("loanInterestInput");
  config.penalty_negative = numberValue("penaltyNegativeInput");
  config.penalty_positive = numberValue("penaltyPositiveInput");
  config.daily_scrums_per_sprint = Math.max(1, numberValue("dailyScrumsInput", 1));
  config.daily_scrum_target = Math.max(1, numberValue("dailyScrumTargetInput", 1));

  syncVisualShapeFromInputs();

  config.product_names = config.product_names.map((_, index) => {
    const input = $(`productNameInput_${index}`);
    return input ? input.value.trim() || `Product ${index + 1}` : `Product ${index + 1}`;
  });

  config.board_ring_values = config.board_ring_values.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`ringValue_${productIndex}_${sprintIndex}`, 1))
  );
  config.board_features = config.board_features.map((row, productIndex) =>
    row.map((_, sprintIndex) => numberValue(`featureValue_${productIndex}_${sprintIndex}`, 1))
  );

  config.dice_rules = config.dice_rules.map((rule, index) => ({
    min_features: Math.max(1, numberValue(`diceMin_${index}`, rule.min_features)),
    max_features: (() => {
      const raw = $(`diceMax_${index}`).value.trim();
      return raw === "" ? null : Math.max(1, Number(raw));
    })(),
    dice_count: Math.max(1, numberValue(`diceCount_${index}`, rule.dice_count)),
    dice_sides: Math.max(2, numberValue(`diceSides_${index}`, rule.dice_sides)),
  }));

  config.refinement.active = $("refinementActiveInput").checked;
  config.refinement.model_name = $("refinementModelInput").value.trim();
  config.refinement.die_sides = Math.max(2, numberValue("refinementDieSidesInput", 20));
  config.refinement.product_rules = config.refinement.product_rules.map((rule, index) => ({
    product_key: normalizeProductKey($(`refinementKey_${index}`).value) || normalizeProductKey(config.product_names[index]),
    increase_rolls: parseNumberList($(`refinementIncrease_${index}`).value),
    decrease_rolls: parseNumberList($(`refinementDecrease_${index}`).value),
  }));

  config.incident.active = $("incidentActiveInput").checked;
  config.incident.allow_player_specific_incidents = $("playerSpecificIncidentsInput").checked;
  config.incident.draw_probability = Number($("incidentDrawProbabilityInput").value);
  config.incident.severity_multiplier = Number($("incidentSeverityMultiplierInput").value);
  config.incident.cards = config.incident.cards.map((card, index) => ({
    card_id: numberValue(`incidentId_${index}`, card.card_id),
    name: $(`incidentName_${index}`).value.trim(),
    description: $(`incidentDescription_${index}`).value.trim(),
    effect_type: $(`incidentEffect_${index}`).value.trim(),
    target_products: $(`incidentTargets_${index}`).value.split(",").map((value) => normalizeProductKey(value)).filter(Boolean),
    delta_money: numberValue(`incidentDelta_${index}`, 0),
    target_sprint: (() => {
      const raw = $(`incidentSprint_${index}`).value.trim();
      if (raw === "") return null;
      const sprintCount = config.board_ring_values[0]?.length || 1;
      return Math.max(1, Math.min(sprintCount, Number(raw)));
    })(),
    set_value_money: (() => {
      const raw = $(`incidentExactValue_${index}`).value.trim();
      return raw === "" ? null : Number(raw);
    })(),
    future_only: $(`incidentFutureOnly_${index}`).checked,
    weight: Number($(`incidentWeight_${index}`).value),
  }));

  return clone(config);
}

/** Render visual metadata. */
export function renderVisualMetadata() {
  const config = state.visualGameConfig;
  $("configNameInput").value = config.config_name;
  $("schemaVersionInput").value = config.schema_version;
  $("configDescriptionInput").value = config.config_description;
  $("playersCountInput").value = config.players_count;
  $("productsCountInput").value = config.product_names.length;
  $("sprintsPerProductInput").value = config.board_ring_values[0]?.length || 1;
  $("maxTurnsInput").value = config.max_turns;
  $("startingMoneyInput").value = config.starting_money;
  $("ringValueInput").value = config.ring_value;
  $("costContinueInput").value = config.cost_continue;
  $("costSwitchMidInput").value = config.cost_switch_mid;
  $("costSwitchAfterInput").value = config.cost_switch_after;
  $("mandatoryLoanInput").value = config.mandatory_loan_amount;
  $("loanInterestInput").value = config.loan_interest;
  $("penaltyNegativeInput").value = config.penalty_negative;
  $("penaltyPositiveInput").value = config.penalty_positive;
  $("dailyScrumsInput").value = config.daily_scrums_per_sprint;
  $("dailyScrumTargetInput").value = config.daily_scrum_target;
}

/** Render visual product names. */
export function renderVisualProductNames() {
  const host = $("productNamesGrid");
  host.innerHTML = "";
  state.visualGameConfig.product_names.forEach((name, index) => {
    const label = document.createElement("label");
    label.className = "field";
    label.innerHTML = `
      <span>Product ${index + 1}</span>
      <input id="productNameInput_${index}" type="text" value="${escapeHtml(name)}" />
    `;
    host.appendChild(label);
  });
}

/** Render visual board matrix. */
export function renderVisualBoardMatrix() {
  const host = $("boardMatrixContainer");
  const sprintCount = state.visualGameConfig.board_ring_values[0]?.length || 1;
  let html = '<table class="matrix-table"><thead><tr><th>Product</th>';
  for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
    html += `<th>Sprint ${sprintIndex + 1}</th>`;
  }
  html += "</tr></thead><tbody>";
  state.visualGameConfig.product_names.forEach((productName, productIndex) => {
    html += `<tr><th>${escapeHtml(productName)}</th>`;
    for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
      html += `
        <td>
          <div class="matrix-cell">
            <label class="field">
              <span>Value</span>
              <input id="ringValue_${productIndex}_${sprintIndex}" type="number" value="${state.visualGameConfig.board_ring_values[productIndex][sprintIndex]}" />
            </label>
            <label class="field">
              <span>Features</span>
              <input id="featureValue_${productIndex}_${sprintIndex}" type="number" min="1" value="${state.visualGameConfig.board_features[productIndex][sprintIndex]}" />
            </label>
          </div>
        </td>
      `;
    }
    html += "</tr>";
  });
  html += "</tbody></table>";
  host.innerHTML = html;
}

/** Render visual dice rules. */
export function renderVisualDiceRules() {
  const host = $("diceRulesList");
  host.innerHTML = "";
  state.visualGameConfig.dice_rules.forEach((rule, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Rule ${index + 1}</strong>
        <button class="button secondary guest-hide" type="button" data-remove-dice="${index}" style="${isGuest() ? "display:none" : ""}">Remove</button>
      </div>
      <div class="list-row-grid">
        <label class="field"><span>Min Features</span><input id="diceMin_${index}" type="number" min="1" value="${rule.min_features}" /></label>
        <label class="field"><span>Max Features</span><input id="diceMax_${index}" type="number" min="1" value="${rule.max_features ?? ""}" placeholder="blank = no max" /></label>
        <label class="field"><span>Dice Count</span><input id="diceCount_${index}" type="number" min="1" value="${rule.dice_count}" /></label>
        <label class="field"><span>Dice Sides</span><input id="diceSides_${index}" type="number" min="2" value="${rule.dice_sides}" /></label>
      </div>
    `;
    host.appendChild(row);
  });
}

/** Render visual refinement rules. */
export function renderVisualRefinementRules() {
  const config = state.visualGameConfig;
  $("refinementActiveInput").checked = Boolean(config.refinement.active);
  $("refinementModelInput").value = config.refinement.model_name;
  $("refinementDieSidesInput").value = config.refinement.die_sides;
  const host = $("refinementRulesList");
  host.innerHTML = "";
  config.refinement.product_rules.forEach((rule, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Product Rule ${index + 1}</strong>
      </div>
      <div class="list-row-grid">
        <label class="field"><span>Product Key</span><input id="refinementKey_${index}" type="text" value="${escapeHtml(rule.product_key)}" /></label>
        <label class="field"><span>Increase Rolls</span><input id="refinementIncrease_${index}" type="text" value="${rule.increase_rolls.join(", ")}" /></label>
        <label class="field"><span>Decrease Rolls</span><input id="refinementDecrease_${index}" type="text" value="${rule.decrease_rolls.join(", ")}" /></label>
      </div>
    `;
    host.appendChild(row);
  });
}

/** Render visual incident cards. */
export function renderVisualIncidentCards() {
  const config = state.visualGameConfig;
  $("incidentActiveInput").checked = Boolean(config.incident.active);
  $("playerSpecificIncidentsInput").checked = Boolean(config.incident.allow_player_specific_incidents);
  $("incidentDrawProbabilityInput").value = config.incident.draw_probability;
  $("incidentSeverityMultiplierInput").value = config.incident.severity_multiplier;
  const host = $("incidentCardsList");
  host.innerHTML = "";
  config.incident.cards.forEach((card, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Incident Card ${index + 1}</strong>
        <button class="button secondary guest-hide" type="button" data-remove-incident="${index}" style="${isGuest() ? "display:none" : ""}">Remove</button>
      </div>
      <div class="grid three">
        <label class="field hidden"><span>Card ID</span><input id="incidentId_${index}" type="number" value="${card.card_id}" /></label>
        <label class="field"><span>Name</span><input id="incidentName_${index}" type="text" value="${escapeHtml(card.name)}" /></label>
        <label class="field"><span>Effect Type</span><input id="incidentEffect_${index}" type="text" value="${escapeHtml(card.effect_type)}" /></label>
        <label class="field span-2"><span>Description</span><textarea id="incidentDescription_${index}" rows="2">${escapeHtml(card.description)}</textarea></label>
        <label class="field"><span>Target Products</span><input id="incidentTargets_${index}" type="text" value="${escapeHtml(card.target_products.join(", "))}" placeholder="comma-separated product keys" /></label>
        <label class="field"><span>Delta Money</span><input id="incidentDelta_${index}" type="number" value="${card.delta_money}" /></label>
        <label class="field"><span>Target Sprint</span><input id="incidentSprint_${index}" type="number" min="1" value="${card.target_sprint ?? ""}" /></label>
        <label class="field"><span>Set Exact Value</span><input id="incidentExactValue_${index}" type="number" value="${card.set_value_money ?? ""}" /></label>
        <label class="field"><span>Weight</span><input id="incidentWeight_${index}" type="number" min="0.1" step="0.1" value="${card.weight}" /></label>
        <label class="field checkbox-inline">
          <input id="incidentFutureOnly_${index}" type="checkbox" ${card.future_only ? "checked" : ""} />
          <span>Future Only</span>
        </label>
      </div>
    `;
    host.appendChild(row);
  });
}

/** Synchronize game json editor from visual. */
export function syncGameJsonEditorFromVisual() {
  try {
    const canonical = readVisualEditorIntoState();
    $("gameConfigEditor").value = formatJson(canonical);
    $("summaryProducts").textContent = String(canonical.product_names.length);
    $("summarySprints").textContent = String(canonical.board_ring_values[0]?.length || 0);
    $("summaryActions").textContent = String(canonical.product_names.length + 1);
    $("summaryIncidentCards").textContent = String(canonical.incident.cards.length);
    if (!$("gameConfigFileNameInput").value.trim()) {
      $("gameConfigFileNameInput").value = String(canonical.config_name || "game_config")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
    }
  } catch (_error) {
    // Keep the current editor text untouched until the input becomes valid again.
  }
}

/** Render visual editor. */
export function renderVisualEditor() {
  ensureVisualShapeConsistency();
  renderVisualMetadata();
  renderVisualProductNames();
  renderVisualBoardMatrix();
  renderVisualDiceRules();
  renderVisualRefinementRules();
  renderVisualIncidentCards();
  syncGameJsonEditorFromVisual();

  // After every render pass, lock all inputs in the Design page for guests.
  // The CSS body.guest-mode rule blocks mouse interaction; this covers keyboard.
  if (isGuest()) {
    const page = document.getElementById("page-rules");
    if (page) {
      page.querySelectorAll("input:not([type='checkbox']):not([type='file']), textarea").forEach((el) => {
        el.readOnly = true;
      });
      page.querySelectorAll("select").forEach((el) => {
        el.disabled = true;
      });
    }
  }
}
