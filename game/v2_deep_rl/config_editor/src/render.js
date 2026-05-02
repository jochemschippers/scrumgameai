import { state } from './state.js';
import { $, escapeHtml, slugifyFileName } from './utils.js';
import { canonicalConfig } from './form.js';

export function renderMetadata() {
  $("configNameInput").value = state.config_name;
  $("schemaVersionInput").value = state.schema_version;
  $("configDescriptionInput").value = state.config_description;
  $("playersCountInput").value = state.players_count;
  $("productsCountInput").value = state.product_names.length;
  $("sprintsPerProductInput").value = state.board_ring_values[0]?.length || 1;
  $("maxTurnsInput").value = state.max_turns;
  $("startingMoneyInput").value = state.starting_money;
  $("ringValueInput").value = state.ring_value;
  $("costContinueInput").value = state.cost_continue;
  $("costSwitchMidInput").value = state.cost_switch_mid;
  $("costSwitchAfterInput").value = state.cost_switch_after;
  $("mandatoryLoanInput").value = state.mandatory_loan_amount;
  $("loanInterestInput").value = state.loan_interest;
  $("penaltyNegativeInput").value = state.penalty_negative;
  $("penaltyPositiveInput").value = state.penalty_positive;
  $("dailyScrumsInput").value = state.daily_scrums_per_sprint;
  $("dailyScrumTargetInput").value = state.daily_scrum_target;
}

export function renderProductNames() {
  const host = $("productNamesGrid");
  host.innerHTML = "";

  state.product_names.forEach((name, index) => {
    const label = document.createElement("label");
    label.className = "field";
    label.innerHTML = `
      <span>Product ${index + 1}</span>
      <input id="productNameInput_${index}" type="text" value="${escapeHtml(name)}" />
    `;
    host.appendChild(label);
  });
}

export function renderBoardMatrix() {
  const host = $("boardMatrixContainer");
  const sprintCount = state.board_ring_values[0]?.length || 1;

  let html = '<table class="matrix-table"><thead><tr><th>Product</th>';
  for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
    html += `<th>Sprint ${sprintIndex + 1}</th>`;
  }
  html += "</tr></thead><tbody>";

  state.product_names.forEach((productName, productIndex) => {
    html += `<tr><th>${escapeHtml(productName)}</th>`;
    for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
      html += `
        <td>
          <div class="matrix-cell">
            <label class="field">
              <span>Value</span>
              <input id="ringValue_${productIndex}_${sprintIndex}" type="number" value="${state.board_ring_values[productIndex][sprintIndex]}" />
            </label>
            <label class="field">
              <span>Features</span>
              <input id="featureValue_${productIndex}_${sprintIndex}" type="number" min="1" value="${state.board_features[productIndex][sprintIndex]}" />
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

export function renderDiceRules() {
  const host = $("diceRulesList");
  host.innerHTML = "";

  state.dice_rules.forEach((rule, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Rule ${index + 1}</strong>
        <button class="button danger" type="button" data-remove-dice="${index}">Remove</button>
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

export function renderRefinementRules() {
  $("refinementActiveInput").checked = Boolean(state.refinement.active);
  $("refinementModelInput").value = state.refinement.model_name;
  $("refinementDieSidesInput").value = state.refinement.die_sides;

  const host = $("refinementRulesList");
  host.innerHTML = "";

  state.refinement.product_rules.forEach((rule, index) => {
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

export function renderIncidentCards() {
  $("incidentActiveInput").checked = Boolean(state.incident.active);
  $("playerSpecificIncidentsInput").checked = Boolean(state.incident.allow_player_specific_incidents);
  $("incidentDrawProbabilityInput").value = state.incident.draw_probability;
  $("incidentSeverityMultiplierInput").value = state.incident.severity_multiplier;

  const host = $("incidentCardsList");
  host.innerHTML = "";

  state.incident.cards.forEach((card, index) => {
    const row = document.createElement("div");
    row.className = "list-row";
    row.innerHTML = `
      <div class="list-row-head">
        <strong>Incident Card ${index + 1}</strong>
        <button class="button danger" type="button" data-remove-incident="${index}">Remove</button>
      </div>
      <div class="grid three">
        <label class="field"><span>Card ID</span><input id="incidentId_${index}" type="number" value="${card.card_id}" /></label>
        <label class="field"><span>Name</span><input id="incidentName_${index}" type="text" value="${escapeHtml(card.name)}" /></label>
        <label class="field"><span>Effect Type</span><input id="incidentEffect_${index}" type="text" value="${escapeHtml(card.effect_type)}" /></label>
        <label class="field span-2"><span>Description</span><textarea id="incidentDescription_${index}" rows="2">${escapeHtml(card.description)}</textarea></label>
        <label class="field"><span>Target Products</span><input id="incidentTargets_${index}" type="text" value="${escapeHtml(card.target_products.join(", "))}" placeholder="comma-separated product keys" /></label>
        <label class="field"><span>Delta Money</span><input id="incidentDelta_${index}" type="number" value="${card.delta_money}" /></label>
        <label class="field"><span>Target Sprint</span><input id="incidentSprint_${index}" type="number" min="1" value="${card.target_sprint ?? ""}" /></label>
        <label class="field"><span>Set Exact Value</span><input id="incidentExactValue_${index}" type="number" value="${card.set_value_money ?? ""}" /></label>
        <label class="field"><span>Weight</span><input id="incidentWeight_${index}" type="number" min="0.1" step="0.1" value="${card.weight}" /></label>
        <label class="field checkbox-field">
          <input id="incidentFutureOnly_${index}" type="checkbox" ${card.future_only ? "checked" : ""} />
          <span>Future Only</span>
        </label>
      </div>
    `;
    host.appendChild(row);
  });
}

export function updatePreview() {
  const config = canonicalConfig();
  $("jsonPreview").textContent = JSON.stringify(config, null, 2);
  $("summaryProducts").textContent = String(config.product_names.length);
  $("summarySprints").textContent = String(config.board_ring_values[0]?.length || 0);
  $("summaryActions").textContent = String(config.product_names.length + 1);
  $("summaryIncidentCards").textContent = String(config.incident.cards.length);
  if (!$("downloadFileNameInput").dataset.touched) {
    $("downloadFileNameInput").value = `${slugifyFileName(config.config_name)}.json`;
  }
}

export function renderAll() {
  renderMetadata();
  renderProductNames();
  renderBoardMatrix();
  renderDiceRules();
  renderRefinementRules();
  renderIncidentCards();
  updatePreview();
}
