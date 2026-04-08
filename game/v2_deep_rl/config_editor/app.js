(function () {
  "use strict";

  const DEFAULT_CONFIG = {
    schema_version: "1.0",
    config_name: "Advanced Classical DDQN",
    config_description: "Default advanced single-player Scrum Game rules for the deep-RL branch.",
    players_count: 1,
    product_names: ["Yellow", "Blue", "Red", "Orange", "Green", "Purple", "Black"],
    max_turns: 6,
    starting_money: 25000,
    ring_value: 5000,
    cost_continue: 0,
    cost_switch_mid: 5000,
    cost_switch_after: 0,
    mandatory_loan_amount: 50000,
    loan_interest: 5000,
    penalty_negative: 1000,
    penalty_positive: 5000,
    daily_scrums_per_sprint: 5,
    daily_scrum_target: 12,
    board_ring_values: [
      [4, 2, 1, 1],
      [5, 3, 2, 1],
      [6, 4, 3, 2],
      [5, 3, 2, 1],
      [4, 3, 2, 1],
      [5, 3, 2, 1],
      [7, 4, 3, 2],
    ],
    board_features: [
      [3, 3, 2, 1],
      [2, 2, 1, 1],
      [1, 1, 1, 1],
      [2, 2, 2, 1],
      [3, 2, 2, 1],
      [2, 2, 1, 1],
      [1, 1, 1, 1],
    ],
    dice_rules: [
      { min_features: 1, max_features: 1, dice_count: 1, dice_sides: 20 },
      { min_features: 2, max_features: 2, dice_count: 2, dice_sides: 10 },
      { min_features: 3, max_features: null, dice_count: 3, dice_sides: 6 },
    ],
    refinement: {
      active: true,
      model_name: "Standard (ID 301)",
      die_sides: 20,
      product_rules: [
        { product_key: "yellow", increase_rolls: [1, 2], decrease_rolls: [19, 20] },
        { product_key: "blue", increase_rolls: [1, 2, 3, 4], decrease_rolls: [19, 20] },
        { product_key: "red", increase_rolls: [1, 2], decrease_rolls: [19, 20] },
        { product_key: "orange", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
        { product_key: "green", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
        { product_key: "purple", increase_rolls: [1, 2, 3], decrease_rolls: [19, 20] },
        { product_key: "black", increase_rolls: [1], decrease_rolls: [20] },
      ],
    },
    incident: {
      active: true,
      allow_player_specific_incidents: false,
      draw_probability: 1.0,
      severity_multiplier: 1.0,
      cards: [
        {
          card_id: 401,
          name: "Demand Collapse Red",
          description: "All future red sprints are worth zero.",
          effect_type: "set_future_product_to_zero",
          target_products: ["red"],
          delta_money: 0,
          target_sprint: null,
          set_value_money: null,
          future_only: true,
          weight: 1.0,
        },
        {
          card_id: 402,
          name: "New Competitors",
          description: "Orange and blue products lose value due to new competitors.",
          effect_type: "adjust_future_products",
          target_products: ["orange", "blue"],
          delta_money: -5000,
          target_sprint: null,
          set_value_money: null,
          future_only: true,
          weight: 1.0,
        },
        {
          card_id: 403,
          name: "Government Subsidy",
          description: "All first sprints gain a subsidy bonus.",
          effect_type: "adjust_specific_sprint_globally",
          target_products: [],
          delta_money: 5000,
          target_sprint: 1,
          set_value_money: null,
          future_only: true,
          weight: 1.0,
        },
        {
          card_id: 404,
          name: "Yellow Demand Boost",
          description: "All future yellow sprints gain value.",
          effect_type: "adjust_future_products",
          target_products: ["yellow"],
          delta_money: 5000,
          target_sprint: null,
          set_value_money: null,
          future_only: true,
          weight: 1.0,
        },
        {
          card_id: 405,
          name: "Black Product Breakthrough",
          description: "The fourth black sprint becomes worth 100000.",
          effect_type: "set_specific_sprint_exact",
          target_products: ["black"],
          delta_money: 0,
          target_sprint: 4,
          set_value_money: 100000,
          future_only: true,
          weight: 1.0,
        },
      ],
    },
    reserved_fields: {},
  };

  const state = structuredClone(DEFAULT_CONFIG);

  function $(id) {
    return document.getElementById(id);
  }

  function normalizeProductKey(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  }

  function slugifyFileName(value) {
    const slug = String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
    return slug || "my_custom_config";
  }

  function numberValue(inputId, fallback = 0) {
    const value = Number($(inputId).value);
    return Number.isFinite(value) ? value : fallback;
  }

  function parseNumberList(value) {
    return String(value || "")
      .split(",")
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isFinite(item));
  }

  function rebuildBoardIfNeeded(productCount, sprintCount) {
    const nextRingValues = [];
    const nextFeatures = [];

    for (let productIndex = 0; productIndex < productCount; productIndex += 1) {
      const ringRow = [];
      const featureRow = [];
      for (let sprintIndex = 0; sprintIndex < sprintCount; sprintIndex += 1) {
        ringRow.push(state.board_ring_values?.[productIndex]?.[sprintIndex] ?? 1);
        featureRow.push(state.board_features?.[productIndex]?.[sprintIndex] ?? 1);
      }
      nextRingValues.push(ringRow);
      nextFeatures.push(featureRow);
    }

    state.board_ring_values = nextRingValues;
    state.board_features = nextFeatures;
  }

  function rebuildProductNamesIfNeeded(productCount) {
    const nextNames = [];
    for (let index = 0; index < productCount; index += 1) {
      nextNames.push(state.product_names?.[index] ?? `Product ${index + 1}`);
    }
    state.product_names = nextNames;
  }

  function rebuildRefinementRulesFromProducts() {
    state.refinement.product_rules = state.product_names.map((name) => ({
      product_key: normalizeProductKey(name),
      increase_rolls: [1, 2],
      decrease_rolls: [19, 20],
    }));
  }

  function ensureShapeConsistencyFromState() {
    const productCount = Math.max(1, state.product_names?.length || state.board_ring_values?.length || 1);
    const sprintCount = Math.max(1, state.board_ring_values?.[0]?.length || state.board_features?.[0]?.length || 1);
    rebuildProductNamesIfNeeded(productCount);
    rebuildBoardIfNeeded(productCount, sprintCount);
    if (!Array.isArray(state.refinement?.product_rules) || state.refinement.product_rules.length !== productCount) {
      rebuildRefinementRulesFromProducts();
    }
  }

  function syncShapeFromInputs() {
    const productCount = Math.max(1, numberValue("productsCountInput", state.product_names.length));
    const sprintCount = Math.max(1, numberValue("sprintsPerProductInput", state.board_ring_values[0]?.length || 1));

    rebuildProductNamesIfNeeded(productCount);
    rebuildBoardIfNeeded(productCount, sprintCount);

    if (!Array.isArray(state.refinement.product_rules) || state.refinement.product_rules.length !== productCount) {
      rebuildRefinementRulesFromProducts();
    }
  }

  function readFormIntoState() {
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

  function canonicalConfig() {
    readFormIntoState();
    return structuredClone(state);
  }

  function renderMetadata() {
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

  function renderProductNames() {
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

  function renderBoardMatrix() {
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

  function renderDiceRules() {
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

  function renderRefinementRules() {
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

  function renderIncidentCards() {
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

  function updatePreview() {
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

  function renderAll() {
    renderMetadata();
    renderProductNames();
    renderBoardMatrix();
    renderDiceRules();
    renderRefinementRules();
    renderIncidentCards();
    updatePreview();
  }

  function downloadJson() {
    const fileName = $("downloadFileNameInput").value.trim() || "custom_game_config.json";
    const blob = new Blob([JSON.stringify(canonicalConfig(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName.endsWith(".json") ? fileName : `${fileName}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function copyJson() {
    const text = JSON.stringify(canonicalConfig(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      window.alert("Config JSON copied to clipboard.");
    } catch (error) {
      window.alert("Clipboard copy failed. Use Download JSON instead.");
    }
  }

  function importJsonFile(file) {
    const reader = new FileReader();
    reader.onload = function () {
      try {
        const parsed = JSON.parse(String(reader.result || "{}"));
        Object.keys(state).forEach((key) => delete state[key]);
        Object.assign(state, structuredClone(parsed));
        delete $("downloadFileNameInput").dataset.touched;
        ensureShapeConsistencyFromState();
        renderAll();
      } catch (error) {
        window.alert("Could not parse that JSON file.");
      }
    };
    reader.readAsText(file);
  }

  function resetDefaults() {
    Object.keys(state).forEach((key) => delete state[key]);
    Object.assign(state, structuredClone(DEFAULT_CONFIG));
    delete $("downloadFileNameInput").dataset.touched;
    renderAll();
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function attachListeners() {
    document.addEventListener("input", (event) => {
      if (event.target.matches("input, textarea, select")) {
        if (event.target.id === "productsCountInput" || event.target.id === "sprintsPerProductInput") {
          syncShapeFromInputs();
          renderAll();
          return;
        }
        updatePreview();
      }
    });

    document.addEventListener("click", (event) => {
      const removeDiceIndex = event.target.getAttribute("data-remove-dice");
      if (removeDiceIndex !== null) {
        state.dice_rules.splice(Number(removeDiceIndex), 1);
        renderAll();
      }

      const removeIncidentIndex = event.target.getAttribute("data-remove-incident");
      if (removeIncidentIndex !== null) {
        state.incident.cards.splice(Number(removeIncidentIndex), 1);
        renderAll();
      }
    });

    $("addDiceRuleButton").addEventListener("click", () => {
      state.dice_rules.push({
        min_features: 1,
        max_features: null,
        dice_count: 1,
        dice_sides: 6,
      });
      renderAll();
    });

    $("resetRefinementRulesButton").addEventListener("click", () => {
      readFormIntoState();
      rebuildRefinementRulesFromProducts();
      renderAll();
    });

    $("addIncidentCardButton").addEventListener("click", () => {
      const sprintCount = state.board_ring_values[0]?.length || 1;
      state.incident.cards.push({
        card_id: Date.now(),
        name: "Custom Incident",
        description: "",
        effect_type: "adjust_future_products",
        target_products: [],
        delta_money: 0,
        target_sprint: Math.min(1, sprintCount),
        set_value_money: null,
        future_only: true,
        weight: 1.0,
      });
      renderAll();
    });

    $("downloadJsonButton").addEventListener("click", downloadJson);
    $("copyJsonButton").addEventListener("click", copyJson);
    $("resetDefaultsButton").addEventListener("click", resetDefaults);
    $("downloadFileNameInput").addEventListener("input", () => {
      $("downloadFileNameInput").dataset.touched = "true";
    });
    $("importConfigInput").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) {
        importJsonFile(file);
      }
      event.target.value = "";
    });
  }

  attachListeners();
  renderAll();
})();
