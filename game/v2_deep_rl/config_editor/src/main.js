/**
 * Main Entry Point for the modular Configuration Editor.
 * 
 * This module coordinates the startup boot sequence and attaches global event listeners
 * to the DOM. It wires user actions (imports, exports, updates) to the shared state
 * and triggers UI re-renders.
 * 
 * Connections:
 *   - Entry Point: Script tag in `config_editor/index.html` (type="module")
 *   - Imports: `state.js`, `utils.js`, `board.js`, `form.js`, `render.js`, `actions.js`.
 */

import { state } from './state.js';
import { $ } from './utils.js';
import { syncShapeFromInputs, rebuildRefinementRulesFromProducts } from './board.js';
import { readFormIntoState } from './form.js';
import { renderAll, updatePreview } from './render.js';
import { downloadJson, copyJson, importJsonFile, resetDefaults } from './actions.js';

/** Attach global and button-specific DOM event listeners. */
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

// ── Boot ──────────────────────────────────────────────────────────────────────

attachListeners();
renderAll();
