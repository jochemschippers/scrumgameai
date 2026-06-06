/**
 * Configuration Editor Board Reshaping & Syncing.
 * 
 * This module manages the dimensional integrity of the board matrix in the editor.
 * When the user changes the number of products or sprints, this module resizes the
 * multidimensional matrices (`board_ring_values` and `board_features`) and adjusts
 * product rules (like refinement rolls) to prevent out-of-bounds index errors.
 * 
 * Connections:
 *   - Imports: `state` from `state.js`, `normalizeProductKey` and `numberValue` from `utils.js`.
 *   - Exported functions: `syncShapeFromInputs` and `ensureShapeConsistencyFromState` are called by `main.js`, `actions.js`, and `form.js`
 *     to reconcile state shape after uploads or form modifications.
 */

import { state } from './state.js';
import { normalizeProductKey, numberValue } from './utils.js';

/**
 * Rebuilds the board matrices (`board_ring_values` and `board_features`) to match the desired dimensions.
 * Retains existing values where available, and fills new cells with default values (1).
 * 
 * @param {number} productCount - The target number of rows (products).
 * @param {number} sprintCount - The target number of columns (sprints).
 */
export function rebuildBoardIfNeeded(productCount, sprintCount) {
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

/**
 * Rebuilds the array of product names to match the target product count, appending new default names if expanded.
 * 
 * @param {number} productCount - The target number of products.
 */
export function rebuildProductNamesIfNeeded(productCount) {
  const nextNames = [];
  for (let index = 0; index < productCount; index += 1) {
    nextNames.push(state.product_names?.[index] ?? `Product ${index + 1}`);
  }
  state.product_names = nextNames;
}

/**
 * Resets/rebuilds standard refinement product rules (model ID 301) for all products.
 * Set default die rolls: 1-2 for increase, 19-20 for decrease.
 */
export function rebuildRefinementRulesFromProducts() {
  state.refinement.product_rules = state.product_names.map((name) => ({
    product_key: normalizeProductKey(name),
    increase_rolls: [1, 2],
    decrease_rolls: [19, 20],
  }));
}

/**
 * Verifies and repairs the shapes of state variables (product names, board matrices, and refinement rules)
 * to make sure they are fully synchronous, typically called after a raw JSON import.
 */
export function ensureShapeConsistencyFromState() {
  const productCount = Math.max(1, state.product_names?.length || state.board_ring_values?.length || 1);
  const sprintCount = Math.max(1, state.board_ring_values?.[0]?.length || state.board_features?.[0]?.length || 1);
  rebuildProductNamesIfNeeded(productCount);
  rebuildBoardIfNeeded(productCount, sprintCount);
  if (!Array.isArray(state.refinement?.product_rules) || state.refinement.product_rules.length !== productCount) {
    rebuildRefinementRulesFromProducts();
  }
}

/**
 * Reads the current product count and sprints count from DOM inputs, resizes state matrices,
 * and maintains matching shapes for refinement rules.
 */
export function syncShapeFromInputs() {
  const productCount = Math.max(1, numberValue("productsCountInput", state.product_names.length));
  const sprintCount = Math.max(1, numberValue("sprintsPerProductInput", state.board_ring_values[0]?.length || 1));

  rebuildProductNamesIfNeeded(productCount);
  rebuildBoardIfNeeded(productCount, sprintCount);

  if (!Array.isArray(state.refinement.product_rules) || state.refinement.product_rules.length !== productCount) {
    rebuildRefinementRulesFromProducts();
  }
}
