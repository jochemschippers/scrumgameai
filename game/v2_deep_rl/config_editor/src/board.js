import { state } from './state.js';
import { normalizeProductKey, numberValue } from './utils.js';

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

export function rebuildProductNamesIfNeeded(productCount) {
  const nextNames = [];
  for (let index = 0; index < productCount; index += 1) {
    nextNames.push(state.product_names?.[index] ?? `Product ${index + 1}`);
  }
  state.product_names = nextNames;
}

export function rebuildRefinementRulesFromProducts() {
  state.refinement.product_rules = state.product_names.map((name) => ({
    product_key: normalizeProductKey(name),
    increase_rolls: [1, 2],
    decrease_rolls: [19, 20],
  }));
}

export function ensureShapeConsistencyFromState() {
  const productCount = Math.max(1, state.product_names?.length || state.board_ring_values?.length || 1);
  const sprintCount = Math.max(1, state.board_ring_values?.[0]?.length || state.board_features?.[0]?.length || 1);
  rebuildProductNamesIfNeeded(productCount);
  rebuildBoardIfNeeded(productCount, sprintCount);
  if (!Array.isArray(state.refinement?.product_rules) || state.refinement.product_rules.length !== productCount) {
    rebuildRefinementRulesFromProducts();
  }
}

export function syncShapeFromInputs() {
  const productCount = Math.max(1, numberValue("productsCountInput", state.product_names.length));
  const sprintCount = Math.max(1, numberValue("sprintsPerProductInput", state.board_ring_values[0]?.length || 1));

  rebuildProductNamesIfNeeded(productCount);
  rebuildBoardIfNeeded(productCount, sprintCount);

  if (!Array.isArray(state.refinement.product_rules) || state.refinement.product_rules.length !== productCount) {
    rebuildRefinementRulesFromProducts();
  }
}
