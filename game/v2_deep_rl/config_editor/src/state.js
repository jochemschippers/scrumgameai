import { DEFAULT_CONFIG } from './constants.js';

// Single mutable state object — imported and mutated directly by all modules.
export const state = structuredClone(DEFAULT_CONFIG);
