/**
 * Configuration Editor State Management.
 * 
 * This module manages the live, mutable state of the configuration editor. It initializes
 * a clone of `DEFAULT_CONFIG` which is modified in-place by form interactions and actions,
 * providing a single source of truth across all components of the editor.
 * 
 * Connections:
 *   - Imports: `DEFAULT_CONFIG` from `constants.js`.
 *   - Exported state: Imported and read/written by `main.js`, `actions.js`, `board.js`, `form.js`, and `render.js`.
 */

import { DEFAULT_CONFIG } from './constants.js';

// Single mutable state object — imported and mutated directly by all modules.
export const state = structuredClone(DEFAULT_CONFIG);
