/**
 * Configuration Editor Utilities.
 * 
 * This module provides clean, lightweight utility helpers for DOM selection, HTML escaping,
 * string sanitization, and parsing form inputs (such as numeric inputs and lists of integers).
 * 
 * Connections:
 *   - Used across all Configuration Editor frontend modules (`main.js`, `actions.js`, `board.js`, `form.js`, `render.js`).
 */

/**
 * Shorthand for document.getElementById.
 * 
 * @param {string} id - The DOM element ID.
 * @returns {HTMLElement|null} The DOM element.
 */
export function $(id) {
  return document.getElementById(id);
}

/**
 * Escapes HTML characters in a string to prevent XSS vulnerability when rendering templates.
 * 
 * @param {any} value - The input value to escape.
 * @returns {string} The escaped safe HTML string.
 */
export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Normalizes a product name/key into lowercase alphanumeric chars for routing and internal dict matching.
 * 
 * @param {string} value - The product name/key.
 * @returns {string} Normalized alphanumeric key.
 */
export function normalizeProductKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

/**
 * Converts a string name into a snake_case slug suitable for a filename.
 * 
 * @param {string} value - The input string (e.g. config name).
 * @returns {string} Safe snake_case file name.
 */
export function slugifyFileName(value) {
  const slug = String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug || "my_custom_config";
}

/**
 * Safely parses the value of a numeric input field, falling back to a default value if invalid.
 * 
 * @param {string} inputId - DOM ID of the input field.
 * @param {number} fallback - The fallback value if parsing fails.
 * @returns {number} The parsed finite number.
 */
export function numberValue(inputId, fallback = 0) {
  const value = Number($(inputId).value);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Parses a comma-separated list of numbers (e.g. "1, 2, 3") into an array of integers.
 * 
 * @param {string} value - The raw comma-separated string.
 * @returns {number[]} Array of parsed finite numbers.
 */
export function parseNumberList(value) {
  return String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}
