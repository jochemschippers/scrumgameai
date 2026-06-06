/**
 * API HTTP Client and Session Manager.
 * 
 * This module manages the client-side authentication states (JWT keys, user role levels: admin/guest)
 * stored in `localStorage` and provides a centralized wrapper for firing HTTP requests (`apiRequest`)
 * to the FastAPI backend. It handles bearer tokens, timeouts, and redirects to the login screen upon session expiry.
 * 
 * Connections:
 *   - Imports: Auth constants from `constants/defaults.js`, `state` from `state/store.js`, and `showLoginScreen` from `components/auth.js`.
 *   - Exported functions: Imported and called by components to interact with the backend API.
 */

import { AUTH_TOKEN_KEY, AUTH_ROLE_KEY } from '../constants/defaults.js';
import { state } from '../state/store.js';
import { showLoginScreen } from '../components/auth.js';

/**
 * Retrieves the currently saved session token from localStorage.
 * 
 * @returns {string|null} The Bearer token if logged in.
 */
export function getToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Persists the session token to localStorage.
 * 
 * @param {string} token - The signed JWT string.
 */
export function setToken(token) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

/**
 * Removes the session token from localStorage.
 */
export function clearToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

/**
 * Retrieves the currently saved user role (e.g. "admin" or "guest") from localStorage.
 * 
 * @returns {string|null} The user's role.
 */
export function getRole() {
  return localStorage.getItem(AUTH_ROLE_KEY) || null;
}

/**
 * Persists the user's role level to localStorage.
 * 
 * @param {string} role - The role string ("admin" or "guest").
 */
export function setRole(role) {
  localStorage.setItem(AUTH_ROLE_KEY, role);
}

/**
 * Removes the user's role level from localStorage.
 */
export function clearRole() {
  localStorage.removeItem(AUTH_ROLE_KEY);
}

/**
 * Determines whether the currently authenticated session is a read-only guest session.
 * 
 * @returns {boolean} True if the role is guest.
 */
export function isGuest() {
  return getRole() === "guest";
}

/**
 * Dispatches an HTTP request to the backend with Bearer authentication and abort signal handles.
 * Automatically clears session and forces login redirect if a 401 Unauthorized response is received.
 * 
 * @param {string} path - URL path (relative to api base).
 * @param {Object} options - Standard fetch init options (headers, body, method).
 * @param {number} timeoutMs - Request abort threshold in milliseconds.
 * @returns {Promise<any>} Parsed JSON response.
 * @throws {Error} If request fails, times out, or returns a non-OK status.
 */
export async function apiRequest(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const token = getToken();
    const response = await fetch(`${state.apiBaseUrl}${path}`, {
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        ...(options.headers || {}),
      },
      ...options,
    });

    if (response.status === 401) {
      clearToken();
      clearRole();
      showLoginScreen();
      throw new Error("Session expired. Please log in again.");
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }

    return response.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out: ${path}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
