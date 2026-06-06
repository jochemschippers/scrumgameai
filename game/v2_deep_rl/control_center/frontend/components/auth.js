/**
 * Frontend Authentication UI Component.
 * 
 * This module controls the visibility toggles for the login screen overlay and the main app shell,
 * and handles user logout cleanup.
 * 
 * Connections:
 *   - Imports: Storage cleaner utilities from `api/client.js` and global `state` from `state/store.js`.
 *   - Exported functions: `showLoginScreen`, `hideLoginScreen`, and `logout`. Called by the global entrypoint `main.js`.
 */

import { clearToken as clearStorageToken, clearRole } from '../api/client.js';
import { state } from '../state/store.js';

function $(id) {
  return document.getElementById(id);
}

/** Show login screen. */
export function showLoginScreen() {
  const overlay = document.getElementById("loginOverlay");
  const shell = document.getElementById("appShell");
  if (overlay) overlay.style.display = "flex";
  if (shell) shell.style.display = "none";
  setTimeout(() => {
    const u = document.getElementById("loginUsername");
    if (u) u.focus();
  }, 50);
}

/** Hide login screen. */
export function hideLoginScreen() {
  const overlay = document.getElementById("loginOverlay");
  const shell = document.getElementById("appShell");
  if (overlay) overlay.style.display = "none";
  if (shell) shell.style.display = "flex";
}

/** Handle logout. */
export function logout() {
  clearStorageToken();
  clearRole();
  state.userRole = null;
  showLoginScreen();
}
