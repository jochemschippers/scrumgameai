import { clearToken as clearStorageToken, clearRole } from '../api/client.js';
import { state } from '../state/store.js';

function $(id) {
  return document.getElementById(id);
}

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

export function hideLoginScreen() {
  const overlay = document.getElementById("loginOverlay");
  const shell = document.getElementById("appShell");
  if (overlay) overlay.style.display = "none";
  if (shell) shell.style.display = "flex";
}

export function logout() {
  clearStorageToken();
  clearRole();
  state.userRole = null;
  showLoginScreen();
}
