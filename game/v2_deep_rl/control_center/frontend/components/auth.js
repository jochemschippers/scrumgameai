import { clearToken as clearStorageToken } from '../api/client.js';

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
  showLoginScreen();
}
