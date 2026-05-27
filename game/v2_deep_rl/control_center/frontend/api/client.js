import { AUTH_TOKEN_KEY, AUTH_ROLE_KEY } from '../constants/defaults.js';
import { state } from '../state/store.js';
import { showLoginScreen } from '../components/auth.js';

export function getToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

export function getRole() {
  return localStorage.getItem(AUTH_ROLE_KEY) || null;
}

export function setRole(role) {
  localStorage.setItem(AUTH_ROLE_KEY, role);
}

export function clearRole() {
  localStorage.removeItem(AUTH_ROLE_KEY);
}

/** Returns true when the logged-in user is a guest (read-only). */
export function isGuest() {
  return getRole() === "guest";
}

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
