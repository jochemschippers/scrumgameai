import { state } from '../../state/store.js';
import { $, showMessage } from '../../utils/helpers.js';
import { apiRequest } from '../../api/client.js';

export function defaultSeatName(type, index) {
  if (type === "human") return "Player";
  if (type === "model-expert") return `AI Expert ${index}`;
  if (type === "model-balanced") return `AI Balanced ${index}`;
  if (type === "model-beginner") return `AI Beginner ${index}`;
  if (type === "heuristic") return `Heuristic AI ${index}`;
  return `Random AI ${index}`;
}

export function playSeatPayload(draft, index) {
  const displayName = String(draft.display_name || "").trim() || defaultSeatName(draft.type, index + 1);
  if (draft.type === "human") return { type: "human", display_name: displayName };
  if (draft.type === "heuristic") return { type: "heuristic", display_name: displayName };
  if (draft.type === "random") return { type: "random", display_name: displayName };
  if (draft.type?.startsWith("model-")) {
    return {
      type: "model",
      checkpoint_id: state.activeCheckpointId,
      profile_name: draft.type.replace("model-", ""),
      display_name: displayName,
    };
  }
  return null;
}

export function latestPlayTurn() {
  const rows = state.playSession?.turn_log || [];
  return rows.length ? rows[rows.length - 1] : null;
}

export function productNameById(productId) {
  const product = state.playSession?.board?.products?.find((item) => Number(item.product_id) === Number(productId));
  return product?.name || `Product ${productId || "-"}`;
}

export async function createPlaySession(event) {
  event.preventDefault();
  if (!state.activeGameConfigId) {
    showMessage("Select a game config first.", "error");
    return;
  }
  const seats = state.playSeatDrafts
    .map((draft, index) => playSeatPayload(draft, index))
    .filter(Boolean);
  if (!seats.length) {
    showMessage("Add at least one seat.", "error");
    return;
  }
  if (seats.length > 4) {
    showMessage("Shared play supports at most 4 seats.", "error");
    return;
  }
  if (seats.filter((seat) => seat.type === "human").length > 1) {
    showMessage("Shared play supports at most one human seat.", "error");
    return;
  }
  if (seats.some((seat) => seat.type === "model") && !state.activeCheckpointId) {
    showMessage("Select an active brain before adding Active Brain seats.", "error");
    return;
  }
  state.playSession = await apiRequest("/play/session", {
    method: "POST",
    body: JSON.stringify({
      mode: "shared",
      game_config_id: state.activeGameConfigId,
      base_seed: Number($("playSeedInput").value),
      seats,
    }),
  }, 120000);
  document.dispatchEvent(new CustomEvent("playSessionUpdated"));
  showMessage("Play session started.");
}

export async function advancePlayRound(humanAction = null) {
  if (!state.playSession?.id) {
    showMessage("Start a play session first.", "error");
    return;
  }
  document.dispatchEvent(new CustomEvent("showPlayDiceOverlay"));
  const humanSeat = state.playSession.seats?.find((seat) => seat.controller.type === "human" && !seat.done);
  const payload = humanAction === null
    ? {}
    : { human_actions: { [humanSeat?.id || "seat_1"]: Number(humanAction) } };
  state.playSession = await apiRequest(`/play/session/${encodeURIComponent(state.playSession.id)}/action`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  document.dispatchEvent(new CustomEvent("playSessionUpdated"));
}

export async function refreshPlaySession() {
  if (!state.playSession?.id) {
    document.dispatchEvent(new CustomEvent("playSessionUpdated"));
    return;
  }
  state.playSession = await apiRequest(`/play/session/${encodeURIComponent(state.playSession.id)}`);
  document.dispatchEvent(new CustomEvent("playSessionUpdated"));
}
