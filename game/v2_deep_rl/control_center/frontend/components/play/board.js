import { state } from '../../state/store.js';
import { $, showMessage, buildOptions } from '../../utils/helpers.js';
import { escapeHtml, formatNumber, formatCurrency } from '../../utils/formatting.js';
import { defaultSeatName, latestPlayTurn, productNameById } from './session.js';
import { renderPlayDiceZone, renderPlayDicePreview } from './dice.js';

export function renderPlaySeatEditor() {
  const host = $("playSeatEditor");
  if (!host) return;
  host.innerHTML = state.playSeatDrafts.map((seat, index) => `
    <article class="list-row play-seat-row" data-seat-index="${index}">
      <div class="list-row-head">
        <strong>Seat ${index + 1}</strong>
        <button class="button secondary remove-play-seat-button" type="button" data-seat-index="${index}" ${state.playSeatDrafts.length <= 1 ? "disabled" : ""}>Remove</button>
      </div>
      <div class="grid two">
        <label class="field">
          <span>Seat Type</span>
          <select class="play-seat-type" data-seat-index="${index}">
            <option value="human" ${seat.type === "human" ? "selected" : ""}>Human</option>
            <option value="model-expert" ${seat.type === "model-expert" ? "selected" : ""}>Active Brain Expert</option>
            <option value="model-balanced" ${seat.type === "model-balanced" ? "selected" : ""}>Active Brain Balanced</option>
            <option value="model-beginner" ${seat.type === "model-beginner" ? "selected" : ""}>Active Brain Beginner</option>
            <option value="heuristic" ${seat.type === "heuristic" ? "selected" : ""}>Heuristic AI</option>
            <option value="random" ${seat.type === "random" ? "selected" : ""}>Random AI</option>
          </select>
        </label>
        <label class="field">
          <span>Name</span>
          <input class="play-seat-name" data-seat-index="${index}" type="text" value="${escapeHtml(seat.display_name || "")}" />
        </label>
      </div>
    </article>
  `).join("");

  host.querySelectorAll(".play-seat-type").forEach((select) => {
    select.addEventListener("change", (event) => {
      const index = Number(event.target.dataset.seatIndex);
      state.playSeatDrafts[index].type = event.target.value;
      state.playSeatDrafts[index].display_name = defaultSeatName(event.target.value, index + 1);
      renderPlaySeatEditor();
    });
  });
  host.querySelectorAll(".play-seat-name").forEach((input) => {
    input.addEventListener("input", (event) => {
      const index = Number(event.target.dataset.seatIndex);
      state.playSeatDrafts[index].display_name = event.target.value;
    });
  });
  host.querySelectorAll(".remove-play-seat-button").forEach((button) => {
    button.addEventListener("click", () => {
      if (state.playSeatDrafts.length <= 1) return;
      state.playSeatDrafts.splice(Number(button.dataset.seatIndex), 1);
      renderPlaySeatEditor();
    });
  });
}

export function renderPlayBoard() {
  const host = $("playBoardCard");
  const label = $("playBoardStatus");
  if (!host || !label) return;
  const board = state.playSession?.board;
  if (!board) {
    label.textContent = "No session";
    host.className = "empty-state";
    host.textContent = "Start a shared session to see the board.";
    return;
  }

  const products = board.products || [];
  const cellWidth = 112;
  const rowHeight = 58;
  const labelWidth = 120;
  const width = labelWidth + cellWidth * Math.max(1, products[0]?.cells?.length || 1) + 24;
  const height = 36 + rowHeight * products.length + 12;
  const seatColors = ["#557287", "#4f8d68", "#c58b2f", "#cc5f5f"];
  const seatsById = Object.fromEntries((state.playSession.seats || []).map((seat, index) => [
    seat.id,
    { seat, index, color: seatColors[index % seatColors.length] },
  ]));
  const rows = products.map((product, rowIndex) => {
    const y = 32 + rowIndex * rowHeight;
    const cells = (product.cells || []).map((cell, cellIndex) => {
      const x = labelWidth + cellIndex * cellWidth;
      const classes = ["play-board-cell"];
      if (cell.completed) classes.push("is-complete");
      if (cell.active) classes.push("is-active");
      const badges = [
        cell.incident_delta || cell.incident_override !== null ? "I" : "",
        cell.refinement_delta ? "R" : "",
      ].filter(Boolean);
      const seatMarkers = (cell.active_seats || [])
        .map((seatId, markerIndex) => {
          const meta = seatsById[seatId];
          if (!meta) return "";
          const markerX = x + cellWidth - 24 - markerIndex * 18;
          return `<circle cx="${markerX}" cy="${y + 32}" r="7" fill="${meta.color}"><title>Seat ${meta.index + 1}: ${escapeHtml(meta.seat.controller.display_name)}</title></circle>`;
        })
        .join("");
      return `
        <g>
          <rect class="${classes.join(" ")}" x="${x}" y="${y}" width="${cellWidth - 10}" height="44" rx="6"></rect>
          <text class="play-board-sprint" x="${x + 10}" y="${y + 17}">S${cell.sprint}</text>
          <text class="play-board-meta" x="${x + 10}" y="${y + 34}">${formatNumber(cell.sprint_value, 0)} / f${cell.features_required}</text>
          ${badges.map((badge, badgeIndex) => `<text class="play-board-badge" x="${x + cellWidth - 28 - badgeIndex * 16}" y="${y + 17}">${badge}</text>`).join("")}
          ${seatMarkers}
        </g>
      `;
    }).join("");
    return `
      <g>
        <text class="play-board-product" x="10" y="${y + 25}">${escapeHtml(product.name)}</text>
        ${cells}
      </g>
    `;
  }).join("");

  label.textContent = board.incident?.active ? `Incident: ${board.incident.name}` : "Shared board";
  host.className = "play-board-wrap";
  host.innerHTML = `
    <svg class="play-board-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Shared play board">
      <text class="play-board-header" x="${labelWidth}" y="18">Shared products and sprints</text>
      ${rows}
    </svg>
  `;
}

export function renderPlayTopbar() {
  const sessionCode = $("playSessionCode");
  const roundCode = $("playRoundCode");
  const incidentBanner = $("playIncidentBanner");
  if (!sessionCode || !roundCode || !incidentBanner) return;

  const session = state.playSession;
  sessionCode.textContent = session?.id || "No session";
  roundCode.textContent = session ? String(session.round_number) : "-";

  const incident = session?.board?.incident;
  if (incident?.active) {
    incidentBanner.classList.add("is-active");
    incidentBanner.textContent = `Incident: ${incident.name || "Unknown"}`;
  } else {
    incidentBanner.classList.remove("is-active");
    const latest = session?.round_incidents?.slice(-1)[0];
    incidentBanner.textContent = latest ? `Last incident: ${latest.name}` : "No active incident";
  }
}

export function renderPlayStandings() {
  const host = $("playStandingsCard");
  if (!host) return;
  const rows = state.playSession?.standings || [];
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "Standings will appear after a session starts.";
    return;
  }
  const seatsById = Object.fromEntries((state.playSession?.seats || []).map((seat) => [seat.id, seat]));
  host.className = "play-standings-list";
  host.innerHTML = rows.map((row) => {
    const seat = seatsById[row.seat_id];
    const product = productNameById(seat?.state?.current_product);
    const bank = Number(row.ending_money || 0);
    return `
      <article class="play-score-card">
        <div class="play-score-top">
          <span class="play-score-name">${escapeHtml(row.controller)}</span>
          <strong class="play-bank ${bank < 0 ? "is-negative" : ""}">${formatCurrency(bank)}</strong>
        </div>
        <div class="play-score-meta">
          <span class="tag">${escapeHtml(row.type)}</span>
          <span class="tag">${escapeHtml(product)} S${escapeHtml(String(seat?.state?.current_sprint ?? "-"))}</span>
          <span class="tag ${seat?.state?.loan_active ? "bad" : "good"}">${seat?.state?.loan_active ? "loan" : "cash"}</span>
          <span class="tag ${row.done ? "bad" : "good"}">${row.done ? "done" : "active"}</span>
        </div>
      </article>
    `;
  }).join("");
}

export function renderPlayTurnLog() {
  const host = $("playTurnLogCard");
  if (!host) return;
  const rows = state.playSession?.turn_log || [];
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "Turns will appear after the first round.";
    return;
  }
  host.className = "play-turn-log-list";
  host.innerHTML = rows.slice(-24).reverse().map((row) => `
    <article class="play-log-row">
      <strong>R${escapeHtml(String(row.round))} - ${escapeHtml(row.controller)} - ${escapeHtml(row.action)}</strong>
      <div class="play-log-meta">
        <span class="tag ${row.outcome === "Success" ? "good" : row.outcome === "Invalid" ? "bad" : ""}">${escapeHtml(row.outcome)}</span>
        <span class="tag">bank ${formatCurrency(row.bank)}</span>
        <span class="tag ${Number(row.reward) >= 0 ? "good" : "bad"}">reward ${formatCurrency(row.reward)}</span>
      </div>
      <p>${escapeHtml(row.product || "-")} sprint ${escapeHtml(String(row.sprint ?? "-"))} · ${escapeHtml(row.refinement || "none")}</p>
    </article>
  `).join("");
}

export function renderPlayActionButtons(humanSeat) {
  const host = $("playActionButtonGrid");
  const select = $("playHumanActionSelect");
  if (!host || !select) return;

  if (!humanSeat) {
    host.innerHTML = "";
    return;
  }

  const validActions = new Map((humanSeat.valid_actions || []).map((action) => [Number(action.action_id), action]));
  const products = state.playSession?.board?.products || [];
  const allActions = [
    { action_id: 0, label: "Continue", hint: "Play current sprint" },
    ...products.map((product) => ({
      action_id: Number(product.product_id),
      label: `Switch to ${product.name}`,
      hint: `Start S${humanSeat.state?.target_next_sprints?.[Number(product.product_id) - 1] || 1}`,
    })),
  ];
  const selected = Number(select.value || (humanSeat.valid_actions || [])[0]?.action_id || 0);

  host.innerHTML = allActions.map((action) => {
    const valid = validActions.has(Number(action.action_id));
    const label = validActions.get(Number(action.action_id))?.label || action.label;
    return `
      <button class="play-action-button ${selected === Number(action.action_id) ? "is-selected" : ""}" type="button"
        data-action-id="${action.action_id}" ${valid ? "" : "disabled"}>
        <strong>${escapeHtml(label)}</strong>
        <span>${valid ? escapeHtml(action.hint || "Available") : "Locked"}</span>
      </button>
    `;
  }).join("");

  host.querySelectorAll(".play-action-button").forEach((button) => {
    button.addEventListener("click", () => {
      select.value = button.dataset.actionId;
      renderPlayActionButtons(humanSeat);
      renderPlayDicePreview(humanSeat, Number(button.dataset.actionId));
    });
  });
}

export function renderPlaySession() {
  const card = $("playSessionCard");
  const humanWrap = $("playHumanActionWrap");
  const setupForm = $("playSessionForm");
  const addSeatButton = $("addPlaySeatButton");
  const actionPanel = document.querySelector(".play-action-panel");
  renderPlayTopbar();
  if (!state.playSession) {
    actionPanel?.classList.remove("is-playing");
    actionPanel?.classList.add("is-setup");
    setupForm?.classList.remove("hidden");
    if (addSeatButton) addSeatButton.classList.remove("hidden");
    card.className = "play-session-summary hidden";
    card.textContent = "";
    humanWrap.className = "play-action-wrap hidden";
    renderPlayBoard();
    renderPlayStandings();
    renderPlayTurnLog();
    renderPlayDiceZone();
    return;
  }

  actionPanel?.classList.remove("is-setup");
  actionPanel?.classList.add("is-playing");
  setupForm?.classList.add("hidden");
  if (addSeatButton) addSeatButton.classList.add("hidden");
  card.className = "play-session-summary play-session-summary-card";
  card.innerHTML = `
    <div class="play-score-meta">
      <span class="tag">${state.playSession.done ? "complete" : "in progress"}</span>
      <span class="tag">${state.playSession.seats.length} seats</span>
      <span class="tag">seed ${state.playSession.base_seed}</span>
    </div>
  `;

  const humanSeat = state.playSession.seats.find((seat) => seat.controller.type === "human" && !seat.done);
  renderPlayBoard();
  renderPlayStandings();
  renderPlayTurnLog();
  renderPlayDiceZone();
  if (!humanSeat) {
    humanWrap.className = "play-action-wrap hidden";
    renderPlayActionButtons(null);
    return;
  }

  humanWrap.className = "play-action-wrap";
  buildOptions("playHumanActionSelect", humanSeat.valid_actions || [], "action_id", "label", "No actions");
  renderPlayActionButtons(humanSeat);
}
