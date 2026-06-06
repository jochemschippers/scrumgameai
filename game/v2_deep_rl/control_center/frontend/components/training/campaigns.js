/** Implement campaigns user-interface behavior. */

import { state } from '../../state/store.js';
import { $, showMessage } from '../../utils/helpers.js';
import { escapeHtml } from '../../utils/formatting.js';
import { apiRequest } from '../../api/client.js';

/** Render campaign panel. */
export function renderCampaignPanel() {
  const card = $("campaignCard");
  const label = $("campaignStatusLabel");
  const stopButton = $("stopCampaignButton");
  const escalateButton = $("escalateCampaignButton");
  if (!card || !label || !stopButton || !escalateButton) return;

  const active = state.campaigns.find((campaign) => campaign.status === "running");
  const completed = state.campaigns.find((campaign) => campaign.status === "completed");
  const display = active || completed || null;

  if (!display) {
    state.activeCampaignId = null;
    label.textContent = "-";
    card.className = "empty-state";
    card.textContent = "No active campaign.";
    stopButton.style.display = "none";
    escalateButton.style.display = "none";
    return;
  }

  state.activeCampaignId = display.campaign_id;
  const done = Number(display.variations_completed || 0);
  const total = Number(display.max_variations || 0);
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const history = display.variation_history || [];

  label.textContent = display.status;
  card.className = "list-card";
  card.innerHTML = `
    <h4>${escapeHtml(display.campaign_id)}</h4>
    <div class="card-meta">
      <span class="tag">${escapeHtml(display.status)}</span>
      <span class="tag">${done} / ${total} variations</span>
      <span class="tag">${pct}%</span>
      ${display.escalate_mode ? `<span class="tag warn">escalated</span>` : ""}
    </div>
    <div class="card-meta">
      <span class="tag">base ${escapeHtml(display.base_run_id || "-")}</span>
      <span class="tag">current ${escapeHtml(display.current_run_id || "-")}</span>
    </div>
    ${history.length
      ? `<div class="decision-list" style="margin-top:0.5rem;">${history.map((item) => `
          <div class="decision-row">
            <div class="decision-row-head">
              <span class="tag">v${escapeHtml(String(item.index ?? "-"))}</span>
              ${item.escalate ? `<span class="tag warn">escalate</span>` : ""}
              <span class="tag">${escapeHtml(item.to_run || "-")}</span>
            </div>
            <p class="decision-reason">${escapeHtml(item.reason || "")}</p>
          </div>
        `).join("")}</div>`
      : `<p class="muted">Waiting for the first plateau stop.</p>`
    }
  `;
  stopButton.style.display = display.status === "running" ? "" : "none";
  escalateButton.style.display = display.status === "completed" ? "" : "none";
}

/** Refresh campaigns. */
export async function refreshCampaigns() {
  try {
    state.campaigns = await apiRequest("/campaigns");
  } catch (_error) {
    state.campaigns = [];
  }
  renderCampaignPanel();
}
