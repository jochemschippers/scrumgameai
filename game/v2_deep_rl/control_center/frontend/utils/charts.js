/** Implement charts behavior for the utils package. */

import { escapeHtml, formatNumber } from './formatting.js';
import { $ } from './helpers.js';

/** Build polyline. */
export function buildPolyline(points, width, height) {
  if (!points.length) return "";
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
}

/** Render table. */
export function renderTable(hostId, columns, rows) {
  const host = $(hostId);
  if (!rows.length) {
    host.className = "empty-state";
    host.textContent = "No rows yet.";
    return;
  }
  host.className = "data-table-wrap";
  const header = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((column) => `<td>${escapeHtml(String(row[column.key] ?? "-"))}</td>`)
          .join("")}</tr>`
    )
    .join("");
  host.innerHTML = `<table class="data-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

/** Render line chart. */
export function renderLineChart(hostId, series, valueKey, lineColor, caption) {
  const host = $(hostId);
  const filtered = (series || []).filter((item) => Number.isFinite(item?.[valueKey]));
  if (!filtered.length) {
    host.className = "empty-state";
    host.textContent = caption || "No data yet.";
    return;
  }

  const width = 640;
  const height = 180;
  const padding = 18;
  const values = filtered.map((item) => Number(item[valueKey]));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const range = maxValue - minValue || 1;
  const points = filtered.map((item, index) => {
    const x =
      padding +
      (filtered.length === 1 ? 0 : (index / (filtered.length - 1)) * (width - padding * 2));
    const normalized = (Number(item[valueKey]) - minValue) / range;
    const y = height - padding - normalized * (height - padding * 2);
    return { x, y };
  });
  const polyline = buildPolyline(points, width, height);
  const last = filtered[filtered.length - 1];
  host.className = "";
  host.innerHTML = `
    <svg class="mini-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(valueKey)} chart">
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="rgba(170,177,195,0.25)" stroke-width="1" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="rgba(170,177,195,0.25)" stroke-width="1" />
      <polyline fill="none" stroke="${lineColor}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />
    </svg>
    <div class="chart-caption">Latest ${escapeHtml(valueKey)}: ${escapeHtml(formatNumber(last[valueKey]))}${caption ? ` | ${escapeHtml(caption)}` : ""}</div>
  `;
}

/** Render bar chart. */
export function renderBarChart(hostId, rows, valueKey, labelKey, positiveColor, negativeColor, caption) {
  const host = $(hostId);
  const filtered = (rows || []).filter((item) => Number.isFinite(Number(item?.[valueKey])));
  if (!filtered.length) {
    host.className = "empty-state";
    host.textContent = caption || "No data yet.";
    return;
  }

  const width = 640;
  const height = 220;
  const padding = 24;
  const values = filtered.map((item) => Number(item[valueKey]));
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1);
  const zeroY = height / 2;
  const barWidth = Math.max(18, Math.min(60, (width - padding * 2) / filtered.length - 8));
  const gap = ((width - padding * 2) - barWidth * filtered.length) / Math.max(filtered.length - 1, 1);
  const bars = filtered
    .map((item, index) => {
      const value = Number(item[valueKey]);
      const magnitude = Math.abs(value) / maxAbs;
      const barHeight = magnitude * (height / 2 - padding);
      const x = padding + index * (barWidth + gap);
      const y = value >= 0 ? zeroY - barHeight : zeroY;
      const fill = value >= 0 ? positiveColor : negativeColor;
      const seedLabel = String(item[labelKey] ?? index + 1);
      return `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="3" fill="${fill}" />
        <text x="${(x + barWidth / 2).toFixed(1)}" y="${height - 8}" text-anchor="middle" font-size="10" fill="#667085">${escapeHtml(seedLabel)}</text>
      `;
    })
    .join("");
  host.className = "";
  host.innerHTML = `
    <svg class="bar-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(valueKey)} bar chart">
      <line x1="${padding}" y1="${zeroY}" x2="${width - padding}" y2="${zeroY}" stroke="rgba(102,112,133,0.35)" stroke-width="1" />
      ${bars}
    </svg>
    <div class="chart-caption">${escapeHtml(caption || `Values by ${labelKey}`)}</div>
  `;
}
