"""Live learner dashboard — a tiny stdlib HTTP server over runs/.

Tails the newest `events_*.jsonl` and `examples_*/` in the runs directory and
serves a self-refreshing page with: current state, rolling probe MSE chart,
latest probe grid image, per-action means, and the raw event stream. No extra
dependencies — pure `http.server`.

Usage:
    python scripts/dashboard.py                   # http://127.0.0.1:8765
    python scripts/dashboard.py --port 9000

Leave it running in one terminal, then start the learner in another:
    python -m learner --config configs/hardware_run.yaml

The dashboard auto-picks whichever session is newest, so it keeps following
the latest run without restart.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO_ROOT / "runs"


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>canvas-autonomous-learner — live</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --ink:#e6e8ef; --muted:#8a93a6; --accent:#5aa9ff; --warn:#ff9c5a; --ok:#65d88c; --bad:#ff6464; --purple:#d0a3ff; }
  html, body { background: var(--bg); color: var(--ink); font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; min-width: 480px; }
  header { padding: 14px 20px; border-bottom: 1px solid #2a2f3b; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; }
  header .session { color: var(--muted); font-size: 12px; }
  .pill { font-size: 11px; padding: 3px 9px; border-radius: 12px; background:#2a2f3b; color: var(--muted); }
  .pill.ok { background:#12361e; color: var(--ok); }
  .pill.warn { background:#3a2a12; color: var(--warn); }
  .pill.bad { background:#3a1212; color: var(--bad); }
  .pill.purple { background:#2a1a3b; color: var(--purple); }
  main { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; padding: 14px 20px; }
  .panel { background: var(--panel); border: 1px solid #232735; border-radius: 8px; padding: 12px 14px; }
  .panel h2 { font-size: 11px; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); }
  .metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  .metric { flex: 1 1 88px; background: #1d2230; border-left: 3px solid var(--accent); padding: 6px 10px; border-radius: 4px; }
  .metric b { display: block; font-size: 17px; color: var(--ink); font-variant-numeric: tabular-nums; }
  .metric span { font-size: 11px; color: var(--muted); }
  img.probe { width: 100%; max-width: 720px; display: block; border: 1px solid #2a2f3b; border-radius: 4px; }
  /* SVGs fill the container width up to a per-chart max, and use their
     inline `height` attribute for vertical size (no height: auto — that
     was blowing the coverage histogram to ~360px tall and the locked-val
     trajectory to ~600px tall on wide viewports). The default SVG
     `preserveAspectRatio="xMidYMid meet"` letterboxes horizontally if the
     container is wider than viewBox × (height/viewBox_h), which is fine. */
  /* SVGs fill their panel width. Per-chart max-widths were removed so the
     charts scale with the responsive grid instead of sitting at a fixed
     size in a stretched panel. */
  svg { display: block; width: 100%; }
  .events { max-height: 320px; overflow-y: auto; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; color: #c8cddc; }
  .events .row { padding: 2px 0; border-bottom: 1px dashed #232735; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .events .name { color: var(--accent); }
  .events .name.probe { color: var(--ok); }
  .events .name.verify_summary { color: var(--warn); }
  .events .name.state { color: var(--purple); }
  .events .name.explore_start, .events .name.explore_done, .events .name.retrain_accepted, .events .name.checkpoint_swapped { color: var(--warn); }
  .events .name.explore_failed, .events .name.retrain_rejected, .events .name.retrain_rejected_or_failed { color: var(--bad); }
  .ts { color: var(--muted); margin-right: 8px; }
  .empty { color: var(--muted); font-style: italic; }
  code { background: #1d2230; padding: 1px 5px; border-radius: 3px; font-size: 12px; color: #c8cddc; }
  /* Curriculum dashboard additions */
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; padding: 12px 20px; background: #12151c; border-bottom: 1px solid #2a2f3b; }
  .card { background: #1d2230; border-left: 3px solid var(--accent); padding: 6px 12px; border-radius: 4px; }
  .card b { display: block; font-size: 17px; color: var(--ink); font-variant-numeric: tabular-nums; }
  .card span { font-size: 11px; color: var(--muted); }
  .phase-pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .phase-pill.explore { background: #12361e; color: var(--ok); }
  .phase-pill.verify { background: #2a1a3b; color: var(--purple); }
  .phase-pill.train_diffusion { background: #3a2a12; color: var(--warn); }
  .phase-pill.evaluate { background: #12263a; color: var(--accent); }
  .phase-pill.processing, .phase-pill.idle, .phase-pill.done { background: #2a2f3b; color: var(--muted); }
  main .full { grid-column: 1 / -1; }
  .heatmap { display: grid; gap: 1px; background: #232735; padding: 1px; border-radius: 4px; font-size: 9px; }
  .heatmap .hcell { background: #1d2230; padding: 4px 2px; text-align: center; color: #c8cddc; position: relative; }
  .heatmap .hhdr { background: #1d2230; color: var(--muted); padding: 3px; text-align: center; font-weight: 600; font-size: 10px; }
  .heatmap .hhdr.active { background: #12263a; color: var(--accent); }
  .sub-bursts { list-style: none; margin: 0; padding: 0; }
  .sub-bursts li { padding: 6px 10px; background: #1d2230; margin-bottom: 4px; border-radius: 3px; display: flex; justify-content: space-between; font-size: 12px; }
  .sub-bursts li .rng { color: var(--accent); font-variant-numeric: tabular-nums; }
  .sub-bursts li .n { color: var(--muted); }
  #action-canvas-gallery { display: flex; flex-direction: column; gap: 12px; max-width: 900px; margin: 0 auto; }
  #action-canvas-gallery figure { margin: 0; background: #1d2230; border: 1px solid #232735; border-radius: 4px; padding: 10px; }
  #action-canvas-gallery figure img { width: 100%; display: block; border-radius: 2px; }
  #action-canvas-gallery figcaption { font-size: 10px; color: var(--muted); margin-top: 4px; text-align: center; font-family: ui-monospace, Menlo, Consolas, monospace; }
  /* `auto-fit` in the main grid already handles reflow across widths,
     so no explicit breakpoint is needed. */
</style></head><body>

<header>
  <h1>canvas-autonomous-learner</h1>
  <span class="session" id="session">—</span>
  <span class="pill" id="state-pill">init</span>
  <span class="pill" id="connected-pill">connecting…</span>
  <span class="session" id="duration"></span>
</header>

<div class="strip">
  <div class="card"><b id="m-cycle">—</b><span>cycle</span></div>
  <div class="card"><b id="m-eps">0</b><span>episodes</span></div>
  <div class="card"><b id="m-explore-progress">—</b><span>explore</span></div>
  <div class="card"><b id="m-retrains">0</b><span>retrains</span></div>
  <div class="card"><b id="m-phase"><span class="phase-pill">—</span></b><span>phase</span></div>
  <div class="card"><b id="m-active-range">—</b><span>active range</span></div>
  <div class="card"><b id="m-arm-a">—</b><span>Arm A target</span></div>
</div>

<main>
  <div class="panel full">
    <h2>Locked-val MSE trajectory</h2>
    <div id="locked-val-chart"></div>
    <div id="locked-val-empty" class="empty" style="display: none;">(no locked_val_measured events yet)</div>
  </div>

  <div class="panel">
    <h2>State × action error heatmap</h2>
    <div id="heatmap"></div>
    <div id="heatmap-empty" class="empty" style="display: none;">(no probes with motor_state in this session)</div>
  </div>

  <div class="panel">
    <h2>Coverage histogram</h2>
    <div id="coverage"></div>
  </div>

  <div class="panel">
    <h2>Last EXPLORE sub-bursts</h2>
    <ul class="sub-bursts" id="sub-bursts"><li class="empty">(no explore burst yet)</li></ul>
  </div>

  <div class="panel">
    <h2>EXPLORE actions taken</h2>
    <div id="joint-series"><div class="empty">(no joint data yet)</div></div>
    <div id="explore-actions" style="margin-top:10px;"><div class="empty">(no actions yet)</div></div>
  </div>

  <!-- Training panel (only rendered when a retrain is in progress or
       recently completed — hidden on cold start before first retrain). -->
  <div class="panel full" id="panel-training" style="display: none;">
    <h2>Training progress · <span id="training-meta" class="session"></span></h2>
    <div id="training-chart"></div>
    <div class="metrics" style="margin-top: 10px;">
      <div class="metric"><b id="t-epoch">—</b><span>epoch</span></div>
      <div class="metric"><b id="t-train-loss">—</b><span>train loss</span></div>
      <div class="metric"><b id="t-val-loss">—</b><span>val loss</span></div>
      <div class="metric"><b id="t-best-val">—</b><span>best val</span></div>
      <div class="metric"><b id="t-train-canvases">—</b><span>train canvases</span></div>
      <div class="metric"><b id="t-val-canvases">—</b><span>val canvases</span></div>
    </div>
  </div>

  <div class="panel full" id="panel-action-canvas-gallery">
    <h2>Most recent <span id="action-canvas-gallery-n">5</span> action canvases · before | predicted | actual</h2>
    <div id="action-canvas-gallery"><div class="empty">(no action canvases yet)</div></div>
  </div>

  <div class="panel">
    <h2>Verification MSE · rolling (last 30 probes)
      <button id="btn-verify-now" style="float:right;background:#2c7fb8;color:#fff;border:0;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px;">Verify now</button>
    </h2>
    <div id="chart"></div>
    <div class="metrics">
      <div class="metric"><b id="m-probes">0</b><span>probes</span></div>
      <div class="metric"><b id="m-probes-cycle">0</b><span>probes this cycle</span></div>
      <div class="metric"><b id="m-mean">—</b><span>mean</span></div>
      <div class="metric"><b id="m-last">—</b><span>last</span></div>
      <div class="metric"><b id="m-best">—</b><span>best</span></div>
      <div class="metric"><b id="m-worst">—</b><span>worst</span></div>
    </div>
  </div>

  <div class="panel full">
    <h2>Verification MSE · all-time
      <select id="alltime-range" style="float:right;background:#1d2230;color:#dfe6f2;border:1px solid #2b3246;padding:3px 6px;font-size:11px;border-radius:3px;">
        <option value="5">last 5 min</option>
        <option value="15">last 15 min</option>
        <option value="60">last 1 hour</option>
        <option value="240">last 4 hours</option>
        <option value="0" selected>all time</option>
      </select>
    </h2>
    <div id="alltime-chart"></div>
  </div>

  <div class="panel">
    <h2>Per-action mean MSE</h2>
    <div id="action-chart"></div>
  </div>

  <div class="panel">
    <h2>Event stream</h2>
    <div class="events" id="events"><div class="empty">(waiting for events…)</div></div>
  </div>
</main>

<script>
const POLL_MS = 15000;   // 15 seconds — matches ~14s-per-episode explore rate
const MAX_EVENTS = 80;
const ACTION_CANVAS_GALLERY_SIZE = 5;

function fmtMse(v) { return (v == null) ? "—" : v.toFixed(5); }
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function lineChart(values, width, height, opts) {
  opts = opts || {};
  const padL = 48, padR = 12, padT = 10, padB = 24;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  if (!values.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no probes yet</text></svg>`;
  }
  const ymin = Math.min.apply(null, values);
  const ymax = Math.max.apply(null, values);
  const pad = (ymax - ymin) * 0.1 || Math.max(1e-6, ymax * 0.1);
  const y0 = ymin - pad, y1 = ymax + pad;
  const xMax = Math.max(1, values.length - 1);
  const X = i => padL + (i / xMax) * plotW;
  const Y = v => padT + (1 - (v - y0) / (y1 - y0)) * plotH;
  const pts = values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
  const circles = values.map((v, i) => `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="2.6" fill="#5aa9ff"/>`).join("");
  let ticks = "";
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = y0 + frac * (y1 - y0);
    const py = padT + (1 - frac) * plotH;
    ticks += `<line x1="${padL}" y1="${py.toFixed(1)}" x2="${padL + plotW}" y2="${py.toFixed(1)}" stroke="#232735"/>
              <text x="${padL - 6}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(4)}</text>`;
  }
  let hlineSvg = "";
  if (opts.hline != null && opts.hline >= y0 && opts.hline <= y1) {
    const hy = Y(opts.hline);
    hlineSvg = `<line x1="${padL}" y1="${hy.toFixed(1)}" x2="${padL + plotW}" y2="${hy.toFixed(1)}" stroke="#ff9c5a" stroke-dasharray="4,4"/>
                <text x="${padL + plotW - 4}" y="${(hy - 4).toFixed(1)}" text-anchor="end" fill="#ff9c5a" font-size="10">mean ${opts.hline.toFixed(5)}</text>`;
  }
  // Optional tau reference lines (green = tau_low "good enough", red = tau_high "must explore")
  let tauSvg = "";
  if (opts.tau_low != null && opts.tau_low >= y0 && opts.tau_low <= y1) {
    const ly = Y(opts.tau_low);
    tauSvg += `<line x1="${padL}" y1="${ly.toFixed(1)}" x2="${padL + plotW}" y2="${ly.toFixed(1)}" stroke="#7cd992" stroke-dasharray="2,3"/>
               <text x="${padL + 4}" y="${(ly - 3).toFixed(1)}" fill="#7cd992" font-size="10">τ_low ${opts.tau_low.toFixed(4)}</text>`;
  }
  if (opts.tau_high != null && opts.tau_high >= y0 && opts.tau_high <= y1) {
    const hy = Y(opts.tau_high);
    tauSvg += `<line x1="${padL}" y1="${hy.toFixed(1)}" x2="${padL + plotW}" y2="${hy.toFixed(1)}" stroke="#ff6b6b" stroke-dasharray="2,3"/>
               <text x="${padL + 4}" y="${(hy - 3).toFixed(1)}" fill="#ff6b6b" font-size="10">τ_high ${opts.tau_high.toFixed(4)}</text>`;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${ticks}
    ${hlineSvg}
    ${tauSvg}
    <polyline points="${pts}" fill="none" stroke="#5aa9ff" stroke-width="1.8"/>
    ${circles}
  </svg>`;
}

function timeSeriesChart(points, width, height, opts) {
  // points: [{t: seconds_since_epoch, value: number}, ...]
  opts = opts || {};
  const padL = 52, padR = 12, padT = 10, padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  if (!points.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no probes in range</text></svg>`;
  }
  const tmin = points[0].t;
  const tmax = points[points.length - 1].t;
  const tspan = Math.max(1, tmax - tmin);
  const vals = points.map(p => p.value);
  const ymin = Math.min.apply(null, vals);
  const ymax = Math.max.apply(null, vals);
  const pad = (ymax - ymin) * 0.1 || Math.max(1e-6, ymax * 0.1);
  const y0 = ymin - pad, y1 = ymax + pad;
  const X = t => padL + ((t - tmin) / tspan) * plotW;
  const Y = v => padT + (1 - (v - y0) / (y1 - y0)) * plotH;
  const pts = points.map(p => `${X(p.t).toFixed(1)},${Y(p.value).toFixed(1)}`).join(" ");
  const circles = points.length <= 200
    ? points.map(p => `<circle cx="${X(p.t).toFixed(1)}" cy="${Y(p.value).toFixed(1)}" r="1.8" fill="#5aa9ff"/>`).join("")
    : "";
  let yticks = "";
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = y0 + frac * (y1 - y0);
    const py = padT + (1 - frac) * plotH;
    yticks += `<line x1="${padL}" y1="${py.toFixed(1)}" x2="${padL + plotW}" y2="${py.toFixed(1)}" stroke="#232735"/>
               <text x="${padL - 6}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(4)}</text>`;
  }
  // X-axis labels: 5 ticks with elapsed-time labels (mm:ss or Hh)
  let xticks = "";
  function fmtElapsed(sec) {
    if (sec < 60) return sec.toFixed(0) + "s";
    if (sec < 3600) return (sec / 60).toFixed(0) + "m";
    return (sec / 3600).toFixed(1) + "h";
  }
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const t = tmin + frac * tspan;
    const px = padL + frac * plotW;
    const label = i === 0 ? "start" : ("-" + fmtElapsed(tmax - t));
    xticks += `<text x="${px.toFixed(1)}" y="${(padT + plotH + 14).toFixed(1)}" text-anchor="middle" fill="#8a93a6" font-size="10">${label}</text>`;
  }
  let tauSvg = "";
  if (opts.tau_low != null && opts.tau_low >= y0 && opts.tau_low <= y1) {
    const ly = Y(opts.tau_low);
    tauSvg += `<line x1="${padL}" y1="${ly.toFixed(1)}" x2="${padL + plotW}" y2="${ly.toFixed(1)}" stroke="#7cd992" stroke-dasharray="2,3"/>
               <text x="${padL + 4}" y="${(ly - 3).toFixed(1)}" fill="#7cd992" font-size="10">τ_low ${opts.tau_low.toFixed(4)}</text>`;
  }
  if (opts.tau_high != null && opts.tau_high >= y0 && opts.tau_high <= y1) {
    const hy = Y(opts.tau_high);
    tauSvg += `<line x1="${padL}" y1="${hy.toFixed(1)}" x2="${padL + plotW}" y2="${hy.toFixed(1)}" stroke="#ff6b6b" stroke-dasharray="2,3"/>
               <text x="${padL + 4}" y="${(hy - 3).toFixed(1)}" fill="#ff6b6b" font-size="10">τ_high ${opts.tau_high.toFixed(4)}</text>`;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${yticks}
    ${xticks}
    ${tauSvg}
    <polyline points="${pts}" fill="none" stroke="#5aa9ff" stroke-width="1.4"/>
    ${circles}
  </svg>`;
}

function barChart(bars, width, height) {
  const padL = 48, padR = 12, padT = 10, padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  if (!bars.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no probes yet</text></svg>`;
  }
  const vmax = Math.max.apply(null, bars.map(b => b.value)) || 1e-9;
  const slot = plotW / bars.length;
  const bw = slot * 0.55;
  let ticks = "";
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = frac * vmax;
    const py = padT + (1 - frac) * plotH;
    ticks += `<line x1="${padL}" y1="${py.toFixed(1)}" x2="${padL + plotW}" y2="${py.toFixed(1)}" stroke="#232735"/>
              <text x="${padL - 6}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(4)}</text>`;
  }
  const rects = bars.map((b, i) => {
    const x = padL + i * slot + (slot - bw) / 2;
    const h = (b.value / vmax) * plotH;
    const y = padT + plotH - h;
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="#5aa9ff"/>
            <text x="${(x + bw/2).toFixed(1)}" y="${(padT + plotH + 16).toFixed(1)}" text-anchor="middle" fill="#c8cddc" font-size="11">${b.label}</text>
            <text x="${(x + bw/2).toFixed(1)}" y="${(y - 4).toFixed(1)}" text-anchor="middle" fill="#c8cddc" font-size="10">${b.value.toFixed(4)}</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${ticks}
    ${rects}
  </svg>`;
}

function stateClass(state) {
  if (state === "EXPLORE") return "bad";
  if (state === "RETRAIN") return "warn";
  if (state === "VERIFY") return "ok";
  if (state === "IDLE") return "purple";
  return "";
}

function renderHeatmap(hm, activeRange) {
  if (!hm || !hm.cells || !hm.cells.length) {
    return null;
  }
  const nBins = Math.max(1, (hm.state_bins || []).length - 1);
  const actions = hm.actions || [1, 2, 3];
  const maxMse = hm.max_mse || 1e-9;
  // Index cells by [bin, action] for fast lookup
  const grid = {};
  hm.cells.forEach(c => {
    grid[c.state_bin + ":" + c.action] = c;
  });
  const active = activeRange || [null, null];

  // Build a grid using CSS grid: 1 header row + nBins rows, 1 header col + 3 action cols
  let html = `<div class="heatmap" style="grid-template-columns: 72px repeat(${actions.length}, 1fr);">`;
  html += `<div class="hhdr">state →</div>`;
  actions.forEach(a => {
    const label = a === 1 ? "1 (+)" : a === 2 ? "2 (−)" : "3 (hold)";
    html += `<div class="hhdr">${label}</div>`;
  });
  for (let b = 0; b < nBins; b++) {
    const binLo = hm.state_bins[b];
    const binHi = hm.state_bins[b + 1];
    const inActive = (active[0] != null && active[1] != null &&
                      binHi >= active[0] - 0.001 && binLo <= active[1] + 0.001);
    const hdrCls = inActive ? "hhdr active" : "hhdr";
    html += `<div class="${hdrCls}">[${binLo.toFixed(0)},${binHi.toFixed(0)}]</div>`;
    actions.forEach(a => {
      const c = grid[b + ":" + a];
      if (!c) {
        html += `<div class="hcell" style="background:#171a21;color:#555;">·</div>`;
        return;
      }
      const t = Math.min(1.0, c.mean_mse / maxMse);
      // Red-to-green colormap: high t = red, low t = green
      const r = Math.round(40 + 200 * t);
      const g = Math.round(160 - 120 * t);
      const bl = 40;
      const opacity = Math.min(1.0, 0.3 + c.n / 5.0);
      const tooltip = `state [${binLo.toFixed(0)}, ${binHi.toFixed(0)}], action=${a}: n=${c.n} mse=${c.mean_mse.toFixed(4)}`;
      html += `<div class="hcell" style="background:rgba(${r},${g},${bl},${opacity.toFixed(2)});" title="${tooltip}">${c.n}</div>`;
    });
  }
  html += `</div>`;
  return html;
}

function renderCoverageHist(cov, activeRange, width, height) {
  width = width || 480;
  height = height || 180;
  if (!cov || !cov.counts || !cov.counts.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no coverage data yet</text></svg>`;
  }
  const padL = 40, padR = 10, padT = 8, padB = 24;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const nBins = cov.counts.length;
  const slot = plotW / nBins;
  const maxCount = Math.max.apply(null, cov.counts) || 1;
  const active = activeRange || [null, null];
  let bars = "";
  for (let i = 0; i < nBins; i++) {
    const binLo = cov.bin_edges[i];
    const binHi = cov.bin_edges[i + 1];
    const inActive = (active[0] != null && active[1] != null &&
                      binHi >= active[0] - 0.001 && binLo <= active[1] + 0.001);
    const color = inActive ? "#5aa9ff" : "#3a4055";
    const h = (cov.counts[i] / maxCount) * plotH;
    const x = padL + i * slot + 1;
    const y = padT + plotH - h;
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(slot - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${color}"/>`;
  }
  // X-axis labels at start / mid / end
  const labels = [0, Math.floor(nBins / 2), nBins - 1].map(i => {
    const binLo = cov.bin_edges[i];
    const x = padL + i * slot + slot / 2;
    return `<text x="${x.toFixed(1)}" y="${(padT + plotH + 14).toFixed(1)}" text-anchor="middle" fill="#8a93a6" font-size="10">${binLo.toFixed(0)}</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${bars}
    ${labels}
  </svg>`;
}

function renderLockedValChart(history, armA, rangeHistory, width, height) {
  width = width || 780;
  height = height || 240;
  if (!history || !history.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no locked_val_measured events yet</text></svg>`;
  }
  const padL = 56, padR = 20, padT = 14, padB = 30;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xs = history.map(h => h.total_eps);
  const ys = history.map(h => Number(h.locked_val));
  const allY = ys.slice();
  if (armA != null) allY.push(Number(armA));
  const ymin = Math.min.apply(null, allY);
  const ymax = Math.max.apply(null, allY);
  const pad = (ymax - ymin) * 0.1 || 1e-6;
  const y0 = ymin - pad, y1 = ymax + pad;
  const xMin = Math.min.apply(null, xs);
  const xMax = Math.max.apply(null, xs);
  const xRange = Math.max(1, xMax - xMin);
  const X = v => padL + ((v - xMin) / xRange) * plotW;
  const Y = v => padT + (1 - (v - y0) / (y1 - y0)) * plotH;
  const pts = history.map((h, i) => `${X(h.total_eps).toFixed(1)},${Y(h.locked_val).toFixed(1)}`).join(" ");
  const circles = history.map((h, i) => {
    const fill = h.accepted ? "#5aa9ff" : "#ff6464";
    return `<circle cx="${X(h.total_eps).toFixed(1)}" cy="${Y(h.locked_val).toFixed(1)}" r="3" fill="${fill}"/>`;
  }).join("");
  // Y ticks
  let ticks = "";
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = y0 + frac * (y1 - y0);
    const py = padT + (1 - frac) * plotH;
    ticks += `<line x1="${padL}" y1="${py.toFixed(1)}" x2="${padL + plotW}" y2="${py.toFixed(1)}" stroke="#232735"/>
              <text x="${padL - 6}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(4)}</text>`;
  }
  // X ticks (start / mid / end)
  let xticks = "";
  [0, Math.floor(history.length / 2), history.length - 1].forEach(i => {
    const v = history[i].total_eps;
    xticks += `<text x="${X(v).toFixed(1)}" y="${(padT + plotH + 14).toFixed(1)}" text-anchor="middle" fill="#8a93a6" font-size="10">${v} eps</text>`;
  });
  // Arm A reference line
  let armASvg = "";
  if (armA != null && armA >= y0 && armA <= y1) {
    const hy = Y(armA);
    armASvg = `<line x1="${padL}" y1="${hy.toFixed(1)}" x2="${padL + plotW}" y2="${hy.toFixed(1)}" stroke="#ff6464" stroke-dasharray="4,4"/>
               <text x="${padL + plotW - 4}" y="${(hy - 4).toFixed(1)}" text-anchor="end" fill="#ff6464" font-size="10">Arm A ${Number(armA).toFixed(5)}</text>`;
  }
  // Range expansion vertical markers
  let rangeMarkers = "";
  (rangeHistory || []).forEach(r => {
    if (r.total_eps == null) return;
    const rx = X(r.total_eps);
    if (rx < padL || rx > padL + plotW) return;
    const label = `→ [${r.new_range[0]},${r.new_range[1]}]`;
    rangeMarkers += `<line x1="${rx.toFixed(1)}" y1="${padT}" x2="${rx.toFixed(1)}" y2="${padT + plotH}" stroke="#d0a3ff" stroke-dasharray="2,3"/>
                     <text x="${rx.toFixed(1)}" y="${(padT - 2).toFixed(1)}" text-anchor="middle" fill="#d0a3ff" font-size="9">${label}</text>`;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${ticks}
    ${xticks}
    ${armASvg}
    ${rangeMarkers}
    <polyline points="${pts}" fill="none" stroke="#5aa9ff" stroke-width="1.8"/>
    ${circles}
  </svg>`;
}

function renderSubBursts(sub) {
  if (!sub || !sub.length) {
    return '<li class="empty">(no explore burst yet)</li>';
  }
  return sub.map(b => {
    const lo = b.range ? Number(b.range[0]).toFixed(1) : "?";
    const hi = b.range ? Number(b.range[1]).toFixed(1) : "?";
    return `<li><span class="rng">[${lo}, ${hi}]</span><span class="n">${b.n_eps} eps</span></li>`;
  }).join("");
}

function renderJointSeries(series, w, h) {
  const joints = Object.keys(series || {});
  if (!joints.length) return '<div class="empty">(no joint data yet)</div>';
  const colors = ["#6ea8fe","#f7a072","#7cd992","#d18ce5","#ffd866","#ff6b9d","#90e0ef"];
  const pad = {l: 36, r: 90, t: 8, b: 22};
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  let tmin = Infinity, tmax = -Infinity, vmin = Infinity, vmax = -Infinity;
  for (const j of joints) {
    for (const pt of series[j]) {
      if (pt.t < tmin) tmin = pt.t;
      if (pt.t > tmax) tmax = pt.t;
      if (pt.value < vmin) vmin = pt.value;
      if (pt.value > vmax) vmax = pt.value;
    }
  }
  if (tmax === tmin) tmax = tmin + 1;
  if (vmax === vmin) { vmax = vmin + 1; vmin = vmin - 1; }
  const vpad = (vmax - vmin) * 0.1;
  vmin -= vpad; vmax += vpad;
  const xsc = t => pad.l + (t - tmin) / (tmax - tmin) * iw;
  const ysc = v => pad.t + (1 - (v - vmin) / (vmax - vmin)) * ih;
  let svg = `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;">`;
  svg += `<rect x="${pad.l}" y="${pad.t}" width="${iw}" height="${ih}" fill="#0f1218" stroke="#1f2530"/>`;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * ih;
    const v = vmax - (i / 4) * (vmax - vmin);
    svg += `<line x1="${pad.l}" y1="${y}" x2="${pad.l + iw}" y2="${y}" stroke="#1f2530" stroke-dasharray="2,3"/>`;
    svg += `<text x="${pad.l - 4}" y="${y + 3}" text-anchor="end" font-size="9" fill="#8a93a6">${v.toFixed(1)}</text>`;
  }
  joints.forEach((j, i) => {
    const color = colors[i % colors.length];
    const pts = series[j].map(p => `${xsc(p.t).toFixed(1)},${ysc(p.value).toFixed(1)}`).join(" ");
    svg += `<polyline fill="none" stroke="${color}" stroke-width="1.5" points="${pts}"/>`;
    const last = series[j][series[j].length - 1];
    const ly = ysc(last.value);
    svg += `<circle cx="${xsc(last.t).toFixed(1)}" cy="${ly.toFixed(1)}" r="2.5" fill="${color}"/>`;
    svg += `<text x="${pad.l + iw + 4}" y="${pad.t + 10 + i * 12}" font-size="10" fill="${color}">${j} <tspan fill="#8a93a6">${last.value.toFixed(1)}</tspan></text>`;
  });
  svg += `<text x="${pad.l}" y="${h - 6}" font-size="9" fill="#8a93a6">t=0s</text>`;
  svg += `<text x="${pad.l + iw}" y="${h - 6}" text-anchor="end" font-size="9" fill="#8a93a6">t=${tmax.toFixed(0)}s</text>`;
  svg += `</svg>`;
  return svg;
}

function renderExploreActions(ea) {
  if (!ea || (!ea.this_cycle || !ea.this_cycle.length) && (!ea.total || !ea.total.length)) {
    return '<div class="empty">(no actions yet)</div>';
  }
  function rows(list) {
    if (!list || !list.length) return '<div class="empty" style="padding:4px 0;">(none)</div>';
    return '<ul class="sub-bursts" style="margin-top:4px;">' + list.map(r => {
      const sign = r.direction === "negative" ? "−" : "+";
      const label = `${r.joint} ${sign}${Number(r.magnitude).toFixed(1)}`;
      return `<li><span class="rng">${label}</span><span class="n">${r.count}×</span></li>`;
    }).join("") + '</ul>';
  }
  return `
    <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;">This cycle</div>
    ${rows(ea.this_cycle)}
    <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;margin-top:10px;">Total</div>
    ${rows(ea.total)}
  `;
}

function renderPhasePill(phase) {
  const p = (phase || "").toString();
  const label = p || "—";
  return `<span class="phase-pill ${p}">${label}</span>`;
}

function renderRangeText(active, full) {
  if (!active || active.length !== 2) return "—";
  const [lo, hi] = active;
  const [fmin, fmax] = full || [-60, 60];
  const pct = Math.max(0, Math.min(100, ((hi - lo) / (fmax - fmin)) * 100));
  return `${Number(lo).toFixed(0)}° → ${Number(hi).toFixed(0)}° <span style="color:#8a93a6;font-size:11px;">(${pct.toFixed(0)}%)</span>`;
}

function renderTrainingChart(progress, epochsTarget, width, height) {
  width = width || 780;
  height = height || 240;
  if (!progress || !progress.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">waiting for first epoch…</text></svg>`;
  }
  const padL = 56, padR = 140, padT = 14, padB = 30;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const trainVals = progress.map(p => Number(p.train_loss));
  const valVals = progress.map(p => Number(p.val_loss));
  const all = trainVals.concat(valVals).filter(v => Number.isFinite(v));
  if (!all.length) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
      <text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#8a93a6" font-size="12">no numeric losses yet</text></svg>`;
  }
  const ymin = Math.min.apply(null, all);
  const ymax = Math.max.apply(null, all);
  const pad = (ymax - ymin) * 0.1 || Math.max(1e-6, ymax * 0.1);
  const y0 = ymin - pad, y1 = ymax + pad;
  // X-axis: 1..epochsTarget if known, otherwise just 1..len(progress).
  const xMax = Math.max(1, Math.max(epochsTarget || 0, progress.length));
  const X = e => padL + ((e - 1) / Math.max(1, xMax - 1)) * plotW;
  const Y = v => padT + (1 - (v - y0) / (y1 - y0)) * plotH;

  const pathForSeries = (vals) => progress.map((p, i) => {
    const x = X(p.epoch || (i + 1));
    const y = Y(Number(vals[i]));
    return (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");

  const trainPath = pathForSeries(trainVals);
  const valPath = pathForSeries(valVals);

  // Y ticks
  let yticks = "";
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const yv = y0 + frac * (y1 - y0);
    const py = padT + (1 - frac) * plotH;
    yticks += `<line x1="${padL}" y1="${py.toFixed(1)}" x2="${padL + plotW}" y2="${py.toFixed(1)}" stroke="#232735"/>
               <text x="${padL - 6}" y="${(py + 3).toFixed(1)}" text-anchor="end" fill="#8a93a6" font-size="10">${yv.toFixed(4)}</text>`;
  }
  // X ticks
  let xticks = "";
  const currentEpoch = progress[progress.length - 1].epoch || progress.length;
  [1, Math.max(1, Math.floor(xMax / 2)), xMax].forEach(v => {
    xticks += `<text x="${X(v).toFixed(1)}" y="${(padT + plotH + 14).toFixed(1)}" text-anchor="middle" fill="#8a93a6" font-size="10">${v}</text>`;
  });
  // Current-epoch marker
  const cx = X(currentEpoch);
  const progressMarker = `<line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${padT + plotH}" stroke="#65d88c" stroke-dasharray="3,3" opacity="0.6"/>`;

  // Legend
  const legendX = padL + plotW + 10;
  const legend = `
    <rect x="${legendX}" y="${padT + 6}" width="10" height="10" fill="#e07b39"/>
    <text x="${legendX + 14}" y="${padT + 15}" fill="#c8cddc" font-size="11">train loss</text>
    <rect x="${legendX}" y="${padT + 26}" width="10" height="10" fill="#5aa9ff"/>
    <text x="${legendX + 14}" y="${padT + 35}" fill="#c8cddc" font-size="11">val loss</text>
    <line x1="${legendX}" y1="${padT + 50}" x2="${legendX + 12}" y2="${padT + 50}" stroke="#65d88c" stroke-dasharray="3,3"/>
    <text x="${legendX + 16}" y="${padT + 53}" fill="#c8cddc" font-size="11">epoch ${currentEpoch}/${xMax}</text>
  `;

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="${height}">
    <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="#1d2230" stroke="#232735"/>
    ${yticks}
    ${xticks}
    ${progressMarker}
    <path d="${trainPath}" fill="none" stroke="#e07b39" stroke-width="1.8"/>
    <path d="${valPath}" fill="none" stroke="#5aa9ff" stroke-width="1.8"/>
    ${legend}
  </svg>`;
}

function renderActionCanvasGallery(imageNames) {
  if (!imageNames || !imageNames.length) {
    return '<div class="empty">(no action canvases yet)</div>';
  }
  return imageNames.map(name => {
    const src = "/canvas/" + encodeURIComponent(name) + "?t=" + Date.now();
    return `<figure>
      <img src="${src}" alt="${name}"/>
      <figcaption>${name}</figcaption>
    </figure>`;
  }).join("");
}

function eventDetail(e) {
  if (e.event === "probe") return `action=${e.action} mse=${Number(e.mse).toFixed(5)}`;
  if (e.event === "verify_summary") return `mean_err=${Number(e.mean_err).toFixed(5)} n=${e.n}`;
  if (e.event === "state") return `${e.state} (iter ${e.iteration})`;
  const rest = {};
  for (const k in e) if (k !== "t" && k !== "event") rest[k] = e[k];
  return Object.keys(rest).length ? JSON.stringify(rest) : "";
}

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    const connPill = document.getElementById("connected-pill");
    connPill.textContent = "live";
    connPill.className = "pill ok";

    if (!s.session) {
      document.getElementById("session").textContent = "(no session yet — waiting for runs/events_*.jsonl)";
      const statePill = document.getElementById("state-pill");
      statePill.textContent = "waiting";
      statePill.className = "pill";
      document.getElementById("chart").innerHTML = lineChart([], 640, 220);
      document.getElementById("action-chart").innerHTML = barChart([], 500, 200);
      return;
    }

    document.getElementById("session").textContent = s.session;
    const state = s.current_state || "—";
    const pill = document.getElementById("state-pill");
    pill.textContent = state;
    pill.className = "pill " + stateClass(state);

    if (s.duration_seconds != null) {
      document.getElementById("duration").textContent = `${s.duration_seconds.toFixed(1)}s`;
    }

    // --- Metrics strip ---
    document.getElementById("m-cycle").textContent = s.cycle_count != null ? s.cycle_count : "—";
    // Episodes: total (+delta this cycle). Both update live as
    // explore_episode_progress events land.
    const total = s.total_eps != null ? s.total_eps : 0;
    const this_cycle = s.episodes_this_cycle != null ? s.episodes_this_cycle : 0;
    document.getElementById("m-eps").innerHTML =
      `${total} <span style="color:var(--muted);font-size:11px;">(+${this_cycle} cycle)</span>`;
    document.getElementById("m-retrains").textContent = s.retrain_count != null ? s.retrain_count : 0;

    // --- Explore progress card ---
    const exCard = document.getElementById("m-explore-progress");
    const ex = s.explore || {};
    if (ex.in_progress && ex.total_episodes != null) {
      const done = ex.episode_index != null ? (ex.episode_index + 1) : 0;
      exCard.innerHTML = `${done}/${ex.total_episodes} <span style="color:var(--ok);font-size:10px;">● live</span>`;
    } else if (ex.total_episodes != null) {
      exCard.textContent = "—";
    } else {
      exCard.textContent = "—";
    }
    document.getElementById("m-phase").innerHTML = renderPhasePill(s.current_phase);
    document.getElementById("m-active-range").innerHTML = renderRangeText(s.active_range, s.range_full);
    document.getElementById("m-arm-a").textContent = s.arm_a_locked_val_mse != null
      ? Number(s.arm_a_locked_val_mse).toFixed(5)
      : "(not measured)";

    // --- Locked val trajectory ---
    const lvHistory = s.locked_val_history || [];
    const lvEmpty = document.getElementById("locked-val-empty");
    if (lvHistory.length === 0) {
      document.getElementById("locked-val-chart").innerHTML =
        renderLockedValChart([], s.arm_a_locked_val_mse, s.range_history, 780, 220);
      lvEmpty.style.display = "block";
    } else {
      document.getElementById("locked-val-chart").innerHTML =
        renderLockedValChart(lvHistory, s.arm_a_locked_val_mse, s.range_history, 780, 260);
      lvEmpty.style.display = "none";
    }

    // --- Heatmap ---
    const hmHtml = renderHeatmap(s.heatmap, s.active_range);
    if (hmHtml == null) {
      document.getElementById("heatmap").innerHTML = "";
      document.getElementById("heatmap-empty").style.display = "block";
    } else {
      document.getElementById("heatmap").innerHTML = hmHtml;
      document.getElementById("heatmap-empty").style.display = "none";
    }

    // --- Coverage histogram ---
    document.getElementById("coverage").innerHTML =
      renderCoverageHist(s.coverage, s.active_range, 480, 180);

    // --- Sub-bursts list ---
    document.getElementById("sub-bursts").innerHTML =
      renderSubBursts(s.last_explore_sub_bursts);

    // --- Explore actions breakdown ---
    document.getElementById("explore-actions").innerHTML =
      renderExploreActions(s.explore_actions);
    document.getElementById("joint-series").innerHTML =
      renderJointSeries(s.joint_state_series, 520, 200);

    // --- Verification MSE rolling chart (last 30 probes) + all-time chart ---
    const allProbes = s.probes || [];
    const rollingProbes = allProbes.slice(-30);
    const mses = rollingProbes.map(p => Number(p.mse));
    const hline = mses.length ? mses.reduce((a, b) => a + b, 0) / mses.length : null;
    document.getElementById("chart").innerHTML = lineChart(mses, 640, 220, {
      hline,
      tau_low: s.thresholds ? s.thresholds.tau_low : null,
      tau_high: s.thresholds ? s.thresholds.tau_high : null,
    });

    // All-time chart: filter by the selected time window in the dropdown.
    const rangeSel = document.getElementById("alltime-range");
    const rangeMin = rangeSel ? Number(rangeSel.value) : 0;
    const nowSec = Date.now() / 1000;
    const cutoff = rangeMin > 0 ? nowSec - rangeMin * 60 : 0;
    const filteredPoints = allProbes
      .filter(p => p.t != null && (rangeMin === 0 || Number(p.t) >= cutoff))
      .map(p => ({ t: Number(p.t), value: Number(p.mse) }));
    document.getElementById("alltime-chart").innerHTML = timeSeriesChart(
      filteredPoints, 900, 240,
      {
        tau_low: s.thresholds ? s.thresholds.tau_low : null,
        tau_high: s.thresholds ? s.thresholds.tau_high : null,
      },
    );

    document.getElementById("m-probes").textContent = s.probes.length;
    document.getElementById("m-mean").textContent = fmtMse(hline);
    document.getElementById("m-last").textContent = fmtMse(mses.length ? mses[mses.length - 1] : null);
    document.getElementById("m-best").textContent = fmtMse(mses.length ? Math.min.apply(null, mses) : null);
    document.getElementById("m-worst").textContent = fmtMse(mses.length ? Math.max.apply(null, mses) : null);

    const buckets = {};
    s.probes.forEach(p => {
      const a = p.action;
      if (!buckets[a]) buckets[a] = [];
      buckets[a].push(Number(p.mse));
    });
    const labelOf = { 1: "1 (+)", 2: "2 (−)", 3: "3 (hold)" };
    const bars = Object.keys(buckets).sort().map(k => ({
      label: labelOf[k] || k,
      value: buckets[k].reduce((a, b) => a + b, 0) / buckets[k].length,
    }));
    document.getElementById("action-chart").innerHTML = barChart(bars, 500, 200);

    // --- Action canvas gallery (last N canvases across verify+explore) ---
    const gallery = document.getElementById("action-canvas-gallery");
    const galleryImages = (s.latest_action_canvases || []).slice(0, ACTION_CANVAS_GALLERY_SIZE);
    const topName = galleryImages[0] || "";
    if (gallery && gallery.getAttribute("data-top") !== topName) {
      gallery.innerHTML = renderActionCanvasGallery(galleryImages);
      gallery.setAttribute("data-top", topName);
    }
    const galleryNSpan = document.getElementById("action-canvas-gallery-n");
    if (galleryNSpan) galleryNSpan.textContent = String(ACTION_CANVAS_GALLERY_SIZE);

    // --- Per-cycle probe counter in the MSE card row ---
    const probesCycleEl = document.getElementById("m-probes-cycle");
    if (probesCycleEl) {
      probesCycleEl.textContent = s.probes_this_cycle != null
        ? String(s.probes_this_cycle) : "0";
    }

    // --- Training panel (shown only when a retrain is current/recent) ---
    const trainPanel = document.getElementById("panel-training");
    const t = s.training || {};
    const hasTrainingData = (t.progress && t.progress.length > 0) || !!t.in_progress;
    if (hasTrainingData) {
      trainPanel.style.display = "";
      const progress = t.progress || [];
      const current = progress.length > 0 ? progress[progress.length - 1] : null;
      document.getElementById("training-chart").innerHTML =
        renderTrainingChart(progress, t.epochs_target, 780, 240);
      document.getElementById("t-epoch").textContent =
        current ? `${current.epoch}/${t.epochs_target || "?"}` : "—";
      document.getElementById("t-train-loss").textContent =
        current && current.train_loss != null ? Number(current.train_loss).toFixed(5) : "—";
      document.getElementById("t-val-loss").textContent =
        current && current.val_loss != null ? Number(current.val_loss).toFixed(5) : "—";
      document.getElementById("t-best-val").textContent =
        current && current.best_val != null ? Number(current.best_val).toFixed(5) : "—";
      const ds = t.dataset_size || {};
      document.getElementById("t-train-canvases").textContent =
        ds.train_canvases != null ? String(ds.train_canvases) : "—";
      document.getElementById("t-val-canvases").textContent =
        ds.val_canvases != null ? String(ds.val_canvases) : "—";
      const inProgressTag = t.in_progress ? " · live" : " · done";
      const fromScratchTag = t.from_scratch ? "cold start" : "fine-tune";
      document.getElementById("training-meta").textContent =
        `cycle ${t.cycle != null ? t.cycle : "?"} · ${fromScratchTag}${inProgressTag}`;
    } else {
      trainPanel.style.display = "none";
    }

    const evs = (s.events || []).slice(-MAX_EVENTS).reverse();
    const rows = evs.map(e => {
      const cls = e.event || "";
      const t = fmtTime(e.t);
      const detail = eventDetail(e);
      return `<div class="row"><span class="ts">${t}</span><span class="name ${cls}">${e.event}</span> ${detail}</div>`;
    }).join("");
    document.getElementById("events").innerHTML = rows || '<div class="empty">(no events yet)</div>';
  } catch (e) {
    const p = document.getElementById("connected-pill");
    p.textContent = "disconnected";
    p.className = "pill bad";
  }
}

poll();
setInterval(poll, POLL_MS);

document.addEventListener("change", (ev) => {
  if (ev.target && ev.target.id === "alltime-range") {
    poll();
  }
});

document.addEventListener("click", async (ev) => {
  if (ev.target && ev.target.id === "btn-verify-now") {
    const btn = ev.target;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Triggered…";
    try {
      const r = await fetch("/trigger/verify", { method: "POST" });
      btn.textContent = r.ok ? "Queued ✓" : "Failed";
    } catch (e) {
      btn.textContent = "Failed";
    }
    setTimeout(() => { btn.disabled = false; btn.textContent = orig; }, 2500);
  }
});
</script>
</body></html>
"""


# ------------------------------------------------------------------ state ---

def _latest_session(runs_dir: Path) -> str | None:
    """Newest events_*.jsonl by mtime so the dashboard follows whichever run
    is currently active (including after the learner is restarted)."""
    candidates = list(runs_dir.glob("events_*.jsonl"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.stem.removeprefix("events_")


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latest_action_canvas(examples_dir: Path) -> str | None:
    if not examples_dir.exists():
        return None
    imgs = list(examples_dir.glob("action_canvas_*.png")) + list(
        examples_dir.glob("probe_*.png")  # back-compat: pre-rename files
    )
    if not imgs:
        return None
    return max(imgs, key=lambda p: p.stat().st_mtime).name


def _latest_action_canvases(examples_dir: Path, n: int = 5) -> list[str]:
    """Return the newest `n` action-canvas image filenames, newest first.

    Matches both the new `action_canvas_*.png` prefix and the legacy
    `probe_*.png` prefix so in-flight sessions started before the rename
    still render.
    """
    if not examples_dir.exists():
        return []
    imgs = sorted(
        list(examples_dir.glob("action_canvas_*.png"))
        + list(examples_dir.glob("probe_*.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [p.name for p in imgs[:n]]


def _current_state(events: list[dict]) -> str | None:
    """Walk the log in reverse and return the most recent state name.
    A `shutdown` event means the run finished — collapse to IDLE."""
    for e in reversed(events):
        ev = e.get("event")
        if ev == "shutdown":
            return "IDLE"
        if ev == "state":
            return e.get("state")
    return None


def _current_phase(events: list[dict]) -> str | None:
    """Finer-grained phase than `_current_state` — walks events in reverse
    and returns one of `explore | train_diffusion | evaluate | verify |
    idle | done` based on the most recent subprocess_start / state /
    shutdown event. Used for the dashboard phase pill.
    """
    for e in reversed(events):
        ev = e.get("event")
        if ev in ("experiment_done", "shutdown"):
            return "done"
        if ev == "subprocess_start":
            tag = e.get("tag") or ""
            if "train_diffusion" in tag:
                return "train_diffusion"
            if "evaluate" in tag:
                return "evaluate"
            if "create_dataset" in tag or "combine_datasets" in tag:
                return "processing"
        if ev == "subprocess_done":
            continue  # look past it, not informative on its own
        if ev == "explore_start":
            return "explore"
        if ev in ("explore_done", "explore_failed"):
            return "idle"
        if ev == "state":
            s = (e.get("state") or "").lower()
            if s:
                return s
    return None


def _experiment_info(events: list[dict]) -> dict:
    """Pull aggregated counters and end-state from the events stream.

    `total_eps` and `episodes_this_cycle` are computed LIVE from
    `explore_episode_progress` events — they tick up as each episode
    completes, even while the learner is still blocked inside the
    recording subprocess. Without this, the counters would only update
    at `cycle_start` events (once every ~15 min), making the dashboard
    feel frozen during EXPLORE.
    """
    cycle_count = 0
    retrain_count = 0
    baseline_total_eps = 0           # from latest cycle_start
    active_range: list | None = None
    last_retrain_start_t: float | None = None
    last_retrain_duration_s: float | None = None
    termination_reason: str | None = None

    # State machine for computing live episode deltas within the current
    # cycle. `cycle_baseline` is the total_eps the orchestrator logged at
    # the most recent cycle_start. `sub_bursts_done_this_cycle` accumulates
    # episodes from sub-bursts that have already completed
    # (explore_done → sum of n_eps from their explore_start). `current_in_flight`
    # tracks the latest `explore_episode_progress.episode_index + 1` for the
    # CURRENTLY-running sub-burst, which resets at every new `explore_start`.
    cycle_baseline = 0
    sub_bursts_done_this_cycle = 0
    current_sub_burst_n_eps = 0
    current_in_flight = 0
    current_sub_burst_repo: str | None = None

    for e in events:
        ev = e.get("event")
        if ev == "cycle_start":
            cycle_count = max(cycle_count, int(e.get("cycle", 0)) + 1)
            cycle_baseline = int(e.get("total_eps", 0))
            baseline_total_eps = cycle_baseline
            # Reset per-cycle accumulators.
            sub_bursts_done_this_cycle = 0
            current_sub_burst_n_eps = 0
            current_in_flight = 0
            current_sub_burst_repo = None
            if isinstance(e.get("active_range"), list):
                active_range = e["active_range"]
        elif ev == "explore_start":
            current_sub_burst_repo = e.get("repo_id")
            current_sub_burst_n_eps = int(e.get("episodes", 0))
            current_in_flight = 0
        elif ev == "explore_episode_progress":
            # Live counter for the running sub-burst.
            current_in_flight = int(e.get("episode_index", 0)) + 1
        elif ev == "explore_done":
            # Sub-burst finished: roll its n_eps into the cycle accumulator
            # and reset the in-flight counter.
            sub_bursts_done_this_cycle += current_sub_burst_n_eps
            current_sub_burst_n_eps = 0
            current_in_flight = 0
            current_sub_burst_repo = None
        elif ev == "retrain_start":
            last_retrain_start_t = float(e.get("t", 0.0))
        elif ev == "retrain_done":
            retrain_count += 1
            if last_retrain_start_t is not None:
                last_retrain_duration_s = float(e.get("t", 0.0)) - last_retrain_start_t
        elif ev == "range_expanded" and isinstance(e.get("new_range"), list):
            active_range = e["new_range"]
        elif ev == "experiment_start" and isinstance(e.get("active_range"), list):
            active_range = e["active_range"]
        elif ev == "experiment_done":
            termination_reason = e.get("reason")

    episodes_this_cycle = sub_bursts_done_this_cycle + current_in_flight
    live_total_eps = cycle_baseline + episodes_this_cycle

    return {
        "cycle_count": cycle_count,
        "retrain_count": retrain_count,
        "total_eps": live_total_eps,
        "episodes_this_cycle": episodes_this_cycle,
        "active_range": active_range,
        "last_retrain_duration_s": last_retrain_duration_s,
        "termination_reason": termination_reason,
    }


def _locked_val_history(events: list[dict]) -> list[dict]:
    """Extract locked-val trajectory for the Arm-A-vs-Arm-B chart."""
    out = []
    for e in events:
        if e.get("event") != "locked_val_measured":
            continue
        out.append({
            "cycle": e.get("cycle"),
            "total_eps": e.get("total_eps"),
            "locked_val": e.get("locked_val_mse"),
            "train_val": e.get("train_val_mse"),
            "accepted": bool(e.get("accepted", True)),
        })
    return out


def _range_history(events: list[dict]) -> list[dict]:
    """List of `range_expanded` events for curriculum visualization."""
    out = []
    for e in events:
        if e.get("event") != "range_expanded":
            continue
        out.append({
            "cycle": e.get("cycle"),
            "total_eps": e.get("total_eps"),
            "new_range": e.get("new_range"),
            "old_range": e.get("old_range"),
        })
    return out


def _last_sub_bursts(events: list[dict]) -> list[dict]:
    """Return the sub-burst plan from the most recent EXPLORE phase."""
    for e in reversed(events):
        if e.get("event") != "explore_sub_bursts_planned":
            continue
        sub = e.get("sub_bursts") or []
        return [
            {"n_eps": int(b.get("n_eps", 0)), "range": b.get("range")}
            for b in sub
        ]
    return []


def _training_progress(events: list[dict]) -> dict:
    """Extract per-epoch training metrics for the most recent (or current)
    retrain, plus the dataset size used.

    The "current" retrain is whichever retrain has a `retrain_start` event
    without a matching subsequent `retrain_done` / `retrain_failed`. If no
    retrain is currently in-flight, we return the progress of the most
    recent completed retrain so the dashboard can still render the final
    loss curve.
    """
    # Find the index of the most recent retrain_start. Everything after
    # that is "this retrain"; everything before is older.
    start_idx: Optional[int] = None
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "retrain_start":
            start_idx = i
            break
    if start_idx is None:
        return {
            "progress": [],
            "dataset_size": None,
            "in_progress": False,
            "cycle": None,
            "epochs_target": None,
            "from_scratch": None,
        }

    retrain_start = events[start_idx]
    tail = events[start_idx + 1:]

    # Is training still running? We check for a terminal event after the
    # start. `subprocess_done` with tag=train_diffusion marks the end of
    # the training subprocess specifically. `retrain_done`/`retrain_failed`
    # mark the end of the whole retrain (train + eval).
    in_progress = True
    for e in tail:
        ev = e.get("event")
        if ev in ("retrain_done", "retrain_failed"):
            in_progress = False
            break

    progress = [
        {
            "epoch": int(e.get("epoch", 0)),
            "total_epochs": int(e.get("total_epochs", 0)),
            "train_loss": float(e["train_loss"]) if e.get("train_loss") is not None else None,
            "val_loss": float(e["val_loss"]) if e.get("val_loss") is not None else None,
            "lr": e.get("lr"),
            "best_val": e.get("best_val"),
        }
        for e in tail
        if e.get("event") == "training_progress"
    ]

    dataset_size = None
    for e in tail:
        if e.get("event") == "training_dataset_size":
            dataset_size = {
                "train_canvases": int(e.get("train_canvases", 0)),
                "val_canvases": int(e.get("val_canvases", 0)),
            }
            break

    return {
        "progress": progress,
        "dataset_size": dataset_size,
        "in_progress": in_progress,
        "cycle": retrain_start.get("cycle"),
        "epochs_target": retrain_start.get("epochs"),
        "from_scratch": retrain_start.get("from_scratch"),
        "num_accumulated_dirs": retrain_start.get("num_accumulated_dirs"),
        "total_eps_at_start": retrain_start.get("total_eps"),
    }


def _joint_state_series(events: list[dict], window: int = 60) -> dict:
    """Rolling window of commanded joint values from `explore_joint_state`
    events. Returns a dict of {joint_name: [{t, value}, ...]} trimmed to the
    last `window` samples per joint."""
    series: dict[str, list[dict]] = {}
    t0: Optional[float] = None
    for e in events:
        if e.get("event") != "explore_joint_state":
            continue
        state = e.get("state") or {}
        t = float(e.get("t", 0.0) or 0.0)
        if t0 is None:
            t0 = t
        for joint, val in state.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            series.setdefault(joint, []).append({"t": t - t0, "value": v})
    for j, rows in series.items():
        if len(rows) > window:
            series[j] = rows[-window:]
    return series


def _explore_action_counts(events: list[dict], current_cycle: int) -> dict:
    """Tally `explore_action_taken` events into per-(joint, direction, magnitude)
    counts for the current cycle and across the whole session.

    The learner emits one such event per episode as it streams the recorder's
    stdout. Grouping by (joint, direction, magnitude) keeps the panel compact
    even when multiple joints or step sizes are in play.
    """
    total: dict[tuple, int] = {}
    this_cycle: dict[tuple, int] = {}
    live_cycle = -1
    for e in events:
        ev = e.get("event")
        if ev == "cycle_start":
            live_cycle = int(e.get("cycle", -1))
            continue
        if ev != "explore_action_taken":
            continue
        key = (
            str(e.get("joint", "?")),
            str(e.get("direction", "?")),
            float(e.get("magnitude", 0.0) or 0.0),
        )
        total[key] = total.get(key, 0) + 1
        if live_cycle == current_cycle:
            this_cycle[key] = this_cycle.get(key, 0) + 1

    def _serialize(d: dict) -> list[dict]:
        out = [
            {"joint": k[0], "direction": k[1], "magnitude": k[2], "count": v}
            for k, v in d.items()
        ]
        out.sort(key=lambda r: -r["count"])
        return out

    return {
        "this_cycle": _serialize(this_cycle),
        "total": _serialize(total),
    }


def _probe_counts(events: list[dict], current_cycle: int) -> dict:
    """Tally probe events: how many this cycle, how many total."""
    total = 0
    this_cycle = 0
    for e in events:
        if e.get("event") != "probe":
            continue
        total += 1
        if e.get("cycle") == current_cycle:
            this_cycle += 1
    return {"probes_this_cycle": this_cycle, "probes_total": total}


def _explore_progress(events: list[dict]) -> dict:
    """Live progress of whichever EXPLORE subprocess is currently running.

    Walks back to the most recent `explore_start` and collects subsequent
    `explore_episode_progress` / `explore_done` events so the dashboard can
    show episode N/total, even while the recording subprocess is still
    blocking the learner.
    """
    start_idx: Optional[int] = None
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "explore_start":
            start_idx = i
            break
    if start_idx is None:
        return {
            "in_progress": False,
            "episode_index": None,
            "total_episodes": None,
            "repo_id": None,
        }
    start = events[start_idx]
    tail = events[start_idx + 1:]

    in_progress = True
    latest_episode: Optional[int] = None
    for e in tail:
        ev = e.get("event")
        if ev in ("explore_done", "explore_failed"):
            in_progress = False
        if ev == "explore_episode_progress":
            latest_episode = int(e.get("episode_index", 0))
    return {
        "in_progress": in_progress,
        "episode_index": latest_episode,
        "total_episodes": start.get("episodes"),
        "repo_id": start.get("repo_id"),
    }


def _build_heatmap(
    probes: list[dict],
    full_min: float,
    full_max: float,
    control_joint_idx: int,
    n_bins: int = 12,
) -> dict:
    """State-bin × action mean-MSE grid, backward-compatible with old
    probes that lack `motor_state` (those are skipped).
    """
    if full_max <= full_min or n_bins <= 0:
        return {"state_bins": [], "actions": [], "cells": [], "max_mse": 0.0}
    bin_width = (full_max - full_min) / n_bins
    cells: dict[tuple[int, int], list[float]] = {}
    for p in probes:
        motor_state = p.get("motor_state")
        if motor_state is None or control_joint_idx >= len(motor_state):
            continue
        pos = float(motor_state[control_joint_idx])
        if pos < full_min or pos > full_max:
            continue
        b = min(n_bins - 1, max(0, int((pos - full_min) / bin_width)))
        key = (b, int(p.get("action", 0)))
        cells.setdefault(key, []).append(float(p.get("mse", 0.0)))
    flat = []
    max_mse = 0.0
    for (b, a), mses in cells.items():
        mean = sum(mses) / len(mses)
        max_mse = max(max_mse, mean)
        flat.append({"state_bin": b, "action": a, "n": len(mses), "mean_mse": mean})
    return {
        "state_bins": [full_min + i * bin_width for i in range(n_bins + 1)],
        "actions": [1, 2, 3],
        "cells": flat,
        "max_mse": max_mse,
    }


def _build_coverage(
    probes: list[dict],
    full_min: float,
    full_max: float,
    control_joint_idx: int,
    n_bins: int = 12,
) -> dict:
    """1D histogram of probe starting positions across the full joint range."""
    if full_max <= full_min or n_bins <= 0:
        return {"bin_edges": [], "counts": []}
    bin_width = (full_max - full_min) / n_bins
    counts = [0] * n_bins
    for p in probes:
        motor_state = p.get("motor_state")
        if motor_state is None or control_joint_idx >= len(motor_state):
            continue
        pos = float(motor_state[control_joint_idx])
        if pos < full_min or pos > full_max:
            continue
        b = min(n_bins - 1, max(0, int((pos - full_min) / bin_width)))
        counts[b] += 1
    return {
        "bin_edges": [full_min + i * bin_width for i in range(n_bins + 1)],
        "counts": counts,
    }


def _read_arm_a_result(runs_dir: Path) -> float | None:
    """Best-effort read of `runs/arm_a_result.json` for the reference line."""
    path = runs_dir / "arm_a_result.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    val = data.get("val_mse_visual") or data.get("val_mse")
    return float(val) if val is not None else None


def _infer_context_from_events(events: list[dict]) -> dict:
    """Try to reconstruct the control joint + full-range bounds + thresholds
    from events the learner emits at startup. Falls back to defaults if
    the session predates those events.
    """
    control_joint = "shoulder_pan"
    control_joint_idx = 0
    full_min = -60.0
    full_max = 60.0
    tau_low = 0.02
    tau_high = 0.06
    for e in events:
        ev = e.get("event")
        if ev == "range_init" and isinstance(e.get("full"), list):
            full_min, full_max = float(e["full"][0]), float(e["full"][1])
        if ev == "experiment_start":
            if e.get("tau_low") is not None:
                tau_low = float(e["tau_low"])
            if e.get("tau_high") is not None:
                tau_high = float(e["tau_high"])
    return {
        "control_joint": control_joint,
        "control_joint_idx": control_joint_idx,
        "full_min": full_min,
        "full_max": full_max,
        "tau_low": tau_low,
        "tau_high": tau_high,
    }


def build_state_payload(runs_dir: Path) -> dict:
    session = _latest_session(runs_dir)
    if session is None:
        return {"session": None, "probes": [], "events": []}
    events_path = runs_dir / f"events_{session}.jsonl"
    examples_dir = runs_dir / f"examples_{session}"
    events = _read_events(events_path)
    probes = [e for e in events if e.get("event") == "probe"]
    start_t = events[0]["t"] if events else None
    end_t = events[-1]["t"] if events else None
    duration = (end_t - start_t) if (start_t and end_t) else None

    ctx = _infer_context_from_events(events)
    exp = _experiment_info(events)
    heatmap = _build_heatmap(
        probes,
        full_min=ctx["full_min"],
        full_max=ctx["full_max"],
        control_joint_idx=ctx["control_joint_idx"],
    )
    coverage = _build_coverage(
        probes,
        full_min=ctx["full_min"],
        full_max=ctx["full_max"],
        control_joint_idx=ctx["control_joint_idx"],
    )

    # Probe counts (per-cycle + total). Use the current cycle from the
    # experiment_info pass — exp["cycle_count"] is the NEXT cycle index,
    # so subtract 1 to get the "currently running" cycle.
    current_cycle = max(0, int(exp["cycle_count"]) - 1)
    counts = _probe_counts(events, current_cycle)
    action_counts = _explore_action_counts(events, current_cycle)

    return {
        "session": session,
        "current_state": _current_state(events),
        "current_phase": _current_phase(events),
        "cycle_count": exp["cycle_count"],
        "retrain_count": exp["retrain_count"],
        "total_eps": exp["total_eps"],
        "episodes_this_cycle": exp["episodes_this_cycle"],
        "last_retrain_duration_s": exp["last_retrain_duration_s"],
        "termination_reason": exp["termination_reason"],
        "active_range": exp["active_range"],
        "range_full": [ctx["full_min"], ctx["full_max"]],
        "control_joint": ctx["control_joint"],
        "control_joint_idx": ctx["control_joint_idx"],
        "thresholds": {
            "tau_low": ctx["tau_low"],
            "tau_high": ctx["tau_high"],
        },
        "arm_a_locked_val_mse": _read_arm_a_result(runs_dir),
        "probes": probes,
        "events": events,
        "latest_action_canvas": _latest_action_canvas(examples_dir),
        "latest_action_canvases": _latest_action_canvases(examples_dir, n=5),
        "probes_this_cycle": counts["probes_this_cycle"],
        "probes_total": counts["probes_total"],
        "duration_seconds": duration,
        "heatmap": heatmap,
        "coverage": coverage,
        "locked_val_history": _locked_val_history(events),
        "range_history": _range_history(events),
        "last_explore_sub_bursts": _last_sub_bursts(events),
        "training": _training_progress(events),
        "explore": _explore_progress(events),
        "explore_actions": action_counts,
        "joint_state_series": _joint_state_series(events, window=60),
    }


# ------------------------------------------------------------------ server ---

def make_handler(runs_dir: Path):
    runs_root_resolved = runs_dir.resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default access log
            return

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                try:
                    payload = build_state_payload(runs_dir)
                    body = json.dumps(payload, default=str).encode("utf-8")
                    self._send(200, body, "application/json")
                except Exception as e:
                    self._send(500, str(e).encode("utf-8"), "text/plain")
                return
            if path.startswith("/canvas/") or path.startswith("/probe/"):
                prefix = "/canvas/" if path.startswith("/canvas/") else "/probe/"
                name = unquote(path[len(prefix):])
                session = _latest_session(runs_dir)
                if session is None:
                    self.send_error(404)
                    return
                img_path = (runs_dir / f"examples_{session}" / name)
                # Path-traversal guard: resolved file must sit under runs_dir.
                try:
                    resolved = img_path.resolve()
                    resolved.relative_to(runs_root_resolved)
                except (ValueError, OSError):
                    self.send_error(403)
                    return
                if not resolved.exists():
                    self.send_error(404)
                    return
                self._send(200, resolved.read_bytes(), "image/png")
                return
            self.send_error(404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/trigger/verify":
                # Drop a sentinel file; the orchestrator polls for it
                # during IDLE sleeps and short-circuits straight to VERIFY.
                try:
                    runs_dir.mkdir(parents=True, exist_ok=True)
                    (runs_dir / "trigger_verify.flag").write_text("1")
                    self._send(200, b'{"ok":true}', "application/json")
                except Exception as e:
                    self._send(500, str(e).encode("utf-8"), "text/plain")
                return
            self.send_error(404)

    return Handler


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(runs_dir: Path, port: int, host: str = "127.0.0.1") -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    handler = make_handler(runs_dir)
    with _ThreadedServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"dashboard: {url}  (runs_dir={runs_dir})")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down…")


def main():
    p = argparse.ArgumentParser(description="Live canvas-autonomous-learner dashboard")
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    serve(Path(args.runs_dir), args.port, args.host)


if __name__ == "__main__":
    main()
