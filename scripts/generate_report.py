"""Generate a self-contained HTML report for a learner session.

Reads `runs/events_<session>.jsonl` and `runs/examples_<session>/*.png`,
optionally pulls an offline training `timing.json` to plot a supplementary
loss curve, and writes `runs/report_<session>.html`. Charts are inline SVG
so the report works offline and has no chart-library dependency.

Usage:
    python scripts/generate_report.py --session 20260412_134201
    python scripts/generate_report.py --session 20260412_134201 \
        --timing ../canvas-world-model/local/checkpoints/hold_exp/iter1/diff_finetune/timing.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------- inputs

def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_timing(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def embed_png(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------- charts

def _svg_line_chart(
    series: list[tuple[str, list[float], str]],
    width: int = 720,
    height: int = 260,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    hline: float | None = None,
    hline_label: str = "",
) -> str:
    """Minimal inline SVG line chart for one or more series.

    series items: (label, y_values, stroke_color)
    """
    pad_l, pad_r, pad_t, pad_b = 50, 120, 30, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_y: list[float] = []
    for _, ys, _ in series:
        all_y.extend(ys)
    if hline is not None:
        all_y.append(hline)
    if not all_y:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">no data</text></svg>'
    y_min, y_max = min(all_y), max(all_y)
    if y_max == y_min:
        y_max = y_min + 1e-9
    y_range = y_max - y_min
    y_min -= 0.05 * y_range
    y_max += 0.05 * y_range

    max_len = max(len(ys) for _, ys, _ in series)
    x_max = max(1, max_len - 1)

    def _x(i: int) -> float:
        return pad_l + (i / x_max) * plot_w

    def _y(v: float) -> float:
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="-apple-system,Segoe UI,Roboto,sans-serif" font-size="11">')
    # Background
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    # Plot frame
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d0d0d0"/>')
    # Y gridlines + labels (5 ticks)
    for i in range(5):
        frac = i / 4
        yv = y_min + frac * (y_max - y_min)
        py = _y(yv)
        parts.append(f'<line x1="{pad_l}" y1="{py}" x2="{pad_l + plot_w}" y2="{py}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{py + 3}" text-anchor="end" fill="#555">{yv:.4g}</text>')
    # X tick labels (start / mid / end)
    for i in (0, max_len // 2, max_len - 1):
        if i < 0:
            continue
        parts.append(f'<text x="{_x(i)}" y="{pad_t + plot_h + 14}" text-anchor="middle" fill="#555">{i}</text>')
    # Horizontal threshold line
    if hline is not None and y_min <= hline <= y_max:
        hy = _y(hline)
        parts.append(f'<line x1="{pad_l}" y1="{hy}" x2="{pad_l + plot_w}" y2="{hy}" stroke="#d94141" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{pad_l + plot_w + 6}" y="{hy + 3}" fill="#d94141">{hline_label}</text>')
    # Series
    legend_y = pad_t + 4
    for label, ys, color in series:
        if not ys:
            continue
        pts = " ".join(f"{_x(i)},{_y(v)}" for i, v in enumerate(ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        for i, v in enumerate(ys):
            parts.append(f'<circle cx="{_x(i)}" cy="{_y(v)}" r="2.2" fill="{color}"/>')
        # Legend entry
        parts.append(f'<rect x="{pad_l + plot_w + 6}" y="{legend_y - 8}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{pad_l + plot_w + 20}" y="{legend_y + 1}" fill="#333">{html.escape(label)}</text>')
        legend_y += 16
    # Axis labels
    if title:
        parts.append(f'<text x="{width / 2}" y="16" text-anchor="middle" font-weight="600" fill="#222">{html.escape(title)}</text>')
    if xlabel:
        parts.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 6}" text-anchor="middle" fill="#555">{html.escape(xlabel)}</text>')
    if ylabel:
        parts.append(f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#555" transform="rotate(-90 14 {pad_t + plot_h / 2})">{html.escape(ylabel)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_bar_chart(
    bars: list[tuple[str, float]],
    width: int = 480,
    height: int = 220,
    title: str = "",
    ylabel: str = "",
    color: str = "#2c7fb8",
) -> str:
    pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    if not bars:
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">no data</text></svg>'
    vmax = max(v for _, v in bars)
    vmax = vmax if vmax > 0 else 1e-9

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="-apple-system,Segoe UI,Roboto,sans-serif" font-size="11">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d0d0d0"/>')
    # Y ticks
    for i in range(5):
        frac = i / 4
        yv = frac * vmax
        py = pad_t + (1 - frac) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{py}" x2="{pad_l + plot_w}" y2="{py}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{py + 3}" text-anchor="end" fill="#555">{yv:.4g}</text>')
    slot = plot_w / len(bars)
    bw = slot * 0.6
    for i, (label, v) in enumerate(bars):
        x = pad_l + i * slot + (slot - bw) / 2
        h_px = (v / vmax) * plot_h
        y = pad_t + plot_h - h_px
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{h_px}" fill="{color}"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{pad_t + plot_h + 14}" text-anchor="middle" fill="#555">{html.escape(label)}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{y - 4}" text-anchor="middle" fill="#333">{v:.4f}</text>')
    if title:
        parts.append(f'<text x="{width / 2}" y="16" text-anchor="middle" font-weight="600" fill="#222">{html.escape(title)}</text>')
    if ylabel:
        parts.append(f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#555" transform="rotate(-90 14 {pad_t + plot_h / 2})">{html.escape(ylabel)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------- main

def _format_ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def build_report(
    session: str,
    runs_dir: Path,
    timing_path: Path | None,
    cfg_summary: dict | None = None,
) -> Path:
    events_path = runs_dir / f"events_{session}.jsonl"
    examples_dir = runs_dir / f"examples_{session}"
    if not events_path.exists():
        raise FileNotFoundError(f"events log not found: {events_path}")

    events = load_events(events_path)
    probes = [e for e in events if e["event"] == "probe"]
    verify_summaries = [e for e in events if e["event"] == "verify_summary"]
    state_changes = [e for e in events if e["event"] == "state"]
    startup = next((e for e in events if e["event"] == "startup"), None)

    # Per-probe MSE trace
    probe_mses = [p["mse"] for p in probes]
    probe_actions = [p["action"] for p in probes]
    run_mean = (sum(probe_mses) / len(probe_mses)) if probe_mses else 0.0

    # Per-action mean MSE bars
    per_action: dict[int, list[float]] = {}
    for p in probes:
        per_action.setdefault(p["action"], []).append(p["mse"])
    action_labels = {1: "1 (+)", 2: "2 (-)", 3: "3 (hold)"}
    action_bars = [
        (action_labels.get(a, str(a)), sum(v) / len(v))
        for a, v in sorted(per_action.items())
    ]

    # Example probe images — embed inline so the HTML is fully portable.
    probe_pngs = sorted(examples_dir.glob("probe_*.png")) if examples_dir.exists() else []
    probe_images_html_parts = []
    for img_path in probe_pngs:
        data_uri = embed_png(img_path)
        probe_images_html_parts.append(
            f'<figure><img src="{data_uri}" alt="{html.escape(img_path.name)}"/>'
            f'<figcaption>{html.escape(img_path.name)}</figcaption></figure>'
        )

    # Online verification MSE chart
    online_chart = _svg_line_chart(
        [("probe MSE", probe_mses, "#2c7fb8")],
        title="Online verification MSE per probe (this session)",
        ylabel="MSE",
        xlabel="probe index",
        hline=run_mean if probe_mses else None,
        hline_label=f"mean={run_mean:.4f}",
    )

    action_chart = _svg_bar_chart(
        action_bars,
        title="Mean MSE per action (this session)",
        ylabel="mean MSE",
    )

    # Offline training loss curves (supplementary — from whichever timing.json
    # the caller pointed at). This is context for "here's what a training run
    # looks like" rather than the history of the live checkpoint itself.
    timing = load_timing(timing_path)
    if timing:
        val_hist = timing.get("val_loss_history", [])
        train_hist = timing.get("train_loss_history", [])
        training_chart = _svg_line_chart(
            [
                ("train loss", train_hist, "#e07b39"),
                ("val loss", val_hist, "#2c7fb8"),
            ],
            title=f"Training curves ({timing_path.parent.name})",
            ylabel="loss",
            xlabel="epoch",
        )
        training_note = (
            f"Source: <code>{html.escape(str(timing_path))}</code> "
            f"(<b>{timing.get('num_epochs_run', '?')}</b> epochs, "
            f"best val loss <b>{timing.get('best_val_loss', 0):.5f}</b> "
            f"at epoch <b>{timing.get('best_epoch', '?')}</b>)."
        )
    else:
        training_chart = "<p><i>No training curves supplied — pass <code>--timing &lt;path&gt;</code> to include offline training history.</i></p>"
        training_note = ""

    # Config summary
    cfg_html = ""
    if cfg_summary:
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in cfg_summary.items()
        )
        cfg_html = f"<table class='kv'>{rows}</table>"

    start_t = events[0]["t"] if events else 0
    end_t = events[-1]["t"] if events else 0
    duration = end_t - start_t if events else 0

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>canvas-autonomous-learner — session {session}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #777; margin-top: 0; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 36px; }}
  .kv {{ border-collapse: collapse; }}
  .kv td {{ padding: 3px 10px 3px 0; vertical-align: top; }}
  .kv td:first-child {{ color: #666; }}
  figure {{ display: inline-block; margin: 8px 8px 16px 0; }}
  figure img {{ display: block; width: 100%; max-width: 720px; border: 1px solid #ddd; }}
  figcaption {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
  .metrics {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .metric {{ background: #f4f8fb; border-left: 3px solid #2c7fb8; padding: 8px 14px; }}
  .metric b {{ font-size: 18px; display: block; }}
  .metric span {{ color: #666; font-size: 12px; }}
</style></head><body>
<h1>canvas-autonomous-learner</h1>
<p class="sub">Session <code>{html.escape(session)}</code> &middot; {_format_ts(start_t)} &rarr; {_format_ts(end_t)} &middot; {duration:.1f}s</p>

<div class="metrics">
  <div class="metric"><b>{len(probes)}</b><span>probes</span></div>
  <div class="metric"><b>{len(verify_summaries)}</b><span>verify bursts</span></div>
  <div class="metric"><b>{run_mean:.5f}</b><span>mean MSE (run)</span></div>
  <div class="metric"><b>{min(probe_mses) if probe_mses else 0:.5f}</b><span>best probe MSE</span></div>
  <div class="metric"><b>{max(probe_mses) if probe_mses else 0:.5f}</b><span>worst probe MSE</span></div>
</div>

<h2>Run configuration</h2>
{cfg_html}

<h2>Online verification MSE over this session</h2>
<p>Each point is one probe: predict next frame with the live CWM checkpoint, execute the action on the real SO-101, capture the actual frame, compute MSE on the last-frame visual region (normalized to [0,1]). This is the active-learning signal the state machine thresholds on.</p>
{online_chart}

<h2>Per-action mean MSE</h2>
<p>Which discrete action is currently hardest for the model to predict.</p>
{action_chart}

<h2>Example probe grids (before &middot; predicted &middot; actual)</h2>
<p>Each image is a 2&times;3 grid: base camera on top, wrist camera on bottom; columns are <b>before</b> (context frame), <b>predicted</b> next frame, and <b>actual</b> next frame after executing the action. Labelled header shows the action and the computed MSE.</p>
<div class="grid">
{''.join(probe_images_html_parts) if probe_images_html_parts else '<p><i>No probe images saved.</i></p>'}
</div>

<h2>Offline training loss curves (supplementary)</h2>
<p>{training_note}</p>
{training_chart}

</body></html>
"""

    out_path = runs_dir / f"report_{session}.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def _find_session(runs_dir: Path) -> str | None:
    """Return the most recent session name based on events_*.jsonl."""
    candidates = sorted(runs_dir.glob("events_*.jsonl"))
    if not candidates:
        return None
    latest = candidates[-1]
    return latest.stem.removeprefix("events_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default=None,
        help="Session tag (e.g. 20260412_134201). Defaults to the latest runs/ entry.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(REPO_ROOT / "runs"),
        help="Directory containing events_*.jsonl and examples_*/.",
    )
    parser.add_argument(
        "--timing",
        default=None,
        help="Optional path to a training timing.json for the loss curves section.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to the YAML config used for the run (summarized in the report).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    session = args.session or _find_session(runs_dir)
    if session is None:
        print(f"ERROR: no sessions found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    timing_path = Path(args.timing) if args.timing else None

    cfg_summary = None
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config) as f:
                raw = yaml.safe_load(f)
            cfg_summary = {
                "config_file": args.config,
                "live_checkpoint": raw.get("paths", {}).get("live_checkpoint"),
                "base_canvas": raw.get("paths", {}).get("base_canvas"),
                "control_joint": raw.get("robot", {}).get("control_joint"),
                "step_size": raw.get("robot", {}).get("step_size"),
                "probes_per_verify": raw.get("cadence", {}).get("probes_per_verify"),
                "tau_high": raw.get("thresholds", {}).get("tau_high"),
                "tau_low": raw.get("thresholds", {}).get("tau_low"),
            }
        except Exception as e:
            print(f"warning: could not parse config: {e}", file=sys.stderr)

    out = build_report(session, runs_dir, timing_path, cfg_summary)
    print(f"report written: {out}")


if __name__ == "__main__":
    main()
