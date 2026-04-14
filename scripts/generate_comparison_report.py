"""Generate the Arm A vs Arm B comparison report.

Reads:
- `runs/arm_a_result.json` — the offline baseline's locked-val MSE (single point)
- `runs/events_<session>.jsonl` — Arm B's full trajectory (locked_val_measured events)
- `runs/examples_<session>/` — a few late-run probe PNGs for visual sanity check

Writes `runs/comparison_report.html` — self-contained (inline SVG, base64 PNGs).

Usage:
    python scripts/generate_comparison_report.py                    # latest Arm B session
    python scripts/generate_comparison_report.py --session 20260412_201500
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
sys.path.insert(0, str(REPO_ROOT.parent))  # so `scripts._charts` resolves
sys.path.insert(0, str(REPO_ROOT))

from scripts._charts import svg_line_chart, svg_bar_chart  # noqa: E402


def _load_events(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _latest_session_with_locked_val(runs_dir: Path) -> str | None:
    """Pick the newest events_*.jsonl that contains a locked_val_measured event."""
    candidates = sorted(runs_dir.glob("events_*.jsonl"), key=lambda p: p.stat().st_mtime)
    for p in reversed(candidates):
        try:
            events = _load_events(p)
        except Exception:
            continue
        if any(e.get("event") == "locked_val_measured" for e in events):
            return p.stem.removeprefix("events_")
    return None


def _embed_png(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _fmt_ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _build_armB_summary(events: list[dict]) -> dict:
    locked = [e for e in events if e.get("event") == "locked_val_measured"]
    done = next((e for e in events if e.get("event") == "experiment_done"), None)
    start = events[0]["t"] if events else None
    end = events[-1]["t"] if events else None
    return {
        "cycles": len(locked),
        "total_eps": locked[-1]["total_eps"] if locked else 0,
        "final_locked_val_mse": locked[-1]["locked_val_mse"] if locked else None,
        "reason": done["reason"] if done else "unknown",
        "start_t": start,
        "end_t": end,
        "duration_s": (end - start) if (start and end) else None,
        "locked_val_trajectory": [
            {
                "cycle": e["cycle"],
                "total_eps": e["total_eps"],
                "locked_val_mse": e["locked_val_mse"],
                "train_val_mse": e.get("train_val_mse"),
                "accepted": e.get("accepted", True),
            }
            for e in locked
        ],
    }


def build_report(
    runs_dir: Path,
    session: str,
    arm_a_path: Path,
    out_path: Path,
) -> Path:
    events_path = runs_dir / f"events_{session}.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"events log not found: {events_path}")
    events = _load_events(events_path)
    armB = _build_armB_summary(events)

    if not arm_a_path.exists():
        raise FileNotFoundError(f"arm A result missing: {arm_a_path}")
    with open(arm_a_path) as f:
        arm_a = json.load(f)
    arm_a_mse = arm_a.get("val_mse_visual") or arm_a.get("val_mse")

    # --- Locked-val trajectory chart -----------------------------------------
    traj = armB["locked_val_trajectory"]
    accepted_traj = [p for p in traj if p.get("locked_val_mse") is not None]
    x_eps = [p["total_eps"] for p in accepted_traj]
    y_locked = [float(p["locked_val_mse"]) for p in accepted_traj]
    y_train = [
        float(p["train_val_mse"]) if p.get("train_val_mse") is not None else None
        for p in accepted_traj
    ]
    # Filter out None train values for plotting
    y_train_clean = [v for v in y_train if v is not None]

    series = [("Arm B locked val MSE", y_locked, "#2c7fb8")]
    if y_train_clean and len(y_train_clean) == len(y_locked):
        series.append(("Arm B train val MSE", y_train_clean, "#e07b39"))

    locked_chart = svg_line_chart(
        series,
        width=780, height=300,
        title="Locked-val MSE vs episodes collected",
        xlabel="episodes collected",
        ylabel="MSE (last-frame visual)",
        hline=arm_a_mse,
        hline_label=f"Arm A: {arm_a_mse:.5f}" if arm_a_mse is not None else "Arm A: n/a",
        x_values=x_eps,
    )

    # --- Per-cycle episode budget bar chart ----------------------------------
    cycle_bars = [
        (f"c{p['cycle']}", float(p["locked_val_mse"]))
        for p in accepted_traj
    ]
    cycle_chart = svg_bar_chart(
        cycle_bars, width=780, height=240,
        title="Locked-val MSE by retrain cycle",
        ylabel="locked val MSE",
    )

    # --- Online verification probe trace (sanity check) ---------------------
    probes = [e for e in events if e.get("event") == "probe"]
    probe_chart_html = ""
    if probes:
        probe_chart_html = svg_line_chart(
            [("online verify MSE", [float(p["mse"]) for p in probes], "#65d88c")],
            width=780, height=220,
            title="Online verification MSE across the run",
            ylabel="probe MSE",
            xlabel="probe #",
        )

    # --- Late-run probe image grids ------------------------------------------
    examples_dir = runs_dir / f"examples_{session}"
    probe_imgs = []
    if examples_dir.exists():
        imgs = sorted(examples_dir.glob("probe_*.png"))[-6:]
        for img in imgs:
            probe_imgs.append(
                f'<figure><img src="{_embed_png(img)}" alt="{html.escape(img.name)}"/>'
                f'<figcaption>{html.escape(img.name)}</figcaption></figure>'
            )
    probe_imgs_html = "".join(probe_imgs) if probe_imgs else "<p><i>No probe images captured.</i></p>"

    # --- Headline table ------------------------------------------------------
    arm_a_ckpt = arm_a.get("checkpoint", "n/a")
    headline_rows = [
        ("Arm A checkpoint", arm_a_ckpt),
        ("Arm A locked val MSE", f"{arm_a_mse:.5f}" if arm_a_mse is not None else "n/a"),
        ("Arm B final locked val MSE", f"{armB['final_locked_val_mse']:.5f}" if armB['final_locked_val_mse'] is not None else "n/a"),
        ("Arm B cycles", armB["cycles"]),
        ("Arm B episodes collected", armB["total_eps"]),
        ("Arm B termination reason", armB["reason"]),
        ("Arm B duration", f"{armB['duration_s']:.1f}s" if armB['duration_s'] else "n/a"),
        ("Arm B session", session),
    ]
    if arm_a_mse is not None and armB["final_locked_val_mse"] is not None:
        ratio = armB["final_locked_val_mse"] / arm_a_mse
        verdict = "Arm B wins" if ratio <= 1.0 else "Arm A wins" if ratio > 1.1 else "within 10%"
        headline_rows.insert(
            0,
            ("Verdict", f"{verdict} (Arm B / Arm A = {ratio:.2f}×)"),
        )
    headline_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in headline_rows
    )

    duration_str = (
        f"{_fmt_ts(armB['start_t'])} → {_fmt_ts(armB['end_t'])} "
        f"({armB['duration_s']:.1f}s)"
        if armB["start_t"] and armB["end_t"] else "n/a"
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Arm A vs Arm B — comparison report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #777; margin-top: 0; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 36px; }}
  table.kv {{ border-collapse: collapse; }}
  table.kv td {{ padding: 4px 14px 4px 0; vertical-align: top; }}
  table.kv td:first-child {{ color: #666; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
  figure {{ display: inline-block; margin: 6px 8px 16px 0; }}
  figure img {{ display: block; width: 100%; max-width: 720px; border: 1px solid #ddd; }}
  figcaption {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 4px; }}
</style></head><body>
<h1>Comparison report — autonomous learner vs offline baseline</h1>
<p class="sub">{duration_str}</p>

<h2>Headline</h2>
<table class="kv">{headline_html}</table>

<h2>Locked-val MSE over the Arm B run</h2>
<p>Each point is one retrain cycle of Arm B. The dashed red line is Arm A (baseline <code>diff_iter4_wider</code>) measured on the same locked val set — Arm B wins any point that sits below the line. X-axis is total episodes collected, not cycle number, so the x-scale reflects actual data cost.</p>
{locked_chart}

<h2>Locked-val MSE by cycle</h2>
{cycle_chart}

<h2>Online verification probes</h2>
<p>Sanity-check telemetry: online MSE from real-time robot probes across the run (when <code>verify_after_retrain</code> is enabled).</p>
{probe_chart_html if probe_chart_html else "<p><i>No VERIFY telemetry in this run.</i></p>"}

<h2>Late-run probe grids (before · predicted · actual)</h2>
<div class="grid">
{probe_imgs_html}
</div>

</body></html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument(
        "--session",
        default=None,
        help="Arm B session tag (auto-detected if omitted).",
    )
    p.add_argument(
        "--arm-a",
        default=str(REPO_ROOT / "runs" / "arm_a_result.json"),
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "runs" / "comparison_report.html"),
    )
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    session = args.session or _latest_session_with_locked_val(runs_dir)
    if session is None:
        print(f"ERROR: no Arm B session with locked_val_measured events in {runs_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[comparison_report] session: {session}")

    out = build_report(runs_dir, session, Path(args.arm_a), Path(args.output))
    print(f"[comparison_report] written: {out}")


if __name__ == "__main__":
    main()
