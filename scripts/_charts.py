"""Inline-SVG chart helpers shared by the report generators."""

from __future__ import annotations

import html


def svg_line_chart(
    series: list[tuple[str, list[float], str]],
    width: int = 720,
    height: int = 260,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    hline: float | None = None,
    hline_label: str = "",
    x_values: list[float] | None = None,
) -> str:
    """Minimal inline SVG line chart for one or more series.

    series items: (label, y_values, stroke_color).
    If `x_values` is provided, it indexes the x-axis explicitly (same length
    as the longest y series) — otherwise the points are spaced uniformly.
    """
    pad_l, pad_r, pad_t, pad_b = 60, 140, 30, 38
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

    if x_values is not None:
        xs = list(x_values)
        x_min = min(xs)
        x_max = max(xs)
        if x_max == x_min:
            x_max = x_min + 1e-9
        def _x(i: int) -> float:
            return pad_l + ((xs[i] - x_min) / (x_max - x_min)) * plot_w
        xlabel_points = xs
    else:
        x_max_idx = max(1, max_len - 1)
        def _x(i: int) -> float:
            return pad_l + (i / x_max_idx) * plot_w
        xlabel_points = list(range(max_len))

    def _y(v: float) -> float:
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="-apple-system,Segoe UI,Roboto,sans-serif" font-size="11">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="#fafafa" stroke="#d0d0d0"/>'
    )

    # Y gridlines + labels (5 ticks)
    for i in range(5):
        frac = i / 4
        yv = y_min + frac * (y_max - y_min)
        py = _y(yv)
        parts.append(
            f'<line x1="{pad_l}" y1="{py}" x2="{pad_l + plot_w}" y2="{py}" stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{py + 3}" text-anchor="end" fill="#555">{yv:.4g}</text>'
        )

    # X tick labels (start / mid / end)
    if xlabel_points:
        picks = [0, max_len // 2, max_len - 1]
        for i in picks:
            if 0 <= i < max_len:
                label = f"{xlabel_points[i]:g}" if isinstance(xlabel_points[i], (int, float)) else str(xlabel_points[i])
                parts.append(
                    f'<text x="{_x(i)}" y="{pad_t + plot_h + 14}" text-anchor="middle" fill="#555">{label}</text>'
                )

    # Horizontal threshold line
    if hline is not None and y_min <= hline <= y_max:
        hy = _y(hline)
        parts.append(
            f'<line x1="{pad_l}" y1="{hy}" x2="{pad_l + plot_w}" y2="{hy}" '
            f'stroke="#d94141" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w + 6}" y="{hy + 3}" fill="#d94141">{html.escape(hline_label)}</text>'
        )

    # Series
    legend_y = pad_t + 4
    for label, ys, color in series:
        if not ys:
            continue
        pts = " ".join(f"{_x(i)},{_y(v)}" for i, v in enumerate(ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        for i, v in enumerate(ys):
            parts.append(f'<circle cx="{_x(i)}" cy="{_y(v)}" r="2.4" fill="{color}"/>')
        parts.append(
            f'<rect x="{pad_l + plot_w + 6}" y="{legend_y - 8}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w + 20}" y="{legend_y + 1}" fill="#333">{html.escape(label)}</text>'
        )
        legend_y += 16

    if title:
        parts.append(
            f'<text x="{width / 2}" y="16" text-anchor="middle" font-weight="600" fill="#222">{html.escape(title)}</text>'
        )
    if xlabel:
        parts.append(
            f'<text x="{pad_l + plot_w / 2}" y="{height - 6}" text-anchor="middle" fill="#555">{html.escape(xlabel)}</text>'
        )
    if ylabel:
        parts.append(
            f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#555" '
            f'transform="rotate(-90 14 {pad_t + plot_h / 2})">{html.escape(ylabel)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_bar_chart(
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
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="-apple-system,Segoe UI,Roboto,sans-serif" font-size="11">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="#fafafa" stroke="#d0d0d0"/>'
    )
    for i in range(5):
        frac = i / 4
        yv = frac * vmax
        py = pad_t + (1 - frac) * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{py}" x2="{pad_l + plot_w}" y2="{py}" stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{py + 3}" text-anchor="end" fill="#555">{yv:.4g}</text>'
        )
    slot = plot_w / len(bars)
    bw = slot * 0.6
    for i, (label, v) in enumerate(bars):
        x = pad_l + i * slot + (slot - bw) / 2
        h_px = (v / vmax) * plot_h
        y = pad_t + plot_h - h_px
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{h_px}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bw / 2}" y="{pad_t + plot_h + 14}" text-anchor="middle" fill="#555">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x + bw / 2}" y="{y - 4}" text-anchor="middle" fill="#333">{v:.4f}</text>'
        )
    if title:
        parts.append(
            f'<text x="{width / 2}" y="16" text-anchor="middle" font-weight="600" fill="#222">{html.escape(title)}</text>'
        )
    if ylabel:
        parts.append(
            f'<text x="14" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#555" '
            f'transform="rotate(-90 14 {pad_t + plot_h / 2})">{html.escape(ylabel)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
