"""Draws the ablation as a timeline of what the user hears, as a standalone SVG."""

from __future__ import annotations

import json
from pathlib import Path

from .simulate import ABLATIONS, SimConfig, Simulator, interruption_metrics

REPLY, STALE, NEXT = "#2a78d6", "#eb6834", "#1baf7a"
LEGEND = ((REPLY, "Interrupted reply"), (STALE, "Stale audio"), (NEXT, "Next reply"))
GRID = "#e4e3de"
MUTED = "#52514e"
LEFT, PLOT_WIDTH, NOTES_X = 112, 306, 428
TOP, ROW_HEIGHT, BAR_HEIGHT = 46, 32, 12
START_S, END_S = -2.0, 6.2  # Axis range, in seconds since the user started speaking.


def _notes(metrics: dict) -> dict[str, tuple[str, str, str]]:
    """Row label and a two-line description of the failure, from the measured values."""
    return {
        "none": (
            "Full protocol",
            f"Audio stops {metrics['none']['stop_latency_ms']:,.0f} ms after",
            "the user starts speaking",
        ),
        "cancel": (
            "No cancel",
            f"Plays for {metrics['cancel']['stop_latency_ms']:,.0f} ms more; next",
            f"reply waits {metrics['cancel']['next_reply_delay_ms']:,.0f} ms",
        ),
        "run_identity": (
            "No run identity",
            f"{metrics['run_identity']['stale_audio_ms']:,.0f} ms of stale audio plays",
            "after the stop command",
        ),
        "playback_progress": (
            "No playback progress",
            "Sounds normal, but history claims",
            f"{metrics['playback_progress']['unheard_chars_in_history']:,.0f} characters never played",
        ),
        "session": (
            "No session",
            "Sounds normal, but the next",
            "request has no earlier messages",
        ),
    }


def _merged(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Join chunks that play back to back into one continuous interval."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _x(seconds: float) -> float:
    return LEFT + (seconds - START_S) / (END_S - START_S) * PLOT_WIDTH


def _bar(row_y: float, start_ms: float, end_ms: float, onset_ms: float, color: str, tip: str) -> str:
    x0, x1 = _x((start_ms - onset_ms) / 1000), _x((end_ms - onset_ms) / 1000)
    return (
        f'<rect x="{x0:.1f}" y="{row_y:.1f}" width="{x1 - x0:.1f}" height="{BAR_HEIGHT}" '
        f'rx="2" fill="{color}"><title>{tip}: {start_ms - onset_ms:,.0f} to '
        f'{end_ms - onset_ms:,.0f} ms</title></rect>'
    )


def timeline_svg(scenario: dict, config: SimConfig) -> str:
    index, turn = next(
        (i, t) for i, t in enumerate(scenario["turns"]) if "speech_started_ms" in t
    )
    onset_ms = turn["speech_started_ms"]
    interrupted, next_run = index, index + 1

    sims = {
        name: Simulator(scenario, config, primitives).run()
        for name, primitives in ABLATIONS.items()
    }
    metrics = {name: interruption_metrics(sim, scenario) for name, sim in sims.items()}
    notes = _notes(metrics)
    height = TOP + ROW_HEIGHT * len(sims) + 34

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 {height}" role="img" '
        'aria-label="Timeline of what the user hears after interrupting, with each protocol '
        'primitive removed in turn" style="max-width:100%;height:auto;font-size:11px">'
    ]
    legend_x = float(LEFT)
    for color, label in LEGEND:
        parts.append(f'<rect x="{legend_x:.0f}" y="4" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 14:.0f}" y="13" fill="currentColor">{label}</text>')
        legend_x += 14 + len(label) * 5.6 + 18

    bottom = TOP + ROW_HEIGHT * len(sims)
    for tick in range(-2, 7, 2):
        x = _x(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 13}" text-anchor="middle" fill="{MUTED}">{tick}</text>'
        )
    parts.append(
        f'<text x="{_x((START_S + END_S) / 2):.1f}" y="{bottom + 28}" text-anchor="middle" '
        f'fill="{MUTED}">Seconds since the user started speaking</text>'
    )

    for row, (name, sim) in enumerate(sims.items()):
        center = TOP + ROW_HEIGHT * row + ROW_HEIGHT / 2
        label, line_one, line_two = notes[name]
        parts.append(
            f'<text x="{LEFT - 8}" y="{center + 4:.1f}" text-anchor="end" '
            f'fill="currentColor">{label}</text>'
        )
        parts.append(f'<text x="{NOTES_X}" y="{center - 2:.1f}" fill="{MUTED}">{line_one}</text>')
        parts.append(f'<text x="{NOTES_X}" y="{center + 11:.1f}" fill="{MUTED}">{line_two}</text>')
        bar_y = center - BAR_HEIGHT / 2
        flushed_at = sim.player.flushed_at.get(interrupted)
        spans = [(s, e) for run, s, e in sim.player.spans if run == interrupted]
        before = [(s, e) for s, e in spans if flushed_at is None or s < flushed_at]
        stale = [(s, e) for s, e in spans if flushed_at is not None and s >= flushed_at]
        for start, end in _merged(before):
            parts.append(_bar(bar_y, start, end, onset_ms, REPLY, "Interrupted reply"))
        for start, end in _merged(stale):
            parts.append(_bar(bar_y, start, end, onset_ms, STALE, "Stale audio"))
        following = [(s, e) for run, s, e in sim.player.spans if run == next_run]
        for start, end in _merged(following):
            parts.append(_bar(bar_y, start, end, onset_ms, NEXT, "Next reply"))

    zero = _x(0)
    parts.append(
        f'<line x1="{zero:.1f}" y1="{TOP - 8}" x2="{zero:.1f}" y2="{bottom}" '
        'stroke="currentColor" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{zero:.1f}" y="{TOP - 12}" text-anchor="middle" fill="currentColor">'
        'User starts speaking</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_timeline(scenario_path: str | Path, config: SimConfig, out_dir: str | Path) -> Path:
    scenario = json.loads(Path(scenario_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_timeline.svg"
    path.write_text(timeline_svg(scenario, config))
    return path
