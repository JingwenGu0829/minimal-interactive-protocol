"""Tests on a real chat model the two primitives that change what the model is told.

The user interrupts a reply, then asks what the agent last said. The model can answer
only from the history the protocol gave it, so the answer shows whether that history
matches what actually played. This is a small demonstration with one fixed prompt, not
an estimate of how often real systems fail.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .live import run_dialog

CONDITIONS = {
    "none": {},
    "playback_progress": {"playback_progress": False},
    "session": {"session": False},
}
FIRST_TURN = "Tell me three unrelated facts about Paris, one short sentence each."
PROBE_TURN = "Sorry, what was the last thing you said before I cut you off? Quote it exactly."


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def classify(reply: str, planned: list[str], fully_played: list[str]) -> str:
    """Label a reply. Quoting anything that did not fully play takes priority."""
    reply = _normalized(reply)
    not_played = [sentence for sentence in planned if sentence not in fully_played]
    if any(_normalized(sentence) in reply for sentence in not_played):
        return "quoted_not_played"
    if any(_normalized(sentence) in reply for sentence in fully_played):
        return "quoted_played"
    return "no_quote"


async def run_probe(config: dict, repeats: int = 10, interrupt_after_ms: int = 2000) -> dict:
    trials = []
    for removed, primitives in CONDITIONS.items():
        for _ in range(repeats):
            dialog = {
                **config,
                "primitives": primitives,
                "turns": [
                    {"user": FIRST_TURN, "interrupt_after_ms": interrupt_after_ms},
                    {"user": PROBE_TURN},
                ],
            }
            first, second = (await run_dialog(dialog))["runs"]
            reply = " ".join(second["planned"])
            failed = "failed" in (first["end_reason"], second["end_reason"])
            label = "failed" if failed else classify(
                reply, first["planned"], first["fully_played"]
            )
            trials.append(
                {
                    "removed_primitive": removed,
                    "label": label,
                    "planned": first["planned"],
                    "fully_played": first["fully_played"],
                    "history": first["heard"],
                    "reply": reply,
                    "first_text_ms": [first["first_text_ms"], second["first_text_ms"]],
                    "errors": [first["error"], second["error"]],
                }
            )
    latencies = [ms for trial in trials for ms in trial["first_text_ms"] if ms is not None]
    return {
        "repeats": repeats,
        "interrupt_after_ms": interrupt_after_ms,
        "counts": {
            removed: dict(Counter(t["label"] for t in trials if t["removed_primitive"] == removed))
            for removed in CONDITIONS
        },
        "first_text_ms": {
            "requests": len(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies, default=None),
            "max": max(latencies, default=None),
        },
        "trials": trials,
    }


def write_summary(summary: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "probe.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return path
