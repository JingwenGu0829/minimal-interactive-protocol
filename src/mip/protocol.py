"""Protocol vocabulary: the primitives, the run lifecycle, and session commands.

This module and `session.py` do no I/O. The same protocol can therefore be driven
by the deterministic simulator (`simulate.py`) and by HTTP services (`live.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Primitives:
    """Each flag is one protocol primitive. Turning one off is an ablation."""

    session: bool = True  # Conversation context survives across replies.
    cancel: bool = True  # In-progress work and playback can be stopped.
    run_identity: bool = True  # Every event names the reply it belongs to.
    playback_progress: bool = True  # The player reports what was actually played.


class RunState(str, Enum):
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class EndReason(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class StartGeneration:
    run_id: int
    messages: tuple[dict, ...]


@dataclass(frozen=True)
class Synthesize:
    run_id: int
    sentence_id: int
    text: str


@dataclass(frozen=True)
class CancelWork:
    run_id: int


@dataclass(frozen=True)
class DeliverAudio:
    run_id: int
    sentence_id: int
    duration_ms: float
    pcm: bytes = b""


@dataclass(frozen=True)
class EndOfAudio:
    run_id: int


@dataclass(frozen=True)
class FlushPlayback:
    run_id: int


Command = StartGeneration | Synthesize | CancelWork | DeliverAudio | EndOfAudio | FlushPlayback

# An English terminator only ends a sentence once whitespace follows it, so a
# streamed "3." does not split before its "5" arrives. CJK terminators end one at once.
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)|[。！？]+[”’）]*")
_ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "vs.", "e.g.", "i.e.")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return the complete sentences in `buffer` and the unfinished tail."""
    sentences = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        candidate = buffer[start : match.end()].strip()
        if candidate.endswith(_ABBREVIATIONS):
            continue
        sentences.append(candidate)
        start = match.end()
    return sentences, buffer[start:].lstrip()
