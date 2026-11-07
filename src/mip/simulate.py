"""Deterministic discrete-event simulation of a conversation, plus protocol ablation.

Time is simulated, so every number is reproducible and can be checked by hand. The
results test the protocol logic only. They are not measurements of real models.
"""

from __future__ import annotations

import csv
import heapq
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from .protocol import (
    CancelWork,
    Command,
    DeliverAudio,
    EndOfAudio,
    FlushPlayback,
    Primitives,
    StartGeneration,
    Synthesize,
    split_sentences,
)
from .session import Session

ABLATIONS = {
    "none": Primitives(),
    "cancel": Primitives(cancel=False),
    "run_identity": Primitives(run_identity=False),
    "playback_progress": Primitives(playback_progress=False),
    "session": Primitives(session=False),
}


@dataclass(frozen=True)
class SimConfig:
    ms_per_char: int = 50  # Speech duration per character of text.
    chunk_ms: int = 200  # Audio duration of one chunk.
    synth_speed: float = 2.0  # Synthesis speed as a multiple of real time.
    llm_first_ms: int = 300  # Time from a request to the first sentence.
    llm_sentence_ms: int = 100  # Time between later sentences.
    net_ms: int = 50  # One-way delay on every link.
    vad_ms: int = 100  # Time to detect that the user started speaking.


class SimPlayer:
    """Plays chunks in arrival order, like a client with a simple audio queue."""

    def __init__(self, sim: "Simulator"):
        self.sim = sim
        self.queue: deque[DeliverAudio] = deque()
        self.playing: tuple[DeliverAudio, float] | None = None
        self.token = 0
        self.ended: set[int] = set()
        self.spans: list[tuple[int, float, float]] = []  # (run_id, start, end)
        self.first_audio: dict[int, float] = {}
        self.flushed_at: dict[int, float] = {}

    def played_ms(self, run_id: int) -> float:
        return sum(end - start for run, start, end in self.spans if run == run_id)

    def enqueue(self, chunk: DeliverAudio) -> None:
        self.queue.append(chunk)
        if self.playing is None:
            self._play_next()

    def _play_next(self) -> None:
        if self.queue:
            chunk = self.queue.popleft()
            self.token += 1
            self.playing = (chunk, self.sim.now)
            self.first_audio.setdefault(chunk.run_id, self.sim.now)
            self.sim.after(chunk.duration_ms, self._chunk_finished, self.token)
        else:
            self._report_finished_runs()

    def _chunk_finished(self, token: int) -> None:
        if token != self.token:
            return  # The chunk was cut off by a flush.
        self._record_playing()
        self._play_next()

    def _record_playing(self) -> None:
        chunk, start = self.playing
        if self.sim.now > start:  # A chunk flushed the moment it starts played nothing.
            self.spans.append((chunk.run_id, start, self.sim.now))
        self.playing = None

    def flush(self, run_id: int) -> None:
        self.flushed_at.setdefault(run_id, self.sim.now)
        if self.playing is not None and self.playing[0].run_id == run_id:
            self._record_playing()
            self.token += 1
        self.queue = deque(chunk for chunk in self.queue if chunk.run_id != run_id)
        self.sim.receipt(run_id, self.played_ms(run_id), completed=False)
        if self.playing is None:
            self._play_next()

    def end_of_audio(self, run_id: int) -> None:
        self.ended.add(run_id)
        self._report_finished_runs()

    def _report_finished_runs(self) -> None:
        if self.playing is not None:
            return
        for run_id in sorted(self.ended):
            self.ended.discard(run_id)
            self.sim.receipt(run_id, self.played_ms(run_id), completed=True)


class Simulator:
    def __init__(self, scenario: dict, config: SimConfig, primitives: Primitives):
        self.config = config
        self.session = Session(scenario["system"], primitives)
        self.player = SimPlayer(self)
        self.now = 0.0
        self.requests: dict[int, tuple[dict, ...]] = {}
        self._replies = iter(turn["reply"] for turn in scenario["turns"])
        self._cancelled: set[int] = set()
        self._events: list = []
        self._sequence = 0
        for turn in scenario["turns"]:
            if "speech_started_ms" in turn:
                started = turn["speech_started_ms"] + config.vad_ms
                self._at(started, self._input, self.session.user_speech_started)
            self._at(turn["ready_ms"], self._input, self.session.user_turn_ready, turn["user"])

    def run(self) -> "Simulator":
        while self._events:
            self.now, _, callback, args = heapq.heappop(self._events)
            callback(*args)
        return self

    def after(self, delay: float, callback, *args) -> None:
        self._at(self.now + delay, callback, *args)

    def receipt(self, run_id: int, played_ms: float, completed: bool) -> None:
        self.after(
            self.config.net_ms, self._input, self.session.playback_stopped,
            run_id, played_ms, completed,
        )

    def _at(self, time: float, callback, *args) -> None:
        # The sequence number keeps events at the same instant in scheduling order.
        heapq.heappush(self._events, (time, self._sequence, callback, args))
        self._sequence += 1

    def _input(self, method, *args) -> None:
        self._dispatch(method(*args))

    def _dispatch(self, commands: list[Command]) -> None:
        net = self.config.net_ms
        for command in commands:
            match command:
                case StartGeneration():
                    self.after(net, self._generate, command)
                case Synthesize():
                    self.after(net, self._synthesize, command)
                case CancelWork():
                    self.after(net, self._cancelled.add, command.run_id)
                case DeliverAudio():
                    self.after(net, self.player.enqueue, command)
                case EndOfAudio():
                    self.after(net, self.player.end_of_audio, command.run_id)
                case FlushPlayback():
                    self.after(net, self.player.flush, command.run_id)

    def _generate(self, command: StartGeneration) -> None:
        self.requests[command.run_id] = command.messages
        sentences, _ = split_sentences(next(self._replies) + " ")
        delay = self.config.llm_first_ms
        for sentence in sentences:
            self.after(delay, self._emit, command.run_id,
                       self.session.text_delta, sentence + " ")
            delay += self.config.llm_sentence_ms
        self.after(delay, self._emit, command.run_id, self.session.generation_done)

    def _synthesize(self, command: Synthesize) -> None:
        config = self.config
        total_ms = len(command.text) * config.ms_per_char
        full_chunks, remainder = divmod(total_ms, config.chunk_ms)
        chunks = [config.chunk_ms] * full_chunks + ([remainder] if remainder else [])
        delay = 0.0
        for duration in chunks:
            delay += duration / config.synth_speed
            self.after(delay, self._emit, command.run_id, self.session.audio_chunk,
                       command.sentence_id, duration)
        self.after(delay, self._emit, command.run_id, self.session.synthesis_done,
                   command.sentence_id)

    def _emit(self, run_id: int, method, *args) -> None:
        """A service produces an event, unless a cancel has already reached it."""
        if run_id not in self._cancelled:
            self.after(self.config.net_ms, self._input, method, run_id, *args)


def interruption_metrics(sim: Simulator, scenario: dict) -> dict:
    """Measure the first interruption in the scenario."""
    index, turn = next(
        (i, t) for i, t in enumerate(scenario["turns"]) if "speech_started_ms" in t
    )
    # Run ids start at 1, so the reply being interrupted by turn `index` is run `index`.
    interrupted, next_run = index, index + 1
    spans = [(start, end) for run, start, end in sim.player.spans if run == interrupted]
    flushed_at = sim.player.flushed_at.get(interrupted)
    stale_ms = 0.0
    if flushed_at is not None:
        stale_ms = sum(max(0.0, end - max(start, flushed_at)) for start, end in spans)
    truly_heard_chars = sim.player.played_ms(interrupted) / sim.config.ms_per_char
    history_chars = sum(len(text) for text in sim.session.runs[interrupted].heard or [])
    return {
        "stop_latency_ms": max(end for _, end in spans) - turn["speech_started_ms"],
        "stale_audio_ms": stale_ms,
        "unheard_chars_in_history": max(0.0, history_chars - truly_heard_chars),
        # Everything in the next request except the system prompt and the new turn.
        "prior_messages_in_next_request": len(sim.requests[next_run]) - 2,
        "next_reply_delay_ms": sim.player.first_audio[next_run] - turn["ready_ms"],
    }


def run_ablation(scenario_path: str | Path, config: SimConfig) -> list[dict]:
    scenario = json.loads(Path(scenario_path).read_text())
    rows = []
    for removed, primitives in ABLATIONS.items():
        sim = Simulator(scenario, config, primitives).run()
        rows.append({"removed_primitive": removed, **interruption_metrics(sim, scenario)})
    return rows


def write_rows(rows: list[dict], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def format_rows(rows: list[dict]) -> str:
    names = list(rows[0])
    widths = [max(len(name), 17) for name in names]
    lines = ["  ".join(name.ljust(width) for name, width in zip(names, widths))]
    for row in rows:
        cells = [
            str(row[name]) if isinstance(row[name], str) else f"{row[name]:g}" for name in names
        ]
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))
    return "\n".join(lines)
