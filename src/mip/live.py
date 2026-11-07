"""Runs a scripted conversation against real HTTP endpoints, in real time.

This drives the same `Session` as the simulator. Only the execution of its commands
differs: asyncio tasks and HTTP streams replace simulated events.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from .mock_server import MockAPI
from .protocol import (
    CancelWork,
    Command,
    DeliverAudio,
    EndOfAudio,
    FlushPlayback,
    Primitives,
    RunState,
    StartGeneration,
    Synthesize,
)
from .services import Endpoint, OpenAIChat, OpenAISpeech, OpenAITranscription
from .session import Session


@dataclass
class _Chunk:
    start: float
    duration_ms: float
    pcm: bytes


class ClockPlayer:
    """Stands in for a loudspeaker: audio "plays" by the clock, and nothing else."""

    def __init__(self, on_stopped):
        self.on_stopped = on_stopped
        self.chunks: dict[int, list[_Chunk]] = defaultdict(list)
        self.play_until = 0.0
        self.finish_timer: asyncio.TimerHandle | None = None
        # What actually played, kept even when the session is not told (for evaluation).
        self.stopped_ms: dict[int, float] = {}

    def enqueue(self, command: DeliverAudio) -> None:
        start = max(time.monotonic(), self.play_until)
        self.play_until = start + command.duration_ms / 1000
        self.chunks[command.run_id].append(_Chunk(start, command.duration_ms, command.pcm))

    def end_of_audio(self, run_id: int) -> None:
        delay = max(0.0, self.play_until - time.monotonic())
        self.stopped_ms[run_id] = self.played_ms(run_id, at_end=True)
        self.finish_timer = asyncio.get_running_loop().call_later(
            delay, self.on_stopped, run_id, self.stopped_ms[run_id], True
        )

    def flush(self, run_id: int) -> None:
        if self.finish_timer is not None:
            self.finish_timer.cancel()
            self.finish_timer = None
        now = time.monotonic()
        self.play_until = now
        played = self.played_ms(run_id)
        self.stopped_ms[run_id] = played
        self.chunks[run_id] = [chunk for chunk in self.chunks[run_id] if chunk.start < now]
        self.on_stopped(run_id, played, False)

    def played_ms(self, run_id: int, at_end: bool = False) -> float:
        chunks = self.chunks[run_id]
        if at_end:
            return sum(chunk.duration_ms for chunk in chunks)
        now = time.monotonic()
        return sum(
            min(chunk.duration_ms, max(0.0, (now - chunk.start) * 1000)) for chunk in chunks
        )

    def pcm(self, run_id: int) -> bytes:
        return b"".join(chunk.pcm for chunk in self.chunks[run_id])


class LiveRunner:
    def __init__(self, session: Session, chat: OpenAIChat, speech: OpenAISpeech):
        self.session = session
        self.chat = chat
        self.speech = speech
        self.player = ClockPlayer(self._playback_stopped)
        self.tasks: dict[int, set[asyncio.Task]] = defaultdict(set)
        self.started_at: dict[int, float] = {}
        self.first_text_ms: dict[int, float] = {}
        self.first_audio_ms: dict[int, float] = {}

    def _elapsed_ms(self, run_id: int) -> float:
        return (time.monotonic() - self.started_at[run_id]) * 1000

    def dispatch(self, commands: list[Command]) -> None:
        for command in commands:
            match command:
                case StartGeneration():
                    self.started_at[command.run_id] = time.monotonic()
                    self._spawn(command.run_id, self._generate(command))
                case Synthesize():
                    self._spawn(command.run_id, self._synthesize(command))
                case CancelWork():
                    for task in self.tasks[command.run_id]:
                        if task is not asyncio.current_task():
                            task.cancel()  # Closes the HTTP stream to the service.
                case DeliverAudio():
                    self.first_audio_ms.setdefault(
                        command.run_id, self._elapsed_ms(command.run_id)
                    )
                    self.player.enqueue(command)
                case EndOfAudio():
                    self.player.end_of_audio(command.run_id)
                case FlushPlayback():
                    self.player.flush(command.run_id)

    async def wait_closed(self, run_id: int, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self.session.runs[run_id].state is not RunState.CLOSED:
            if time.monotonic() > deadline:
                raise TimeoutError(f"run {run_id} did not close within {timeout_s} s")
            await asyncio.sleep(0.01)

    def _spawn(self, run_id: int, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks[run_id].add(task)
        task.add_done_callback(self.tasks[run_id].discard)

    def _playback_stopped(self, run_id: int, played_ms: float, completed: bool) -> None:
        self.dispatch(self.session.playback_stopped(run_id, played_ms, completed))

    async def _generate(self, command: StartGeneration) -> None:
        try:
            async for text in self.chat.stream(list(command.messages)):
                self.first_text_ms.setdefault(command.run_id, self._elapsed_ms(command.run_id))
                self.dispatch(self.session.text_delta(command.run_id, text))
            self.dispatch(self.session.generation_done(command.run_id))
        except Exception as error:  # Any service failure must end the run, not hang it.
            self.dispatch(self.session.run_failed(command.run_id, repr(error)))

    async def _synthesize(self, command: Synthesize) -> None:
        run_id, sentence_id = command.run_id, command.sentence_id
        try:
            async for pcm in self.speech.stream(command.text):
                duration = self.speech.duration_ms(pcm)
                self.dispatch(self.session.audio_chunk(run_id, sentence_id, duration, pcm))
            self.dispatch(self.session.synthesis_done(run_id, sentence_id))
        except Exception as error:  # Any service failure must end the run, not hang it.
            self.dispatch(self.session.run_failed(run_id, repr(error)))


@contextlib.asynccontextmanager
async def _resolve_mocks(services: dict):
    """Serve any endpoint whose `base_url` is "mock" from the bundled mock server."""
    if all(service["base_url"] != "mock" for service in services.values()):
        yield services
        return
    api = MockAPI()
    base_url = await api.start()
    try:
        yield {
            role: {**service, "base_url": base_url} if service["base_url"] == "mock" else service
            for role, service in services.items()
        }
    finally:
        await api.stop()


async def run_dialog(config: dict, out_dir: str | Path | None = None) -> dict:
    """Run `config["turns"]` in order and return the transcript and each run's outcome.

    A turn gives `user` text, or `audio_path` to transcribe a WAV file first. It may
    set `interrupt_after_ms` to speak over the reply that the turn produces. An optional
    `primitives` object turns protocol primitives off, for example
    `{"session": false}`.
    """
    primitives = Primitives(**config.get("primitives", {}))
    async with _resolve_mocks(config["services"]) as services, aiohttp.ClientSession() as http:
        chat = OpenAIChat(http, Endpoint(**services["chat"]))
        speech = OpenAISpeech(http, Endpoint(**services["speech"]), **config.get("speech", {}))
        asr = OpenAITranscription(http, Endpoint(**services["transcription"]))
        session = Session(config["system"], primitives)
        runner = LiveRunner(session, chat, speech)
        for turn in config["turns"]:
            text = turn.get("user")
            if text is None:
                text = await asr.transcribe(Path(turn["audio_path"]).read_bytes())
            runner.dispatch(session.user_turn_ready(text))
            run_id = session.latest.run_id
            if "interrupt_after_ms" in turn:
                await asyncio.sleep(turn["interrupt_after_ms"] / 1000)
                runner.dispatch(session.user_speech_started())
            await runner.wait_closed(run_id)
    result = {
        "history": session.history,
        "runs": [
            {
                "run_id": run.run_id,
                "end_reason": run.end_reason.value,
                "error": run.error,
                "first_text_ms": runner.first_text_ms.get(run.run_id),
                "first_audio_ms": runner.first_audio_ms.get(run.run_id),
                "played_ms": run.played_ms,
                "planned": [sentence.text for sentence in run.sentences],
                "heard": run.heard,
                "fully_played": Session.fully_played(
                    run, runner.player.stopped_ms.get(run.run_id, 0)
                ),
            }
            for run in session.runs.values()
        ],
    }
    if out_dir is not None:
        _write_audio(Path(out_dir), runner.player, session, speech.sample_rate)
    return result


def _write_audio(out_dir: Path, player: ClockPlayer, session: Session, sample_rate: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_dir / "live.wav"), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        for run_id in session.runs:
            file.writeframes(player.pcm(run_id))
