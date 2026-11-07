"""The session coordinator: the only place conversation and run state change.

Every method takes one input event and returns the commands it causes. Whoever
drives the session (simulator or HTTP runner) executes those commands and feeds
the resulting events back in.

The design adapts ideas from the SGLang-Omni interactive runtime: persistent sessions,
response identity that invalidates work in flight, and history based on playback
(https://www.sglang.io/blog/omni-interactive-runtime). No SGLang-Omni code is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import (
    CancelWork,
    Command,
    DeliverAudio,
    EndOfAudio,
    EndReason,
    FlushPlayback,
    Primitives,
    RunState,
    StartGeneration,
    Synthesize,
    split_sentences,
)


@dataclass
class Sentence:
    sentence_id: int
    text: str
    audio_ms: float = 0
    sent: bool = False
    synth_done: bool = False


@dataclass
class Run:
    """One reply, from the generation request to its unique terminal state."""

    run_id: int
    state: RunState = RunState.RUNNING
    end_reason: EndReason | None = None
    error: str | None = None
    sentences: list[Sentence] = field(default_factory=list)
    tail: str = ""
    synthesizing: bool = False
    generation_done: bool = False
    end_of_audio_sent: bool = False
    played_ms: float | None = None
    # None means unknown; [] means confirmed that nothing was heard.
    heard: list[str] | None = None


class Session:
    def __init__(self, system_prompt: str, primitives: Primitives = Primitives()):
        self.primitives = primitives
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self.runs: dict[int, Run] = {}
        self.latest: Run | None = None
        self.rejected_events = 0
        self._pending_users: list[str] = []

    # --- Events from the user or client ---------------------------------------

    def user_speech_started(self) -> list[Command]:
        """Stop the reply at once; the model is not consulted."""
        if self.latest is None:
            return []
        return self.cancel(self.latest.run_id)

    def cancel(self, run_id: int) -> list[Command]:
        """Stop a run. Repeating this never changes an earlier outcome."""
        run = self.runs.get(run_id)
        if run is None or run.state is not RunState.RUNNING or not self.primitives.cancel:
            return []
        return self._stop(run, EndReason.CANCELLED)

    def user_turn_ready(self, text: str) -> list[Command]:
        """Queue the turn; it starts once the previous reply has closed."""
        self._pending_users.append(text)
        if self.latest is not None and self.latest.state is not RunState.CLOSED:
            return []
        return self._start_run()

    def playback_stopped(
        self, run_id: int, played_ms: float, completed: bool = False
    ) -> list[Command]:
        """Receipt from the player. A late one adds detail but changes no outcome."""
        if not self.primitives.playback_progress:
            return []
        run = self.runs.get(run_id)
        if run is None:
            return []
        run.played_ms = played_ms
        if run.state is RunState.CLOSED:
            return []
        if completed:
            heard = [sentence.text for sentence in run.sentences]
        else:
            heard = self.fully_played(run, played_ms)
        reason = run.end_reason or (EndReason.COMPLETED if completed else EndReason.CANCELLED)
        return self._close(run, reason, heard)

    def closing_timeout(self, run_id: int) -> list[Command]:
        """The player never confirmed the stop, so what was heard stays unknown."""
        run = self.runs.get(run_id)
        if run is None or run.state is not RunState.CLOSING:
            return []
        return self._close(run, run.end_reason, heard=None)

    # --- Events from model services -------------------------------------------

    def text_delta(self, run_id: int, text: str) -> list[Command]:
        run = self._event_run(run_id)
        if run is None:
            return []
        run.tail += text
        complete, run.tail = split_sentences(run.tail)
        self._add_sentences(run, complete)
        return self._next_synthesis(run)

    def generation_done(self, run_id: int) -> list[Command]:
        run = self._event_run(run_id)
        if run is None:
            return []
        if run.tail.strip():
            self._add_sentences(run, [run.tail.strip()])
            run.tail = ""
        run.generation_done = True
        return self._next_synthesis(run) + self._finish_if_ready(run)

    def audio_chunk(
        self, run_id: int, sentence_id: int, duration_ms: float, pcm: bytes = b""
    ) -> list[Command]:
        run = self._event_run(run_id)
        if run is None:
            return []
        if sentence_id < len(run.sentences):
            sentence = run.sentences[sentence_id]
            sentence.audio_ms += duration_ms
            sentence.sent = True
        return [DeliverAudio(run.run_id, sentence_id, duration_ms, pcm)]

    def synthesis_done(self, run_id: int, sentence_id: int) -> list[Command]:
        run = self._event_run(run_id)
        if run is None:
            return []
        run.sentences[sentence_id].synth_done = True
        run.synthesizing = False
        return self._next_synthesis(run) + self._finish_if_ready(run)

    def run_failed(self, run_id: int, error: str) -> list[Command]:
        run = self._event_run(run_id)
        if run is None:
            return []
        run.error = error
        return self._stop(run, EndReason.FAILED)

    # --- Internal transitions -------------------------------------------------

    def _event_run(self, run_id: int) -> Run | None:
        """Find the run an event belongs to, rejecting events from a stopped run."""
        if not self.primitives.run_identity:
            # Without identity, a late event cannot be told apart from a current one.
            return self.latest
        run = self.runs.get(run_id)
        if run is None or run.state is not RunState.RUNNING:
            self.rejected_events += 1
            return None
        return run

    def _start_run(self) -> list[Command]:
        for text in self._pending_users:
            self.history.append({"role": "user", "content": text})
        current_user = self.history[-1]
        self._pending_users.clear()
        run = Run(run_id=len(self.runs) + 1)
        self.runs[run.run_id] = run
        self.latest = run
        messages = self.history if self.primitives.session else [self.history[0], current_user]
        return [StartGeneration(run.run_id, tuple(messages))]

    @staticmethod
    def _add_sentences(run: Run, texts: list[str]) -> None:
        for text in texts:
            run.sentences.append(Sentence(len(run.sentences), text))

    @staticmethod
    def _next_synthesis(run: Run) -> list[Command]:
        """Synthesize one sentence at a time so that audio stays in order."""
        if run.synthesizing:
            return []
        for sentence in run.sentences:
            if not sentence.synth_done:
                run.synthesizing = True
                return [Synthesize(run.run_id, sentence.sentence_id, sentence.text)]
        return []

    def _finish_if_ready(self, run: Run) -> list[Command]:
        ready = run.generation_done and not run.synthesizing and not run.end_of_audio_sent
        if run.state is not RunState.RUNNING or not ready:
            return []
        run.end_of_audio_sent = True
        sent = [sentence.text for sentence in run.sentences if sentence.sent]
        if not sent:
            return self._close(run, EndReason.COMPLETED, heard=[])
        if self.primitives.playback_progress:
            return [EndOfAudio(run.run_id)]
        # With no receipts coming, "sent" is the best knowledge the session has.
        return [EndOfAudio(run.run_id)] + self._close(run, EndReason.COMPLETED, heard=sent)

    def _stop(self, run: Run, reason: EndReason) -> list[Command]:
        run.state = RunState.CLOSING
        run.end_reason = reason
        commands: list[Command] = [CancelWork(run.run_id), FlushPlayback(run.run_id)]
        if not self.primitives.playback_progress:
            sent = [sentence.text for sentence in run.sentences if sentence.sent]
            commands += self._close(run, reason, heard=sent)
        return commands

    @staticmethod
    def fully_played(run: Run, played_ms: float) -> list[str]:
        """Keep only sentences whose audio fully played.

        This is the fallback rule from the SGLang-Omni post linked above, used when
        no finer alignment between audio and text is available.
        """
        heard = []
        elapsed = 0.0
        for sentence in run.sentences:
            elapsed += sentence.audio_ms
            if not sentence.synth_done or elapsed > played_ms:
                break
            heard.append(sentence.text)
        return heard

    def _close(self, run: Run, reason: EndReason, heard: list[str] | None) -> list[Command]:
        run.state = RunState.CLOSED
        run.end_reason = reason
        run.heard = heard
        # A reply nobody heard leaves no entry. An empty assistant message is not
        # neutral: see the empty-reply replay reported by the `baseline` target.
        if heard:
            self.history.append({"role": "assistant", "content": " ".join(heard)})
        return self._start_run() if self._pending_users else []
