"""The session coordinator: the only place conversation and run state change.

Every method takes one input event and returns the commands it causes. Whoever
drives the session (simulator or HTTP runner) executes those commands and feeds
the resulting events back in.

The design adapts ideas from the SGLang-Omni interactive runtime: persistent sessions,
response identity that invalidates work in flight, and history based on playback
(https://www.sglang.io/blog/omni-interactive-runtime). No SGLang-Omni code is used.

With `speech_as_tool=True` the model speaks by calling a `speak` tool instead of
writing text. An interruption then closes that one tool call, and what the user heard
comes back to the model as the call's result. Other tools are not cancelled by an
interruption: their results are addressed by call id and land in the history whenever
they arrive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .protocol import (
    SPEAK,
    CallTool,
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


RUNNING = json.dumps({"status": "running"})  # Placeholder until a tool's result arrives.


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict
    result: str | None = None  # Each call gets exactly one result.
    message: dict | None = None  # Its tool message, once it is in the history.


@dataclass
class Run:
    """One model inference and the speech it produces, up to a unique terminal state."""

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
    # Speech-as-a-tool mode only.
    calls: list[ToolCall] = field(default_factory=list)
    speak: ToolCall | None = None
    text: str = ""  # Plain text the model wrote. It is never spoken.


class Session:
    def __init__(
        self,
        system_prompt: str,
        primitives: Primitives = Primitives(),
        speech_as_tool: bool = False,
        max_inferences: int = 4,
    ):
        self.primitives = primitives
        self.speech_as_tool = speech_as_tool
        self.max_inferences = max_inferences  # Per user turn, so a turn cannot loop.
        self._inferences = 0
        self._call_runs: dict[str, Run] = {}
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
        if self.speech_as_tool:
            run.text += text
            return []
        run.tail += text
        complete, run.tail = split_sentences(run.tail)
        self._add_sentences(run, complete)
        return self._next_synthesis(run)

    def tool_call(self, run_id: int, call_id: str, name: str, arguments: dict) -> list[Command]:
        """The model called a tool. `speak` starts speech; anything else is dispatched."""
        run = self._event_run(run_id)
        if run is None:
            return []
        call = ToolCall(call_id, name, arguments)
        run.calls.append(call)
        if name != SPEAK:
            self._call_runs[call_id] = run
            return [CallTool(run.run_id, call_id, name, arguments)]
        if run.speak is not None:
            # One speech at a time: a second call must not silently replace the first.
            call.result = json.dumps({"status": "failed", "error": "BUSY"})
            return []
        run.speak = call
        sentences, tail = split_sentences(str(arguments.get("text", "")) + " ")
        self._add_sentences(run, sentences + ([tail.strip()] if tail.strip() else []))
        return self._next_synthesis(run)

    def tool_result(self, call_id: str, content: str) -> list[Command]:
        """A tool finished. The result is kept even if its turn was interrupted."""
        run = self._call_runs.get(call_id)
        call = next((c for c in run.calls if c.call_id == call_id), None) if run else None
        if call is None or call.result is not None:
            self.rejected_events += 1
            return []
        call.result = content
        if call.message is not None:
            call.message["content"] = content  # Replace the running placeholder.
        return self._continue_turn(self.latest)

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
        """Start an inference: for the queued user turns, or as a follow-up in a turn."""
        if self._pending_users:
            for text in self._pending_users:
                self.history.append({"role": "user", "content": text})
            self._pending_users.clear()
            self._inferences = 0
        self._inferences += 1
        run = Run(run_id=len(self.runs) + 1)
        self.runs[run.run_id] = run
        self.latest = run
        messages = self.history
        if not self.primitives.session:
            # Without a session, the model sees only the current turn.
            last_user = max(i for i, m in enumerate(self.history) if m["role"] == "user")
            messages = [self.history[0]] + self.history[last_user:]
        # Copies, so a result that arrives later cannot rewrite a request already sent.
        return [StartGeneration(run.run_id, tuple(dict(m) for m in messages))]

    @property
    def idle(self) -> bool:
        """True when nothing is running and no follow-up inference is on its way."""
        run = self.latest
        if run is None:
            return True
        waiting = self._wants_follow_up(run) and not self._results_in(run)
        return run.state is RunState.CLOSED and not self._pending_users and not waiting

    def _wants_follow_up(self, run: Run) -> bool:
        """The model called a tool other than speech and finished speaking normally."""
        return (
            self.speech_as_tool
            and run.end_reason is EndReason.COMPLETED
            and any(call.name != SPEAK for call in run.calls)
            and self._inferences < self.max_inferences
        )

    @staticmethod
    def _results_in(run: Run) -> bool:
        return all(call.result is not None for call in run.calls)

    def _continue_turn(self, run: Run | None) -> list[Command]:
        if run is None or run is not self.latest or run.state is not RunState.CLOSED:
            return []
        if self._pending_users:
            return self._start_run()
        if self._wants_follow_up(run) and self._results_in(run):
            return self._start_run()
        return []

    def _record_calls(self, run: Run, reason: EndReason, heard: list[str] | None) -> None:
        """Write the inference into the history as tool calls and their results.

        The speak call keeps its full text. What was heard goes into its result, so
        the record of what the model wrote and the fact of what played stay separate.
        """
        if not run.calls:
            return
        if run.speak is not None:
            if reason is EndReason.FAILED:
                result = {"status": "failed", "error": run.error}
            elif heard is None:
                result = {"status": "failed", "error": "playback position unknown"}
            elif reason is EndReason.COMPLETED:
                result = {"status": "completed"}
            else:
                result = {"status": "interrupted", "heard_prefix": " ".join(heard)}
            run.speak.result = json.dumps(result)
        self.history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in run.calls
                ],
            }
        )
        for call in run.calls:
            content = call.result if call.result is not None else RUNNING
            call.message = {"role": "tool", "tool_call_id": call.call_id, "content": content}
            self.history.append(call.message)

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
        if self.speech_as_tool:
            self._record_calls(run, reason, heard)
        elif heard:
            self.history.append({"role": "assistant", "content": " ".join(heard)})
        return self._continue_turn(run)
