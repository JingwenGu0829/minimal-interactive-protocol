"""Unit tests for the protocol state machine, with no timing involved."""

import unittest

from mip.protocol import (
    CancelWork,
    EndOfAudio,
    EndReason,
    FlushPlayback,
    Primitives,
    RunState,
    StartGeneration,
    Synthesize,
    split_sentences,
)
from mip.session import Session


def started_session(primitives=Primitives()):
    """A session with run 1 synthesizing "One." (100 ms) and "Two." (100 ms) queued."""
    session = Session("system", primitives)
    session.user_turn_ready("hi")
    session.text_delta(1, "One. Two. ")
    session.generation_done(1)
    session.audio_chunk(1, 0, 100)
    return session


class SplitSentences(unittest.TestCase):
    def test_splits_on_terminators_and_keeps_tail(self):
        self.assertEqual(split_sentences("One. Two! Thr"), (["One.", "Two!"], "Thr"))

    def test_does_not_split_decimals_or_abbreviations(self):
        self.assertEqual(split_sentences("Pi is 3.14 today. "), (["Pi is 3.14 today."], ""))
        self.assertEqual(split_sentences("Ask Dr. Lee now. "), (["Ask Dr. Lee now."], ""))

    def test_waits_for_whitespace_after_a_period(self):
        self.assertEqual(split_sentences("Version 3."), ([], "Version 3."))

    def test_cjk_terminator_needs_no_whitespace(self):
        self.assertEqual(split_sentences("你好。再见"), (["你好。"], "再见"))


class RunLifecycle(unittest.TestCase):
    def test_first_turn_starts_a_run_with_the_user_message(self):
        session = Session("system")
        commands = session.user_turn_ready("hi")
        self.assertIsInstance(commands[0], StartGeneration)
        self.assertEqual(commands[0].messages[-1], {"role": "user", "content": "hi"})

    def test_sentences_synthesize_one_at_a_time_and_in_order(self):
        session = Session("system")
        session.user_turn_ready("hi")
        first = session.text_delta(1, "One. Two. ")
        self.assertEqual(first, [Synthesize(1, 0, "One.")])
        session.audio_chunk(1, 0, 100)
        second = session.synthesis_done(1, 0)
        self.assertEqual(second, [Synthesize(1, 1, "Two.")])

    def test_completion_waits_for_the_playback_receipt(self):
        session = started_session()
        session.synthesis_done(1, 0)
        session.audio_chunk(1, 1, 100)
        commands = session.synthesis_done(1, 1)
        self.assertEqual(commands, [EndOfAudio(1)])
        self.assertIs(session.runs[1].state, RunState.RUNNING)
        session.playback_stopped(1, 200, completed=True)
        self.assertIs(session.runs[1].end_reason, EndReason.COMPLETED)
        self.assertEqual(session.history[-1]["content"], "One. Two.")

    def test_reply_with_no_audio_closes_without_waiting(self):
        session = Session("system")
        session.user_turn_ready("hi")
        self.assertEqual(session.generation_done(1), [])
        self.assertIs(session.runs[1].state, RunState.CLOSED)
        self.assertEqual(session.runs[1].heard, [])


class Interruption(unittest.TestCase):
    def test_speech_stops_the_reply_without_consulting_the_model(self):
        session = started_session()
        commands = session.user_speech_started()
        self.assertEqual(commands, [CancelWork(1), FlushPlayback(1)])
        self.assertIs(session.runs[1].state, RunState.CLOSING)

    def test_repeated_cancel_keeps_the_first_outcome(self):
        session = started_session()
        session.user_speech_started()
        session.playback_stopped(1, 50)
        self.assertEqual(session.cancel(1), [])
        self.assertIs(session.runs[1].end_reason, EndReason.CANCELLED)

    def test_events_from_a_stopped_run_are_rejected(self):
        session = started_session()
        session.user_speech_started()
        self.assertEqual(session.audio_chunk(1, 0, 100), [])
        self.assertEqual(session.text_delta(1, "Late. "), [])
        self.assertEqual(session.rejected_events, 2)

    def test_without_identity_a_late_event_passes_as_current(self):
        session = started_session(Primitives(run_identity=False))
        session.user_speech_started()
        self.assertEqual(len(session.audio_chunk(1, 0, 100)), 1)
        self.assertEqual(session.rejected_events, 0)

    def test_history_keeps_only_fully_played_sentences(self):
        session = started_session()
        session.synthesis_done(1, 0)
        session.audio_chunk(1, 1, 100)
        session.user_speech_started()
        session.playback_stopped(1, 150)
        self.assertEqual(session.runs[1].heard, ["One."])
        self.assertEqual(session.history[-1], {"role": "assistant", "content": "One."})

    def test_nothing_heard_leaves_no_assistant_entry(self):
        session = started_session()
        session.user_speech_started()
        session.playback_stopped(1, 0)
        self.assertEqual(session.runs[1].heard, [])
        self.assertEqual(session.history[-1]["role"], "user")

    def test_turn_during_closing_waits_and_follows_the_heard_reply(self):
        session = started_session()
        session.synthesis_done(1, 0)
        session.user_speech_started()
        self.assertEqual(session.user_turn_ready("wait"), [])
        commands = session.playback_stopped(1, 100)
        self.assertIsInstance(commands[0], StartGeneration)
        roles = [message["role"] for message in commands[0].messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_closing_timeout_records_unknown_rather_than_nothing(self):
        session = started_session()
        session.user_speech_started()
        session.closing_timeout(1)
        self.assertIs(session.runs[1].state, RunState.CLOSED)
        self.assertIsNone(session.runs[1].heard)

    def test_late_receipt_adds_detail_but_changes_no_outcome(self):
        session = started_session()
        session.user_speech_started()
        session.closing_timeout(1)
        self.assertEqual(session.playback_stopped(1, 80, completed=True), [])
        self.assertEqual(session.runs[1].played_ms, 80)
        self.assertIs(session.runs[1].end_reason, EndReason.CANCELLED)
        self.assertIsNone(session.runs[1].heard)

    def test_failure_stops_the_run_and_records_the_error(self):
        session = started_session()
        commands = session.run_failed(1, "HTTP 500")
        self.assertEqual(commands, [CancelWork(1), FlushPlayback(1)])
        session.playback_stopped(1, 0)
        self.assertIs(session.runs[1].end_reason, EndReason.FAILED)
        self.assertEqual(session.runs[1].error, "HTTP 500")


if __name__ == "__main__":
    unittest.main()
