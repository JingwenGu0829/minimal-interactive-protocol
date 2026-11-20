"""Tests for speech-as-a-tool mode: the session, the simulated ablation, and HTTP.

Simulation numbers are worked out by hand from `testdata/tool_dialog.json` with the
default timings (50 ms per character, 200 ms chunks at 2x real time, 50 ms per link,
100 ms to detect speech).

Turn 2 is ready at 4000 ms. The model calls `speak` with three sentences of 25, 29 and
30 characters (4200 ms of audio), then `get_weather`. First audio plays at 4650 ms, as
in the text dialog. The user starts speaking at 6500 ms and the flush reaches the
player at 6650 ms, so 2000 ms played (40 characters): all of sentence one and part of
sentence two. The weather result arrives near 7100 ms, after the interruption and
before the next turn is ready at 7500 ms.
"""

import json
import unittest
from pathlib import Path

from mip import tools
from mip.live import run_dialog
from mip.mock_server import MockAPI
from mip.protocol import CallTool, EndReason, StartGeneration, Synthesize
from mip.session import RUNNING, Session
from mip.simulate import ABLATIONS, SimConfig, Simulator, interruption_metrics

SCENARIO = json.loads((Path(__file__).parent / "testdata" / "tool_dialog.json").read_text())


def speaking_session():
    """Run 1 said "One." (100 ms, synthesized) and has "Two." queued."""
    session = Session("system", speech_as_tool=True)
    session.user_turn_ready("hi")
    session.tool_call(1, "s1", "speak", {"text": "One. Two."})
    session.audio_chunk(1, 0, 100)
    session.synthesis_done(1, 0)
    return session


def tool_messages(session):
    return {m["tool_call_id"]: json.loads(m["content"]) for m in session.history
            if m["role"] == "tool"}


class ToyTools(unittest.TestCase):
    def test_results_are_json_and_errors_do_not_raise(self):
        self.assertEqual(json.loads(tools.execute("multiply", {"a": 17, "b": 23})),
                         {"product": 391})
        self.assertEqual(json.loads(tools.execute("get_weather", {"city": "Paris"}))["temp_c"], 18)
        self.assertIn("error", json.loads(tools.execute("get_weather", {"city": "Atlantis"})))
        self.assertIn("error", json.loads(tools.execute("multiply", {"a": 1})))
        self.assertIn("error", json.loads(tools.execute("missing", {})))

    def test_speak_is_offered_first(self):
        names = [schema["function"]["name"] for schema in tools.schemas()]
        self.assertEqual(names, ["speak", "get_weather", "multiply"])


class SpeechAsTool(unittest.TestCase):
    def test_speak_starts_synthesis_and_plain_text_is_not_spoken(self):
        session = Session("system", speech_as_tool=True)
        session.user_turn_ready("hi")
        self.assertEqual(session.text_delta(1, "Hello there. "), [])
        commands = session.tool_call(1, "s1", "speak", {"text": "One. Two."})
        self.assertEqual(commands, [Synthesize(1, 0, "One.")])

    def test_completed_speech_is_recorded_as_a_call_and_ends_the_turn(self):
        session = speaking_session()
        session.audio_chunk(1, 1, 100)
        session.synthesis_done(1, 1)
        session.generation_done(1)
        commands = session.playback_stopped(1, 200, completed=True)
        self.assertEqual(commands, [])  # Only speech was called: no follow-up inference.
        self.assertTrue(session.idle)
        call = session.history[-2]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"text": "One. Two."})
        self.assertEqual(tool_messages(session), {"s1": {"status": "completed"}})

    def test_interruption_keeps_the_full_text_and_reports_what_was_heard(self):
        session = speaking_session()
        session.user_speech_started()
        session.playback_stopped(1, 100)
        self.assertEqual(session.runs[1].end_reason, EndReason.CANCELLED)
        call = session.history[-2]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"text": "One. Two."})
        self.assertEqual(
            tool_messages(session), {"s1": {"status": "interrupted", "heard_prefix": "One."}}
        )

    def test_a_second_speak_in_one_inference_is_refused(self):
        session = speaking_session()
        self.assertEqual(session.tool_call(1, "s2", "speak", {"text": "Three."}), [])
        session.user_speech_started()
        session.playback_stopped(1, 100)
        self.assertEqual(tool_messages(session)["s2"], {"status": "failed", "error": "BUSY"})

    def test_a_result_that_arrives_during_speech_waits_then_starts_a_follow_up(self):
        session = speaking_session()
        call = session.tool_call(1, "w1", "get_weather", {"city": "Paris"})
        self.assertEqual(call, [CallTool(1, "w1", "get_weather", {"city": "Paris"})])
        self.assertEqual(session.tool_result("w1", '{"temp_c": 18}'), [])  # Still speaking.
        session.audio_chunk(1, 1, 100)
        session.synthesis_done(1, 1)
        session.generation_done(1)
        commands = session.playback_stopped(1, 200, completed=True)
        self.assertIsInstance(commands[0], StartGeneration)
        self.assertEqual(commands[0].messages[-1]["content"], '{"temp_c": 18}')

    def test_a_slow_result_starts_the_follow_up_when_it_arrives(self):
        session = speaking_session()
        session.tool_call(1, "w1", "get_weather", {"city": "Paris"})
        session.audio_chunk(1, 1, 100)
        session.synthesis_done(1, 1)
        session.generation_done(1)
        self.assertEqual(session.playback_stopped(1, 200, completed=True), [])
        self.assertFalse(session.idle)  # A follow-up is still owed.
        self.assertEqual(session.history[-1]["content"], RUNNING)
        commands = session.tool_result("w1", '{"temp_c": 18}')
        self.assertIsInstance(commands[0], StartGeneration)

    def test_an_interruption_does_not_cancel_or_lose_a_tool_result(self):
        session = speaking_session()
        session.tool_call(1, "w1", "get_weather", {"city": "Paris"})
        commands = session.user_speech_started()
        self.assertFalse(any(isinstance(command, CallTool) for command in commands))
        session.playback_stopped(1, 100)
        self.assertEqual(tool_messages(session)["w1"], {"status": "running"})
        self.assertTrue(session.idle)  # The user has the floor: no follow-up.
        self.assertEqual(session.tool_result("w1", '{"temp_c": 18}'), [])
        self.assertEqual(tool_messages(session)["w1"], {"temp_c": 18})
        self.assertEqual(session.rejected_events, 0)

    def test_each_call_gets_one_result(self):
        session = speaking_session()
        session.tool_call(1, "w1", "get_weather", {"city": "Paris"})
        session.tool_result("w1", '{"temp_c": 18}')
        session.tool_result("w1", '{"temp_c": 99}')
        session.tool_result("unknown", "{}")
        self.assertEqual(session.runs[1].calls[1].result, '{"temp_c": 18}')
        self.assertEqual(session.rejected_events, 2)

    def test_a_request_already_sent_is_not_rewritten_by_a_late_result(self):
        session = speaking_session()
        session.tool_call(1, "w1", "get_weather", {"city": "Paris"})
        session.user_speech_started()
        session.playback_stopped(1, 100)
        request = session.user_turn_ready("next")[0].messages
        session.tool_result("w1", '{"temp_c": 18}')
        self.assertEqual(request[-2]["content"], RUNNING)

    def test_inferences_per_turn_are_capped(self):
        session = Session("system", speech_as_tool=True, max_inferences=2)
        session.user_turn_ready("hi")
        for run_id in (1, 2):
            session.tool_call(run_id, f"m{run_id}", "multiply", {"a": 1, "b": 2})
            session.tool_result(f"m{run_id}", '{"product": 2}')
            commands = session.generation_done(run_id)
        self.assertEqual(commands, [])
        self.assertTrue(session.idle)


class ToolAblation(unittest.TestCase):
    def measure(self, removed):
        sim = Simulator(SCENARIO, SimConfig(), ABLATIONS[removed]).run()
        return interruption_metrics(sim, SCENARIO), sim

    def test_full_protocol_stops_speech_and_keeps_the_tool_result(self):
        metrics, sim = self.measure("none")
        self.assertEqual(metrics["stop_latency_ms"], 100 + 50)
        self.assertEqual(metrics["stale_audio_ms"], 0)
        self.assertEqual(metrics["unheard_chars_in_history"], 0)
        self.assertEqual(metrics["tool_results_in_next_request"], 1)
        self.assertEqual(
            tool_messages(sim.session)["call_2_0"],
            {"status": "interrupted", "heard_prefix": "Let me check the weather."},
        )

    def test_without_cancel_the_whole_speech_plays(self):
        metrics, _ = self.measure("cancel")
        self.assertEqual(metrics["stop_latency_ms"], 4650 + 4200 - 6500)

    def test_without_run_identity_stale_audio_plays(self):
        metrics, _ = self.measure("run_identity")
        self.assertEqual(metrics["stale_audio_ms"], 200)

    def test_without_playback_progress_the_result_claims_unheard_text(self):
        metrics, _ = self.measure("playback_progress")
        self.assertEqual(metrics["unheard_chars_in_history"], 25 + 29 + 30 - 40)

    def test_without_session_the_tool_result_is_lost_too(self):
        metrics, _ = self.measure("session")
        self.assertEqual(metrics["prior_messages_in_next_request"], 0)
        self.assertEqual(metrics["tool_results_in_next_request"], 0)


class LiveTools(unittest.IsolatedAsyncioTestCase):
    async def test_tool_calls_stream_over_http_and_the_result_reaches_the_model(self):
        api = MockAPI(replies=[
            [{"name": "speak", "arguments": {"text": "One moment."}},
             {"name": "multiply", "arguments": {"a": 17, "b": 23}}],
            [{"name": "speak", "arguments": {"text": "It is 391."}}],
        ])
        service = {"base_url": await api.start(), "model": "mock"}
        try:
            result = await run_dialog({
                "system": "system",
                "speech_as_tool": True,
                "services": {"chat": service, "speech": service, "transcription": service},
                "turns": [{"user": "What is 17 times 23?"}],
            })
        finally:
            await api.stop()
        first, second = result["runs"]
        self.assertEqual(first["heard"], ["One moment."])
        self.assertEqual(json.loads(first["calls"][1]["result"]), {"product": 391})
        self.assertEqual(second["heard"], ["It is 391."])
        self.assertEqual([t["function"]["name"] for t in api.chat_requests[0]["tools"]][0], "speak")
        follow_up = api.chat_requests[1]["messages"]
        self.assertEqual([m["role"] for m in follow_up],
                         ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(json.loads(follow_up[-1]["content"]), {"product": 391})


if __name__ == "__main__":
    unittest.main()
