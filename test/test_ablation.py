"""Checks the ablation against numbers worked out by hand from the test dialog.

Settings: 50 ms of speech per character, 200 ms chunks synthesized at 2x real time,
50 ms on every link, 100 ms to detect speech. The second reply has three sentences of
24, 35 and 38 characters (1200, 1750 and 1900 ms of audio).

Turn 2 is ready at 4000 ms. Its first audio plays 650 ms later, at 4650 ms:
50 (link) + 300 (LLM) + 50 (link) + 50 (link) + 100 (one chunk) + 50 + 50 (links).

The user starts speaking at 6500 ms. Speech is detected at 6600 ms and the flush
reaches the player at 6650 ms, so 2000 ms of audio played (40 characters): all of
sentence one and part of sentence two.
"""

import json
import unittest
from pathlib import Path

from mip.figures import STALE, timeline_svg
from mip.simulate import ABLATIONS, SimConfig, Simulator, interruption_metrics

SCENARIO = json.loads((Path(__file__).parent / "testdata" / "interrupt_dialog.json").read_text())


def measure(removed: str) -> tuple[dict, Simulator]:
    sim = Simulator(SCENARIO, SimConfig(), ABLATIONS[removed]).run()
    return interruption_metrics(sim, SCENARIO), sim


class FullProtocol(unittest.TestCase):
    def setUp(self):
        self.metrics, self.sim = measure("none")

    def test_audio_stops_after_detection_and_one_link(self):
        self.assertEqual(self.metrics["stop_latency_ms"], 100 + 50)

    def test_late_chunk_is_rejected_and_never_played(self):
        self.assertEqual(self.metrics["stale_audio_ms"], 0)
        self.assertEqual(self.sim.session.rejected_events, 1)

    def test_history_claims_nothing_the_user_did_not_hear(self):
        self.assertEqual(self.sim.player.played_ms(2), 2000)
        self.assertEqual(self.sim.session.runs[2].heard, ["Paris sits on the Seine."])
        self.assertEqual(self.metrics["unheard_chars_in_history"], 0)

    def test_next_request_carries_the_conversation(self):
        self.assertEqual(self.metrics["prior_messages_in_next_request"], 4)
        request = self.sim.requests[3]
        self.assertEqual(request[-2]["content"], "Paris sits on the Seine.")
        self.assertEqual(request[-1]["content"], "Wait, how big is it?")

    def test_next_reply_starts_on_the_normal_schedule(self):
        self.assertEqual(self.metrics["next_reply_delay_ms"], 650)

    def test_every_run_reaches_a_terminal_state(self):
        reasons = [run.end_reason.value for run in self.sim.session.runs.values()]
        self.assertEqual(reasons, ["completed", "cancelled", "completed"])


class RemovingEachPrimitive(unittest.TestCase):
    def test_without_cancel_the_whole_reply_plays_out(self):
        metrics, _ = measure("cancel")
        # Audio ends at 4650 + 1200 + 1750 + 1900 = 9500 ms.
        self.assertEqual(metrics["stop_latency_ms"], 9500 - 6500)
        # The next run starts when the receipt arrives at 9550 ms, then takes 650 ms.
        self.assertEqual(metrics["next_reply_delay_ms"], 9550 + 650 - 7500)

    def test_without_identity_a_chunk_in_flight_plays_after_the_flush(self):
        metrics, _ = measure("run_identity")
        # That chunk reaches the player at 6725 ms and plays for 200 ms.
        self.assertEqual(metrics["stale_audio_ms"], 200)
        self.assertEqual(metrics["stop_latency_ms"], 6925 - 6500)

    def test_without_progress_history_records_what_was_sent(self):
        metrics, sim = measure("playback_progress")
        self.assertEqual(len(sim.session.runs[2].heard), 3)
        # History holds 24 + 35 + 38 = 97 characters, but only 40 were heard.
        self.assertEqual(metrics["unheard_chars_in_history"], 97 - 40)

    def test_without_session_the_next_request_has_no_context(self):
        metrics, sim = measure("session")
        self.assertEqual(metrics["prior_messages_in_next_request"], 0)
        self.assertEqual([message["role"] for message in sim.requests[3]], ["system", "user"])


class TimelineFigure(unittest.TestCase):
    def test_stale_audio_is_drawn_once_plus_its_legend_swatch(self):
        svg = timeline_svg(SCENARIO, SimConfig())
        self.assertEqual(svg.count(f'fill="{STALE}"'), 2)
        self.assertIn("200 ms of stale audio", svg)


if __name__ == "__main__":
    unittest.main()
