"""End-to-end test over real HTTP against the local mock server."""

import asyncio
import io
import tempfile
import unittest
import wave
from pathlib import Path

from mip import baseline
from mip.live import run_dialog
from mip.mock_server import MockAPI
from mip.probe import CONDITIONS, classify, run_probe

TESTDATA = Path(__file__).parent / "testdata"


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0" * 320)
    return buffer.getvalue()


class LiveDialog(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = MockAPI(replies=["One two three. Four five six.", "Short answer."])
        base_url = await self.api.start()
        service = {"base_url": base_url, "model": "mock"}
        self.services = {"chat": service, "speech": service, "transcription": service}

    async def asyncTearDown(self):
        await self.api.stop()

    async def test_interruption_closes_the_stream_and_the_next_turn_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "turn.wav"
            audio_path.write_bytes(wav_bytes())
            config = {
                "system": "system",
                "services": self.services,
                "turns": [
                    {"audio_path": str(audio_path), "interrupt_after_ms": 100},
                    {"user": "next question"},
                ],
            }
            result = await run_dialog(config, out_dir=directory)
            with wave.open(str(Path(directory) / "live.wav"), "rb") as file:
                self.assertGreater(file.getnframes(), 0)

        first, second = result["runs"]
        self.assertEqual(first["end_reason"], "cancelled")
        self.assertEqual(second["end_reason"], "completed")
        self.assertEqual(second["heard"], ["Short answer."])

        # The server saw the closed connection: cancel reached the service over HTTP.
        for _ in range(100):
            if self.api.speech_disconnects:
                break
            await asyncio.sleep(0.01)
        self.assertGreaterEqual(self.api.speech_disconnects, 1)

        # The transcribed turn reached the model, with no reply text the user never heard.
        second_request = self.api.chat_requests[1]["messages"]
        self.assertEqual(second_request[1], {"role": "user", "content": "hello"})
        self.assertEqual(
            [message["role"] for message in second_request], ["system", "user", "user"]
        )

    async def test_mock_base_url_and_primitives_come_from_the_config(self):
        mock = {"base_url": "mock", "model": "mock"}
        config = {
            "system": "system",
            "services": {"chat": mock, "speech": mock, "transcription": mock},
            "primitives": {"session": False},
            "turns": [{"user": "first"}, {"user": "second"}],
        }
        result = await run_dialog(config)
        self.assertEqual(
            [run["end_reason"] for run in result["runs"]], ["completed", "completed"]
        )
        self.assertEqual(len(result["runs"][0]["planned"]), 2)
        self.assertGreater(result["runs"][0]["first_audio_ms"], result["runs"][0]["first_text_ms"])

    async def test_request_params_reach_the_service(self):
        chat = {**self.services["chat"], "params": {"temperature": 0}}
        config = {
            "system": "system",
            "services": {**self.services, "chat": chat},
            "turns": [{"user": "hello"}],
        }
        await run_dialog(config)
        self.assertEqual(self.api.chat_requests[0]["temperature"], 0)

    async def test_failed_service_ends_in_a_failed_run(self):
        config = {
            "system": "system",
            "services": {
                **self.services,
                "speech": {"base_url": self.services["speech"]["base_url"] + "/missing", "model": "mock"},
            },
            "turns": [{"user": "hello"}],
        }
        result = await run_dialog(config)
        self.assertEqual(result["runs"][0]["end_reason"], "failed")
        self.assertIn("404", result["runs"][0]["error"])


class Probe(unittest.IsolatedAsyncioTestCase):
    PLANNED = ["Paris is old.", "The Seine is long."]

    def test_quoting_a_played_sentence(self):
        label = classify('I said: "paris is old"', self.PLANNED, ["Paris is old."])
        self.assertEqual(label, "quoted_played")

    def test_quoting_an_unplayed_sentence_takes_priority(self):
        reply = "Paris is old. The Seine is long."
        self.assertEqual(classify(reply, self.PLANNED, ["Paris is old."]), "quoted_not_played")

    def test_quoting_nothing(self):
        self.assertEqual(classify("I said nothing.", self.PLANNED, []), "no_quote")

    async def test_counts_cover_every_trial(self):
        mock = {"base_url": "mock", "model": "mock"}
        config = {
            "system": "system",
            "services": {"chat": mock, "speech": mock, "transcription": mock},
        }
        summary = await run_probe(config, repeats=1, interrupt_after_ms=50)
        self.assertEqual(set(summary["counts"]), set(CONDITIONS))
        self.assertEqual(sum(sum(c.values()) for c in summary["counts"].values()), 3)
        self.assertEqual(summary["first_text_ms"]["requests"], 6)


class Baseline(unittest.TestCase):
    def test_summarize_reads_both_files(self):
        summary = baseline.summarize(TESTDATA / "baseline")
        self.assertEqual(summary["cascade_capability_fields"], 3)
        self.assertEqual(
            summary["empty_replies_by_history_form"],
            {
                "keep_entry": {"empty": 2, "samples": 4},
                "drop_entry": {"empty": 0, "samples": 4},
            },
        )


if __name__ == "__main__":
    unittest.main()
