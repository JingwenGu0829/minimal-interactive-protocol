"""A local OpenAI-compatible server with scripted output, used by the tests.

It exercises the real HTTP path (streaming, multipart upload, client disconnects)
without needing a model, a GPU, or an API key.
"""

from __future__ import annotations

import asyncio
import itertools
import json

from aiohttp import web

DEFAULT_REPLY = "This is a mock reply. It has two short sentences."


class MockAPI:
    def __init__(
        self, replies: list[str] | None = None, transcript: str = "hello",
        ms_per_char: int = 20, sample_rate: int = 24000,
    ):
        self.replies = iter(replies) if replies else itertools.repeat(DEFAULT_REPLY)
        self.transcript = transcript
        self.ms_per_char = ms_per_char
        self.sample_rate = sample_rate
        self.chat_requests: list[dict] = []
        self.speech_disconnects = 0
        self._runner: web.AppRunner | None = None

    async def start(self) -> str:
        """Start on a free local port and return the base URL."""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/v1/audio/speech", self._speech)
        app.router.add_post("/v1/audio/transcriptions", self._transcriptions)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        return f"http://127.0.0.1:{port}/v1"

    async def stop(self) -> None:
        await self._runner.cleanup()

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.chat_requests.append(body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for word in next(self.replies).split(" "):
            event = {"choices": [{"delta": {"content": word + " "}}]}
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def _speech(self, request: web.Request) -> web.StreamResponse:
        text = (await request.json())["input"]
        total_bytes = len(text) * self.ms_per_char * self.sample_rate * 2 // 1000
        step = self.sample_rate // 50  # 10 ms of audio, sent in 5 ms: twice real time.
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        try:
            for _ in range(0, total_bytes, step):
                await response.write(b"\0" * step)
                await asyncio.sleep(0.005)
            await response.write_eof()
        except ConnectionResetError:
            self.speech_disconnects += 1  # The client cancelled by closing the stream.
        return response

    async def _transcriptions(self, request: web.Request) -> web.Response:
        form = await request.post()
        if "file" not in form or "model" not in form:
            raise web.HTTPBadRequest(text="file and model are required")
        return web.json_response({"text": self.transcript})
