"""HTTP adapters for OpenAI-compatible endpoints: the model side of the protocol.

A hosted API and a self-served model look the same here: a base URL and a model
name. Closing the HTTP stream is the one cancel that every such endpoint supports.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key_env: str | None = None
    params: dict | None = None  # Extra request fields, such as temperature.

    @property
    def headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return {"Authorization": f"Bearer {key}"} if key else {}


class OpenAIChat:
    def __init__(self, http: aiohttp.ClientSession, endpoint: Endpoint):
        self.http = http
        self.endpoint = endpoint

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        url = f"{self.endpoint.base_url}/chat/completions"
        body = {
            "model": self.endpoint.model,
            "messages": messages,
            "stream": True,
            **(self.endpoint.params or {}),
        }
        async with self.http.post(url, json=body, headers=self.endpoint.headers) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                text = (choices[0].get("delta") or {}).get("content") if choices else None
                if text:
                    yield text


class OpenAISpeech:
    """Streams 16-bit mono PCM. Chunks carry 100 ms of audio."""

    def __init__(
        self,
        http: aiohttp.ClientSession,
        endpoint: Endpoint,
        voice: str = "alloy",
        sample_rate: int = 24000,
    ):
        self.http = http
        self.endpoint = endpoint
        self.voice = voice
        self.sample_rate = sample_rate

    def duration_ms(self, pcm: bytes) -> float:
        return len(pcm) / (2 * self.sample_rate) * 1000

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        url = f"{self.endpoint.base_url}/audio/speech"
        body = {
            "model": self.endpoint.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            **(self.endpoint.params or {}),
        }
        chunk_bytes = self.sample_rate // 5  # 100 ms at two bytes per sample.
        async with self.http.post(url, json=body, headers=self.endpoint.headers) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(chunk_bytes):
                yield chunk


class OpenAITranscription:
    def __init__(self, http: aiohttp.ClientSession, endpoint: Endpoint):
        self.http = http
        self.endpoint = endpoint

    async def transcribe(self, wav: bytes) -> str:
        form = aiohttp.FormData()
        form.add_field("model", self.endpoint.model)
        form.add_field("file", wav, filename="turn.wav", content_type="audio/wav")
        url = f"{self.endpoint.base_url}/audio/transcriptions"
        async with self.http.post(url, data=form, headers=self.endpoint.headers) as response:
            response.raise_for_status()
            return (await response.json())["text"]
