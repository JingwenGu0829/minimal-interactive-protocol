"""Toy tools for speech-as-a-tool mode, and the schema of the `speak` tool itself.

The tools are deterministic and offline, so tests and the simulator can use them.
Each has an artificial delay: `get_weather` is slow enough that its result arrives
while the agent is still speaking, or after the user has interrupted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .protocol import SPEAK

SPEAK_INSTRUCTIONS = (
    "The user only hears what you pass to the speak tool. Call speak once per reply, "
    "with the complete text to say. Plain text you write is never heard."
)

_WEATHER = {
    "paris": {"temp_c": 18, "sky": "cloudy"},
    "san diego": {"temp_c": 22, "sky": "sunny"},
    "tokyo": {"temp_c": 9, "sky": "rain"},
}


def get_weather(city: str) -> dict:
    report = _WEATHER.get(city.strip().casefold())
    if report is None:
        return {"error": f"no weather data for {city}"}
    return {"city": city, **report}


def multiply(a: float, b: float) -> dict:
    return {"product": a * b}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema properties.
    run: Callable[..., dict]
    delay_ms: int


TOOLS = {
    tool.name: tool
    for tool in (
        Tool(
            "get_weather",
            "Look up the current weather in a city. Slow: takes a few seconds.",
            {"city": {"type": "string"}},
            get_weather,
            delay_ms=2500,
        ),
        Tool(
            "multiply",
            "Multiply two numbers.",
            {"a": {"type": "number"}, "b": {"type": "number"}},
            multiply,
            delay_ms=0,
        ),
    )
}


def _schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters),
            },
        },
    }


def schemas() -> list[dict]:
    """The `tools` field of a chat request: speech first, then the toy tools."""
    speak = _schema(SPEAK, "Say text aloud to the user.", {"text": {"type": "string"}})
    return [speak] + [_schema(t.name, t.description, t.parameters) for t in TOOLS.values()]


def execute(name: str, arguments: dict) -> str:
    """Run a tool and return its result as the content of a tool message."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return json.dumps(tool.run(**arguments))
    except TypeError as error:  # Wrong or missing arguments from the model.
        return json.dumps({"error": str(error)})
