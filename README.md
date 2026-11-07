# A Minimal Protocol for Turning Any Model API into an Interruptible Voice Agent

DSC 180A Quarter 1 Project, UC San Diego. **Status: checkpoint.**

A voice agent built from separate speech recognition, language, and speech synthesis
models needs rules for interruption, cancellation, and conversation history. This
project asks which of those rules are actually necessary. It defines a small protocol
that treats every model as an HTTP endpoint, then removes each protocol primitive in
turn and measures what breaks.

## Quick start

Requires Python 3.11 or newer (tested on 3.11 and 3.14). No GPU, API key, or network
connection is needed for the tests, once the dependencies are installed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py test
```

`python run.py test` takes a few seconds. It runs the unit tests and then the ablation on
the small test dialog, and writes its results to `outputs/test/`.

## The four primitives

| Primitive | What it guarantees |
| --- | --- |
| `session` | Conversation context survives across replies. |
| `cancel` | In-progress generation, synthesis, and playback can be stopped. |
| `run_identity` | Every event names the reply it belongs to, so late events can be rejected. |
| `playback_progress` | The player reports what was actually played, so history records what was heard. |

A fifth primitive, capability declaration, is planned but not implemented yet.

## Current results

`python run.py ablation` removes one primitive at a time from a three-turn scripted
dialog in which the user interrupts the second reply:

| Removed primitive | Stop latency (ms) | Stale audio (ms) | Unheard chars in history | Prior messages in next request | Next reply delay (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 150 | 0 | 0 | 4 | 650 |
| `cancel` | 3000 | 0 | 0 | 4 | 2700 |
| `run_identity` | 425 | 200 | 0 | 4 | 650 |
| `playback_progress` | 150 | 0 | 57 | 4 | 650 |
| `session` | 150 | 0 | 0 | 0 | 650 |

**These numbers come from a simulation with assumed timings, not from real models or
speech.** They show that the protocol logic behaves as intended and that each removal
causes a different failure. `test/test_ablation.py` checks each value against numbers
worked out by hand.

`python run.py probe` tests the same session code on a real chat model, with speech still
simulated. The user interrupts a reply after two seconds and asks what the agent last
said. These are the counts from 10 trials per condition on one hosted chat model:

| Removed primitive | Quoted a fully played sentence | Quoted a sentence not fully played | Quoted no sentence from the reply |
| --- | ---: | ---: | ---: |
| none | 8 | 0 | 2 |
| `playback_progress` | 0 | 10 | 0 |
| `session` | 0 | 0 | 10 |

In the 2 trials in the first row, no sentence had fully played, so the model said it had
said nothing, even though the user had heard part of a sentence. In 7 trials in the last
row, the model invented a sentence it never said. This is one prompt on one model, so it
shows a direction rather than a failure rate. Whether the effects hold on recorded speech
is the question for the rest of the quarter.

## All build targets

| Command | What it does | Needs |
| --- | --- | --- |
| `python run.py test` | Unit tests, then ablation and baseline on unit test data | Nothing beyond `requirements.txt` |
| `python run.py ablation` | Writes `outputs/ablation.csv` and `outputs/ablation_timeline.svg` | Nothing beyond `requirements.txt` |
| `python run.py data` | Downloads three small files to `data/raw/` | Network |
| `python run.py baseline` | Summarizes them into `outputs/baseline.json` | `data` target |
| `python run.py all` | `data`, `baseline`, and `ablation` | Network |
| `python run.py live` | Runs one conversation from `config/live.json` | That file, plus keys for any real endpoints |
| `python run.py probe` | Writes `outputs/probe.json` from 30 conversations | That file, plus keys for any real endpoints |
| `python run.py clean` | Deletes `outputs/` and `data/` | Nothing |

## Data

Raw data is never committed; `data/` is in `.gitignore`.

- **Unit test data** (`test/testdata/`) is small, synthetic, and committed. The dialog is
  short enough to check by hand. The `baseline/` files are made up and only mimic the
  shape of the real ones.
- **Published benchmark runs.** `python run.py data` downloads three small metadata files
  from the Hugging Face dataset
  [`gavinlaw/sglang-omni-fdb-native-runs`](https://huggingface.co/datasets/gavinlaw/sglang-omni-fdb-native-runs)
  (CC BY-NC 4.0), pinned to the revision in `config/default.json`. They are the dataset
  card, the capability declaration of a cascade deployment, and a replay comparing
  history forms. No account is needed.
- **Planned.** [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
  v1.0 and v1.5 recordings will be used for the real evaluation. They are not needed yet.

## Using real models

Any OpenAI-compatible endpoint can fill each role, whether it is a hosted API or a model
you serve yourself (for example with SGLang or SGLang-Omni). Start from the example,
which uses the bundled mock server for every role and so runs offline:

```bash
cp config/live.example.json config/live.json
```

```bash
python run.py live
```

This writes `outputs/live.json` and `outputs/live.wav`. Audio "plays" by the clock rather
than through a loudspeaker.

To use a real endpoint, replace that role's entry in `config/live.json`. Any other role
can stay on `"mock"`. Keep the key in the environment variable named by `api_key_env`,
never in the file. `params` adds fields to each request:

```json
"chat": {
  "base_url": "https://api.openai.com/v1",
  "model": "YOUR_MODEL",
  "api_key_env": "OPENAI_API_KEY",
  "params": {"temperature": 0}
}
```

**Limitation:** the HTTP path has been tested against one real hosted chat API and the
bundled mock server, but not yet against real speech synthesis or recognition endpoints.
The speech adapter requests 16-bit PCM and assumes the `sample_rate` in the config.

## Code layout, and why

```
run.py                  Build script. Contains no protocol logic.
config/                 Parameters likely to change: timings, dataset revision, endpoints.
src/mip/protocol.py     Primitives, run states, commands, sentence splitting.
src/mip/session.py      The coordinator. The only place conversation state changes.
src/mip/simulate.py     Deterministic simulator and the ablation.
src/mip/figures.py      Draws the ablation as a timeline.
src/mip/services.py     HTTP adapters for chat, speech, and transcription.
src/mip/live.py         Runs the same session over real HTTP in real time.
src/mip/probe.py        Tests the session and playback progress primitives on a real model.
src/mip/mock_server.py  Scripted OpenAI-compatible server for tests and offline runs.
src/mip/baseline.py     Downloads and summarizes the published run metadata.
test/                   Unit tests and committed unit test data.
```

The main decision is that `protocol.py` and `session.py` do no I/O. Each session method
takes one event and returns the commands it causes. As a result, the simulator and the
HTTP runner share one implementation, so the simulated results test the same logic that
talks to real services. It also keeps the project independent of any serving runtime.

SGLang-Omni is deliberately not a dependency. It pins specific PyTorch and CUDA packages
and is needed only as one possible endpoint, which the protocol reaches over HTTP.

## Scope so far

Turn detection is outside this protocol: the caller reports when the user starts
speaking and when a turn is ready. So far, all four primitives have been evaluated on one
scripted dialog in simulation, and two of them on one prompt with one real chat model.
Tool calling is planned for Quarter 2.

## Attribution

- The design adapts ideas from the
  [SGLang-Omni interactive runtime](https://www.sglang.io/blog/omni-interactive-runtime):
  persistent sessions, response identity, and playback-based history. No SGLang-Omni code
  is used. The author also contributes to SGLang-Omni.
- The planned evaluation uses
  [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench).
- The empty-reply replay and capability file come from the published run dataset above.
- The `/v1/chat/completions`, `/v1/audio/speech`, and `/v1/audio/transcriptions` request
  formats follow the [OpenAI API](https://developers.openai.com/api/reference).
