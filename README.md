# gt7-race-engineer

A fully local, voice-in/voice-out AI race engineer for Gran Turismo 7, running on macOS (Apple silicon).

It ingests GT7 UDP telemetry into a live car and race state with a rolling frame history. Background analyst agents continuously assess that data, using calculated metrics plus a local decision model to judge whether something is worth the engineer's attention. A single race engineer agent receives those assessments and decides what to say and when. It also converses with the driver by voice and gives a briefing at every new lap.

Voices are switchable in config (`[tts] engine`): Qwen3-TTS 0.6B, Kokoro-82M, or the robotic macOS `say`. Models are reached through OpenAI-compatible endpoints (LM Studio, Ollama or similar), so they can be swapped. "Local" here means no cloud: everything runs on your Mac or on your own LAN (e.g. an Ollama server on another machine). The optional hosted Jev decision model is the one exception and sends only the analysts' calculated numbers to OpenRouter. Open source.

## v1 success criteria

- Run on a Mac, point it at the PlayStation's IP, and get callouts within a lap of starting a race.
- Accurate fuel-to-finish and "pit by lap N" calls in a multi-lap race.
- Tells the driver when a lap is off their best pace and where in the lap time was lost.
- Driver can ask "how's my fuel?" by voice and get a correct spoken answer from live data.
- Setup takes about 10 minutes from this README.

## Quickstart

Requires macOS on Apple silicon, [uv](https://docs.astral.sh/uv/), and an OpenAI-compatible server (LM Studio, Ollama, ...) with a chat model loaded.

```sh
uv sync --extra audio                  # installs deps incl. mic capture + MLX Whisper
cp config.example.toml config.toml     # set ps_ip and your model name
uv run gt7-engineer run                # on the PS5: start a race; the engineer connects automatically
```

GT7 telemetry is UDP: the Mac and PlayStation must be on the same network, and the Mac must allow incoming UDP on port 33740.

`run` shows a live dashboard: telemetry health and car state, mic and engineer status, and the conversation (what the analysts flagged, what the engineer said, what it heard from you). Use `--plain` for plain text lines (also used automatically when piped or with `--text`).

The telemetry badge at the top of the Car panel tells you whether data is arriving:

| Badge | Meaning |
|---|---|
| `● LIVE 60 Hz` | packets are arriving and decoding |
| `● WAITING` | nothing received yet: is a race running, is the IP right, same network? |
| `● BAD DATA` | packets arrive but fail decryption: wrong key or packet format |
| `● PAUSED` | data arriving but the car isn't on track (paused, menu, loading) |
| `● STALE` | data stopped for more than 2 s |

Lap length is measured from the car's position changes (the median of clean complete laps, so pit laps don't skew it) and the dashboard shows how far through the current lap you are. It needs one complete lap, and it is approximate.

Pit stops are detected from the telemetry itself: a refuel (stationary, fuel climbing) and a tyre change (all four temperatures snap to the same lower value) are found independently, because a stop can be fuel only, tyres only, or both. Either one is announced when the car drives away, a tyre change clears an out-of-date wear report, and the dashboard shows how many laps old the tyres are.

The race state resets automatically when a new race starts (the lap counter drops back and stays there), or press `r` on the dashboard to reset it yourself.

If you wear headphones, the mic stays open while the engineer talks and you can interrupt it by speaking (`[audio] barge_in = true`, the default). On speakers set it to `false` so the engineer can't hear itself.

To keep a race for analysis: `--record-raw race.raw.jsonl.gz` saves every field of every packet (60 Hz, about 7 MB for 25 minutes) and `--record race.jsonl` saves a small readable summary (lap, fuel, position, speed, 10 Hz). Play a raw recording back through the whole app with `--replay race.raw.jsonl.gz` (`--fast` skips real-time pacing).

Try it without a PlayStation, microphone or model:

```sh
uv run gt7-engineer run --simulate --no-audio --no-llm --laps 5 --lap-seconds 10
uv run gt7-engineer run --simulate --text      # type to the engineer instead of speaking
uv run gt7-engineer watch --ps-ip 192.168.1.50 # raw telemetry readout only
uv run pytest                                  # tests
```

## Status

Everything above runs end to end against the built-in simulator and is covered by tests. Not yet verified against real hardware or real local models:

- The GT7 packet layout and heartbeat (`telemetry/packet.py`) follow community documentation and are round-trip tested only.
- The Jev (OpenRouter) adapter follows OpenRouter's tutorial and is tested against a mocked response only; it has not been called for real. Local Jev-Style is not integrated.
- Whisper and TTS quality and latency on the target Mac.

Known issues:

- Whisper transcription runs in-process (in a worker thread). A long transcription could stall the event loop that handles telemetry and the engineer. Untested; if it shows up as telemetry lag or dropped frames, move STT to a separate child process behind an OpenAI-compatible `/v1/audio/transcriptions` interface so the app doesn't care whether it's local or external.

Code layout: `telemetry/` (decode, UDP ingest, simulator), `state.py` (state + frame history), `agents/` (analysts), `engineer.py` (arbiter and conversation), `audio/` (Whisper STT, `say` TTS), `app.py` (wiring), `cli.py`.

## Architecture

```
PS5 --UDP--> Telemetry ingest (decrypt/decode)
                     |
                     v
       State store: car state + race state + frame ring buffer
                     |   (queryable back in time)
       +-------------+--------------+----------------+
       v             v              v                v
   Fuel agent   Pace/lap agent   (other agents)   Lap-trace agent
                                                  (LLM reviews a lap's
                                                   telemetry at lap end)
       +-------------+------+-------+----------------+
                            v  assessments + priority
                   Race Engineer agent  <-- new-lap ping
                   (decides whether/when to speak)
                      ^               |
         Whisper (STT)|               v  local TTS
                   Driver speech    Spoken output
```

- **State store.** Current car and race state, plus a ring buffer of frames so any agent can look back (e.g. fuel burn over the last 3 laps).
- **Fuel burn tracking.** Fuel burnt on each completed lap is tracked, with a trend (heavier, lighter or steady) from a least-squares slope over the last 5 laps (needs 3). The fuel analyst uses it alongside the average, and it's in the engineer's state, so "is my burn getting worse?" gets a real answer. Pit-stop laps (fuel level rising) are excluded from the stats.
- **Analyst agents.** Run continuously and independently, each publishing timestamped assessments. Each has two stages: calculate parameters (fuel burn per lap, pace delta), then a local decision model picks from a fixed set of answers (e.g. `ignore / mention / urgent`) given those numbers and instructions. Only "mention" and "urgent" reach the engineer.
- **Lap-trace agent.** At the end of each lap, a chat LLM reviews that lap's telemetry and publishes a coaching assessment.
- **Race engineer.** The only voice. Receives all assessments and rate-limits, prioritises and decides when to speak. Answers the driver and is pinged at each new lap for an update.
- **Driver-reported data.** Where telemetry has gaps, the engineer asks the driver. Tyre wear isn't in GT7 telemetry, so the engineer can ask for it (e.g. "how are the fronts looking?"), and the spoken answer is parsed and stored in the race state with a timestamp, so analysts can use it like any other input.
- **Voice.** `[tts] engine = "qwen3" | "kokoro" | "say"`. The neural engines run through mlx-audio, stream audio as it is generated and are warmed up at startup (first call is much slower). The first run downloads the model.
- **Models.** Every LLM role (engineer, lap reviewer, decision model) is configured with an OpenAI-compatible base URL, headers and model name. Whisper (STT) and TTS run locally and are swappable behind interfaces.

### Using a ChatGPT/Codex subscription (Pi)

Set `kind = "pi"` on `[engineer]` (and `[lap_reviewer]` inherits it) to route chat through the [Pi](https://pi.dev) coding-agent harness in RPC mode. Pi handles the subscription login, so no API key is needed. The app keeps one long-lived `pi --mode rpc` child process with tools, extensions, skills and MCP switched off, and starts a fresh session per call. Setup: run `pi`, `/login`, choose ChatGPT Plus/Pro (Codex); then `pi --list-models` for the model id. See `config.example.toml`. This is cloud: prompts including race state go to OpenAI via Pi.

### Decision model

Optional. Without one, analysts use built-in threshold rules. Three ways to plug one in via `[decision]` in `config.toml`:

- **Hosted Jev via OpenRouter** (`kind = "jev"`): calls the [Decisions API](https://openrouter.ai/docs/guides/community/jev) (`typesafe/jev-1.13`) with a `choice` question and uses the returned choice. **This is not local**: the analysts' calculated numbers (fuel, lap times) are sent to OpenRouter. Set `OPENROUTER_API_KEY` and `api_key_env` in the config.
- **Local small model** (default `kind = "chat"`): any OpenAI-compatible endpoint asked to answer with one word.
- **Local Jev-Style** ([jev-style](https://pypi.org/project/jev-style/)): not integrated; its server API is unverified here.

## Telemetry limits

GT7's own "car on track" flag (bit 0) is not set during real races (observed flags `0x0418` while driving), so the app treats the game as running unless the paused or loading bits are set.

GT7 telemetry covers the player's own car (fuel, speeds, temperatures, lap times, lap counts). The race position and car-count fields are documented as pre-race values, but real GT7 telemetry shows them updating live, so they are used as current position. It does not provide gaps to other cars, rivals or flags, and it does not report tyre wear (the driver reports it by voice instead; the same voice report is the fallback for position in modes that don't send it).

## Non-goals (for now)

- Gaps to other cars, rival tracking, flag alerts.
- Tuning or setup advice.
- Non-Mac platforms.

## TODO / future

- Canned fast paths for common questions (fuel, lap time) to cut voice latency.
- Tyre strategy and automatic tyre wear estimation, building on driver-reported wear.
- Basic race position awareness.
