# gt7-race-engineer

A fully local, voice-in/voice-out AI race engineer for Gran Turismo 7, running on macOS (Apple silicon).

It ingests GT7 UDP telemetry into a live car and race state with a rolling frame history. Background analyst agents continuously assess that data, using calculated metrics plus a local decision model to judge whether something is worth the engineer's attention. A single race engineer agent receives those assessments and decides what to say and when. It also converses with the driver by voice and gives a briefing at every new lap.

Models are reached through OpenAI-compatible endpoints (LM Studio, Ollama or similar), so they can be swapped. Nothing leaves the machine. Open source.

## v1 success criteria

- Run on a Mac, point it at the PlayStation's IP, and get callouts within a lap of starting a race.
- Accurate fuel-to-finish and "pit by lap N" calls in a multi-lap race.
- Tells the driver when a lap is off their best pace and where in the lap time was lost.
- Driver can ask "how's my fuel?" by voice and get a correct spoken answer from live data.
- Setup takes about 10 minutes from this README.

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
- **Analyst agents.** Run continuously and independently, each publishing timestamped assessments. Each has two stages: calculate parameters (fuel burn per lap, pace delta), then a local decision model picks from a fixed set of answers (e.g. `ignore / mention / urgent`) given those numbers and instructions. Only "mention" and "urgent" reach the engineer.
- **Lap-trace agent.** At the end of each lap, a chat LLM reviews that lap's telemetry and publishes a coaching assessment.
- **Race engineer.** The only voice. Receives all assessments and rate-limits, prioritises and decides when to speak. Answers the driver and is pinged at each new lap for an update.
- **Driver-reported data.** Where telemetry has gaps, the engineer asks the driver. Tyre wear isn't in GT7 telemetry, so the engineer can ask for it (e.g. "how are the fronts looking?"), and the spoken answer is parsed and stored in the race state with a timestamp, so analysts can use it like any other input.
- **Models.** Every LLM role (engineer, lap reviewer, decision model) is configured with an OpenAI-compatible base URL, headers and model name. Whisper (STT) and TTS run locally and are swappable behind interfaces.

### Decision model

Candidate: [Jev-Style](https://pypi.org/project/jev-style/), small local decision models (0.8B / 2B, MLX on Apple silicon) that return a probability over predefined answers. It runs its own server, so it may need a dedicated adapter rather than a plain chat endpoint. Verify before relying on it, and check the license of the specific model.

## Telemetry limits

GT7 telemetry covers the player's own car (fuel, speeds, temperatures, lap times) plus race position and car count. It does not provide gaps to other cars, rivals or flags, and it does not report tyre wear (the driver reports it by voice instead).

## Non-goals (for now)

- Gaps to other cars, rival tracking, flag alerts.
- Tuning or setup advice.
- Non-Mac platforms.

## TODO / future

- Canned fast paths for common questions (fuel, lap time) to cut voice latency.
- Tyre strategy and automatic tyre wear estimation, building on driver-reported wear.
- Basic race position awareness.
