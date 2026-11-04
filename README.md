# Haru

A voice assistant with a face, split into a **thin terminal** and a **GPU backend**.

The terminal is a Raspberry Pi with a speakerphone and a monitor. It streams the mic, plays audio and
shows an animated Live2D avatar, and it holds no models or logic. The backend is a Linux box with an
NVIDIA GPU that does everything else: voice activity detection, speech-to-text, the conversation and
its tools, text-to-speech, long-term memory, and driving the avatar. The LLM is any OpenAI-compatible
chat API; Haru was built and tuned against `gpt-oss-120b`.

End of speech → Haru talking takes about 1.7 s (≈ 500 ms VAD + 330 ms STT + 850 ms to the LLM's first sentence + synthesis).

```
protocol/   haru-protocol: the versioned wire format (stdlib only, shared by both sides)
terminal/   haru-terminal: the Pi services (systemd, native venv)
backend/    the backend stack (docker compose, deployed to /opt/haru)
scripts/    deploy-backend.sh, deploy-terminal.sh
tests/      pytest suite (real sockets, few mocks)
```

## Architecture

| Where | Service | What |
|---|---|---|
| Pi | `haru-terminal` | WebSocket link to the backend, mic capture (`arecord`), speaker playback (`aplay`) |
| Pi | `haru-display` | cage + Chromium kiosk showing the backend's face page full-screen |
| Backend | `gateway` | terminal link, Silero VAD, conversation loop, tools, TTS streaming, avatar events |
| Backend | `stt` | [Speaches](https://speaches.ai): faster-whisper STT and Kokoro TTS, OpenAI-compatible API, GPU |
| Backend | `memory` | long-term memory writer: mines the chat log into a [Relevan](https://relevan.dev) index |
| Backend | `face` | static nginx page: Live2D avatar (PixiJS + pixi-live2d-display) |

Every component sits behind a plain API (OpenAI-compatible for LLM/STT/TTS, a small JSON+binary
WebSocket protocol for terminals), so each can be swapped independently.

## Setup

**Hardware it was built on:** Raspberry Pi 5, Jabra Speak2 40 UC speakerphone, an HDMI monitor;
backend with an RTX 5070 Ti (16 GB). STT + TTS use about 3 GB of VRAM.

**Backend host:** Docker with the NVIDIA container runtime, reachable over SSH with passwordless sudo.

```sh
cp backend/config/gateway.env.example backend/config/gateway.env   # LLM URL, time zone, location...
cp backend/config/persona.md.example  backend/config/persona.md    # Haru's system prompt
scripts/deploy-backend.sh <backend-ssh-host>
# then put the LLM key and any optional API keys in /opt/haru/config/secrets.env on the backend
# (see backend/config/secrets.env.example) and re-run the deploy.

scripts/deploy-terminal.sh <pi-ssh-host> <backend-ssh-host>
# then edit ~/haru-terminal/config/config.toml on the Pi (backend URL, device id, ALSA devices)
```

`HARU_BACKEND_HOST` / `HARU_PI_HOST` can stand in for the arguments. The first backend deploy mints
`HARU_TERMINAL_TOKEN` in `/opt/haru/config/secrets.env` (mode 600) on the server; the terminal deploy
pipes it server → Pi over SSH, so it never touches the deploying machine's disk or shell history.
The Pi config is created from `terminal/config/example.toml` on first deploy and never overwritten.

`gateway.env`, `persona.md` and `secrets.env` are gitignored; only their `*.example` templates are committed.

## Speech-to-text

`haru_gateway/stt.py` runs a Silero VAD (v5 ONNX, CPU, baked into the image by checksum) per
stream. An utterance starts at speech onset with 300 ms of pre-roll and ends after 500 ms of
silence (`HARU_VAD_MIN_SILENCE_MS`, the main snappiness vs. cut-off knob), with a 30 s cap. Blips
under 250 ms are dropped. Utterances are transcribed in order by the `stt` container over the
OpenAI-compatible `/v1/audio/transcriptions` API, reachable only on the compose network.

- Model `deepdml/faster-whisper-large-v3-turbo-ct2`, pre-loaded at gateway start and kept in VRAM
  (`STT_MODEL_TTL=-1`). If it's missing, the gateway asks Speaches to download it.
- Whisper hallucinates on silence ("Thank you.", "You"). The VAD keeps silence out, and any
  segment with `no_speech_prob > 0.6` and `avg_logprob < -1` is dropped.
- There's about 350 ms of fixed per-request overhead (`small.en` is only ~35 ms faster), so it's
  overhead rather than model compute. Unresolved.
- Test without talking: `haru-terminal-cli --device-id wav-test --stream-wav clip.wav`
  (16 kHz mono) streams a file through the live pipeline as a fake terminal.

## Conversation

`haru_gateway/conversation.py`: each transcript goes to the LLM with the system prompt from
`backend/config/persona.md`. There's one conversation per terminal (`device_id`): a sliding window
of 24 messages that resets after 15 idle minutes.

- Replies **stream** and are split into sentences as they arrive, so TTS starts on sentence one.
- **Split utterances / barge-in:** a transcript arriving mid-reply cancels that reply and answers the
  merged turn ("what's the weather" + "in London"). Nothing cancelled is kept in history.
- **Failures:** LLM errors or timeouts (20 s to first token, 60 s total) drop that turn with a
  warning and flip `llm: unavailable` in `backend.status`. The next success flips it back.
- The gateway probes `{base}/models` every 30 s and reports
  `llm: ok | no_api_key | unauthorized | model_missing | unavailable`.
- **Cloudflare gotcha:** an LLM endpoint behind Cloudflare may answer `403 error code: 1010`
  to Python's default `Python-urllib` User-Agent. Every request sets an explicit `User-Agent`
  (`llm.USER_AGENT`). The OpenAI SDK sets one, so you only hit this with raw HTTP clients.
- **Open WebUI** proxies the OpenAI API at `/api`, not `/v1`: set `HARU_LLM_BASE_URL` to the base
  that `{base}/chat/completions` and `{base}/models` hang off.
- **Privacy:** every utterance the mic hears is transcribed, logged and sent to the LLM. Possible
  mitigations: don't log transcript text, add a wake-word gate before the LLM, a hardware mute on
  the Pi, or a LAN-only LLM.

## Speech output

`haru_gateway/tts.py`: each sentence is synthesized by **Kokoro** (`speaches-ai/Kokoro-82M-v1.0-ONNX`,
voice `HARU_TTS_VOICE`, default `af_heart`) via the same Speaches container's `/v1/audio/speech`. Audio
is requested in exactly the format the Pi advertised in its hello (24 kHz mono s16le) and sent as one
`audio.output` stream per reply in 200 ms frames. The Pi buffers it and plays it in real time.

- **Pipelining:** sentence N+1 is synthesized while N plays.
- **Barge-in:** every new transcript sends `audio.output cancel`, which silences the Pi at once. It triggers
  on *transcripts*, not raw VAD, so a cough doesn't cut Haru off.
- **Echo:** the Jabra's hardware echo cancellation keeps Haru's voice out of the mic. As a backstop, a
  transcript that's ≥70% words Haru said in the last 15 s is dropped as echo.
- **Kokoro silently ran on the CPU** about half the time: Speaches builds its ONNX provider list from a
  Python set, so the order follows hash randomization. `PYTHONHASHSEED=0` in `stt.env` puts CUDA first,
  and the gateway warns at startup if TTS is slower than 8× realtime.
- **Cold start on Blackwell:** the Speaches image predates sm_120, so CUDA JIT-compiles kernels on first use
  (~85 s Kokoro, ~25 s Whisper). `data/cuda-cache` persists them; later starts take ~1 s.
- **Test without talking:** `haru-terminal-cli --send debug.ask '{"device_id":"living-room","text":"…"}'`
  (as if that terminal heard it), or `debug.say` to speak text directly. Needs `HARU_DEBUG_COMMANDS=1`.

## The face

The avatar is drawn **on the Pi's own GPU**, full-screen on its monitor, but all logic stays on the
backend. `haru-display` (`terminal/haru_terminal/display.py`) waits for the page and opens
`http://<backend>:8766/?device=…&token=…`, served by the `face` container. The model is **Haru**,
a Live2D Inc. sample model (Free Material License) fetched at build time from Live2D's
CubismWebSamples at a pinned commit.

The page opens `ws://<backend>:8765/display` and follows orders:
- `avatar.state`: `idle → listening` (VAD speech start) `→ thinking` (utterance end) `→ speaking`
  (the Pi reports playback started) `→ idle`, plus `emotion`. The LLM starts every reply with an inline
  marker like `<|ACT {"emotion":"sad"}|>` (the convention from [Project AIRI](https://github.com/moeru-ai/airi),
  requested in `persona.md`). Markers are stripped before TTS, so they're never spoken.
- `avatar.mouth`: a 50 fps mouth-openness envelope computed from each synthesized sentence, played from
  playback start plus `mouth_delay_ms` (140 ms by default, for audio still in the Pi's speaker buffer).
- `avatar.gesture` (`{"name": "wave"}`): one-off body motions requested in the marker,
  `<|ACT {"emotion":"happy","motion":"wave"}|>`. The page holds it until speech starts. Haru has no
  authored wave, so it's hand-built in `face.js`: her right arm's "B" pose (`ParamArmRB` ≈ 5 is a raised
  open palm) swinging 4.0–5.0 at 2.2 Hz for 2.6 s. Arm parts swap at the fade midpoint, because
  crossfading the A/B arm parts shows both. Test: `haru-terminal-cli --send debug.gesture
  '{"device_id":"living-room","text":"wave"}'`. `window.__haru` on the page exposes the model for tuning.
- Emotions are eased parameter targets (eye smile, mouth form, blush, brows, head tilt) on top of idle
  motions and auto-blink.

On a 4K monitor the page renders at half resolution (`renderScale`) and stretches. **Don't** use Chromium's
`--force-device-scale-factor` under cage/Wayland: it inflates the viewport and only the top-left quarter
shows. Chromium DevTools listens on 127.0.0.1:9222 on the Pi (loopback only).

## Tools

Tools (`haru_gateway/tools/`) run inside the streaming reply. Calls run in parallel, results go back to
the model for up to 4 rounds, and every call is schema-validated: bad calls, unknown tools, timeouts and
crashes come back to the model as `{"error": …}` so it can correct itself (local models make occasional
malformed calls). Tool rounds are kept (trimmed) in history, so follow-ups work.

- `web_search` / `read_webpage`: Firecrawl v2 (`FIRECRAWL_API_KEY`).
- `get_weather`: Open-Meteo, no key. `HARU_HOME_LOCATION` answers "what's the weather?" with no place given.
- `set_timer`, `set_reminder`, `list_timers`, `cancel_timer`: persisted in `data/state/timers.json` and
  announced out loud on the terminal when due. Ones that came due during a restart are announced late, not dropped.
- `calendar_events` / `calendar_availability`: Google Calendar, **read-only**, via a service account that sees
  only calendars explicitly shared with it ("See all event details"). Key at
  `/opt/haru/config/google-service-account.json`; calendars as `HARU_CALENDARS=name=id,...`. On Google
  Workspace, the admin setting "External sharing options for primary calendars → Share all information"
  must be on for that sharing level to be selectable.
- `find_data_tools` / `run_data_tool`: **Firecrawl Alexandria**, a catalogue of structured-data tools (local
  businesses with hours and ratings, menus, prices, markets, sports...). Guard rails: a per-call price cap
  (`HARU_ALEXANDRIA_MAX_CREDITS_PER_CALL`), a persisted daily budget (`HARU_ALEXANDRIA_DAILY_CREDITS`), inputs
  checked against the tool contract, and providers needing licence terms are reported, never auto-accepted.
  Tool-choice wording matters: without it in web_search's description and the persona, gpt-oss used web
  search for "coffee open near me" and guessed opening hours.
- `github_*`: **GitHub's official MCP server** (bundled in the gateway image, stdio subprocess, `tools/mcp.py`).
  Read-only three ways: a fine-grained read-only token (`GITHUB_PERSONAL_ACCESS_TOKEN`), the server's
  `GITHUB_READ_ONLY=1` plus lockdown mode, and the gateway refusing any tool not annotated `readOnlyHint`.
  The tools are **topic-gated**: only offered when the conversation mentions PRs, issues, CI or repos (~6k tokens
  of schema otherwise sent on every request), and sticky for 3 turns. Unscoped searches are pinned to
  `org:$HARU_GITHUB_ORG`, because otherwise they search all of GitHub and return strangers' PRs as yours.

## Memory

Long-term memory is stored in a [Relevan](https://relevan.dev) index (`RELEVAN_ORG` / `RELEVAN_MEMORY_INDEX`,
`RELEVAN_API_KEY`). Without a key, Haru runs with conversation memory only. The design and what we
learned building it are in [docs/memory-system.md](docs/memory-system.md).

- Every turn goes to `data/conversations/<device>/<date>.jsonl`, including which memories were surfaced.
- The `memory` container (`haru_gateway/memory_writer.py`) waits until a session is quiet for 2 min (or has
  20 new turns), then has the LLM (reasoning `medium`) turn the new turns into add / update / retire operations.
  Fields include `text`, `subject`, `tags`, `kind`, `importance`, `use_count`, timestamps (ranking signals) and `status`.
- Read side (`memory.py`): a session's memories are preloaded in the background from the moment you start
  speaking. Each turn runs a search on the content words with a 250 ms budget, falling back to a local match
  over the cache, so recall never delays a reply.

## Audio on the Pi

The Pi streams the mic **continuously** while connected: 16 kHz mono s16le in 20 ms frames (32 KB/s). There's
no VAD or wake word on the Pi; the backend decides what is speech. Capture is `arecord` into a 1 s
drop-oldest queue, so a network stall loses stale audio instead of piling it up.

**Jabra quirk:** the Speak2 40 UC returns `read error: Input/output error` on capture unless its **playback
stream is open** (its echo canceller seems to run off the output clock). So the Pi keeps `aplay` open
permanently (`audio_output.py`), playing paced silence between replies. `[audio.output]` must be enabled for
the mic to work. Near-digital silence (≈ -85 dBFS) in a quiet room is the Jabra's noise suppression, not a dead mic.

Backend side, `AudioRouter` fans frames out to sinks: `StatsSink` logs fps / loss / level / clipping / latency,
and `WavRecorderSink` is a debug recorder that's **off by default**. Turn it on with
`HARU_AUDIO_RECORD_DIR=/data/recordings` in `/opt/haru/config/overrides.env`, then
`docker compose up -d --force-recreate gateway`. It records an always-on mic, so only enable it temporarily.

## Protocol (v1)

Connection: `ws://<backend>:8765/terminal` with `Authorization: Bearer $HARU_TERMINAL_TOKEN`
(anything else gets HTTP 401; other paths 404; `GET /healthz` is open and returns a terminal count).

Text frames are JSON envelopes:

```json
{"version": 1, "type": "button.press", "timestamp": 1790000000.12, "data": {"button": "main"}}
```

`timestamp` is float Unix seconds. The first message on a connection **must** be `device.status` carrying
`device_id` (plus `software`, `platform`, `capabilities`); the backend replies with `backend.status`. The
terminal sends `device.heartbeat` (`uptime_s`, `rtt_ms`) every 5 s; WebSocket ping/pong (10 s) detects
dead links. Unknown types are logged and ignored, so either side can add types first.

Types — terminal → backend: `audio.input button.press button.release camera.frame device.status device.heartbeat`.
Backend → terminal: `audio.output display.state avatar.state led.state device.command backend.status`.

Binary frames (audio) carry a 16-byte big-endian header, then the payload:
`version u8 | kind u8 (1=audio in, 2=audio out) | stream_id u16 | seq u32 | captured_at u64 (Unix µs)`.
A JSON `audio.input` message brackets each stream:
`{"event":"start","stream_id":1,"encoding":"pcm_s16le","sample_rate":16000,"channels":1,"frame_ms":20}`
… `{"event":"stop","stream_id":1,"reason":"…"}`. Every (re)connection starts a new stream; a `seq`
jump means frames were dropped.

Close codes: `4002` unsupported version, `4003` bad hello, `4004` replaced by a newer connection with the
same `device_id`, `1001` clean terminal shutdown. Display clients connect to
`ws://<backend>:8765/display?device=<id>&token=<terminal token>`.

## Debugging

```sh
curl http://<backend>:8765/healthz
ssh <backend> docker logs -f haru-gateway
ssh <pi> journalctl -fu haru-terminal
# send events as a fake terminal (on the Pi, token from its secrets file):
ssh <pi> 'cd haru-terminal && set -a && . config/secrets.env && set +a && \
  HARU_BACKEND_URL=ws://<backend>:8765/terminal .venv/bin/haru-terminal-cli \
  --send button.press "{\"button\":\"main\"}"'
```

## Tests

```sh
uv venv .venv && uv pip install --python .venv -e protocol -e terminal -e backend/services/gateway pytest
.venv/bin/python -m pytest tests
```

## Security

Designed for a trusted LAN. The gateway requires the bearer token and refuses to start without one.
It binds `0.0.0.0:8765` and the face page `0.0.0.0:8766`; override with `HARU_GATEWAY_BIND` /
`HARU_FACE_BIND` in `/opt/haru/.env`. Put a firewall in front of both if the network isn't trusted.

## Third-party components

- PixiJS 6.5.10 (MIT) and pixi-live2d-display 0.4.0 (MIT), vendored in `backend/services/face/vendor`.
- Live2D Cubism Core, vendored under Live2D's Proprietary Software License as Redistributable Code;
  see the header of `live2dcubismcore.min.js`.
- The Haru model is a Live2D Inc. sample model under the
  [Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html),
  downloaded at image build time and not included in this repository.
- The `<|ACT …|>` emotion-marker convention comes from [Project AIRI](https://github.com/moeru-ai/airi).

## License

Copyright 2026 David Tippett. Licensed under the [Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE).
Third-party components above keep their own licenses.
