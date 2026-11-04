"""Debug client: connect as a (fake) terminal, send events, print replies.

    haru-terminal-cli --url ws://<backend-host>:8765/terminal \
        --send button.press '{"button": "main"}' --listen 3

Reads HARU_TERMINAL_TOKEN from the environment like the real service. Uses
the exact same BackendLink as the service, so it exercises the real path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import wave

from haru_protocol import AUDIO_START, AUDIO_STOP, PCM_S16LE, FrameKind, MessageType, encode_frame

from haru_terminal.config import TerminalConfig
from haru_terminal.transport import BackendLink


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="backend URL (default: HARU_BACKEND_URL)")
    parser.add_argument("--device-id", default="haru-cli")
    parser.add_argument("--send", nargs=2, action="append", default=[], metavar=("TYPE", "JSON"),
                        help="send an event once connected (repeatable)")
    parser.add_argument("--stream-wav", metavar="PATH",
                        help="stream a 16 kHz mono 16-bit WAV as mic audio, in real time")
    parser.add_argument("--listen", type=float, default=2.0,
                        help="seconds to keep printing backend messages before exiting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.url:
        os.environ["HARU_BACKEND_URL"] = args.url
    config = TerminalConfig.load(None)
    config.device_id = args.device_id
    config.backoff_max_s = 2.0
    sys.exit(asyncio.run(run(config, args.send, args.listen, args.stream_wav)))


async def stream_wav(link: BackendLink, path: str) -> None:
    with wave.open(path) as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (16000, 1, 2):
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit")
        pcm = wf.readframes(wf.getnframes())
    frame_bytes = 640  # 20 ms
    await link.send(MessageType.AUDIO_INPUT, {"event": AUDIO_START, "stream_id": 1, "encoding": PCM_S16LE,
                                              "sample_rate": 16000, "channels": 1, "frame_ms": 20})
    print(f"-> streaming {path} ({len(pcm) / 32000:.1f}s)", flush=True)
    t0 = time.monotonic()
    for seq, off in enumerate(range(0, len(pcm) - frame_bytes + 1, frame_bytes)):
        await asyncio.sleep(max(0.0, t0 + seq * 0.02 - time.monotonic()))
        await link.send_frame(encode_frame(FrameKind.AUDIO_INPUT, 1, seq, time.time_ns() // 1000,
                                           pcm[off:off + frame_bytes]))
    await link.send(MessageType.AUDIO_INPUT, {"event": AUDIO_STOP, "stream_id": 1, "reason": "end of file"})


async def run(config: TerminalConfig, sends: list[list[str]], listen_s: float,
              wav: str | None = None) -> int:
    link = BackendLink(config)
    for type_ in MessageType:
        link.on(type_, lambda m: print(f"<- {m.type} {json.dumps(m.data)}", flush=True))
    runner = asyncio.create_task(link.run_forever())
    try:
        try:
            await link.wait_online(timeout=config.connect_timeout_s + 1)
        except TimeoutError:
            print(f"could not connect to {config.backend_url}", file=sys.stderr)
            return 1
        print(f"connected to {config.backend_url} as {config.device_id!r}", flush=True)
        for type_, payload in sends:
            ok = await link.send(type_, json.loads(payload))
            print(f"-> {type_} {payload}{'' if ok else '  (NOT SENT: offline)'}", flush=True)
        if wav:
            await stream_wav(link, wav)
        await asyncio.sleep(listen_s)
        return 0
    finally:
        await link.close()  # completes the close handshake
        runner.cancel()     # stop any pending reconnect sleep
