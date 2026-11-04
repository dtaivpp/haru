"""haru-terminal service entry point (what systemd runs on the Pi).

Stage 1: just hold the link to the backend and log what it says. Audio,
GPIO and display peripherals register themselves on the link in later stages.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from haru_protocol import Message, MessageType

from haru_terminal import __version__
from haru_terminal.audio_input import MicStreamer
from haru_terminal.audio_output import Speaker
from haru_terminal.config import TerminalConfig
from haru_terminal.transport import BackendLink, LinkState

log = logging.getLogger("haru.terminal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Haru physical I/O terminal")
    parser.add_argument("--config", type=Path, default=os.environ.get("HARU_TERMINAL_CONFIG"),
                        help="path to config.toml (or HARU_TERMINAL_CONFIG)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        # journald adds its own timestamps
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("websockets").setLevel(logging.INFO)
    config = TerminalConfig.load(args.config)
    asyncio.run(run(config))


async def run(config: TerminalConfig) -> None:
    capabilities = []
    if config.audio_input.enabled:
        capabilities.append("audio.input")
    if config.audio_output.enabled:
        capabilities.append("audio.output")
    link = BackendLink(config, capabilities=capabilities)
    mic = MicStreamer(link, config.audio_input) if config.audio_input.enabled else None
    # Started even while it only plays silence: some USB speakerphones (the
    # Jabra) deliver no mic audio unless their output stream is open.
    speaker = Speaker(link, config.audio_output) if config.audio_output.enabled else None

    def on_backend_status(msg: Message) -> None:
        log.info("backend status: %s", msg.data)

    def on_state(state: LinkState) -> None:
        # Hook point for a local "offline" indicator once a display/LED exists.
        if state == LinkState.OFFLINE:
            log.info("offline — backend unreachable")

    link.on(MessageType.BACKEND_STATUS, on_backend_status)
    link.on_state(on_state)

    log.info("haru-terminal %s starting as %r -> %s", __version__, config.device_id, config.backend_url)
    loop = asyncio.get_running_loop()
    peripherals = [asyncio.create_task(p.run_forever()) for p in (speaker, mic) if p]

    async def shutdown() -> None:
        if mic:
            await mic.stop("terminal shutting down")
        await link.close()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown()))
    try:
        await link.run_forever()
    finally:
        for task in peripherals:
            task.cancel()
    log.info("shut down")

