"""haru-display: puts the backend's avatar page full-screen on the terminal's monitor.

The Pi only hosts a kiosk browser (cage + Chromium). The page, the avatar and
every decision about what she does come from the backend, so there's nothing to
update here when the face changes. Waits for the page to be reachable before
launching, so a backend that's down at boot doesn't leave an error page up.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from haru_terminal.config import TerminalConfig

log = logging.getLogger("haru.terminal.display")


def main() -> None:
    parser = argparse.ArgumentParser(description="Haru avatar display (kiosk)")
    parser.add_argument("--config", type=Path, default=os.environ.get("HARU_TERMINAL_CONFIG"))
    parser.add_argument("--print-url", action="store_true", help="print the page URL (without token) and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = TerminalConfig.load(args.config)
    d = config.display
    backend_host = urlsplit(config.backend_url).hostname
    backend_port = urlsplit(config.backend_url).port or 8765
    page = d.url or f"http://{backend_host}:8766/"
    query = {"device": config.device_id, "backend": f"{backend_host}:{backend_port}", "model": d.model,
             "mouthDelayMs": d.mouth_delay_ms}
    if d.zoom:
        query["zoom"] = d.zoom
    if d.render_scale:
        query["renderScale"] = d.render_scale
    if args.print_url:
        print(f"{page}?{urlencode(query)}")
        return

    delay = 2.0
    while True:
        try:
            with urllib.request.urlopen(page, timeout=5) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, OSError) as exc:
            log.info("face page %s not reachable yet (%s); retrying in %.0fs", page, exc, delay)
        time.sleep(delay)
        delay = min(15.0, delay * 1.5)

    url = f"{page}?{urlencode({**query, 'token': config.token})}"
    profile = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "chromium-profile"  # tmpfs: no crash bubbles
    cmd = [
        "cage", "--",
        "chromium", "--kiosk", "--ozone-platform=wayland", f"--user-data-dir={profile}",
        "--noerrdialogs", "--disable-infobars", "--no-first-run", "--disable-session-crashed-bubble",
        "--disable-features=Translate,MediaRouter", "--password-store=basic",
        "--enable-gpu-rasterization",
        # Loopback only: lets `ssh <pi>` screenshot/inspect the page. Not reachable off the Pi.
        "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=9222",
        url,
    ]
    log.info("launching kiosk: %s", f"{page}?{urlencode(query)}")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
