"""Terminal config: a TOML file for settings, the environment for secrets.

TOML rather than YAML because tomllib is stdlib — the Pi's only third-party
dependency stays `websockets`.
"""

from __future__ import annotations

import os
import socket
import tomllib
from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class AudioInputConfig:
    enabled: bool = False
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    queue_ms: int = 1000

    @classmethod
    def from_dict(cls, raw: dict) -> AudioInputConfig:
        d = cls()
        return cls(
            enabled=bool(raw.get("enabled", d.enabled)),
            device=str(raw.get("device", d.device)),
            sample_rate=int(raw.get("sample_rate", d.sample_rate)),
            channels=int(raw.get("channels", d.channels)),
            frame_ms=int(raw.get("frame_ms", d.frame_ms)),
            queue_ms=int(raw.get("queue_ms", d.queue_ms)),
        )

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * self.channels * 2 * self.frame_ms // 1000



@dataclass
class AudioOutputConfig:
    enabled: bool = False
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    lead_ms: int = 60  # how far ahead of real time writes may run

    @classmethod
    def from_dict(cls, raw: dict) -> AudioOutputConfig:
        d = cls()
        return cls(
            enabled=bool(raw.get("enabled", d.enabled)),
            device=str(raw.get("device", d.device)),
            sample_rate=int(raw.get("sample_rate", d.sample_rate)),
            channels=int(raw.get("channels", d.channels)),
            frame_ms=int(raw.get("frame_ms", d.frame_ms)),
            lead_ms=int(raw.get("lead_ms", d.lead_ms)),
        )


@dataclass
class DisplayConfig:
    url: str = ""                # face page; default http://<backend host>:8766/
    model: str = "haru"          # avatar: haru | hiyori
    zoom: float = 0.0            # 1 = whole body, higher = upper body; 0 = the model's default
    render_scale: float = 0.0    # avatar render resolution vs screen; 0 = page default (0.5 on 4K)
    mouth_delay_ms: int = 140    # lip-sync offset for audio still in the playback buffer

    @classmethod
    def from_dict(cls, raw: dict) -> DisplayConfig:
        d = cls()
        return cls(url=str(raw.get("url", d.url)), model=str(raw.get("model", d.model)),
                   zoom=float(raw.get("zoom", d.zoom)),
                   render_scale=float(raw.get("render_scale", d.render_scale)),
                   mouth_delay_ms=int(raw.get("mouth_delay_ms", d.mouth_delay_ms)))


@dataclass
class TerminalConfig:
    backend_url: str
    token: str
    device_id: str
    heartbeat_interval_s: float = 5.0
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 10.0
    connect_timeout_s: float = 5.0
    audio_input: AudioInputConfig = field(default_factory=AudioInputConfig)
    audio_output: AudioOutputConfig = field(default_factory=AudioOutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @classmethod
    def load(cls, path: Path | None) -> TerminalConfig:
        raw: dict = {}
        if path is not None:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        backend = raw.get("backend", {})
        device = raw.get("device", {})
        reconnect = raw.get("reconnect", {})

        url = os.environ.get("HARU_BACKEND_URL") or backend.get("url")
        if not url:
            raise SystemExit("no backend URL: set [backend] url in the config or HARU_BACKEND_URL")
        token = os.environ.get("HARU_TERMINAL_TOKEN", "")
        if not token:
            raise SystemExit("HARU_TERMINAL_TOKEN is not set")
        return cls(
            backend_url=url,
            token=token,
            device_id=device.get("id") or socket.gethostname(),
            heartbeat_interval_s=float(device.get("heartbeat_interval_s", 5.0)),
            backoff_initial_s=float(reconnect.get("initial_s", 1.0)),
            backoff_max_s=float(reconnect.get("max_s", 10.0)),
            connect_timeout_s=float(backend.get("connect_timeout_s", 5.0)),
            audio_input=AudioInputConfig.from_dict(raw.get("audio", {}).get("input", {})),
            audio_output=AudioOutputConfig.from_dict(raw.get("audio", {}).get("output", {})),
            display=DisplayConfig.from_dict(raw.get("display", {})),
        )
