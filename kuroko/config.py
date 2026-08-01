"""Kuroko configuration."""

import json
import os
from dataclasses import dataclass, asdict

DEFAULT_TEXT_PROMPT = (
    "You are Reachy Mini, a small friendly expressive robot who lives on a desk. "
    "You speak casually and concisely, with warmth and curiosity, and occasionally "
    "make gentle jokes about being a small robot."
)


@dataclass
class KurokoConfig:
    robot_host: str = "reachy-mini.local"
    server_host: str = "127.0.0.1"      # PersonaPlex is a sidecar: localhost
    server_port: int = 8998
    voice_prompt: str = "NATM2.pt"
    text_prompt: str = DEFAULT_TEXT_PROMPT
    heartbeat_port: int = 8043          # pi reflex supervisor listens for this

    @classmethod
    def load(cls, path: str | None = None) -> "KurokoConfig":
        path = path or os.environ.get("KUROKO_CONFIG", "kuroko.json")
        cfg = cls()
        if os.path.exists(path):
            with open(path) as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        for field_name in ("robot_host", "server_host", "voice_prompt"):
            env = os.environ.get(f"KUROKO_{field_name.upper()}")
            if env:
                setattr(cfg, field_name, env)
        return cfg

    def save(self, path: str = "kuroko.json") -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
