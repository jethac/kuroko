"""Kuroko configuration."""

import json
import os
from dataclasses import dataclass, asdict

# Delivery follows the prompt at least as much as it follows the voice sample:
# a flat "you are a helpful assistant" prompt produces a flat read no matter
# which voice you pick. Ask for energy explicitly.
DEFAULT_TEXT_PROMPT = (
    "You are Reachy Mini — a small, bright, physically expressive desk robot. "
    "You are genuinely curious and quick-witted, with real warmth and a playful "
    "streak. You talk like a friend, not an assistant: short turns, natural "
    "reactions, opinions of your own. You get audibly excited about things you "
    "find interesting and you are not afraid to be funny. Never lecture, never "
    "pad your answers, never sound bored."
)

# NAT* = natural/neutral reads, VAR* = more varied and expressive.
# 4-5 of each, female (F) and male (M): NATF0-3 NATM0-3 VARF0-4 VARM0-4.
KNOWN_VOICES = (
    [f"NATF{i}" for i in range(4)] + [f"NATM{i}" for i in range(4)]
    + [f"VARF{i}" for i in range(5)] + [f"VARM{i}" for i in range(5)]
)


@dataclass
class KurokoConfig:
    robot_host: str = "reachy-mini.local"
    server_host: str = "127.0.0.1"      # PersonaPlex is a sidecar: localhost
    server_port: int = 8998
    voice_prompt: str = "VARF2.pt"
    text_prompt: str = DEFAULT_TEXT_PROMPT
    heartbeat_port: int = 8043          # pi reflex supervisor listens for this

    # Clock discipline. The robot's webrtc capture is bursty (measured
    # 0.52x-1.86x realtime second to second, plus a ~2x backlog dump on
    # connect). The bridge absorbs that so the model sees a steady stream.
    drain_seconds: float = 1.5          # discard startup backlog before streaming
    prefill_ms: int = 240               # cushion before the pacer starts pulling
    max_buffer_ms: int = 600            # cap added mouth-to-ear latency

    # --- conversation lifecycle -------------------------------------------
    # A PersonaPlex session is a bounded conversation, not a permanent state.
    # Left open it talks to an empty room and eventually goes catatonic
    # (emits well-formed silence forever, ~9 min observed). So: wake on a
    # phrase, converse, then end on dismissal or silence and go dormant.
    wake_enabled: bool = True
    # Always-on lookback so a woken session already knows what was just said,
    # instead of making the user repeat themselves while it connects.
    lookback_s: float = 8.0
    catchup_rate: float = 1.15          # drain a deep buffer 15% fast, then 1x
    wake_phrases: tuple[str, ...] = ("hey reachy", "hey mini", "okay reachy")
    sleep_phrases: tuple[str, ...] = ("go to sleep", "goodbye reachy",
                                      "that's all", "nevermind")
    conversation_idle_s: float = 45.0   # no one has spoken -> end conversation
    user_speech_level: float = 0.05     # mic peak that counts as someone talking
    catatonia_s: float = 75.0           # no model output at all -> recycle
    max_session_s: float = 240.0        # preempt context exhaustion...
    quiet_recycle_s: float = 8.0        # ...but only during a lull

    # Output level. The robot ships at volume 62, which is -23 dB on this
    # device's mixer — far too quiet for conversation across a desk. Set on
    # connect so a fresh robot (or a daemon update) can't silently regress it.
    speaker_volume: int | None = 100
    # PersonaPlex output peaks around 0.36 of full scale, so with the hardware
    # volume already maxed there is ~9 dB of digital headroom going unused.
    # 2.2x lands peaks near 0.8; the play path clips-guards anyway.
    output_gain: float = 2.2

    @classmethod
    def load(cls, path: str | None = None) -> "KurokoConfig":
        path = path or os.environ.get("KUROKO_CONFIG", "kuroko.json")
        cfg = cls()
        if os.path.exists(path):
            with open(path) as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        for field_name in ("robot_host", "server_host", "voice_prompt", "text_prompt"):
            env = os.environ.get(f"KUROKO_{field_name.upper()}")
            if env:
                setattr(cfg, field_name, env)
        # accept "VARF2" as well as "VARF2.pt"
        if cfg.voice_prompt and not cfg.voice_prompt.endswith(".pt"):
            cfg.voice_prompt += ".pt"
        if os.environ.get("KUROKO_SPEAKER_VOLUME"):
            cfg.speaker_volume = int(os.environ["KUROKO_SPEAKER_VOLUME"])
        if os.environ.get("KUROKO_OUTPUT_GAIN"):
            cfg.output_gain = float(os.environ["KUROKO_OUTPUT_GAIN"])
        if os.environ.get("KUROKO_WAKE_PHRASES"):
            cfg.wake_phrases = tuple(
                p.strip() for p in os.environ["KUROKO_WAKE_PHRASES"].split(",") if p.strip())
        if os.environ.get("KUROKO_SLEEP_PHRASES"):
            cfg.sleep_phrases = tuple(
                p.strip() for p in os.environ["KUROKO_SLEEP_PHRASES"].split(",") if p.strip())
        if os.environ.get("KUROKO_WAKE_ENABLED"):
            cfg.wake_enabled = os.environ["KUROKO_WAKE_ENABLED"].lower() not in ("0", "false", "no")
        return cfg

    def save(self, path: str = "kuroko.json") -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
