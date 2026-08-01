"""Puppet track — server-authored embodiment.

v0 (P2): compiles a 10-20 Hz keyframe track from the audio energy envelope observed
just before playout, plus text-piece events. Channels are deliberately few:

    wobble    0..1   head sway amplitude while speaking
    nod       pulse  beat gesture on energy onsets
    antenna   0..1   perk while listening / settle at end of turn
    gaze_bias -1..1  where the track *wants* to look (arbiter may override)

P4 upgrades the input from "audio about to play" (lead ~= jitter buffer depth) to the
fork's frame tap (lead = full network+buffer margin, plus turn-state logits), which is
what makes gestures anticipatory rather than merely punctual. The channel format does
not change — only the lead time and the richness of the cues.

The track is also a recordable artifact (list of (t, channel, value)) — an embodiment
codec you can replay, diff, and unit-test without a robot.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Keyframe:
    t: float          # monotonic wall time the value should land
    channel: str
    value: float


@dataclass
class PuppetTrack:
    lead_s: float = 0.15          # how far ahead of playout we schedule
    keyframes: list[Keyframe] = field(default_factory=list)
    _env: float = 0.0             # smoothed energy envelope
    _speaking: bool = False
    _last_nod: float = 0.0

    def on_audio(self, pcm: np.ndarray, sr: int, now: float) -> None:
        """Feed audio that is ABOUT to play (v0 lookahead = the playout buffer)."""
        rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
        self._env = 0.8 * self._env + 0.2 * rms
        speaking = self._env > 0.015

        t_land = now + self.lead_s
        self.keyframes.append(Keyframe(t_land, "wobble", min(1.0, self._env * 12.0)))

        if speaking and not self._speaking:
            self.keyframes.append(Keyframe(t_land, "antenna", 0.2))
        if not speaking and self._speaking:
            # end of utterance: antennas settle, gaze returns to the human
            self.keyframes.append(Keyframe(t_land, "antenna", 0.9))
            self.keyframes.append(Keyframe(t_land, "gaze_bias", 0.0))
        self._speaking = speaking

        # beat nod on a fresh energy onset, rate-limited to feel intentional
        if rms > 2.2 * max(self._env, 1e-4) and now - self._last_nod > 0.9:
            self._last_nod = now
            self.keyframes.append(Keyframe(t_land, "nod", 1.0))

    def on_text(self, piece: str) -> None:
        """Text pieces arrive ahead of their audio; cheap discourse cues live here.
        P4 also strips <nod>/<tilt> gesture tags emitted by the persona itself."""
        if any(w in piece.lower() for w in ("you", "your")):
            self.keyframes.append(Keyframe(time.monotonic() + self.lead_s,
                                           "gaze_bias", 0.0))
        if "?" in piece:
            self.keyframes.append(Keyframe(time.monotonic() + self.lead_s,
                                           "antenna", 1.0))

    def due(self, now: float) -> list[Keyframe]:
        """Pop keyframes whose land time has arrived."""
        ready = [k for k in self.keyframes if k.t <= now]
        self.keyframes = [k for k in self.keyframes if k.t > now]
        return ready


def wobble_pose(amplitude: float, t: float) -> tuple[float, float, float]:
    """Map wobble amplitude to a small (roll, pitch, yaw) offset, phase-locked to t."""
    return (
        0.06 * amplitude * math.sin(2 * math.pi * 1.8 * t),
        0.04 * amplitude * math.sin(2 * math.pi * 0.9 * t + 1.3),
        0.03 * amplitude * math.sin(2 * math.pi * 0.6 * t + 2.1),
    )
