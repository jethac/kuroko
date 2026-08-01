"""The body — turning the puppet track into actual motion.

`puppet.py` compiles cues into keyframes and `arbiter.py` decides where to
look; this module is what finally moves the robot, at a fixed control rate,
through a single writer.

Two rules shape the design:

1. **One writer.** The daemon's own behaviours (wobbling, face tracking) and
   an app posting poses will fight over the same joints and produce twitching.
   So kuroko drives the head itself and leaves daemon wobbling off; face
   tracking stays on only as a *sensor* feeding the arbiter.

2. **Motion must never block audio.** Every SDK call here is fire-and-forget
   from a dedicated task, and failures are swallowed. A stiff neck is a much
   better failure than a stuttering voice.

Channels come from the puppet track: `wobble` (speech-driven sway), `nod`
(beat gesture), `antenna` (perk/settle), `gaze_bias` (where to face). They are
mixed continuously rather than applied as discrete jumps, so the robot reads
as alive rather than as a series of poses.
"""

import asyncio
import logging
import math
import time

import numpy as np
from reachy_mini.utils import create_head_pose

from .arbiter import GazeArbiter
from .puppet import PuppetTrack, wobble_pose

log = logging.getLogger("kuroko.body")

CONTROL_HZ = 30.0


class Body:
    """Drives head + antennas from the puppet track at a fixed rate."""

    def __init__(self, mini, puppet: PuppetTrack, arbiter: GazeArbiter | None = None):
        self.mini = mini
        self.puppet = puppet
        self.arbiter = arbiter or GazeArbiter()
        self.wobble = 0.0          # smoothed speech energy -> sway amplitude
        self.antenna = 0.5
        self.gaze_bias = 0.0
        self._nod_until = 0.0
        self._ok = True            # False once the SDK proves unusable

    # -- channel mixing ------------------------------------------------------

    def _apply_keyframes(self, now: float) -> None:
        for kf in self.puppet.due(now):
            if kf.channel == "wobble":
                # attack fast, release slow: speech onsets should feel crisp,
                # endings should settle rather than snap back
                k = 0.5 if kf.value > self.wobble else 0.08
                self.wobble += k * (kf.value - self.wobble)
            elif kf.channel == "antenna":
                self.antenna += 0.35 * (kf.value - self.antenna)
            elif kf.channel == "gaze_bias":
                self.gaze_bias += 0.25 * (kf.value - self.gaze_bias)
            elif kf.channel == "nod":
                self._nod_until = now + 0.45

    def _pose(self, now: float) -> tuple[float, float, float]:
        """Mix channels into head Euler angles, in DEGREES.

        Amplitudes are deliberately small — this is conversational body
        language, not dancing. Roughly: sway a few degrees, nod ~9 degrees,
        gaze up to ~25 degrees of yaw.
        """
        r, p, y = wobble_pose(self.wobble, now)      # small radians-scale sway
        roll, pitch, yaw = math.degrees(r), math.degrees(p), math.degrees(y)
        if now < self._nod_until:
            # a nod is a decaying pitch impulse on top of the sway
            phase = (self._nod_until - now) / 0.45
            pitch += 9.0 * math.sin(math.pi * (1 - phase)) * math.exp(-1.5 * (1 - phase))
        yaw += 25.0 * float(np.clip(self.arbiter.target_yaw(self.gaze_bias, now), -1.0, 1.0))
        return roll, pitch, yaw

    # -- output --------------------------------------------------------------

    def _write(self, roll: float, pitch: float, yaw: float) -> None:
        """Post one pose. head is a 4x4 homogeneous matrix; antennas radians."""
        if not self._ok:
            return
        try:
            head = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
            # antennas: perk forward as `antenna` rises, mirrored L/R
            a = float(np.clip(self.antenna, 0.0, 1.0))
            ant = [(a - 0.5) * 1.2, -(a - 0.5) * 1.2]
            self.mini.set_target(head=head, antennas=ant)
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def _fail(self, e: Exception) -> None:
        log.warning("motion disabled after error: %s", e)
        self._ok = False

    async def run(self, stop: asyncio.Event) -> None:
        """Fixed-rate control loop; never raises into the audio path."""
        period = 1.0 / CONTROL_HZ
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        log.info("body control loop at %.0f Hz", CONTROL_HZ)
        while not stop.is_set():
            next_t += period
            await asyncio.sleep(max(0.0, next_t - loop.time()))
            now = time.monotonic()
            self._apply_keyframes(now)
            # decay toward rest so the robot settles when nothing is happening
            self.wobble *= 0.97
            self.antenna += 0.02 * (0.5 - self.antenna)
            roll, pitch, yaw = self._pose(now)
            await loop.run_in_executor(None, self._write, roll, pitch, yaw)

    async def watch_face(self, stop: asyncio.Event) -> None:
        """Feed the daemon's face tracker into the arbiter as a sensor."""
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            await asyncio.sleep(0.2)
            try:
                face = await loop.run_in_executor(
                    None, lambda: self.mini.get_tracked_face(wait=False))
            except Exception:  # noqa: BLE001
                continue
            found = bool(face and getattr(face, "detected", False))
            bearing = None
            if found:
                x = getattr(face, "x", None)
                if x is not None:
                    # normalized image x (0..1) -> yaw offset, mild gain
                    bearing = float((x - 0.5) * -0.8)
            self.arbiter.on_face(found, bearing)
