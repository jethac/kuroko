"""Gaze arbiter — DoA nominates, face confirms.

Raw loudest-sound-wins head turning is trivially hijacked (a phone playing speech in
the corner drags the robot's gaze to a wall). The arbiter treats DoA as a *nomination*
and the daemon's face tracking as *confirmation*:

    DoA event  -> GLANCE  (cheap: small yaw bias toward the bearing, antennas up)
    face found within CONFIRM_WINDOW at that bearing -> COMMIT (full head turn + track)
    no face    -> decay back to the puppet track's gaze_bias

While the robot is speaking, nominations are attenuated to glances only — it should
never whip away mid-sentence. While listening, reflexes own the head.

State machine, not ML. Runs at ~20 Hz next to the puppet scheduler.
"""

import time
from dataclasses import dataclass
from enum import Enum


class Gaze(Enum):
    TRACK = "track"      # following puppet track bias / committed face
    GLANCE = "glance"    # DoA nomination, awaiting confirmation
    COMMIT = "commit"    # face-confirmed attention target


CONFIRM_WINDOW_S = 0.7
GLANCE_YAW_FRACTION = 0.35   # a glance only goes 35% of the way to the bearing
SELF_SPEECH_ATTENUATION = 0.3


@dataclass
class GazeArbiter:
    state: Gaze = Gaze.TRACK
    nominated_bearing: float | None = None
    nominated_at: float = 0.0
    committed_bearing: float | None = None

    def on_doa(self, bearing: float, robot_speaking: bool, now: float | None = None) -> None:
        now = now or time.monotonic()
        # P4: also gate on "bearing ~= robot's own speaker cone" using the
        # server's knowledge of its own emission state (self-speech oracle).
        if robot_speaking:
            bearing = self._attenuate(bearing, SELF_SPEECH_ATTENUATION)
        self.nominated_bearing = bearing
        self.nominated_at = now
        if self.state == Gaze.TRACK:
            self.state = Gaze.GLANCE

    def on_face(self, found: bool, bearing: float | None, now: float | None = None) -> None:
        now = now or time.monotonic()
        if self.state == Gaze.GLANCE and found and bearing is not None:
            if now - self.nominated_at <= CONFIRM_WINDOW_S:
                self.state = Gaze.COMMIT
                self.committed_bearing = bearing
        elif self.state == Gaze.COMMIT and not found:
            self.state = Gaze.TRACK
            self.committed_bearing = None

    def target_yaw(self, track_bias: float, now: float | None = None) -> float:
        now = now or time.monotonic()
        if self.state == Gaze.COMMIT and self.committed_bearing is not None:
            return self.committed_bearing
        if self.state == Gaze.GLANCE and self.nominated_bearing is not None:
            if now - self.nominated_at > CONFIRM_WINDOW_S:
                self.state = Gaze.TRACK   # nomination expired unconfirmed
                return track_bias
            return GLANCE_YAW_FRACTION * self.nominated_bearing + (1 - GLANCE_YAW_FRACTION) * track_bias
        return track_bias

    @staticmethod
    def _attenuate(bearing: float, k: float) -> float:
        return bearing * k
