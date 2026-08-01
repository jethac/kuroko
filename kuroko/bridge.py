"""P1 — the voice loop: robot (remote) <-> PersonaPlex (localhost).

Runs on the inference box. Full duplex:

    robot mic  --SDK remote media-->  resample 24k  --sphn opus-->  ws \x01
    ws \x01  --sphn opus-->  24k pcm  --resample-->  robot speaker
    ws \x02  -->  transcript log + puppet-track text events

The websocket protocol is moshi/PersonaPlex `/api/chat`:
    client -> server : b"\x01" + opus
    server -> client : b"\x01" + opus   |   b"\x02" + utf-8 text piece
"""

import asyncio
import json
import logging
import time
import urllib.request
from urllib.parse import quote

import numpy as np
import soxr
import sphn
import websockets

from reachy_mini import ReachyMini

from .config import KurokoConfig
from .listener import PhraseListener
from .puppet import PuppetTrack
from .sdkfix import ensure_audio_send_ready, harden_sdk

MODEL_SR = 24000
FRAME_MS = 80          # one Mimi frame
log = logging.getLogger("kuroko.bridge")


class VoiceBridge:
    """Owns the audio path and the PersonaPlex session for one robot."""

    def __init__(self, cfg: KurokoConfig, puppet: PuppetTrack | None = None):
        self.cfg = cfg
        self.puppet = puppet
        self.stop = asyncio.Event()
        self.mini: ReachyMini | None = None
        self.media = None
        self.opus_w = sphn.OpusStreamWriter(MODEL_SR)
        self.opus_r = sphn.OpusStreamReader(MODEL_SR)
        self.rs_down = None  # mic sr -> 24k
        self.rs_up = None    # 24k -> speaker sr
        self.frames_played = 0  # crude playhead in model frames
        self.stats = {"mic_chunks": 0, "tx_bytes": 0, "rx_bytes": 0, "spk_chunks": 0,
                      "underruns": 0, "dropped": 0, "buf_ms": 0}
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self.handshake = asyncio.Event()
        self.streaming = asyncio.Event()
        # Per-session abort (reconnect) as distinct from self.stop (shut down).
        self.session_stop = asyncio.Event()
        self.last_activity = 0.0
        self.last_user_speech = 0.0
        self.end_reason: str | None = None
        self.in_sr = MODEL_SR
        self.listener = PhraseListener((), ())   # replaced in run()
        # Lookback at the robot's native rate: everything said in the last few
        # seconds, so a freshly woken session starts already knowing what the
        # user said while it was connecting.
        self._ring = np.zeros(0, dtype=np.float32)

    # -- robot ---------------------------------------------------------------

    def _set_speaker_volume(self) -> None:
        if self.cfg.speaker_volume is None:
            return
        try:
            req = urllib.request.Request(
                f"http://{self.cfg.robot_host}:8000/api/volume/set",
                data=json.dumps({"volume": self.cfg.speaker_volume}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                log.info("speaker volume set to %d (HTTP %d)",
                         self.cfg.speaker_volume, r.status)
        except Exception as e:
            log.warning("could not set speaker volume: %s", e)

    def connect_robot(self) -> None:
        harden_sdk()
        self._set_speaker_volume()
        # explicit network mode: auto mode's failed localhost probe leaves
        # debris whose gc collection froze the process (see sdkfix)
        self.mini = ReachyMini(host=self.cfg.robot_host, connection_mode="network")
        self.media = self.mini.media
        in_sr = self.in_sr = self.media.get_input_audio_samplerate()
        out_sr = self.media.get_output_audio_samplerate()
        if in_sr != MODEL_SR:
            self.rs_down = soxr.ResampleStream(in_sr, MODEL_SR, 1, dtype="float32")
        if out_sr != MODEL_SR:
            self.rs_up = soxr.ResampleStream(MODEL_SR, out_sr, 1, dtype="float32")
        self.media.start_recording()
        self.media.start_playing()
        if not ensure_audio_send_ready(self.media):
            raise RuntimeError("robot speaker path unavailable (webrtc send chain)")
        log.info(f"robot {self.cfg.robot_host}: mic {in_sr} Hz, speaker {out_sr} Hz")

    @staticmethod
    def _mono(pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim == 2:
            axis = 0 if pcm.shape[0] <= 8 and pcm.shape[0] < pcm.shape[1] else 1
            pcm = pcm.mean(axis=axis)
        return np.ascontiguousarray(pcm)

    # -- loops ---------------------------------------------------------------

    async def capture_loop(self) -> None:
        """Always-on capture. Runs dormant and active alike.

        The ears cannot be session-scoped: while dormant there is no
        PersonaPlex session, but we still need audio to spot the wake phrase.
        So this runs for the life of the process, always feeding the phrase
        listener, and additionally filling the model's jitter buffer once a
        session has opened its gate.

        It deliberately does NOT send to the model — `pace_loop` does that on
        a strict wall clock, because the robot's webrtc delivery is bursty
        (0.52x-1.86x realtime second to second, plus a ~2x backlog dump on
        connect) and frame arrival IS the model's clock.
        """
        loop = asyncio.get_running_loop()
        # Ceiling must accommodate a pre-roll seed, or the cap would shred the
        # lookback the instant it was handed over. Catch-up pacing is what
        # brings the buffer back down to the steady-state target.
        ceiling_s = max(self.cfg.max_buffer_ms / 1000.0, self.cfg.lookback_s + 1.0)
        max_samples = int(ceiling_s * MODEL_SR)
        while not self.stop.is_set():
            pcm = await loop.run_in_executor(None, self.media.get_audio_sample)
            if pcm is None or np.size(pcm) == 0:
                await asyncio.sleep(0.004)
                continue
            mono = self._mono(pcm)          # native 16 kHz
            if not mono.size:
                continue

            # phrase spotting wants the robot's native 16 kHz — no resampling
            self.listener.feed(mono)

            # Always-on lookback, kept at native rate.
            self._ring = np.concatenate((self._ring, mono))
            ring_max = int(self.cfg.lookback_s * self.in_sr)
            if self._ring.size > ring_max:
                self._ring = self._ring[-ring_max:]
            if float(np.abs(mono).max()) > self.cfg.user_speech_level:
                self.last_user_speech = time.monotonic()

            if self.streaming.is_set():
                m = self.rs_down.resample_chunk(mono) if self.rs_down is not None else mono
                if m.size:
                    self._mic_buf = np.concatenate((self._mic_buf, m))
                    # Bound latency: if the robot burst ahead, discard the
                    # oldest audio rather than grow mouth-to-ear delay forever.
                    if self._mic_buf.size > max_samples:
                        self.stats["dropped"] += self._mic_buf.size - max_samples
                        self._mic_buf = self._mic_buf[-max_samples:]
            await asyncio.sleep(0.001)

    async def gate_loop(self) -> None:
        """Open the model's input gate once the session has handshaked."""
        # The server discards everything until it sends its \x00 handshake
        # (system-prompt loading) — including the opus stream header.
        await self.handshake.wait()
        # Drop the jitter buffer's startup backlog rather than shipping it as
        # a burst the model would experience as time compression.
        await asyncio.sleep(self.cfg.drain_seconds)
        self._mic_buf = self._preroll()
        self.last_user_speech = time.monotonic()
        self.streaming.set()

    def _preroll(self) -> np.ndarray:
        """Seed a new session with what the user already said.

        Waking on a phrase and then opening a PersonaPlex session costs
        several seconds (websocket + voice/text prompt load). Without a
        lookback the user says "hey reachy, what's the weather" and the model
        only ever hears "...weather" — or nothing, and they have to repeat
        themselves. The ring buffer has been recording the whole time, so hand
        the session the recent past instead of starting deaf.

        Leading silence is trimmed so we replay speech, not room tone, and the
        pacer drains the resulting backlog with a gentle speed-up (see
        `catchup_rate`) rather than carrying the extra latency forever.
        """
        if not self.cfg.lookback_s or self._ring.size == 0:
            return np.zeros(0, dtype=np.float32)
        take = min(self._ring.size, int(self.cfg.lookback_s * self.in_sr))
        seg = self._ring[-take:]

        # trim to the first speech, keeping a short run-up for natural onset
        loud = np.flatnonzero(np.abs(seg) > self.cfg.user_speech_level)
        if loud.size == 0:
            return np.zeros(0, dtype=np.float32)
        start = max(0, int(loud[0]) - int(0.2 * self.in_sr))
        seg = seg[start:]

        rs = soxr.ResampleStream(self.in_sr, MODEL_SR, 1, dtype="float32")
        out = np.asarray(rs.resample_chunk(seg, last=True), dtype=np.float32)
        frame = int(MODEL_SR * FRAME_MS / 1000)
        out = out[: (out.size // frame) * frame]        # whole frames only
        if out.size:
            log.info("pre-roll: seeding session with %.1fs of prior speech",
                     out.size / MODEL_SR)
        return out

    async def pace_loop(self, ws) -> None:
        """Send one 80 ms frame per 80 ms of wall clock (with bounded catch-up).

        This is the fix for the bursty capture path: the model receives a
        perfectly regular stream regardless of how the robot delivers it.
        Underruns are filled with silence (the model must never stall waiting
        for us); surplus is absorbed by the buffer and, at the limit, dropped
        in mic_loop.
        """
        loop = asyncio.get_running_loop()
        await self.streaming.wait()
        frame = int(MODEL_SR * FRAME_MS / 1000)   # 1920 samples @ 24 kHz
        period = FRAME_MS / 1000.0

        # Build a cushion first. Without it the buffer sits near empty and every
        # dip in the robot's delivery becomes an underrun (measured ~20%).
        prefill = int(self.cfg.prefill_ms * MODEL_SR / 1000)
        while self._mic_buf.size < prefill and self.alive():
            await asyncio.sleep(0.01)
        log.info("jitter buffer primed (%d ms) — pacing at %.1f fps",
                 self.cfg.prefill_ms, 1000.0 / FRAME_MS)

        next_deadline = loop.time()

        while self.alive():
            # Adaptive pacing: normally exactly realtime, but when the buffer
            # is deep — after a pre-roll seed, or a delivery burst — run a
            # little fast to drain it, so added latency is temporary rather
            # than permanent. Kept mild; this is the very distortion the
            # pacer exists to prevent, just bounded and deliberate.
            deep = self._mic_buf.size > int(self.cfg.prefill_ms * MODEL_SR / 1000) * 2
            period = (FRAME_MS / 1000.0) / (self.cfg.catchup_rate if deep else 1.0)
            next_deadline += period
            delay = next_deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.5:
                # We fell badly behind (e.g. host hiccup); resync rather than
                # sprint to catch up, which would re-create the burst problem.
                log.warning("pacer %.0f ms behind — resyncing clock", -delay * 1000)
                next_deadline = loop.time()

            if self._mic_buf.size >= frame:
                chunk = self._mic_buf[:frame]
                self._mic_buf = self._mic_buf[frame:]
            else:
                chunk = np.zeros(frame, dtype=np.float32)
                if self._mic_buf.size:
                    chunk[:self._mic_buf.size] = self._mic_buf
                    self._mic_buf = np.zeros(0, dtype=np.float32)
                self.stats["underruns"] += 1

            self.stats["buf_ms"] = int(1000 * self._mic_buf.size / MODEL_SR)
            await loop.run_in_executor(None, self.opus_w.append_pcm, chunk)
            self.stats["mic_chunks"] += 1
            data = await loop.run_in_executor(None, self.opus_w.read_bytes)
            if data:
                self.stats["tx_bytes"] += len(data)
                await ws.send(b"\x01" + data)

    async def recv_loop(self, ws) -> None:
        loop = asyncio.get_running_loop()
        pieces: list[str] = []
        async for msg in ws:
            if not self.alive():
                break
            if not isinstance(msg, (bytes, bytearray)) or not msg:
                continue
            kind = msg[0]
            if kind == 0:
                self.handshake.set()
            elif kind == 1:
                self.stats["rx_bytes"] += len(msg) - 1
                await loop.run_in_executor(None, self.opus_r.append_bytes, bytes(msg[1:]))
            elif kind == 2:
                piece = msg[1:].decode("utf-8", errors="replace")
                pieces.append(piece)
                if self.puppet is not None:
                    self.puppet.on_text(piece)
                if any(c in piece for c in ".!?\n") or sum(map(len, pieces)) > 120:
                    log.info("[says] %s", "".join(pieces).strip())
                    pieces.clear()
                self.last_activity = time.monotonic()
        self.session_stop.set()

    async def play_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self.alive():
            # sphn read_pcm blocks when the stream is empty; executor keeps
            # the event loop alive (learned the hard way)
            pcm = await loop.run_in_executor(None, self.opus_r.read_pcm)
            if pcm is None or pcm.size == 0:
                await asyncio.sleep(0.004)
                continue
            out = np.asarray(pcm, dtype=np.float32)
            if self.cfg.output_gain != 1.0:
                out = np.clip(out * self.cfg.output_gain, -1.0, 1.0)
            # Track the LOUDEST chunk in the window, not the latest — most
            # chunks are inter-word silence, which made this read 0.0 and
            # look like a dead speaker path.
            if out.size:
                lvl = float(np.abs(out).max())
                self.stats["out_peak"] = round(max(self.stats.get("out_peak", 0.0), lvl), 3)
                if lvl > 0.02:          # real speech, not inter-word silence
                    self.last_activity = time.monotonic()
            if self.puppet is not None:
                # v0 embodiment: energy envelope of audio just before playout.
                # P4 replaces this with the server tap's pre-playout sidecar.
                self.puppet.on_audio(out, MODEL_SR, time.monotonic())
            if self.rs_up is not None:
                out = self.rs_up.resample_chunk(out)
            if out.size:
                await loop.run_in_executor(None, self.media.push_audio_sample, out)
                self.frames_played += 1
                self.stats["spk_chunks"] += 1

    def alive(self) -> bool:
        """True while the current PersonaPlex session should keep running.

        Distinguishes 'end this session' (session_stop -> reconnect or go
        dormant) from 'shut the bridge down' (stop).
        """
        return not (self.stop.is_set() or self.session_stop.is_set())

    async def watchdog_loop(self) -> None:
        """Recycle a session that has stopped talking.

        Moshi-architecture models have a bounded streaming context. Past it,
        PersonaPlex keeps emitting perfectly well-formed opus — of pure
        silence — and never takes a turn again. Nothing errors, byte counters
        keep climbing, and the robot looks alive while being brain-dead.
        Observed here after ~9 minutes. Detect by absence of model output and
        start a fresh session.
        """
        await self.streaming.wait()
        self.last_activity = time.monotonic()
        session_start = time.monotonic()
        while self.alive():
            await asyncio.sleep(2.0)
            idle = time.monotonic() - self.last_activity
            age = time.monotonic() - session_start
            if idle > self.cfg.catatonia_s:
                log.warning("no model output for %.0fs — session is stale, recycling", idle)
                self.end_reason = "stale"
                self.session_stop.set()
                return
            since_user = time.monotonic() - self.last_user_speech
            if since_user > self.cfg.conversation_idle_s and idle > self.cfg.quiet_recycle_s:
                log.info("no one has spoken for %.0fs — ending conversation", since_user)
                self.end_reason = "idle"
                self.session_stop.set()
                return
            if age > self.cfg.max_session_s and idle > self.cfg.quiet_recycle_s:
                log.info("session %.0fs old and quiet — recycling before context runs out", age)
                self.end_reason = "aged"
                self.session_stop.set()
                return

    def _task_died(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("loop task died: %r", exc, exc_info=exc)
            self.session_stop.set()

    async def _stats_loop(self) -> None:
        prev = dict(self.stats)
        while self.alive():
            await asyncio.sleep(5)
            # frames/s should sit at 12.5 (one 80 ms frame per 80 ms) if the
            # pacer is doing its job; drift here means the clock is slipping.
            fps = (self.stats["mic_chunks"] - prev.get("mic_chunks", 0)) / 5.0
            log.info("io %s  tx_fps=%.1f (target %.1f)",
                     self.stats, fps, 1000.0 / FRAME_MS)
            prev = dict(self.stats)

    # -- session -------------------------------------------------------------

    def _url(self) -> str:
        c = self.cfg
        return (f"ws://{c.server_host}:{c.server_port}/api/chat"
                f"?voice_prompt={quote(c.voice_prompt)}"
                f"&text_prompt={quote(c.text_prompt)}")

    async def wait_for_wake(self) -> None:
        """Dormant: ears open, no model session, robot resting."""
        log.info("dormant — say one of %s to start a conversation",
                 list(self.cfg.wake_phrases))
        try:
            self.mini.goto_sleep()
        except Exception as e:  # noqa: BLE001
            log.debug("goto_sleep: %s", e)
        while not self.stop.is_set():
            if self.listener.poll() == "wake":
                log.info("woken")
                try:
                    self.mini.wake_up()
                except Exception as e:  # noqa: BLE001
                    log.debug("wake_up: %s", e)
                return
            await asyncio.sleep(0.05)

    async def dismissal_loop(self) -> None:
        """End the conversation when the user dismisses the robot."""
        while self.alive():
            if self.listener.poll() == "sleep":
                log.info("dismissed by phrase — ending conversation")
                self.end_reason = "dismissed"
                self.session_stop.set()
                return
            await asyncio.sleep(0.05)

    async def run_session(self) -> None:
        """One bounded conversation."""
        # fresh codec streams, gates and resampler per session
        self.opus_w = sphn.OpusStreamWriter(MODEL_SR)
        self.opus_r = sphn.OpusStreamReader(MODEL_SR)
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self.handshake = asyncio.Event()
        self.streaming = asyncio.Event()
        self.session_stop = asyncio.Event()
        self.end_reason = None
        if self.in_sr != MODEL_SR:
            self.rs_down = soxr.ResampleStream(self.in_sr, MODEL_SR, 1, dtype="float32")

        async with websockets.connect(self._url(), max_size=None) as ws:
            log.info("personaplex session open (voice=%s)", self.cfg.voice_prompt)
            loops = [self.gate_loop(), self.pace_loop(ws), self.recv_loop(ws),
                     self.play_loop(), self.watchdog_loop(), self._stats_loop()]
            if self.listener.available and self.cfg.wake_enabled:
                loops.append(self.dismissal_loop())
            tasks = [asyncio.create_task(t) for t in loops]
            for t in tasks:
                t.add_done_callback(self._task_died)
            done = asyncio.create_task(self.session_stop.wait())
            stop = asyncio.create_task(self.stop.wait())
            await asyncio.wait([done, stop], return_when=asyncio.FIRST_COMPLETED)
            for t in tasks + [done, stop]:
                t.cancel()
        self.streaming.clear()
        log.info("conversation ended (%s)", self.end_reason or "connection")

    async def run(self) -> None:
        self.connect_robot()
        self.listener = PhraseListener(self.cfg.wake_phrases, self.cfg.sleep_phrases)
        self.listener.start()
        gated = self.cfg.wake_enabled and self.listener.available
        if not gated:
            log.info("wake phrases unavailable — running always-on")

        capture = asyncio.create_task(self.capture_loop())
        capture.add_done_callback(self._task_died)
        backoff = 1.0
        try:
            while not self.stop.is_set():
                try:
                    if gated:
                        await self.wait_for_wake()
                        if self.stop.is_set():
                            break
                    await self.run_session()
                    backoff = 1.0
                except (OSError, websockets.WebSocketException) as e:
                    log.warning("session lost (%s); retrying in %.0fs", e, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
        finally:
            capture.cancel()
