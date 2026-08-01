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
import logging
import time
from urllib.parse import quote

import numpy as np
import soxr
import sphn
import websockets

from reachy_mini import ReachyMini

from .config import KurokoConfig
from .puppet import PuppetTrack
from .sdkfix import ensure_audio_send_ready

MODEL_SR = 24000
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
        self.stats = {"mic_chunks": 0, "tx_bytes": 0, "rx_bytes": 0, "spk_chunks": 0}
        self._mic_buf = np.zeros(0, dtype=np.float32)

    # -- robot ---------------------------------------------------------------

    def connect_robot(self) -> None:
        self.mini = ReachyMini(host=self.cfg.robot_host)
        self.media = self.mini.media
        in_sr = self.media.get_input_audio_samplerate()
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

    async def mic_loop(self, ws) -> None:
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            # native call may block; keep it off the event loop
            pcm = await loop.run_in_executor(None, self.media.get_audio_sample)
            if pcm is None or np.size(pcm) == 0:
                await asyncio.sleep(0.004)
                continue
            mono = self._mono(pcm)
            if self.rs_down is not None:
                mono = self.rs_down.resample_chunk(mono)
            if mono.size:
                # opus wants exact frame sizes; feed 80ms (1920 @ 24k) frames
                self._mic_buf = np.concatenate((self._mic_buf, mono))
                while self._mic_buf.size >= 1920:
                    self.opus_w.append_pcm(self._mic_buf[:1920])
                    self._mic_buf = self._mic_buf[1920:]
                    self.stats["mic_chunks"] += 1
            data = self.opus_w.read_bytes()
            if data:
                self.stats["tx_bytes"] += len(data)
                await ws.send(b"\x01" + data)
            await asyncio.sleep(0.001)

    async def recv_loop(self, ws) -> None:
        pieces: list[str] = []
        async for msg in ws:
            if self.stop.is_set():
                break
            if not isinstance(msg, (bytes, bytearray)) or not msg:
                continue
            kind = msg[0]
            if kind == 1:
                self.stats["rx_bytes"] += len(msg) - 1
                self.opus_r.append_bytes(bytes(msg[1:]))
            elif kind == 2:
                piece = msg[1:].decode("utf-8", errors="replace")
                pieces.append(piece)
                if self.puppet is not None:
                    self.puppet.on_text(piece)
                if any(c in piece for c in ".!?\n") or sum(map(len, pieces)) > 120:
                    log.info("[says] %s", "".join(pieces).strip())
                    pieces.clear()
        self.stop.set()

    async def play_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            # sphn read_pcm blocks when the stream is empty; executor keeps
            # the event loop alive (learned the hard way)
            pcm = await loop.run_in_executor(None, self.opus_r.read_pcm)
            if pcm is None or pcm.size == 0:
                await asyncio.sleep(0.004)
                continue
            out = np.asarray(pcm, dtype=np.float32)
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

    def _task_died(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("loop task died: %r", exc, exc_info=exc)
            self.stop.set()

    async def _stats_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(5)
            log.info("io %s", self.stats)

    # -- session -------------------------------------------------------------

    def _url(self) -> str:
        c = self.cfg
        return (f"ws://{c.server_host}:{c.server_port}/api/chat"
                f"?voice_prompt={quote(c.voice_prompt)}"
                f"&text_prompt={quote(c.text_prompt)}")

    async def run(self) -> None:
        self.connect_robot()
        backoff = 1.0
        while not self.stop.is_set():
            try:
                async with websockets.connect(self._url(), max_size=None) as ws:
                    log.info("personaplex session open")
                    backoff = 1.0
                    tasks = [asyncio.create_task(t) for t in
                             (self.mic_loop(ws), self.recv_loop(ws),
                              self.play_loop(), self._stats_loop())]
                    for t in tasks:
                        t.add_done_callback(self._task_died)
                    await self.stop.wait()
                    for t in tasks:
                        t.cancel()
            except (OSError, websockets.WebSocketException) as e:
                # Resilience scene: connection lost. P3 turns this into the
                # sleepy-robot handoff + warm-start reconstruction.
                log.warning("session lost (%s); retrying in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
