"""The control surface — kuroko's public API for anything driving it.

kuroko's own scope is deliberately narrow: PersonaPlex, one Reachy Mini, done
well. It is not the place for knowledge back-ends, model arbitration or
multi-model scheduling. Those belong to a larger sibling (kōken), which drives
kuroko rather than replacing it.

This module is the seam between them. Everything kōken needs is four
operations, and none of them require it to know how the audio path works:

    suspend()      stop feeding the model; it freezes, keeping full context
    resume()       start feeding again; the model never knew it stopped
    inject(pcm)    write audio into the MODEL's input only
    play(pcm)      write audio to the ROBOT's speaker only
    on_text(cb)    subscribe to the model's text stream as it is generated

The two audio paths are genuinely independent, which is what makes a "hold"
possible at all:

    play()   -> robot speaker -> heard by the human, NOT by the model
    inject() -> model input    -> heard by the model, NOT by the human

So hold music can cover a pause the human experiences while an answer is fed
to the model that the human never hears.

Why suspension is free: the PersonaPlex serve loop only advances the model
when audio frames arrive. Stop sending and it does not idle or age — it
freezes. And because frame arrival IS its clock, a suspended session
experiences zero elapsed time: on resume it needs no context reconstruction
and does not know it was gone.
"""

import logging
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger("kuroko.control")

MODEL_SR = 24000


class SessionControl(Protocol):
    """What a driver (e.g. kōken) can do to a live kuroko session."""

    def suspend(self, reason: str = "") -> None: ...
    def resume(self, reason: str = "") -> None: ...
    def is_suspended(self) -> bool: ...
    def inject(self, pcm24k: np.ndarray) -> None: ...
    def play(self, pcm: np.ndarray, sample_rate: int = MODEL_SR) -> None: ...
    def on_text(self, callback: Callable[[str], None]) -> None: ...


class ControlMixin:
    """Implements SessionControl on top of VoiceBridge's internals.

    Kept in its own module so the public surface is obvious and stays small —
    if a driver needs something not here, that is a deliberate API decision
    rather than an accidental reach into private state.
    """

    # -- conversation control ------------------------------------------------

    def suspend(self, reason: str = "") -> None:
        """Freeze the model mid-conversation.

        Stops the pacer. Live mic audio stops accumulating (capture_loop gates
        on the same flag), the ring buffer keeps recording, and the model
        holds its exact state until resume(). Safe to call when already
        suspended.

        Callers who care about clean seams should wait for the model to finish
        its current utterance first — see `quiet_for()`.
        """
        if not self.streaming.is_set():
            return
        self.streaming.clear()
        self._suspended_reason = reason
        log.info("suspended%s", f" ({reason})" if reason else "")

    def resume(self, reason: str = "") -> None:
        """Unfreeze. The model resumes as though no time passed."""
        if self.streaming.is_set():
            return
        self.streaming.set()
        log.info("resumed%s", f" ({reason})" if reason else "")

    def is_suspended(self) -> bool:
        return not self.streaming.is_set()

    def quiet_for(self) -> float:
        """Seconds since the model last produced speech or text.

        Use this to avoid suspending mid-utterance, which would resume
        mid-word.
        """
        import time
        return time.monotonic() - self.last_activity

    # -- the two independent audio paths -------------------------------------

    def inject(self, pcm24k: np.ndarray) -> None:
        """Feed audio to the MODEL as though the user had spoken it.

        The human hears none of this. Intended for handing the model knowledge
        it does not have: TTS an answer, inject it, resume, and the model
        paraphrases it in its own voice.

        Injected audio is queued ahead of live mic input, so the usual pattern
        is: suspend -> inject -> resume.
        """
        pcm = np.asarray(pcm24k, dtype=np.float32).ravel()
        if not pcm.size:
            return
        frame = int(MODEL_SR * 0.08)
        pcm = pcm[: (pcm.size // frame) * frame] if pcm.size >= frame else pcm
        self._mic_buf = np.concatenate((self._mic_buf, pcm))
        log.info("injected %.1fs of audio into the model's input", pcm.size / MODEL_SR)

    def play(self, pcm: np.ndarray, sample_rate: int = MODEL_SR) -> None:
        """Play audio out of the ROBOT's speaker only.

        The model hears none of this — its input comes from the mic path,
        which the hardware AEC already scrubs of speaker output. Intended for
        hold music and earcons during a suspension.
        """
        out = np.asarray(pcm, dtype=np.float32).ravel()
        if not out.size or self.media is None:
            return
        if sample_rate != self._speaker_sr():
            import soxr
            out = np.asarray(
                soxr.resample(out, sample_rate, self._speaker_sr()), dtype=np.float32)
        self.media.push_audio_sample(np.clip(out, -1.0, 1.0))

    def _speaker_sr(self) -> int:
        try:
            return int(self.media.get_output_audio_samplerate())
        except Exception:  # noqa: BLE001
            return MODEL_SR

    # -- observation ---------------------------------------------------------

    def on_text(self, callback: Callable[[str], None]) -> None:
        """Subscribe to the model's text stream, piece by piece as generated.

        This arrives *before* the corresponding audio reaches the speaker, so
        it is the cheapest place to detect intent — e.g. a persona prompted to
        say "let me look that up" announces a knowledge lookup here with no
        ASR involved.
        """
        self._text_subscribers.append(callback)

    def _emit_text(self, piece: str) -> None:
        for cb in self._text_subscribers:
            try:
                cb(piece)
            except Exception as e:  # noqa: BLE001
                log.warning("text subscriber raised: %s", e)
