"""Wake/sleep phrase spotting — the ears that are always on.

A full-duplex model cannot be the thing that decides when a conversation
starts. It holds an open session, takes the floor whenever it hears silence,
and its context is finite: leave it running and it monologues at an empty room
until it goes catatonic (observed at ~9 minutes). So the session must be
*bounded*, and something cheaper has to decide the boundaries.

That is this module. It runs continuously on the inference box, listening to
the robot's mic with a small offline recognizer, and reports two things:

  wake  — the user said the wake phrase, open a PersonaPlex session
  sleep — the user dismissed the robot, close the session politely

Vosk is used rather than a wake-word engine (openWakeWord, Porcupine) because
the user asked for a *configurable* phrase: those engines need a trained model
per keyword, while a grammar-constrained recognizer accepts any phrase you can
spell, changed at runtime with no retraining. Constraining the grammar to just
the phrases we care about (plus [unk]) makes it both fast and far more accurate
than open-vocabulary transcription.

Degrades gracefully: if vosk or the model is unavailable, `available` is False
and the bridge falls back to always-on behaviour rather than going deaf.
"""

import json
import logging
import os
import queue
import threading

import numpy as np

log = logging.getLogger("kuroko.listener")

VOSK_SR = 16000


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


class PhraseListener:
    """Streaming phrase spotter over 16 kHz mono float32 audio."""

    def __init__(self, wake_phrases, sleep_phrases, model_path: str | None = None):
        self.wake = [_norm(p) for p in wake_phrases]
        self.sleep = [_norm(p) for p in sleep_phrases]
        self.available = False
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._events: queue.Queue = queue.Queue()
        self._rec = None
        self._thread = None

        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)
            path = model_path or os.environ.get("VOSK_MODEL", "/models/vosk")
            if not os.path.isdir(path):
                log.warning("vosk model not found at %s — phrase spotting disabled", path)
                return
            model = Model(path)
            # Grammar-constrained: only these phrases (plus unknown) are
            # considered, which is both cheaper and much more accurate.
            grammar = json.dumps(sorted(set(self.wake + self.sleep)) + ["[unk]"])
            self._rec = KaldiRecognizer(model, VOSK_SR, grammar)
            self._rec.SetWords(False)
            self.available = True
            log.info("phrase listener ready — wake=%s sleep=%s", self.wake, self.sleep)
        except ImportError:
            log.warning("vosk not installed — phrase spotting disabled")
        except Exception as e:  # noqa: BLE001
            log.warning("phrase listener unavailable: %s", e)

    def start(self) -> None:
        if not self.available or self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, pcm16k: np.ndarray) -> None:
        """Feed 16 kHz mono float32 audio (non-blocking; drops under pressure)."""
        if not self.available:
            return
        try:
            self._q.put_nowait(np.asarray(pcm16k, dtype=np.float32).copy())
        except queue.Full:
            pass

    def poll(self) -> str | None:
        """Return 'wake', 'sleep', or None."""
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _classify(self, text: str) -> str | None:
        t = _norm(text)
        if not t:
            return None
        # substring match: people say "hey reachy, what's up" in one breath
        if any(p and p in t for p in self.wake):
            return "wake"
        if any(p and p in t for p in self.sleep):
            return "sleep"
        return None

    def _run(self) -> None:
        while True:
            chunk = self._q.get()
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            try:
                if self._rec.AcceptWaveform(pcm16):
                    text = json.loads(self._rec.Result()).get("text", "")
                else:
                    text = json.loads(self._rec.PartialResult()).get("partial", "")
            except Exception as e:  # noqa: BLE001
                log.warning("recognizer error: %s", e)
                continue
            hit = self._classify(text)
            if hit:
                log.info("heard %r -> %s", text, hit)
                self._events.put(hit)
                try:
                    self._rec.Reset()
                except Exception:  # noqa: BLE001
                    pass
