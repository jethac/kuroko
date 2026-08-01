"""Speech probe — does the mic path pass SPEECH, even if it suppresses tones?

probe/echo2.py concluded "mic is deaf", but every stimulus it used was a pure
tone or a chime. The Reachy Mini's mic array is a speech DSP with noise
suppression, non-linear attenuation and AGC all enabled (PP_AGCONOFF=1,
PP_NLATTENONOFF=1, PP_MIN_NS=0.15), and its USB output is routed to the
ASR/processed channel — which is *designed* to suppress non-speech. Meanwhile
DOA_VALUE tracks live bearings, proving the microphones themselves work.

So "silent output" may mean "working as intended, and my test signal was noise".

This probe plays a real speech recording through the robot's speaker over the
same remote push path kuroko uses, and measures what returns:

  speech returns  -> the mic path is FINE; echo2's verdict was a false alarm
                     caused by testing with a tone. (And we then genuinely do
                     need AEC, since the robot hears its own voice.)
  speech silent   -> the processed output really is dead; investigate routing
                     (AUDIO_MGR_OP_L/R) or fall back to raw mic channels.

Usage:
    python -m probe.speech --robot 192.168.1.128 --wav /audio/speech16.wav
"""

import argparse
import time
import wave

import numpy as np

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data, sr


def to_mono(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ax)
    return arr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--gain", type=float, default=1.0)
    args = ap.parse_args()

    speech, wav_sr = load_wav(args.wav)
    print(f"loaded {args.wav}: {speech.size} samples @ {wav_sr} Hz "
          f"({speech.size / wav_sr:.1f}s), rms={rms(speech):.4f}")

    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    in_sr = media.get_input_audio_samplerate()
    out_sr = media.get_output_audio_samplerate()
    print(f"mic {in_sr} Hz, speaker {out_sr} Hz")
    if wav_sr != out_sr:
        print(f"WARNING: wav {wav_sr} Hz != speaker {out_sr} Hz")

    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)
    time.sleep(1.0)

    dur = speech.size / wav_sr

    # baseline of the same length
    print(f"\n[A] baseline ({dur:.1f}s, stay quiet)")
    parts, t_end = [], time.monotonic() + dur
    while time.monotonic() < t_end:
        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            parts.append(to_mono(pcm))
        else:
            time.sleep(0.002)
    base = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    print(f"    rms={rms(base):.6f}  peak={float(np.abs(base).max()) if base.size else 0:.6f}")

    # play speech, paced to realtime, recording throughout
    print(f"\n[B] playing SPEECH through the robot speaker ({dur:.1f}s)")
    chunk = int(out_sr * 0.02)
    parts = []
    pos, t0, next_push = 0, time.monotonic(), time.monotonic()
    while pos < speech.size:
        now = time.monotonic()
        if now >= next_push:
            seg = speech[pos:pos + chunk] * args.gain
            if seg.size:
                media.push_audio_sample(np.ascontiguousarray(seg.astype(np.float32)))
            pos += chunk
            next_push += 0.02
        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            parts.append(to_mono(pcm))
        time.sleep(0.002)
    # keep recording briefly for buffered playout
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            parts.append(to_mono(pcm))
        else:
            time.sleep(0.002)
    heard = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    print(f"    rms={rms(heard):.6f}  peak={float(np.abs(heard).max()) if heard.size else 0:.6f}")

    b, h = rms(base), rms(heard)
    ratio_db = 20 * np.log10(max(h, 1e-9) / max(b, 1e-9))
    print("\n=== VERDICT ===")
    print(f"  speech vs baseline: {ratio_db:+.1f} dB")
    if h > 3 * b:
        print("  MIC HEARS SPEECH. The earlier 'deaf' verdict was a tone artifact —")
        print("  the DSP suppresses non-speech by design. And since the robot hears")
        print("  its own speaker, we DO need echo cancellation.")
    else:
        print("  Mic did not return the speech either. The processed output path")
        print("  is genuinely not delivering audio; investigate AUDIO_MGR_OP_L/R")
        print("  routing or capture raw mic channels instead.")


if __name__ == "__main__":
    main()
