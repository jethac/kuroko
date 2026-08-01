"""Sanity probe — is the audio we feed the model actually correct?

The mic array, AEC and barge-in all check out (probe/duplex.py), so a garbage
conversation points at our own pipeline. The classic killer for a full-duplex
model is TIME: Mimi consumes exactly one 80 ms frame per 80 ms of wall clock.
Feed it audio at 1.5x and it hears chipmunks; feed it at 0.7x and it hears a
drawl. Either way it will not converse, and nothing in the logs looks wrong.

This probe measures the things bridge.py assumes:

  1. TRUE sample rate = samples delivered / wall-clock seconds, compared with
     what the SDK claims (get_input_audio_samplerate). A mismatch here breaks
     everything downstream.
  2. Delivery continuity: gaps between chunks, so we know the stream is
     smooth rather than bursty.
  3. Writes the captured audio BOTH raw and resampled-to-24 kHz exactly the
     way bridge.py does it, so the actual bytes the model receives can be
     inspected offline.

Usage:
    python -m probe.sanity --robot 192.168.1.128 --seconds 12 --out /audio/cap
"""

import argparse
import time
import wave

import numpy as np
import soxr

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk

MODEL_SR = 24000


def to_mono(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ax)
    return arr


def write_wav(path: str, data: np.ndarray, sr: int) -> None:
    pcm = np.clip(data, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="/audio/cap")
    args = ap.parse_args()

    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    claimed_sr = media.get_input_audio_samplerate()
    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)
    time.sleep(1.0)

    print(f"SDK claims mic rate = {claimed_sr} Hz")
    print(f"capturing {args.seconds:.0f}s — please TALK so there is signal to judge\n")

    parts, gaps, sizes = [], [], []
    last = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        pcm = media.get_audio_sample()
        if pcm is None or np.size(pcm) == 0:
            time.sleep(0.001)
            continue
        now = time.monotonic()
        if last is not None:
            gaps.append(now - last)
        last = now
        m = to_mono(pcm)
        sizes.append(m.size)
        parts.append(m)
    wall = time.monotonic() - t0

    raw = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    true_sr = raw.size / wall if wall > 0 else 0.0

    print("--- delivery ---")
    print(f"  wall clock       {wall:.2f}s")
    print(f"  samples received {raw.size}")
    print(f"  chunks           {len(sizes)} (mean {np.mean(sizes):.0f} samples"
          f" = {1000*np.mean(sizes)/max(claimed_sr,1):.1f} ms each)")
    if gaps:
        g = np.array(gaps) * 1000
        print(f"  inter-chunk gap  p50={np.percentile(g,50):.1f}ms "
              f"p95={np.percentile(g,95):.1f}ms max={g.max():.1f}ms")

    print("\n--- SAMPLE RATE TRUTH ---")
    print(f"  claimed {claimed_sr} Hz")
    print(f"  actual  {true_sr:.0f} Hz  (samples / wall second)")
    err = (true_sr / claimed_sr - 1.0) * 100 if claimed_sr else 0.0
    print(f"  error   {err:+.1f}%")
    if abs(err) > 5:
        print("  *** MISMATCH: the model is being fed time-distorted audio. ***")
        print(f"  *** Speech would sound {true_sr/claimed_sr:.2f}x speed to Mimi. ***")
    else:
        print("  rate is truthful; timing is not the problem")

    # resample exactly as bridge.py does, and save both for inspection
    rs = soxr.ResampleStream(claimed_sr, MODEL_SR, 1, dtype="float32")
    out = rs.resample_chunk(raw)
    write_wav(f"{args.out}_raw_{claimed_sr}.wav", raw, claimed_sr)
    write_wav(f"{args.out}_model_{MODEL_SR}.wav", np.asarray(out, dtype=np.float32), MODEL_SR)
    print(f"\n  wrote {args.out}_raw_{claimed_sr}.wav and {args.out}_model_{MODEL_SR}.wav")
    print(f"  captured rms={float(np.sqrt(np.mean(np.square(raw)))) if raw.size else 0:.5f}"
          f"  peak={float(np.abs(raw).max()) if raw.size else 0:.3f}")

    media.stop_recording()
    media.stop_playing()


if __name__ == "__main__":
    main()
