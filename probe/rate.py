"""Rate probe — is the mic surplus a startup burst or continuous drift?

probe/sanity.py found the robot delivers ~15.5% more audio than wall-clock
realtime (18.5 kHz of samples per second while claiming 16 kHz). For a
full-duplex model that is fatal: the server's frame loop consumes whatever
arrives, so an over-fast input stream runs the model's whole clock fast.

But the fix depends entirely on the shape of the surplus:

  STARTUP BURST  - the webrtc jitter buffer dumps its backlog on connect, then
                   settles to realtime. Fix: drain/discard until the stream
                   settles, then pass through.
  CONTINUOUS     - the capture clock genuinely runs fast relative to wall time.
                   Fix: continuous rate control in the bridge (drop or
                   resample against a wall clock), forever.

This measures samples delivered per wall-clock second, so the shape is
obvious. No speech required — sample delivery is independent of content.

Usage:
    python -m probe.rate --robot 192.168.1.128 --seconds 30
"""

import argparse
import time

import numpy as np

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk


def to_mono(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ax)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()

    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    sr = media.get_input_audio_samplerate()
    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)

    print(f"claimed rate {sr} Hz; measuring {args.seconds}s (no need to talk)\n")
    per_sec = [0] * args.seconds
    t0 = time.monotonic()
    while True:
        el = time.monotonic() - t0
        if el >= args.seconds:
            break
        pcm = media.get_audio_sample()
        if pcm is None or np.size(pcm) == 0:
            time.sleep(0.001)
            continue
        per_sec[int(el)] += to_mono(pcm).size

    print("--- samples delivered per wall-clock second ---")
    for i, n in enumerate(per_sec):
        ratio = n / sr if sr else 0
        bar = "#" * min(70, int(ratio * 40))
        flag = ""
        if ratio > 1.05:
            flag = "  FAST"
        elif ratio < 0.95:
            flag = "  slow"
        print(f"  {i:3d}s  {n:6d} samples  {ratio:.2f}x realtime {bar}{flag}")

    arr = np.array(per_sec, dtype=float) / sr
    first3, rest = arr[:3], arr[3:]
    print(f"\n  first 3s mean: {first3.mean():.3f}x realtime")
    print(f"  after 3s mean: {rest.mean():.3f}x realtime")
    print(f"  overall mean:  {arr.mean():.3f}x   std {arr.std():.3f}")

    print("\n=== VERDICT ===")
    if rest.mean() > 1.05:
        print(f"  CONTINUOUS DRIFT: steady-state runs {rest.mean():.3f}x realtime.")
        print("  The bridge must rate-control against a wall clock permanently:")
        print("  pace mic->model at exactly 1x and drop/absorb the surplus.")
    elif first3.mean() > 1.05:
        print("  STARTUP BURST only: steady state is realtime.")
        print("  The bridge should discard the initial backlog before streaming")
        print("  (or simply wait for the stream to settle after connecting).")
    else:
        print("  Rate looks clean here — re-check under conversation load.")

    media.stop_recording()
    media.stop_playing()


if __name__ == "__main__":
    main()
