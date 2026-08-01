"""Duplex probe — can the user be heard WHILE the robot is speaking?

This is the question that decides the whole embodiment design.

probe/speech.py showed the mic array is healthy (it captured room audio at
peak 0.8) but that its output collapses ~28 dB while the robot's speaker is
active. That is echo suppression doing its job. But PersonaPlex is a
FULL-DUPLEX model: it expects a continuous user stream and decides when to
yield the floor based on what it hears. If the user's voice is suppressed
whenever the robot speaks, the model hears silence, keeps talking, and the
suppression never lifts — an endless monologue. Which is exactly the failure
we observed.

This probe records a continuous timeline with per-second levels while the
user talks throughout:

    phase 1 (0-8s)   user talks, speaker SILENT    -> reference level
    phase 2 (8-18s)  user talks, speaker PLAYING   -> can they get through?
    phase 3 (18-24s) user talks, speaker SILENT    -> recovery

If phase 2 levels collapse to the phase-1 noise floor, barge-in is impossible
and the fix is DSP-side (relax suppression / use a less-processed channel),
not "add AEC" — the AEC is already over-performing.

Usage:
    python -m probe.duplex --robot 192.168.1.128 --wav /audio/speech16.wav
"""

import argparse
import time
import wave

import numpy as np

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk


def load_wav(path: str):
    with wave.open(path) as w:
        sr, n = w.getframerate(), w.getnframes()
        d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            d = d.reshape(-1, w.getnchannels()).mean(axis=1)
    return d, sr


def to_mono(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ax)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--quiet-s", type=float, default=8.0)
    ap.add_argument("--play-s", type=float, default=10.0)
    ap.add_argument("--recover-s", type=float, default=6.0)
    args = ap.parse_args()

    speech, wav_sr = load_wav(args.wav)
    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    in_sr, out_sr = media.get_input_audio_samplerate(), media.get_output_audio_samplerate()
    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)
    time.sleep(0.5)

    total = args.quiet_s + args.play_s + args.recover_s
    print(f"\n*** TALK CONTINUOUSLY FOR THE NEXT {total:.0f} SECONDS ***\n", flush=True)
    print(f"  0-{args.quiet_s:.0f}s   speaker silent  (your voice, clean)", flush=True)
    print(f"  {args.quiet_s:.0f}-{args.quiet_s+args.play_s:.0f}s  speaker PLAYING (can you get through?)", flush=True)
    print(f"  {args.quiet_s+args.play_s:.0f}-{total:.0f}s speaker silent  (recovery)\n", flush=True)

    chunk = int(out_sr * 0.02)
    t0 = time.monotonic()
    next_push = None
    pos = 0
    buckets: dict[int, list[np.ndarray]] = {}

    while True:
        now = time.monotonic()
        el = now - t0
        if el >= total:
            break
        playing = args.quiet_s <= el < args.quiet_s + args.play_s
        if playing:
            if next_push is None:
                next_push = now
            if now >= next_push:
                seg = speech[pos % speech.size: (pos % speech.size) + chunk]
                if seg.size < chunk:                      # loop the clip
                    seg = np.concatenate((seg, speech[:chunk - seg.size]))
                media.push_audio_sample(np.ascontiguousarray(seg.astype(np.float32)))
                pos += chunk
                next_push += 0.02
        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            buckets.setdefault(int(el), []).append(to_mono(pcm))
        else:
            time.sleep(0.002)

    print("--- per-second mic level ---")
    phase_rms = {"quiet": [], "playing": [], "recover": []}
    for sec in sorted(buckets):
        seg = np.concatenate(buckets[sec])
        r = float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0
        pk = float(np.abs(seg).max()) if seg.size else 0.0
        if sec < args.quiet_s:
            phase, mark = "quiet", " "
        elif sec < args.quiet_s + args.play_s:
            phase, mark = "playing", ">"
        else:
            phase, mark = "recover", " "
        phase_rms[phase].append(r)
        bar = "#" * min(60, int(r * 600))
        print(f"{mark}{sec:3d}s  rms={r:.5f} peak={pk:.3f}  {bar}")

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    q, p, rc = avg(phase_rms["quiet"]), avg(phase_rms["playing"]), avg(phase_rms["recover"])
    print(f"\n  phase means: quiet={q:.5f}  playing={p:.5f}  recover={rc:.5f}")
    if q > 1e-6:
        print(f"  suppression while speaker active: {20*np.log10(max(p,1e-9)/q):+.1f} dB")
        print(f"  recovery after speaker stops:     {20*np.log10(max(rc,1e-9)/q):+.1f} dB")

    print("\n=== VERDICT ===")
    if q < 0.002:
        print("  No voice detected even in the clean phase — was anyone talking?")
    elif p > 0.4 * q:
        print("  BARGE-IN WORKS: your voice survives while the robot speaks.")
        print("  Full-duplex conversation is viable; the garbage had another cause.")
    else:
        print("  BARGE-IN BLOCKED: your voice is suppressed while the robot speaks.")
        print("  PersonaPlex therefore hears silence exactly when it should hear you,")
        print("  so it never yields the floor. Fix the DSP suppression (or feed the")
        print("  model a less-processed mic channel) — do NOT add more cancellation.")

    media.stop_recording()
    media.stop_playing()


if __name__ == "__main__":
    main()
