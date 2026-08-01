"""Echo probe, controlled edition — three conditions, one answer.

probe/echo.py measured tone leak but could not distinguish these cases:
  (a) AEC is cancelling our echo (good)
  (b) our pushed audio never reaches the speaker (playback path dead)
  (c) the mic stream is gated/muted (capture path deaf)

All three look identical: "quiet mic during playback". So this probe runs a
control the robot cannot fake — the daemon's OWN test sound, triggered over
HTTP, which uses the daemon's local audio path rather than our remote push:

  A. baseline silence               -> room noise floor
  B. daemon test sound (control)    -> proves mic can hear the speaker at all
  C. our pushed tone (paced to realtime, unlike echo.py)  -> the real question

Reading the results:
  B loud, C quiet   -> our remote push is silent OR is being cancelled;
                       check AEC far-end reference to tell which
  B quiet, C quiet  -> mic capture is deaf/gated; echo was never the problem
  B loud, C loud    -> no cancellation on our path; we need AEC (build it)
"""

import argparse
import json
import time
import urllib.request

import numpy as np

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk


def daemon_get(daemon: str, path: str):
    try:
        with urllib.request.urlopen(f"{daemon}{path}", timeout=3) as r:
            return json.load(r)
    except Exception as e:
        return f"<error {e}>"


def daemon_post(daemon: str, path: str, body: dict | None = None):
    try:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{daemon}{path}", data=data, method="POST",
            headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except Exception as e:
        return f"<error {e}>"


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def to_mono(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ax)
    return arr


def record_for(media, seconds: float) -> np.ndarray:
    parts = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        pcm = media.get_audio_sample()
        if pcm is None or np.size(pcm) == 0:
            time.sleep(0.002)
            continue
        parts.append(to_mono(pcm))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--tone-hz", type=float, default=1000.0)
    ap.add_argument("--amplitude", type=float, default=0.4)
    ap.add_argument("--seconds", type=float, default=4.0)
    args = ap.parse_args()
    daemon = f"http://{args.robot}:8000"

    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    in_sr = media.get_input_audio_samplerate()
    out_sr = media.get_output_audio_samplerate()
    print(f"mic {in_sr} Hz, speaker {out_sr} Hz")
    print(f"media status: {daemon_get(daemon, '/api/media/status')}")

    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)
    time.sleep(1.0)

    # ---- A. baseline -----------------------------------------------------
    print(f"\n[A] baseline silence ({args.seconds:.0f}s) — please stay quiet")
    a = record_for(media, args.seconds)
    print(f"    rms={rms(a):.6f}  samples={a.size}")

    # ---- B. control: daemon's own test sound -----------------------------
    print(f"\n[B] CONTROL: daemon test sound (you should HEAR this)")
    status = daemon_post(daemon, "/api/volume/test-sound")
    print(f"    POST /api/volume/test-sound -> {status}")
    b = record_for(media, args.seconds)
    print(f"    rms={rms(b):.6f}  samples={b.size}")

    time.sleep(1.0)

    # ---- C. our pushed tone, paced to realtime ---------------------------
    print(f"\n[C] our remote push: {args.tone_hz:.0f} Hz tone (you should HEAR a beep)")
    spen = []
    chunk = int(out_sr * 0.02)
    phase, step = 0.0, 2 * np.pi * args.tone_hz / out_sr
    parts = []
    t0 = time.monotonic()
    pushed_chunks = 0
    next_push = t0
    while time.monotonic() - t0 < args.seconds:
        now = time.monotonic()
        if now >= next_push:                      # pace to realtime (echo.py did not)
            n = np.arange(chunk)
            tone = (args.amplitude * np.sin(phase + step * n)).astype(np.float32)
            phase = float((phase + step * chunk) % (2 * np.pi))
            media.push_audio_sample(tone)
            pushed_chunks += 1
            next_push += 0.02
        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            parts.append(to_mono(pcm))
        if len(spen) < 8 and pushed_chunks % 25 == 1:
            v = daemon_get(daemon, "/api/audio/config/parameter/AEC_SPENERGY_VALUES")
            spen.append(v.get("values") if isinstance(v, dict) else v)
        time.sleep(0.002)
    c = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    print(f"    pushed {pushed_chunks} chunks ({pushed_chunks * 0.02:.1f}s of audio)")
    print(f"    rms={rms(c):.6f}  samples={c.size}")
    print(f"    AEC_SPENERGY during push: {spen}")

    # ---- verdict ---------------------------------------------------------
    base, ctrl, ours = rms(a), rms(b), rms(c)
    print("\n=== RESULTS ===")
    print(f"  A baseline        rms {base:.6f}")
    print(f"  B daemon sound    rms {ctrl:.6f}   ({20*np.log10(max(ctrl,1e-9)/max(base,1e-9)):+.1f} dB vs baseline)")
    print(f"  C our pushed tone rms {ours:.6f}   ({20*np.log10(max(ours,1e-9)/max(base,1e-9)):+.1f} dB vs baseline)")

    ctrl_heard = ctrl > 3 * base
    ours_heard = ours > 3 * base
    print("\n=== VERDICT ===")
    if not ctrl_heard and not ours_heard:
        print("  MIC IS DEAF to the speaker (control failed too).")
        print("  Either capture is gated, or hardware AEC is removing everything")
        print("  the speaker emits. Echo is NOT the cause of bad conversation.")
    elif ctrl_heard and not ours_heard:
        print("  Control audible, our push inaudible.")
        print("  -> our remote-pushed audio is either not reaching the speaker,")
        print("     or is being cancelled/suppressed on our path specifically.")
    elif ours_heard:
        print("  OUR AUDIO ECHOES BACK into the mic -> real echo problem.")
        print("  Build AEC (server-side, using the model's own output as reference).")


if __name__ == "__main__":
    main()
