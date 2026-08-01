"""Echo probe — is acoustic echo cancellation actually working on our audio path?

The Reachy Mini's mic array has hardware AEC (4 mics, 1 far-end reference; see
the AEC_* registers in reachy_mini.media.audio_control_utils), and the SDK
declares a software webrtcdsp/webrtcechoprobe path for the local backend. Both
need the *speaker* signal to reach them as a reference. Audio we push from a
remote sidecar may or may not land in that reference path.

This probe answers it empirically:

  1. plays a pure tone through the robot's speaker over the SAME path kuroko
     uses (media.push_audio_sample from the inference box)
  2. records the mic array throughout
  3. measures tone-band energy in the mic during playback vs during silence
     -> Echo Return Loss (ERL). High ERL = cancellation working. Low = we are
        feeding our own voice straight back into the model.
  4. samples the DSP's AEC_SPENERGY_VALUES / AEC_CURRENT_IDLE_TIME while the
     tone plays -> tells us whether the hardware far-end reference sees our
     audio at all (the difference between "AEC is broken" and "AEC never got
     the reference signal")
  5. reports per-channel mic data, since averaging raw mics would bypass the
     DSP's cancelled/beamformed output

Interpretation:
  ERL > 20 dB               cancellation is working; echo is not your problem
  ERL < 10 dB, spenergy > 0 AEC has the reference but is not cancelling our path
  ERL < 10 dB, spenergy = 0 our playback bypasses the AEC far-end reference
                            (fix the routing, not the algorithm)

Usage:
    python -m probe.echo --robot 192.168.1.128
"""

import argparse
import json
import time
import urllib.request

import numpy as np

from reachy_mini import ReachyMini

from kuroko.sdkfix import ensure_audio_send_ready, harden_sdk

TONE_HZ = 1000.0
TONE_SECONDS = 4.0
SILENCE_SECONDS = 3.0


def read_param(daemon: str, name: str):
    try:
        with urllib.request.urlopen(f"{daemon}/api/audio/config/parameter/{name}",
                                    timeout=3) as r:
            return json.load(r).get("values")
    except Exception as e:
        return f"<error {e}>"


def band_energy(pcm: np.ndarray, sr: int, hz: float, half_width: float = 60.0) -> float:
    """Energy in a narrow band around `hz` (Goertzel-ish via rFFT)."""
    if pcm.size < 256:
        return 0.0
    win = np.hanning(pcm.size)
    spec = np.abs(np.fft.rfft(pcm * win))
    freqs = np.fft.rfftfreq(pcm.size, 1.0 / sr)
    sel = (freqs > hz - half_width) & (freqs < hz + half_width)
    if not sel.any():
        return 0.0
    return float(np.sqrt(np.mean(np.square(spec[sel]))))


def collect(media, seconds: float, keep_channels: bool = False):
    """Record for `seconds`, returning (mono, per_channel_or_None)."""
    mono_parts, multi_parts = [], []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        pcm = media.get_audio_sample()
        if pcm is None or np.size(pcm) == 0:
            time.sleep(0.002)
            continue
        arr = np.asarray(pcm, dtype=np.float32)
        if arr.ndim == 2:
            ch_axis = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
            if keep_channels:
                multi_parts.append(arr if ch_axis == 1 else arr.T)
            arr = arr.mean(axis=ch_axis)
        mono_parts.append(arr)
    mono = np.concatenate(mono_parts) if mono_parts else np.zeros(0, dtype=np.float32)
    multi = np.concatenate(multi_parts, axis=0) if multi_parts else None
    return mono, multi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local")
    ap.add_argument("--tone-hz", type=float, default=TONE_HZ)
    ap.add_argument("--amplitude", type=float, default=0.25)
    args = ap.parse_args()

    daemon = f"http://{args.robot}:8000"

    harden_sdk()
    mini = ReachyMini(host=args.robot, connection_mode="network")
    media = mini.media
    in_sr = media.get_input_audio_samplerate()
    out_sr = media.get_output_audio_samplerate()
    try:
        in_ch = media.get_input_channels()
        out_ch = media.get_output_channels()
    except Exception:
        in_ch = out_ch = "?"

    print(f"mic: {in_sr} Hz x {in_ch} ch   speaker: {out_sr} Hz x {out_ch} ch")

    media.start_recording()
    media.start_playing()
    ensure_audio_send_ready(media)

    print("\n--- AEC registers at rest ---")
    for p in ("AEC_NUM_MICS", "AEC_NUM_FARENDS", "AEC_SPENERGY_VALUES",
              "AEC_CURRENT_IDLE_TIME"):
        print(f"  {p:24s} {read_param(daemon, p)}")

    # 1) baseline: room noise, nothing playing
    print(f"\nrecording {SILENCE_SECONDS}s of silence (stay quiet)...")
    quiet_mono, quiet_multi = collect(media, SILENCE_SECONDS, keep_channels=True)
    quiet_e = band_energy(quiet_mono, in_sr, args.tone_hz)
    quiet_rms = float(np.sqrt(np.mean(np.square(quiet_mono)))) if quiet_mono.size else 0.0
    print(f"  baseline: rms={quiet_rms:.5f}  tone-band={quiet_e:.5f}")

    # 2) play the tone while recording
    print(f"\nplaying {args.tone_hz:.0f} Hz for {TONE_SECONDS}s while recording...")
    chunk = int(out_sr * 0.02)  # 20 ms
    phase = 0.0
    step = 2 * np.pi * args.tone_hz / out_sr
    t_end = time.monotonic() + TONE_SECONDS
    loud_parts, multi_parts = [], []
    spenergy_seen, idle_seen = [], []
    last_poll = 0.0

    while time.monotonic() < t_end:
        n = np.arange(chunk)
        tone = (args.amplitude * np.sin(phase + step * n)).astype(np.float32)
        phase = float((phase + step * chunk) % (2 * np.pi))
        if out_ch and out_ch != "?" and int(out_ch) > 1:
            media.push_audio_sample(np.tile(tone[:, None], (1, int(out_ch))))
        else:
            media.push_audio_sample(tone)

        pcm = media.get_audio_sample()
        if pcm is not None and np.size(pcm):
            arr = np.asarray(pcm, dtype=np.float32)
            if arr.ndim == 2:
                ax = 0 if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else 1
                multi_parts.append(arr if ax == 1 else arr.T)
                arr = arr.mean(axis=ax)
            loud_parts.append(arr)

        now = time.monotonic()
        if now - last_poll > 0.5:
            last_poll = now
            spenergy_seen.append(read_param(daemon, "AEC_SPENERGY_VALUES"))
            idle_seen.append(read_param(daemon, "AEC_CURRENT_IDLE_TIME"))
        time.sleep(0.005)

    loud_mono = np.concatenate(loud_parts) if loud_parts else np.zeros(0, dtype=np.float32)
    loud_multi = np.concatenate(multi_parts, axis=0) if multi_parts else None
    loud_e = band_energy(loud_mono, in_sr, args.tone_hz)
    loud_rms = float(np.sqrt(np.mean(np.square(loud_mono)))) if loud_mono.size else 0.0

    print(f"  during tone: rms={loud_rms:.5f}  tone-band={loud_e:.5f}")

    print("\n--- AEC far-end reference during playback ---")
    print(f"  AEC_SPENERGY_VALUES samples: {spenergy_seen[:6]}")
    print(f"  AEC_CURRENT_IDLE_TIME samples: {idle_seen[:6]}")

    # 3) verdict
    print("\n=== VERDICT ===")
    if quiet_e <= 0 or loud_e <= 0:
        print("  inconclusive: no usable audio captured")
    else:
        erl_db = 20.0 * np.log10(loud_e / max(quiet_e, 1e-9))
        print(f"  tone-band leak: {erl_db:+.1f} dB above baseline")
        if erl_db < 10:
            print("  -> LITTLE/NO ECHO: cancellation appears to be working")
        elif erl_db < 20:
            print("  -> PARTIAL echo leaking into the mic")
        else:
            print("  -> STRONG ECHO: the robot hears its own speaker essentially raw")

    saw_reference = any(
        isinstance(v, list) and any(abs(float(x)) > 1e-6 for x in v)
        for v in spenergy_seen
    )
    print(f"  hardware AEC far-end reference active during playback: {saw_reference}")
    if not saw_reference:
        print("  -> the DSP never saw our playback as a reference signal;")
        print("     this is a ROUTING problem, not an algorithm problem.")

    if loud_multi is not None and loud_multi.ndim == 2 and loud_multi.shape[1] > 1:
        print(f"\n  per-channel tone-band energy ({loud_multi.shape[1]} channels):")
        for c in range(loud_multi.shape[1]):
            e = band_energy(np.ascontiguousarray(loud_multi[:, c]), in_sr, args.tone_hz)
            print(f"    ch{c}: {e:.5f}")
        print("  (a channel much quieter than the rest is likely the AEC'd/beamformed"
              " output — use it instead of averaging)")

    media.stop_recording()
    media.stop_playing()


if __name__ == "__main__":
    main()
