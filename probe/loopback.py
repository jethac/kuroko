"""P0 — the go/no-go probe for the kuroko topology.

Connects to a Reachy Mini daemon REMOTELY (this script runs on the inference box,
never on the robot), captures mic audio, and immediately plays it back through the
robot's speaker, while measuring what actually matters:

  - mouth-to-ear round trip (chunk capture -> playback push) distribution
  - inter-chunk arrival jitter on the mic path
  - stall count (gaps > 1 chunk period)
  - DoA liveness: does the daemon keep publishing direction-of-arrival while a
    remote client holds both media directions?

If p95 round-trip stays under ~250 ms and DoA stays live for the whole run, the
puppeteer topology is viable and everything else in this repo is worth building.
If not, fall back to gstreamer-UDP media mode and re-run before abandoning ship.

Usage:
    python -m probe.loopback --robot reachy-mini.local --minutes 10
"""

import argparse
import statistics
import time

import numpy as np

from reachy_mini import ReachyMini


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy-mini.local",
                    help="daemon host (LAN name or tailnet address)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--report-every", type=float, default=30.0, help="seconds")
    args = ap.parse_args()

    mini = ReachyMini(host=args.robot)
    media = mini.media
    in_sr = media.get_input_audio_samplerate()
    print(f"connected to {args.robot}; mic sr={in_sr} "
          f"out sr={media.get_output_audio_samplerate()}")

    media.start_recording()
    media.start_playing()
    from kuroko.sdkfix import ensure_audio_send_ready
    if not ensure_audio_send_ready(media):
        print("WARNING: speaker send chain never became ready — loopback will be silent")

    rtts: list[float] = []
    gaps: list[float] = []
    stalls = 0
    doa_alive = 0
    doa_dead = 0
    last_chunk_t: float | None = None
    t_end = time.monotonic() + args.minutes * 60
    t_report = time.monotonic() + args.report_every

    while time.monotonic() < t_end:
        t0 = time.monotonic()
        pcm = media.get_audio_sample()
        if pcm is None or np.size(pcm) == 0:
            time.sleep(0.002)
            continue

        now = time.monotonic()
        if last_chunk_t is not None:
            gap = now - last_chunk_t
            gaps.append(gap)
            chunk_period = np.size(pcm) / max(1, in_sr)
            if gap > 2.5 * max(chunk_period, 0.02):
                stalls += 1
        last_chunk_t = now

        media.push_audio_sample(np.asarray(pcm, dtype=np.float32))
        rtts.append(time.monotonic() - t0)

        try:
            doa = media.get_DoA()
            if doa is not None:
                doa_alive += 1
            else:
                doa_dead += 1
        except Exception:
            doa_dead += 1

        if time.monotonic() >= t_report:
            t_report += args.report_every
            report(rtts, gaps, stalls, doa_alive, doa_dead)

    print("\n=== FINAL ===")
    report(rtts, gaps, stalls, doa_alive, doa_dead)
    media.stop_recording()
    media.stop_playing()


def report(rtts, gaps, stalls, doa_alive, doa_dead) -> None:
    if not rtts:
        print("no audio chunks received yet — remote media path not delivering")
        return
    q = statistics.quantiles(rtts, n=100)
    jq = statistics.quantiles(gaps, n=100) if len(gaps) > 10 else [0] * 99
    doa_pct = 100.0 * doa_alive / max(1, doa_alive + doa_dead)
    print(f"chunks={len(rtts)}  handle p50={q[49]*1e3:.1f}ms p95={q[94]*1e3:.1f}ms "
          f"p99={q[98]*1e3:.1f}ms | arrival-gap p95={jq[94]*1e3:.1f}ms "
          f"| stalls={stalls} | DoA alive {doa_pct:.1f}%")


if __name__ == "__main__":
    main()
