# kuroko 黒子

**Server-side puppeteer for Reachy Mini × PersonaPlex.** The robot installs nothing; the
brain lives next to the model.

In kabuki, the *kuroko* are the black-clad stagehands who operate props and puppets in
full view of the audience — and are, by convention, invisible. Here, the kuroko is a
sidecar process on the inference box (an NVIDIA GB10 / DGX Spark class machine) that
connects to a Reachy Mini Wireless **over the network** via the `reachy_mini` SDK's
remote-daemon path, bridges the robot's mic/speaker to a realtime PersonaPlex-7B
websocket on localhost, and drives the robot's body with **lookahead** the model itself
provides.

## Why not run the app on the robot?

Prior art (e.g. `TwinPeaksTownie/reachy_personaplex`, which this design started from and
owes its existence to) runs everything on the Pi inside the robot: websocket client,
opus codec, motion heuristics, web UI. That shape fights the hardware at every layer:

| | app-on-Pi | kuroko (app-on-GB10) |
|---|---|---|
| deps on the Pi | opus/sphn native builds (fragile) | **zero — stock daemon only** |
| codec CPU | shares 4GB Pi with daemon + GIL | idle Grace cores |
| motion source | reactive to played audio (always late) | authored from frames **before playout** |
| echo handling | none | server holds a perfect far-end reference |
| clock drift | unbounded buffer creep | frame-time servo, hard latency cap |
| wifi drop | robot freezes mid-gesture | Pi brainstem degrades to "sleepy robot" |
| deploy/update | app-store install per robot | `docker compose up` on the server |
| server preempted | crash | scripted goodbye scene, auto-resume |

The asymmetry is the whole design: the GB10 serving PersonaPlex at 46.5 ms per 80 ms
frame has ~40% headroom, ~100 GB free unified RAM, and mostly-idle CPU cores, while the
robot is a 4 GB Pi on 2.4 GHz wifi. Every byte of work you move off the Pi is a failure
class deleted, not just latency saved.

## Architecture

```
┌────────────────────────── GB10 / DGX Spark ──────────────────────────┐
│                                                                       │
│  ┌─────────────────┐  ws (localhost)   ┌────────────────────────────┐ │
│  │ PersonaPlex-7B  │◄─────────────────►│ kuroko sidecar             │ │
│  │ (fork, --fast   │  \x01 opus        │  audio bridge (sphn/gst)   │ │
│  │  w8a16 serving) │  \x02 text        │  puppet-track compiler     │ │
│  │                 │──────────────────►│  gaze arbiter              │ │
│  │  frame tap ─────┤  sidecar UDP:     │  scene director (banto)    │ │
│  │  (energy, text, │  frame_id, rms,   │  flight recorder           │ │
│  │   turn state)   │  turn logits      └──────────┬─────────────────┘ │
│  └─────────────────┘                              │ zenoh + media     │
└───────────────────────────────────────────────────┼───────────────────┘
                                              LAN / tailnet
                                                    │
┌────────────────────────── Reachy Mini ────────────┼───────────────────┐
│  stock reachy_mini daemon (UNMODIFIED)  ◄─────────┘                   │
│    mic array + DoA ──► up      speaker ◄── down     pose/antennas     │
│  kuroko-reflex (~100 lines, OPTIONAL): heartbeat dead-man's switch    │
│    → on brain loss: cancel gestures, hand back to daemon idle         │
└───────────────────────────────────────────────────────────────────────┘
```

Three load-bearing ideas:

1. **Puppeteer topology.** The kuroko connects to the robot daemon remotely (zenoh
   scouting on LAN, explicit endpoint over tailnet). Mic/speaker flow over the SDK's
   remote media path; PersonaPlex is a localhost websocket. The Pi runs stock firmware.

2. **Lookahead embodiment.** The serve loop knows each 80 ms Mimi frame ~100–500 ms
   before the robot's speaker plays it, and samples the text token stream ahead of the
   audio. A small tap exports `{frame_id, rms, voiced, text_token, turn_state}` to the
   kuroko, which compiles a 10–20 Hz **puppet track** — wobble amplitude, beat-nods,
   antenna state, gaze bias — scheduled in *frame time* against a playhead heartbeat, so
   gestures land on (or a breath before) the stressed syllable instead of trailing it.
   Gaze is arbitrated: DoA **nominates** a glance; daemon face-tracking **confirms** a
   committed head-turn; no confirmation decays back to the track's bias.

3. **Resilience as theater.** Failures collapse into rehearsed scenes:
   - GB10 preempted with warning (banto lease): persona says goodbye, head parks, lease
     released, auto-resume + wake scene when the box frees.
   - Anything else (wifi drop, OOM, kill -9): the Pi-side reflex loses the 100 ms
     heartbeat and hands the body back to daemon-native idle — *sleepy robot, never
     frozen robot*.
   - Sessions are keyed to robot identity, not connection: reconnects open a fresh
     PersonaPlex session warm-started from a condensed transcript ("as I was saying…").

## Status

Design + scaffold. The **first artifact to run is the loopback probe**, which answers
the one question that can sink the topology (full-duplex remote media over 2.4 GHz at
conversational latency, without stealing the daemon's mic from DoA):

```
python -m probe.loopback --robot reachy-mini.local --minutes 10
```

Roadmap:

- [ ] **P0 — probe**: remote full-duplex loopback; RTT/jitter/loss histograms; DoA
      liveness check. Go/no-go for the topology.
- [ ] **P1 — voice loop**: `kuroko/bridge.py` — remote mic → sphn opus → PersonaPlex ws
      → opus → remote speaker. No motion. Conversational parity with the web client.
- [ ] **P2 — body**: playhead heartbeat, puppet track v0 (energy envelope), gaze
      arbiter, wobble handoff.
- [ ] **P3 — theater**: Pi reflex supervisor, banto lease scenes, session
      reconstruction, flight recorder.
- [ ] **P4 — server tap**: fork-side sidecar channel (see `server/TAP.md`) for true
      lookahead: text tokens, turn logits, pre-playout energy.

## Layout

```
kuroko/     the sidecar (runs on the inference box)
pi/         optional ~100-line reflex supervisor (the only thing that may touch the Pi)
server/     notes + patch sketch for the PersonaPlex fork's frame tap
probe/      P0 de-risk instruments
```

## Requirements

- Server box: the PersonaPlex fork serving realtime (see NVIDIA/personaplex PRs #102 /
  #103 for GB10-class devices), python 3.10+, `reachy-mini`, `sphn`, `websockets`.
- Robot: Reachy Mini (wireless) with stock daemon, reachable over LAN or tailnet.

## License

MIT
