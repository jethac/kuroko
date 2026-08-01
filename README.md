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

Running. Wake it with a phrase, talk to it, dismiss it.

```
docker build -t kuroko . && ./voice VARF2 100      # on the inference box
```

- [x] **P0 — probe**: remote full-duplex loopback verified at ~19 ms p50, zero stalls
      over 2.4 GHz wifi. DoA is *not* exposed over the remote media path (open issue).
- [x] **P1 — voice loop**: `kuroko/bridge.py`. Conversational, and clock-disciplined
      (see "The clock problem" below — this was the whole ballgame).
- [x] **P2 — body**: puppet track → `kuroko/body.py` at 30 Hz through a single writer;
      gaze arbiter with face-tracking confirmation.
- [x] **lifecycle**: wake/sleep phrases, lookback pre-roll, idle end, catatonia recovery.
- [ ] **P3 — theater**: Pi reflex supervisor, banto lease scenes, flight recorder.
- [ ] **P4 — server tap**: fork-side sidecar channel (see `server/TAP.md`) for true
      lookahead: text tokens, turn logits, pre-playout energy.

## The clock problem (read this before touching the audio path)

PersonaPlex is a Moshi-architecture model: **frame arrival is its clock**. It consumes
whatever shows up, as fast as it shows up. The Reachy's webrtc capture delivers
0.52x–1.86x realtime second to second (std 0.328) plus a ~2x backlog dump on connect.
Forwarding that as-received jerks the model's sense of time around and destroys its
turn-taking — it monologues, talks over you, and reads as broken, while every log and
byte counter looks perfectly healthy.

So the bridge is a **clock master**: `capture_loop` fills a bounded jitter buffer and
`pace_loop` emits exactly one 80 ms frame per 80 ms of wall clock (silence on underrun,
drop-oldest on overflow, gentle 1.15x catch-up when deep). Watch `tx_fps` in the io log
— it must sit at 12.5. If it drifts, conversation quality dies and nothing else will
tell you why.

Two more hard-won things:

- **The robot's audio hardware is excellent — do not "fix" it.** The mic array's
  hardware AEC strips the robot's own voice ~28 dB while leaving yours intact
  (barge-in measured at +3.1 dB). Adding echo cancellation makes things worse.
- **Probe with speech, never tones.** The mic DSP suppresses non-speech by design, so
  a sine-wave test reports a dead microphone that is in fact perfectly healthy.

## Conversation lifecycle

A PersonaPlex session is a *bounded conversation*, not a permanent state. Left open, it
takes the floor whenever it hears silence and eventually goes catatonic — emitting
well-formed opus silence forever while byte counters keep climbing (observed at ~9 min).

    dormant ──"hey mini"──► conversation ──"go to sleep" / 45 s idle──► dormant
                                  └── stale / aged ──► fresh session

The ears (`kuroko/listener.py`, vosk with a grammar constrained to just these phrases)
run continuously, so phrases are configurable at runtime with no keyword-model
training. A ring buffer records the whole time, so a woken session is seeded with what
you already said instead of making you repeat yourself while it connects.

> Phrases must be spelled with words the recognizer knows or they can never fire —
> and fail silently. `reachy` is **not** in the small English model, hence the
> phonetic stand-ins in the defaults. Check new phrases with `python -m probe.vocab`.

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
