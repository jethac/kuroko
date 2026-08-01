# The frame tap (P4)

The unfair advantage of maintaining the PersonaPlex fork: the serve loop in
`moshi/moshi/server.py` holds every cue embodiment wants, *before* the audio is
audible. The `opus_loop` already computes per-frame:

- the sampled **text token** (line ~301) — ahead of its audio by the playout margin
- the decoded **PCM frame** (its RMS = energy envelope, pre-playout)
- **speaking/silent state** (text token 0/3 = padding vs speech)
- via `lm_gen` internals: turn-taking / end-of-turn tendencies

## Design

A ~30-line addition to the serve loop, strictly fire-and-forget (must never touch the
80 ms budget — the GB10 runs at 46.5 ms/frame, but the tap still gets zero blocking
privileges):

```python
# after mimi.decode(...) and text token sampling, once per frame:
tap.emit({
    "frame_id": _frame_count,          # monotonic, shared clock basis
    "rms": float(main_pcm.square().mean().sqrt()),   # pre-playout energy
    "text_token": text_token,          # raw id; kuroko detokenizes
    "voiced": text_token not in (0, 3),
})
```

`tap.emit` = non-blocking UDP datagram to localhost (the kuroko sidecar), JSON or a
16-byte packed struct. Drops are fine; the puppet track is resilient to gaps.

## Frame-time sync (the hard part)

The tap gives cues in **frame_id time**. The robot's speaker plays in **wall time**.
The bridge stamps each opus packet it forwards with the first frame_id it contains;
the kuroko heartbeat to the Pi reflex (`pi/kuroko_reflex.py`) can carry an `echo`
field, giving RTT; the playhead estimate is then:

    playhead(t) ≈ last_pushed_frame_id - buffered_frames(t)

with `buffered_frames` regressed from push rate vs the 12.5 Hz frame clock. The P2
milestone instruments this end-to-end and plots the lead-time histogram BEFORE any
motion code relies on it — if the lead estimate errs by ~100 ms, anticipatory gestures
land late, and trailing motion reads worse than no motion.

## Also unlocked by owning the serve loop

- **Perfect-reference AEC**: the server knows the exact waveform the robot will play;
  echo-cancel the incoming mic stream against it (aligned via the playhead estimate)
  before the model hears it. The robot's mic is centimeters from its speaker; a
  full-duplex model that hears itself will eventually barge in on itself.
- **Self-speech gating for DoA**: while emitting voiced frames, DoA nominations from
  the speaker's bearing are attenuated (see `kuroko/arbiter.py`).
- **Gesture tags**: system-prompt the persona to emit `<nod>`, `<tilt>`; strip them
  from the `\x02` text stream in the tap, forward as puppet keyframes. Log which tags
  the persona actually emits — that corpus is the future fine-tune set.
