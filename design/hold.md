# Hold: giving PersonaPlex a knowledge back-end without KAME

**Status:** design sketch, not built.

## The problem

PersonaPlex is a delightful conversationalist and knows almost nothing. It is a 7B
speech-to-speech model whose job is timing and prosody, not facts. The obvious fix is a
large model behind it — and on this box there is room: PersonaPlex occupies ~16 GiB of
128 GB, leaving ~100 GB for a resident MoE plus KV cache.

[Sakana's KAME](https://github.com/SakanaAI/kame) solves this properly, with a fourth
"oracle" stream that injects LLM knowledge *while the S2S model is speaking*, so it can
revise mid-sentence. We are not doing that, for two reasons:

1. The front-end must be **trained** to consume the oracle stream (their
   `kame_finetune`, Simulated Oracle Augmentation). PersonaPlex-7B has no such stream.
2. KAME's back-end was `gpt-4.1-nano` — small, and remote. Ours is a large **local**
   model on a bandwidth-bound machine. Running it concurrently is precisely the
   contention that breaks the 80 ms frame budget (see below).

So instead of speaking while thinking, we **stop speaking, think, and come back** —
and we make the pause an explicit, motivated part of the interaction rather than a
stall. The robot says it is looking something up, plays hold music, and returns with
an answer.

## Why suspension is free (the load-bearing property)

The serve loop only advances the model when audio frames arrive. Stop sending frames
and PersonaPlex does not idle, poll, or age — it **freezes**. Verified twice already:
a warm-but-unfed session holds `rx_bytes` flat indefinitely, which is what makes
pre-warming work.

The consequence for hold is stronger than it first appears. Because frame arrival *is*
the model's clock, a suspended session experiences **zero elapsed time**. On resume it
does not need context reconstruction, a prompt reload, or an "as I was saying" — it
genuinely does not know it was gone. A 15-second hold costs nothing in coherence.

Meanwhile the MoE gets the entire 273 GB/s to itself.

### Why they must not overlap

With `--w8a16` PersonaPlex streams ~6.5 GB of weights per frame; at 46.5 ms that is
~140 GB/s, over half the machine's peak. `lm_step` (49 of the 53.5 ms measured in
container) is bandwidth-bound, so contention scales it near-linearly. An MoE taking a
third of the bandwidth pushes a frame past 70 ms, and budget misses do not degrade
gracefully — they desynchronise the model's clock and the conversation becomes
incoherent, not merely slow. Serialisation is a correctness requirement.

## States

```
                  ┌──────────────────────────────────────────┐
                  │                                          │
   DORMANT ──wake──► ACTIVE ──trigger phrase──► HOLD ──answer─┘
      ▲               │  ▲                       │
      └──dismiss/idle─┘  └───────────────────────┘
                         (session frozen throughout HOLD)
```

`HOLD` is a sub-state of a live conversation: the websocket stays open, the model stays
loaded and frozen, only the pacer stops.

## Flow

1. **Trigger.** The persona is prompted to say a fixed phrase when it needs facts:
   *"let me look that up"*, followed by its own restatement of the question. Kuroko
   already parses the model's text stream (`\x02` frames), so detection needs no ASR
   and — usefully — the restatement gives us the **query in text form for free**.
   The announcement also motivates the pause: the robot said it was going to do this.

2. **Enter hold.** Wait for the model to finish its utterance (`last_activity` already
   tracks this), then `streaming.clear()`. The pacer stops; the model freezes mid-
   conversation with full context.

3. **Cover.** Start hold audio on the robot's speaker via `push_audio_sample`. This
   path is entirely separate from the model's input, so anything played here is heard
   by the human and *not* by PersonaPlex. Embodiment shifts to a thinking posture —
   gaze off-axis, slow antenna drift — because a robot that stares at you while silent
   reads as frozen, and one that looks away reads as thinking.

4. **Think.** Dispatch the query to the resident MoE. It now has the full machine.

5. **Inject.** TTS the answer and write it into `_mic_buf` — the model's *input* —
   while live mic audio is held back. The human hears only hold music; PersonaPlex
   hears what sounds like someone telling it the answer. This is a poor-man's oracle
   stream: no training required, at the cost of KAME's mid-sentence refinement.

6. **Resume.** Set `streaming`, fade out the hold audio, release live mic. The model
   responds to what it just "heard" and paraphrases it in its own voice. Anything the
   user said during hold is sitting in the ring buffer and arrives as pre-roll — the
   same mechanism that already covers wake latency.

## The injection trick, stated plainly

Kuroko owns two independent audio paths:

| path | written by | heard by |
|---|---|---|
| `push_audio_sample` → robot speaker | hold music, model speech | the human |
| `_mic_buf` → pacer → websocket | live mic, injected TTS | the model |

Nothing requires these to carry the same audio. Hold music goes only to the human;
injected answers go only to the model. Neither is aware of the other's channel.

## Budget

| stage | estimate |
|---|---|
| trigger detection | ~0 (already in the text stream) |
| finish current utterance | 0–2 s |
| MoE prefill + generation | **5–15 s** (dominant) |
| TTS of the answer | ~1 s |
| resume + response onset | <0.5 s |

The MoE dominates and sets the hold length. At ~273 GB/s with a 4-bit MoE streaming
~10 GB of active experts per token, expect tens of tokens/second — so a 200-token
answer is ~7 s before prefill. **This is why hold music is not a gimmick**: the pause
is real and needs covering.

Worth considering if holds feel long:
- stream the MoE output and inject the first sentence early, letting the model start
  talking while the rest arrives (a cheap step toward KAME without retraining)
- a second, smaller resident model for easy questions, reserving the big MoE
- a "still looking" beat at ~8 s so the hold has structure

## Sizing

| component | budget |
|---|---|
| PersonaPlex weights (`--w8a16`) | ~8 GB resident |
| MoE, ~120B class at 4-bit | 60–70 GB |
| KV cache | 30–40 GB |
| headroom / OS | remainder of 128 GB |

Keep the MoE **resident**. PersonaPlex alone takes ~135 s to load 16 GiB; a 60 GB model
loaded per query would make the hold unbearable and the design pointless.

## Failure modes

- **Trigger never fires.** The persona ignores the instruction and confabulates
  instead. Mitigation: prompt discipline, plus a fallback classifier over the model's
  text. Accept that some questions get a confident wrong answer — that is the status
  quo today.
- **Trigger fires constantly.** Everything becomes a hold and the conversation is all
  music. Mitigation: rate-limit, and prompt to only look up specifics.
- **MoE is slow or hangs.** Hold has a hard timeout; on expiry, resume and inject "I
  couldn't find that" so the robot recovers conversationally rather than hanging.
- **Model reacts oddly to the injection** ("who said that?"). Mitigation: frame it in
  the system prompt — the persona has a research assistant who tells it things.
- **Suspension lands mid-utterance.** Resume clips mid-word. Mitigation: the
  finish-utterance wait in step 2.
- **User talks during hold and expects a response.** They are recorded and delivered
  as pre-roll on resume, so nothing is lost, but the reply is late by the hold length.

## Build order

1. **Suspend/resume as a primitive.** `streaming.clear()` / re-`set()` with a
   finish-utterance wait. Verify the model resumes coherently after 15 s — this is the
   whole premise and is one afternoon's work to confirm or kill.
2. **Hold audio.** Play a loop through `push_audio_sample`, confirm the model neither
   hears it nor is disturbed by it. *(Use royalty-free or generated audio — the actual
   Jeopardy think music is very much under copyright.)*
3. **Injection.** TTS a canned sentence into `_mic_buf` and confirm PersonaPlex
   paraphrases it naturally rather than repeating it verbatim or getting confused.
4. **Trigger.** Prompt for the phrase, parse it out of the text stream, capture the
   restated query.
5. **MoE.** Stand it up resident, behind a lock that makes overlap with an unfrozen
   PersonaPlex structurally impossible.
6. **Thinking posture** in `body.py`, and the "still looking" beat.

Steps 1–3 are independently testable against today's stack and answer the risky
questions first. If step 1 or 3 disappoints, the design changes shape before any MoE
has been downloaded.
