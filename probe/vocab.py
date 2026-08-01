"""Check which wake/sleep words the offline recognizer can actually hear.

A grammar-constrained recognizer silently ignores words outside its
vocabulary, so a wake phrase containing an unknown word can never fire — it
just never matches, with no error at runtime. "reachy" is exactly such a word
in the small English model. Run this before choosing phrases.

Usage:
    python -m probe.vocab hey reachy mini ritchie robot buddy
"""

import json
import os
import sys
import tempfile


def main() -> None:
    words = [w.lower() for w in (sys.argv[1:] or [
        "hey", "okay", "reachy", "reach", "ritchie", "richie", "mini", "robot",
        "buddy", "computer", "friend", "hello", "wake", "up", "go", "to",
        "sleep", "goodbye", "that", "is", "all", "never", "mind", "peachy",
        "teacher", "beachy", "reaching",
    ])]

    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(0)  # we WANT the vocabulary warnings

    # capture vosk's C-level stderr
    err_fd = os.dup(2)
    tmp = tempfile.TemporaryFile()
    os.dup2(tmp.fileno(), 2)
    try:
        model = Model(os.environ.get("VOSK_MODEL", "/models/vosk"))
        KaldiRecognizer(model, 16000, json.dumps(words + ["[unk]"]))
    finally:
        os.dup2(err_fd, 2)
        tmp.seek(0)
        log = tmp.read().decode(errors="replace")

    missing = set()
    for line in log.splitlines():
        if "missing in vocabulary" in line:
            missing.add(line.split("'")[-2].lower())

    ok = [w for w in words if w not in missing]
    print("USABLE :", " ".join(ok))
    print("MISSING:", " ".join(sorted(missing)) or "(none)")
    print("\nPick wake phrases built only from USABLE words.")


if __name__ == "__main__":
    main()
