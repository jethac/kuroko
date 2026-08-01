"""kuroko-reflex — the brainstem. The ONLY kuroko code that may run on the robot.

~100 lines, stdlib only (no pip installs on the Pi, ever). Listens for the kuroko
heartbeat over UDP; while the beat is present it does nothing at all. On loss it
executes exactly one reflex: hand the body back to the daemon's native idle behavior
so the robot goes sleepy instead of freezing mid-gesture.

This is deliberately NOT a supervisor of the brain (it cannot restart the GB10 and
should not try). It is a dead-man's switch for dignity.

Install (optional — the system works without it, it just fails less gracefully):
    scp pi/kuroko_reflex.py pollen@reachy-mini.local:
    ssh pollen@reachy-mini.local 'nohup python3 kuroko_reflex.py &'
or wire it into systemd with the unit file in this directory.
"""

import json
import logging
import socket
import time
import urllib.request

HEARTBEAT_PORT = 8043
LOSS_THRESHOLD_S = 0.5
DAEMON = "http://127.0.0.1:8000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s reflex %(message)s")
log = logging.getLogger("kuroko.reflex")


def daemon_post(path: str) -> bool:
    try:
        req = urllib.request.Request(f"{DAEMON}{path}", method="POST")
        with urllib.request.urlopen(req, timeout=2) as r:
            return 200 <= r.status < 300
    except Exception as e:
        log.warning(f"daemon call {path} failed: {e}")
        return False


def go_sleepy() -> None:
    """Cancel whatever the brain was doing; return the body to daemon idle."""
    # Best-effort, order matters: stop app-driven motion, then let the daemon's
    # own idle (face tracking, breathing) take over.
    daemon_post("/api/apps/stop-current-app")
    log.info("brain lost — body handed back to daemon idle (sleepy robot)")


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", HEARTBEAT_PORT))
    sock.settimeout(0.2)
    last_beat: float | None = None
    sleepy = False
    log.info(f"listening for kuroko heartbeat on :{HEARTBEAT_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(256)
            last_beat = time.monotonic()
            if sleepy:
                log.info(f"heartbeat back from {addr[0]} — brain is alive again")
                sleepy = False
            # Heartbeats may carry a playhead report; echo it back so the brain
            # can measure round-trip and drift (P2 frame-time sync).
            try:
                beat = json.loads(data)
                if beat.get("echo"):
                    sock.sendto(data, addr)
            except (ValueError, KeyError):
                pass
        except socket.timeout:
            pass

        if last_beat is not None and not sleepy:
            if time.monotonic() - last_beat > LOSS_THRESHOLD_S:
                sleepy = True
                go_sleepy()


if __name__ == "__main__":
    main()
