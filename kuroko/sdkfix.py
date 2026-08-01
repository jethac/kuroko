"""Workarounds for reachy_mini SDK quirks on the remote (webrtc) media path.

SDK 1.9.0 race: GstWebRTCClient builds its audio *send* chain from the
incoming-audio ``pad-added`` callback. Over a remote connection that callback
can fire before the transceiver is configured SENDRECV, so the setup finds no
OPUS sink pad on webrtcbin, logs "audio send disabled", and never retries —
push_audio_sample() then drops every buffer with "AppSrc is not initialized".

Observed from a GB10 sidecar against daemon/SDK 1.9.0 (see kuroko README).
The fix is simply to retry the (idempotent-on-failure) setup once negotiation
has settled and the pad exists.
"""

import logging
import time

log = logging.getLogger("kuroko.sdkfix")


def harden_sdk() -> None:
    """Neutralize ReachyMini.__del__.

    SDK 1.9.0's destructor calls a synchronous websocket close() that can hang
    forever joining the socket's send thread (observed via py-spy: gc collected
    an instance ~6s into a session and froze the entire process in
    ws_client.disconnect). A long-lived bridge manages its connection lifecycle
    explicitly; a best-effort destructor that can deadlock is strictly worse
    than none. Call once before constructing ReachyMini.
    """
    from reachy_mini.reachy_mini import ReachyMini

    if getattr(ReachyMini, "_kuroko_del_neutered", False):
        return

    def _safe_del(self) -> None:  # noqa: ANN001
        log.debug("ReachyMini.__del__ suppressed (kuroko manages lifecycle)")

    ReachyMini.__del__ = _safe_del
    ReachyMini._kuroko_del_neutered = True
    log.info("SDK hardened: ReachyMini.__del__ neutered")


def ensure_audio_send_ready(media, timeout_s: float = 15.0,
                            retry_every_s: float = 1.0) -> bool:
    """Return True once the webrtc audio send chain is usable.

    No-op (True) on non-webrtc backends. Safe to call repeatedly.
    """
    audio = getattr(media, "audio", None)
    if audio is None or not hasattr(audio, "_audio_send_ready"):
        return True  # local/gstreamer backend: nothing to fix

    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        if getattr(audio, "_appsrc", None) is not None:
            if attempt:
                log.info("audio send chain ready after %d retry(ies)", attempt)
            return True
        attempt += 1
        try:
            # On failure paths the SDK leaves _audio_send_ready False, so the
            # method's re-entry guard lets us call it again.
            audio._setup_audio_send_chain()
        except Exception as e:  # noqa: BLE001 — best-effort retry loop
            log.warning("send-chain retry %d failed: %s", attempt, e)
        time.sleep(retry_every_s)

    log.error("audio send chain never became ready (%.0fs)", timeout_s)
    return False
