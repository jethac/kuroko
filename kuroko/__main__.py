"""Entry point: python -m kuroko"""

import asyncio
import logging

from .bridge import VoiceBridge
from .config import KurokoConfig
from .puppet import PuppetTrack


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = KurokoConfig.load()
    bridge = VoiceBridge(cfg, puppet=PuppetTrack())
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
