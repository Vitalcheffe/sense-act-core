import asyncio
import logging
import math
import time
from typing import Optional

import numpy as np

from signal_processor import SignalProcessor, InfluenceMap, SourceProfile
from shadow_core import ShadowCore, MarketSnap
from genetic_optimizer import GeneticOptimizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def score(text):
    BULL = ["cut", "shortage", "rally", "surge", "bullish", "deficit", "strong", "increase"]
    BEAR = ["glut", "crash", "bearish", "oversupply", "drop", "weak", "flood", "sell"]
    words = [w.strip(".,!?;:") for w in text.lower().split()]
    p = sum(1 for w in words if w in BULL)
    n = sum(1 for w in words if w in BEAR)
    return (p - n) / (p + n) if (p + n) > 0 else 0.0


_FEED = [
    ("reuters",    2_000_000, False, "OPEC cuts supply deficit bullish"),
    ("aramco_eng", 200,       True,  "ras tanura pipeline shortage confirmed"),
    ("oil_trader", 3_500,     False, "crash oversupply bearish"),
    ("bot_1",      800_000,   False, "oil price up bullish rally strong"),
    ("bot_2",      750_000,   False, "oil prices rising bullish strong rally"),
    ("insider",    120,       True,  "OPEC emergency meeting production cut"),
    ("noise",      50,        False, "great weather today"),
]


class MarketFeed:
    def __init__(self):
        self._price = 110.0
        self._rng   = np.random.default_rng(42)
        self._vix   = 0.15

    async def next(self) -> MarketSnap:
        await asyncio.sleep(0.05)
        self._price *= math.exp(0.0001 + 0.0008 * float(self._rng.standard_normal()))
        self._vix    = max(0.05, self._vix * 0.99 + 0.01 * abs(float(self._rng.standard_normal())) * 0.1)
        spread       = self._price * 0.0002 * (1 + self._vix / 10)
        return MarketSnap("XOM", time.time(), self._price, spread, self._vix)


class SignalFeed:
    def __init__(self):
        self._i   = 0
        self._rng = np.random.default_rng(7)

    async def next(self):
        await asyncio.sleep(float(self._rng.uniform(0.3, 1.5)))
        src, fol, hub, text = _FEED[self._i % len(_FEED)]
        self._i += 1
        return src, fol, hub, text, time.time() - float(self._rng.uniform(0, 30))


class Engine:
    def __init__(self):
        imap = InfluenceMap()
        for src, fol, hub, _ in _FEED:
            imap.add(SourceProfile(src, fol, hub, 0.9 if hub else 0.4, 0.8 if hub else 0.5))

        self.proc   = SignalProcessor(influence_map=imap)
        self.core   = ShadowCore()
        self.opt    = GeneticOptimizer()
        self._mfeed = MarketFeed()
        self._sfeed = SignalFeed()
        self.snap: Optional[MarketSnap] = None
        self._on  = False
        self._n   = 0

    async def _market(self):
        while self._on:
            self.snap = await self._mfeed.next()

    async def _signals(self, limit):
        done = 0
        while self._on and done < limit:
            if self.snap is None:
                await asyncio.sleep(0.05)
                continue

            src, fol, hub, text, ts = await self._sfeed.next()
            s = self.proc.process(text, src, fol, score(text), hub, ts)
            self._n += 1
            done    += 1

            if s.is_duplicate:
                log.info("dup  %-18s", src)
                continue

            flag = "!" if abs(s.impact) >= 0.05 else " "
            log.info("%s %-18s  raw=%+.2f  decay=%+.3f  w=%.2f  impact=%+.4f",
                     flag, src, s.raw_score, s.decayed_score, s.influence, s.impact)

            params = self.opt.params
            if abs(s.impact) > params.sent_thresh:
                pos = self.core.submit(
                    s.uid, "XOM", s.impact, self.snap,
                    stop_pct=params.stop_pct,
                    target_pct=params.target_pct,
                )
                if pos:
                    log.info("  -> [%s] %s @%.4f  sl=%.4f  tp=%.4f",
                             pos.pos_id, pos.side.value, pos.entry, pos.sl, pos.tp)

        self._on = False

    async def _mtm(self):
        while self._on:
            await asyncio.sleep(1.0)
            if self.snap and self.core.book.open_count > 0:
                pnl = self.core.mark(self.snap.mid)
                log.info("  mtm  price=%.4f  open=%d  pnl=%+.4f",
                         self.snap.mid, self.core.book.open_count, self.core.book.pnl)

    async def _optimizer(self):
        while self._on:
            await asyncio.sleep(20.0)
            trades = self.core.book.closed
            if trades:
                await self.opt.maybe_run(trades)

    async def run(self, n=12):
        self._on = True
        print("\n=== sense-act ===\n")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._market())
            tg.create_task(self._signals(n))
            tg.create_task(self._mtm())
            tg.create_task(self._optimizer())
        b = self.core.book
        g = self.opt.params
        print(f"\nsignals={self._n}  trades={len(b.closed)}  pnl={b.pnl:+.4f}")
        print(f"ga  thresh={g.sent_thresh:.3f}  T½={g.half_life:.0f}s  stop={g.stop_pct*100:.2f}%\n")


if __name__ == "__main__":
    asyncio.run(Engine().run(n=12))
