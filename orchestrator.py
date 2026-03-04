import asyncio
import logging
import time
from typing import Optional

import feedparser
import yfinance as yf
import numpy as np

from signal_processor import SignalProcessor, InfluenceMap, SourceProfile
from shadow_core import ShadowCore, MarketSnap
from genetic_optimizer import GeneticOptimizer
from scoring import score, finbert_active


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


RSS_FEEDS = [
    ("reuters",   2_000_000, False, "https://feeds.reuters.com/reuters/businessNews"),
    ("ft",        1_500_000, False, "https://www.ft.com/rss/home"),
    ("oilprice",  500_000,   False, "https://oilprice.com/rss/main"),
]

OIL_KEYWORDS = [
    "oil", "crude", "opec", "petroleum", "energy", "barrel",
    "pipeline", "refinery", "brent", "wti", "supply", "production",
    "natural gas", "lng", "shale",
]


def _oil_related(text):
    t = text.lower()
    return any(k in t for k in OIL_KEYWORDS)


class RSSFeed:
    def __init__(self):
        self._seen = set()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def poll(self):
        import hashlib
        while True:
            for source, followers, is_hub, url in RSS_FEEDS:
                try:
                    feed = await asyncio.get_event_loop().run_in_executor(
                        None, feedparser.parse, url
                    )
                    for entry in feed.entries[:15]:
                        title   = entry.get("title", "")
                        summary = entry.get("summary", "")
                        text    = f"{title} {summary}".strip()
                        uid     = hashlib.md5(text.encode()).hexdigest()[:12]

                        if uid in self._seen or not _oil_related(text):
                            continue

                        self._seen.add(uid)
                        published = entry.get("published_parsed")
                        ts = time.mktime(published) if published else time.time()
                        await self._queue.put((source, followers, is_hub, text, ts))
                        log.info("new  %-12s  %s", source, title[:70])

                except Exception as e:
                    log.warning("rss error %s: %s", source, e)

            await asyncio.sleep(60)

    async def next(self):
        return await self._queue.get()


class MarketFeed:
    def __init__(self, ticker="XOM"):
        self._ticker = ticker
        self._cache  = None
        self._last   = 0.0

    async def next(self) -> MarketSnap:
        now = time.time()
        if now - self._last > 30:
            try:
                data = await asyncio.get_event_loop().run_in_executor(None, self._fetch)
                if data:
                    self._cache = data
                    self._last  = now
            except Exception as e:
                log.warning("market feed error: %s", e)
        await asyncio.sleep(1.0)
        if self._cache is None:
            return MarketSnap(self._ticker, time.time(), 110.0, 0.022, 0.18)
        mid, spread, vix = self._cache
        return MarketSnap(self._ticker, time.time(), mid, spread, vix)

    def _fetch(self):
        info   = yf.Ticker(self._ticker).fast_info
        mid    = float(info.last_price)
        spread = mid * 0.0002
        hist   = yf.Ticker(self._ticker).history(period="5d", interval="1h")
        vix    = 0.20
        if not hist.empty:
            rets = hist["Close"].pct_change().dropna()
            vix  = float(rets.std() * (252 * 6.5) ** 0.5)
        return mid, spread, vix


class Engine:
    def __init__(self, ticker="XOM"):
        imap = InfluenceMap()
        for src, fol, hub, _ in RSS_FEEDS:
            imap.add(SourceProfile(src, fol, hub, 0.7, 0.65))

        self.proc   = SignalProcessor(influence_map=imap)
        self.core   = ShadowCore()
        self.opt    = GeneticOptimizer()
        self._mfeed = MarketFeed(ticker)
        self._sfeed = RSSFeed()
        self.snap:  Optional[MarketSnap] = None
        self._on    = False
        self._n     = 0

    async def _market(self):
        while self._on:
            self.snap = await self._mfeed.next()

    async def _signals(self):
        while self._on:
            if self.snap is None:
                await asyncio.sleep(0.5)
                continue

            src, fol, hub, text, ts = await self._sfeed.next()
            raw = score(text)
            s   = self.proc.process(text, src, fol, raw, hub, ts)
            self._n += 1

            if s.is_duplicate:
                log.info("dup  %-14s", src)
                continue

            flag = "!" if abs(s.impact) >= 0.05 else " "
            log.info("%s %-14s  raw=%+.2f  decay=%+.3f  w=%.2f  impact=%+.4f  mode=%s",
                     flag, src, s.raw_score, s.decayed_score, s.influence, s.impact,
                     self.core.mode.value)

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

    async def _mtm(self):
        while self._on:
            await asyncio.sleep(5.0)
            if self.snap and self.core.book.open_count > 0:
                pnl = self.core.mark(self.snap.mid)
                log.info("  mtm  price=%.4f  open=%d  pnl=%+.4f  mode=%s",
                         self.snap.mid, self.core.book.open_count,
                         self.core.book.pnl, self.core.mode.value)

    async def _optimizer(self):
        while self._on:
            await asyncio.sleep(300.0)
            trades = self.core.book.closed
            if len(trades) >= 10:
                g = await self.opt.maybe_run(trades)
                log.info("  ga  thresh=%.3f  T½=%.0fs  stop=%.2f%%  sharpe=%.4f",
                         g.sent_thresh, g.half_life, g.stop_pct * 100, g.fitness)

    async def run(self):
        self._on = True
        finbert = finbert_active()
        print(f"\n=== sense-act (live) ===  finbert={'yes' if finbert else 'keywords'}\n")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._market())
            tg.create_task(self._signals())
            tg.create_task(self._mtm())
            tg.create_task(self._optimizer())
            tg.create_task(self._sfeed.poll())


if __name__ == "__main__":
    asyncio.run(Engine().run())
