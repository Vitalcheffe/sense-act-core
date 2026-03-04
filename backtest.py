import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from signal_processor import SignalProcessor, InfluenceMap, SourceProfile
from shadow_core import ShadowCore, MarketSnap
from genetic_optimizer import sharpe
from scoring import score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


@dataclass
class BacktestResult:
    ticker:       str
    period:       str
    n_signals:    int
    n_trades:     int
    n_wins:       int
    n_losses:     int
    total_pnl:    float
    sharpe_ratio: float
    max_drawdown: float
    win_rate:     float
    avg_win:      float
    avg_loss:     float
    best_trade:   float
    worst_trade:  float
    equity_curve: List[float] = field(default_factory=list)
    daily_pnl:    List[float] = field(default_factory=list)


_BULL_HEADLINES = [
    "OPEC+ agrees to production cut amid supply concerns",
    "oil supply disruption forces prices higher",
    "pipeline outage reduces crude output significantly",
    "OPEC emergency meeting confirms output reduction",
    "refinery capacity constrained supply deficit expected",
    "geopolitical tensions threaten oil supply routes",
    "cold weather drives energy demand surge",
    "inventory drawdown signals tightening market",
    "Saudi Arabia cuts output further to support prices",
    "oil exports fall as sanctions tighten supply",
]
_BEAR_HEADLINES = [
    "OPEC members exceed production quotas oversupply risk",
    "US shale output hits record high inventory builds",
    "demand outlook weakens amid economic slowdown",
    "oil surplus expected as production ramps up",
    "energy demand falls on mild winter temperatures",
    "strategic reserves released to cool prices",
    "recession fears dampen crude oil demand forecast",
    "OPEC+ eases cuts supply flood expected",
    "global oil demand growth slows significantly",
    "inventory builds for third consecutive week",
]
_NEUTRAL_HEADLINES = [
    "oil markets steady ahead of inventory data",
    "traders await OPEC decision on output levels",
    "energy sector mixed as dollar strengthens",
    "crude prices flat on balanced supply demand",
]
_SOURCES = [
    ("reuters",    2_000_000, False),
    ("ft_energy",  1_500_000, False),
    ("oilprice",   500_000,   False),
    ("aramco_eng", 200,       True),
    ("insider",    120,       True),
]


def _generate_news(timestamps, seed=42) -> List[Tuple]:
    rng = random.Random(seed)
    events = []
    for i in range(0, len(timestamps), rng.randint(6, 20)):
        ts = float(timestamps[i])
        src, fol, hub = rng.choice(_SOURCES)
        r = rng.random()
        if r < 0.35:
            text = rng.choice(_BULL_HEADLINES)
        elif r < 0.70:
            text = rng.choice(_BEAR_HEADLINES)
        else:
            text = rng.choice(_NEUTRAL_HEADLINES)
        events.append((ts, src, fol, hub, text))
    return events


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    arr = np.array(equity)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / (np.abs(peak) + 1e-9)
    return float(dd.min())


def run_backtest(ticker="XOM", period="2y") -> BacktestResult:
    import yfinance as yf

    log.info("fetching %s (%s)...", ticker, period)
    hist = yf.Ticker(ticker).history(period=period, interval="1h")
    if hist.empty:
        raise ValueError(f"no data for {ticker}")
    log.info("loaded %d bars", len(hist))

    timestamps = [int(ts.timestamp()) for ts in hist.index]
    closes = {int(ts.timestamp()): float(row["Close"]) for ts, row in hist.iterrows()}
    times  = sorted(closes.keys())

    def price_at(ts):
        idx = min(np.searchsorted(times, ts), len(times) - 1)
        return closes[times[idx]]

    news = _generate_news(timestamps)
    news.sort(key=lambda x: x[0])
    log.info("generated %d news events", len(news))

    imap = InfluenceMap()
    imap.add(SourceProfile("reuters",    2_000_000, False, 0.70, 0.65))
    imap.add(SourceProfile("ft_energy",  1_500_000, False, 0.70, 0.65))
    imap.add(SourceProfile("oilprice",   500_000,   False, 0.65, 0.60))
    imap.add(SourceProfile("aramco_eng", 200,       True,  0.95, 0.88))
    imap.add(SourceProfile("insider",    120,       True,  0.92, 0.85))

    proc = SignalProcessor(influence_map=imap)
    core = ShadowCore()

    equity    = [0.0]
    daily_pnl = []
    prev_day  = None
    day_start = 0.0
    n_signals = 0

    for ts, src, fol, hub, text in news:
        px   = price_at(int(ts))
        snap = MarketSnap(ticker, ts, px, px * 0.0002, 0.15)
        s    = proc.process(text, src, fol, score(text), hub, ts)
        n_signals += 1

        if s.is_duplicate or abs(s.impact) < 0.02:
            continue

        core.submit(s.uid, ticker, s.impact, snap)
        core.mark(px)
        equity.append(core.book.pnl)

        day = int(ts) // 86400
        if prev_day is None:
            prev_day = day
        if day != prev_day:
            daily_pnl.append(core.book.pnl - day_start)
            day_start = core.book.pnl
            prev_day = day

    final_px = float(hist["Close"].iloc[-1])
    core.mark(final_px)
    equity.append(core.book.pnl)

    trades  = core.book.closed
    wins    = [t for t in trades if t["pnl"] > 0]
    losses  = [t for t in trades if t["pnl"] <= 0]
    all_pnl = [t["pnl"] for t in trades]

    result = BacktestResult(
        ticker=ticker, period=period,
        n_signals=n_signals,
        n_trades=len(trades),
        n_wins=len(wins),
        n_losses=len(losses),
        total_pnl=core.book.pnl,
        sharpe_ratio=sharpe([{"pnl": p} for p in all_pnl]) if len(all_pnl) >= 3 else 0.0,
        max_drawdown=_max_drawdown(equity),
        win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
        avg_win=float(np.mean([t["pnl"] for t in wins])) if wins else 0.0,
        avg_loss=float(np.mean([t["pnl"] for t in losses])) if losses else 0.0,
        best_trade=max(all_pnl) if all_pnl else 0.0,
        worst_trade=min(all_pnl) if all_pnl else 0.0,
        equity_curve=equity,
        daily_pnl=daily_pnl,
    )

    _print(result)
    return result


def _print(r: BacktestResult):
    print(f"\n{'='*52}")
    print(f"  BACKTEST  {r.ticker}  ({r.period})")
    print(f"{'='*52}")
    print(f"  signals       {r.n_signals}")
    print(f"  trades        {r.n_trades}")
    print(f"  wins/losses   {r.n_wins} / {r.n_losses}")
    print(f"  win rate      {r.win_rate:.1f}%")
    print(f"  total pnl     {r.total_pnl:+.4f}")
    print(f"  sharpe        {r.sharpe_ratio:.4f}")
    print(f"  max drawdown  {r.max_drawdown:.2%}")
    print(f"  avg win       {r.avg_win:+.4f}")
    print(f"  avg loss      {r.avg_loss:+.4f}")
    print(f"  best trade    {r.best_trade:+.4f}")
    print(f"  worst trade   {r.worst_trade:+.4f}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    run_backtest("XOM", "2y")
