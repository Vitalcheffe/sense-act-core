import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class Mode(Enum):
    NORMAL = "NORMAL"
    CRISIS = "CRISIS"
    HALTED = "HALTED"


@dataclass
class MarketSnap:
    asset:  str
    ts:     float
    mid:    float
    spread: float
    vix:    float


@dataclass
class Position:
    pos_id:    str
    asset:     str
    side:      Side
    entry:     float
    qty:       float
    sl:        float
    tp:        float
    opened_at: float
    pnl:       float = 0.0


class KillSwitch:
    def __init__(self):
        self._spreads = deque(maxlen=300)
        self._vix     = deque(maxlen=300)
        self._mode    = Mode.NORMAL

    def check(self, snap, impact):
        self._spreads.append((snap.ts, snap.spread))
        self._vix.append((snap.ts, snap.vix))
        cutoff = snap.ts - 60.0

        for ts, v in self._spreads:
            if ts >= cutoff:
                if snap.spread >= v * 2.0:
                    self._mode = Mode.HALTED
                    return self._mode
                break

        for ts, v in self._vix:
            if ts >= cutoff:
                if snap.vix >= v * 2.0:
                    self._mode = Mode.HALTED
                    return self._mode
                break

        if abs(impact) >= 0.05:
            self._mode = Mode.CRISIS
        elif self._mode == Mode.CRISIS:
            self._mode = Mode.NORMAL

        return self._mode

    @property
    def mode(self):
        return self._mode


class JitterBuffer:
    def __init__(self):
        self._rng  = np.random.default_rng(0)
        self._tick = 0.01

    def fill(self, price, side) -> Tuple[float, float, bool]:
        lat    = float(max(0.0, self._rng.normal(12.0, 8.0)))
        beaten = bool(self._rng.random() < 0.35)
        adj    = 2 * self._tick if beaten else 0.0
        fill   = price + adj if side == Side.BUY else price - adj
        return fill, lat, beaten


class MCSlippage:
    def __init__(self):
        self._rng = np.random.default_rng(1)

    def p99(self, mid, spread, side):
        if mid <= 0 or spread <= 0:
            return 0.0
        sig  = (spread / mid) * math.sqrt(252 * 23400)
        dt   = 0.05 / (252 * 23400)
        Z    = self._rng.standard_normal(1000)
        ends = mid * np.exp((-0.5 * sig**2 * dt) + sig * math.sqrt(dt) * Z)
        slips = np.maximum(ends - mid, 0) if side == Side.BUY else np.maximum(mid - ends, 0)
        return float(np.percentile(slips, 99))


class ShadowBook:
    def __init__(self):
        self._open:   Dict[str, Position] = {}
        self._closed: List[dict]          = []
        self._pnl = 0.0

    def open(self, asset, side, fill, qty, stop_pct, target_pct, mode):
        if mode == Mode.CRISIS:
            qty      *= 0.5
            stop_pct *= 2.0
        sl = fill * (1 - stop_pct) if side == Side.BUY else fill * (1 + stop_pct)
        tp = fill * (1 + target_pct) if side == Side.BUY else fill * (1 - target_pct)
        pos = Position(str(uuid.uuid4())[:8], asset, side, fill, qty, sl, tp, time.time())
        self._open[pos.pos_id] = pos
        return pos

    def mark(self, price):
        to_close = []
        total    = 0.0
        for pid, pos in self._open.items():
            pnl = (price - pos.entry) * pos.qty if pos.side == Side.BUY \
                  else (pos.entry - price) * pos.qty
            pos.pnl = pnl
            total  += pnl
            if (pos.side == Side.BUY  and price <= pos.sl) or \
               (pos.side == Side.SELL and price >= pos.sl):
                to_close.append((pid, pnl, "SL"))
            elif (pos.side == Side.BUY  and price >= pos.tp) or \
                 (pos.side == Side.SELL and price <= pos.tp):
                to_close.append((pid, pnl, "TP"))
        for pid, pnl, r in to_close:
            self._close(pid, pnl, r)
        return total

    def _close(self, pid, pnl, reason):
        pos = self._open.pop(pid, None)
        if not pos:
            return
        self._closed.append({"pid": pid, "pnl": pnl, "reason": reason, "side": pos.side.value})
        self._pnl += pnl

    @property
    def pnl(self):        return self._pnl
    @property
    def open_count(self): return len(self._open)
    @property
    def closed(self):     return list(self._closed)


class ShadowCore:
    def __init__(self):
        self._ks     = KillSwitch()
        self._jitter = JitterBuffer()
        self._mc     = MCSlippage()
        self._book   = ShadowBook()

    def submit(self, uid, asset, impact, snap, qty=1.0,
               stop_pct=0.005, target_pct=0.015) -> Optional[Position]:
        mode = self._ks.check(snap, impact)
        if mode == Mode.HALTED or abs(impact) < 1e-6:
            return None
        side = Side.BUY if impact > 0 else Side.SELL
        slip = self._mc.p99(snap.mid, snap.spread, side)
        fill, lat, beaten = self._jitter.fill(snap.mid, side)
        fill = fill + slip if side == Side.BUY else fill - slip
        return self._book.open(asset, side, fill, qty, stop_pct, target_pct, mode)

    def mark(self, price):       return self._book.mark(price)
    @property
    def book(self):              return self._book
    @property
    def mode(self):              return self._ks.mode
