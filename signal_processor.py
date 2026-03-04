import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


COSINE_THRESH = 0.95
Z_ALERT = 2.5
HALF_LIFE = 120.0
EMBED_DIM = 128


@dataclass
class SourceProfile:
    sid: str
    followers: int
    is_hub: bool
    domain: float
    accuracy: float


@dataclass
class Signal:
    uid: str
    source: str
    followers: int
    text: str
    raw_score: float
    decayed_score: float
    z_score: float
    influence: float
    impact: float
    is_duplicate: bool
    timestamp: float


class Welford:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._M2 += d * (x - self.mean)
        if self.n < 2:
            return 0.0
        var = self._M2 / (self.n - 1)
        return (x - self.mean) / (math.sqrt(var) if var > 1e-10 else 1e-9)

    @property
    def std(self):
        return math.sqrt(self._M2 / (self.n - 1)) if self.n >= 2 else 0.0


_LAM = math.log(2) / HALF_LIFE


def _decay(ts):
    return math.exp(-_LAM * max(0.0, time.time() - ts))


def _embed(text):
    seed = int(abs(hash(text)) % (2**31))
    v = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class InfluenceMap:
    def __init__(self):
        self._db = {}

    def add(self, p: SourceProfile):
        self._db[p.sid] = p

    def score(self, sid, followers, is_hub):
        p = self._db.get(sid)
        f   = p.followers if p else followers
        hub = p.is_hub    if p else is_hub
        dom = p.domain    if p else 0.5
        acc = p.accuracy  if p else 0.5
        base = math.log10(f + 1) / math.log10(10_000_001)
        return min(base * (2.5 if hub else 1.0) * dom * acc / 2.5, 1.0)


class SemanticDedup:
    def __init__(self):
        self._buf = deque(maxlen=500)
        self._seen = set()

    def check(self, uid, vec):
        if uid in self._seen:
            return True
        if self._buf:
            if float((np.stack(list(self._buf)) @ vec).max()) >= COSINE_THRESH:
                return True
        self._buf.append(vec)
        self._seen.add(uid)
        return False


class SignalProcessor:
    def __init__(self, influence_map=None):
        self._dedup = SemanticDedup()
        self._welford = Welford()
        self._imap = influence_map or InfluenceMap()
        self._n_total = 0
        self._n_dupes = 0
        self._n_regimes = 0
        self._n_anomalies = 0

    def process(self, text, source, followers, score, is_hub=False, ts=None):
        t = ts or time.time()
        uid = hashlib.md5(text.encode()).hexdigest()[:8]
        vec = _embed(text)
        is_dup = self._dedup.check(uid, vec)
        self._n_total += 1

        if is_dup:
            self._n_dupes += 1
            return Signal(uid=uid, source=source, followers=followers, text=text,
                          raw_score=score, decayed_score=0.0, z_score=0.0,
                          influence=0.0, impact=0.0, is_duplicate=True, timestamp=t)

        z = self._welford.update(score)
        df = _decay(t)
        dec = score * df
        w = self._imap.score(source, followers, is_hub)
        impact = dec * w

        if abs(z) > Z_ALERT:
            self._n_anomalies += 1
        if abs(impact) >= 0.05:
            self._n_regimes += 1

        return Signal(uid=uid, source=source, followers=followers, text=text,
                      raw_score=score, decayed_score=dec, z_score=z,
                      influence=w, impact=impact, is_duplicate=False, timestamp=t)

    @property
    def stats(self):
        return {
            "total":     self._n_total,
            "dupes":     self._n_dupes,
            "regimes":   self._n_regimes,
            "anomalies": self._n_anomalies,
            "w_mean":    round(self._welford.mean, 4),
            "w_std":     round(self._welford.std, 4),
        }
