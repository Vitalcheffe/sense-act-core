import asyncio
import copy
import math
import random
import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Genome:
    sent_thresh:   float = 0.10
    exit_timer:    float = 300.0
    half_life:     float = 120.0
    stop_pct:      float = 0.005
    target_pct:    float = 0.015
    cosine_thresh: float = 0.950
    infl_mult:     float = 1.0
    fitness:       float = float("-inf")


BOUNDS = {
    "sent_thresh":   (0.02, 0.50),
    "exit_timer":    (30.0, 3600.0),
    "half_life":     (30.0, 600.0),
    "stop_pct":      (0.001, 0.030),
    "target_pct":    (0.003, 0.060),
    "cosine_thresh": (0.80, 0.99),
    "infl_mult":     (0.5, 3.0),
}
KEYS = list(BOUNDS.keys())


def sharpe(trades):
    pnl = [t["pnl"] for t in trades if "pnl" in t]
    if len(pnl) < 3:
        return float("-inf")
    a  = np.array(pnl)
    sd = a.std(ddof=1)
    if sd < 1e-10:
        return 0.0
    return float(a.mean() / sd * math.sqrt(len(pnl) * 252))


class GA:
    def __init__(self, seed=42):
        self._r = random.Random(seed)

    def rand(self):
        g = Genome()
        for k, (lo, hi) in BOUNDS.items():
            setattr(g, k, self._r.uniform(lo, hi))
        return g

    def pop(self, n=40):
        return [self.rand() for _ in range(n)]

    def select(self, population):
        return max(self._r.sample(population, 3), key=lambda g: g.fitness)

    def cross(self, a, b):
        ca, cb = copy.copy(a), copy.copy(b)
        ca.fitness = cb.fitness = float("-inf")
        if self._r.random() > 0.7:
            return ca, cb
        for k in KEYS:
            if self._r.random() < 0.5:
                setattr(ca, k, getattr(b, k))
                setattr(cb, k, getattr(a, k))
        return ca, cb

    def mutate(self, g):
        g2 = copy.copy(g)
        g2.fitness = float("-inf")
        for k in KEYS:
            if self._r.random() < 0.2:
                lo, hi = BOUNDS[k]
                v = getattr(g2, k) + self._r.gauss(0, (hi - lo) * 0.1)
                setattr(g2, k, max(lo, min(hi, v)))
        return g2

    def evolve(self, population, fit_fn):
        for g in population:
            if g.fitness == float("-inf"):
                g.fitness = fit_fn(g)
        population.sort(key=lambda g: g.fitness, reverse=True)
        elite = population[:max(1, len(population) // 10)]
        next_gen = list(elite)
        while len(next_gen) < len(population):
            ca, cb = self.cross(self.select(population), self.select(population))
            next_gen.append(self.mutate(ca))
            if len(next_gen) < len(population):
                next_gen.append(self.mutate(cb))
        return next_gen


class GeneticOptimizer:
    def __init__(self):
        self._ga   = GA()
        self._best = Genome()
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def maybe_run(self, trades: List[dict]) -> Genome:
        async with self._lock:
            if len(trades) < 10:
                return self._best
            if time.time() - self._last < 3600 and len(trades) < 50:
                return self._best

            folds = min(5, len(trades) // 5)
            fold  = len(trades) // folds
            best_oos = float("-inf")
            best_g   = self._ga.rand()

            for i in range(folds - 1):
                is_t  = trades[i * fold:(i + 2) * fold]
                oos_t = trades[(i + 2) * fold:(i + 3) * fold]
                if not is_t or not oos_t:
                    continue
                population = self._ga.pop()
                for _ in range(20):
                    population = self._ga.evolve(population, lambda g: sharpe(is_t))
                oos = sharpe(oos_t)
                if oos > best_oos:
                    best_oos = oos
                    best_g   = copy.copy(population[0])
                    best_g.fitness = oos

            self._best = best_g
            self._last = time.time()
            return self._best

    @property
    def params(self):
        return self._best
