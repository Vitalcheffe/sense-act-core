import sys, os, math, time, asyncio
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signal_processor import Welford, SemanticDedup, SignalProcessor, _decay, HALF_LIFE
from shadow_core import ShadowBook, ShadowCore, Side, Mode, MarketSnap
from scoring import _keyword as kw
from genetic_optimizer import GA, GeneticOptimizer, sharpe, BOUNDS

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  OK  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}  ->  {e}")
        failed += 1


print("\n=== WELFORD ===")


def t_mean():
    w = Welford()
    for v in [1, 2, 3, 4, 5]:
        w.update(v)
    assert abs(w.mean - 3.0) < 1e-9


def t_std():
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    w = Welford()
    for v in data:
        w.update(v)
    assert abs(w.std - float(np.std(data, ddof=1))) < 1e-9


def t_z():
    w = Welford()
    for _ in range(5):
        w.update(0.0)
    assert w.update(10.0) > 0


def t_single():
    assert Welford().update(5.0) == 0.0


def t_inc():
    data = [1.3, 2.7, 0.5, 3.1, 4.4, 2.2, 1.8]
    w = Welford()
    for v in data:
        w.update(v)
    assert abs(w.mean - float(np.mean(data))) < 1e-9
    assert abs(w.std - float(np.std(data, ddof=1))) < 1e-9


check("mean converges", t_mean)
check("std matches numpy", t_std)
check("zscore positive", t_z)
check("single value = 0", t_single)
check("incremental = batch", t_inc)


print("\n=== DEDUP ===")


def t_exact():
    d = SemanticDedup()
    v = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    d.check("u1", v)
    assert d.check("u1", v) is True


def t_diff():
    d = SemanticDedup()
    rng = np.random.default_rng(99)
    for i in range(5):
        v = rng.standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)
        assert d.check(f"u{i}", v) is False


def t_proc_dup():
    p = SignalProcessor()
    t = "OPEC cuts oil supply deficit bullish"
    assert p.process(t, "r", 2_000_000, 0.8).is_duplicate is False
    assert p.process(t, "r", 2_000_000, 0.8).is_duplicate is True


def t_proc_stats():
    p = SignalProcessor()
    for t in ["supply shortage bullish", "crash bearish", "nice day today"]:
        p.process(t, "src", 1000, 0.1)
    assert p.stats["total"] == 3 and p.stats["dupes"] == 0


check("exact dup blocked", t_exact)
check("different sigs pass", t_diff)
check("processor filters dup", t_proc_dup)
check("stats count correct", t_proc_stats)


print("\n=== DECAY ===")


def t_fresh():
    assert abs(_decay(time.time()) - 1.0) < 0.01


def t_half():
    assert abs(_decay(time.time() - HALF_LIFE) - 0.5) < 0.05


def t_old():
    assert _decay(time.time() - HALF_LIFE * 10) < 0.01


def t_future():
    assert abs(_decay(time.time() + 1000) - 1.0) < 1e-9


check("fresh ~= 1", t_fresh)
check("half-life = 0.5", t_half)
check("old ~= 0", t_old)
check("future clamped", t_future)


print("\n=== SHADOW CORE ===")


def t_tp():
    b = ShadowBook()
    b.open("XOM", Side.BUY, 100.0, 1.0, 0.01, 0.02, Mode.NORMAL)
    b.mark(102.5)
    assert len(b.closed) == 1 and b.closed[0]["reason"] == "TP"


def t_sl():
    b = ShadowBook()
    b.open("XOM", Side.BUY, 100.0, 1.0, 0.01, 0.02, Mode.NORMAL)
    b.mark(98.5)
    assert len(b.closed) == 1 and b.closed[0]["reason"] == "SL"


def t_crisis_qty():
    b = ShadowBook()
    assert b.open("XOM", Side.BUY, 100.0, 2.0, 0.005, 0.01, Mode.CRISIS).qty == 1.0


def t_crisis_stop():
    b = ShadowBook()
    pos = b.open("XOM", Side.BUY, 100.0, 1.0, 0.005, 0.01, Mode.CRISIS)
    assert abs(pos.sl - 99.0) < 1e-9


def t_sell():
    b = ShadowBook()
    b.open("XOM", Side.SELL, 100.0, 1.0, 0.01, 0.02, Mode.NORMAL)
    assert b.mark(98.0) > 0


def t_pnl_acc():
    b = ShadowBook()
    for _ in range(2):
        b.open("XOM", Side.BUY, 100.0, 1.0, 0.01, 0.02, Mode.NORMAL)
        b.mark(102.5)
    assert b.pnl > 0


check("BUY hits TP", t_tp)
check("BUY hits SL", t_sl)
check("crisis halves qty", t_crisis_qty)
check("crisis widens stop", t_crisis_stop)
check("SELL pnl positive", t_sell)
check("pnl accumulates", t_pnl_acc)


print("\n=== SCORING ===")


def t_bull():
    assert kw("OPEC cuts supply deficit bullish") > 0


def t_bear():
    assert kw("crash oversupply bearish decline") < 0


def t_neutral():
    assert kw("the weather is nice today") == 0.0


def t_empty():
    assert kw("") == 0.0


def t_bounded():
    assert -1 <= kw("cut shortage rally surge bullish") <= 1


check("bullish > 0", t_bull)
check("bearish < 0", t_bear)
check("neutral = 0", t_neutral)
check("empty = 0", t_empty)
check("score in [-1,1]", t_bounded)


print("\n=== GENETIC ===")


def t_bounds():
    ga = GA()
    for _ in range(20):
        g = ga.rand()
        for k, (lo, hi) in BOUNDS.items():
            assert lo <= getattr(g, k) <= hi


def t_sharpe_pos():
    trades = [{"pnl": 0.02 if i % 2 == 0 else -0.005} for i in range(40)]
    assert sharpe(trades) > 0


def t_sharpe_neg():
    trades = [{"pnl": -0.02 if i % 2 == 0 else 0.005} for i in range(40)]
    assert sharpe(trades) < 0


def t_sharpe_min():
    assert sharpe([{"pnl": 1.0}, {"pnl": 2.0}]) == float("-inf")


def t_evolve():
    ga = GA(0)
    pop = ga.pop(20)
    for _ in range(5):
        pop = ga.evolve(pop, lambda g: getattr(g, "stop_pct"))
    assert pop[0].fitness != float("-inf")


def t_opt():
    opt = GeneticOptimizer()
    trades = [{"pnl": 0.02 if i % 2 == 0 else -0.005} for i in range(30)]
    g = asyncio.run(opt.maybe_run(trades))
    assert g is not None


check("genome in bounds", t_bounds)
check("sharpe positive", t_sharpe_pos)
check("sharpe negative", t_sharpe_neg)
check("sharpe needs 3", t_sharpe_min)
check("evolve sets fitness", t_evolve)
check("optimizer returns genome", t_opt)


print(f"\n{'='*42}")
print(f"  {passed + failed} tests    OK {passed}    FAIL {failed}")
print(f"{'='*42}\n")
sys.exit(0 if failed == 0 else 1)
