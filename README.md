# SENSE-ACT

<p align="center">
  <b>Sentiment arbitrage engine for oil markets.</b><br>
  <i>Shadow mode — no real money, real architecture.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Tests-30%2F30-brightgreen" />
  <img src="https://img.shields.io/badge/FinBERT-ProsusAI-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

A Ras Tanura pipeline engineer with 200 followers posts something on Twitter. Oil futures move 3 minutes later. Reuters publishes the same story 8 minutes after that. By then it's already priced in.

The signal was never in the follower count. It was in the information type.

---

## What It Does

Shadow-mode sentiment arbitrage engine tuned for oil/energy. Replaces standard follower-count weighting with explicit domain expertise scoring. The architecture is production-grade. The execution is virtual.

---

## Signal Flow

```
RSS/News API -> FinBERT scoring -> Semantic dedup (cosine) -> Welford z-score
-> Half-life decay -> Influence weighting -> Kill-switch check
-> Monte Carlo slippage -> Shadow book
```

---

## Key Components

| Component | What It Does |
|-----------|-------------|
| **FinBERT** | ProsusAI/finbert for financial text sentiment. Handles "maintains output despite pressure" correctly. |
| **Semantic Dedup** | Catches Reuters headline + Bloomberg paraphrase. Cosine similarity, threshold 0.82. |
| **Welford z-score** | Single-pass variance, O(1) storage. Flags anomalies in real-time. |
| **Half-life decay** | Signals lose 50% weight every 120s. Consistent with Hasbrouck (1991). |
| **Kill-switch** | Halts if spread or VIX doubles in 60 seconds. |
| **Monte Carlo slippage** | Realistic execution cost modeling. |

---

## Influence Weighting

```
weight = log10(followers) * hub_boost * domain_expertise * accuracy
```

A hub account with domain expertise gets 2.5x boost. A broadcast megaphone with 800K followers that just reposts Reuters 6 minutes late isn't worth much.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the backtest
python backtest.py

# Launch the dashboard
python dashboard.py

# Start the signal processor
python orchestrator.py
```

## Tests

```bash
python run_tests.py
# 30/30 passing
```

---

## License

MIT.
