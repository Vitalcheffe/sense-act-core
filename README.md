<h1 align="center">SENSE-ACT</h1>
<p align="center">Sentiment arbitrage engine for oil markets.<br/>Shadow mode — no real money, real architecture.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Tests-30%2F30-brightgreen" />
  <img src="https://img.shields.io/badge/FinBERT-ProsusAI-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## The insight

A Ras Tanura pipeline engineer with 200 followers posts something on Twitter. Oil futures move 3 minutes later. Reuters publishes the same story 8 minutes after that. By then it's already priced in.

The signal was never in the follower count. It was in the information type.

## What this does

Shadow-mode sentiment arbitrage engine tuned for oil/energy. Replaces standard follower-count weighting with explicit domain expertise scoring. The architecture is production-grade. The execution is virtual.

## How signals flow

```
RSS/News API → FinBERT scoring → Semantic dedup (cosine) → Welford z-score
→ Half-life decay → Influence weighting → Kill-switch check
→ Monte Carlo slippage → Shadow book
```

### FinBERT sentiment

Uses ProsusAI/finbert — a BERT model fine-tuned on financial text. Handles "maintains output despite pressure" correctly where keyword matching fails. Falls back to keyword scoring if the model isn't available.

### Semantic deduplication

Catches the Reuters headline and the Bloomberg paraphrase of the same OPEC announcement. Cosine similarity on sentence embeddings, threshold at 0.82.

### Welford z-score

Single-pass variance computation, O(1) storage. Flags statistical anomalies in real-time without storing history.

### Half-life decay

Signals lose 50% weight every 120s (default). Consistent with Hasbrouck (1991) on price impact absorption.

### Influence weight

```
weight = log₁₀(followers) × hub_boost × domain_expertise × accuracy
```

A hub account with domain expertise gets 2.5x boost. A broadcast megaphone with 800K followers that just reposts Reuters 6 minutes late isn't worth much.

### Kill-switch

Halts if spread or VIX doubles in 60 seconds. Crisis mode halves position size and widens stops.

### Monte Carlo slippage

P99 adverse fill over 50ms using 1000 GBM paths. Because signal generation is useless if you can't estimate how your order hits the book.

## Genetic optimizer

Tunes 7 parameters (sentiment threshold, half-life, stop/target %, cosine threshold, influence multiplier, exit timer) using walk-forward validation. Reports Sharpe as fitness.

## Run it

```bash
pip install -r requirements.txt

# Tests (30/30)
python run_tests.py

# Backtest (2y XOM)
python backtest.py

# Dashboard
python dashboard.py

# Live engine + Telegram
python telegram_bot.py
```

## Telegram bot

```
/start · /status · /positions · /pnl · /params · /stop
```

Auto-alerts on new positions, mode transitions, and kill-switch triggers.

## References

Araci (2019) · Welford (1962) · Hasbrouck (1991) · Glosten & Milgrom (1985) · Kyle (1985) · Holland (1992) · López de Prado (2018) · Kilian (2009)

---

<p align="center">
  <sub>Amine Harch · 16 · Casablanca · <a href="https://vitalcheffe.github.io">vitalcheffe.github.io</a></sub>
</p>
