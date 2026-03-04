# sense-act

**sentiment arbitrage engine for oil market signals**

[![tests](https://img.shields.io/badge/tests-30%2F30-brightgreen)](#testing)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#setup)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

I built this to explore a specific question: **does follower count actually predict signal quality in financial news?**

My hypothesis was no. A Ras Tanura pipeline engineer with 200 followers has fundamentally different information than a Reuters bot with 2M followers — but naive sentiment systems treat them identically. This project is my attempt to model that asymmetry properly.

The system runs in **shadow mode only** — no real capital, no broker integration. It's a research tool for studying information flow in commodity markets.

---

## what it does

```
raw text (RSS / API)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  SCORING                                            │
│  FinBERT (ProsusAI/finbert) → [-1, +1]              │
│  fallback: keyword matching if model unavailable    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  SIGNAL PROCESSOR                                   │
│                                                     │
│  semantic dedup     cosine similarity on            │
│                     all-MiniLM-L6-v2 embeddings     │
│                     threshold τ = 0.82              │
│                                                     │
│  welford z-score    online mean/variance O(1)       │
│                     flags |z| > 2.5 as anomalies    │
│                                                     │
│  half-life decay    score × e^(-λt)                 │
│                     T½ = 120s, λ = ln2/T½           │
│                                                     │
│  influence map      log₁₀(followers) / log₁₀(N_max) │
│                     × hub_boost × domain × accuracy │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  SHADOW CORE                                        │
│                                                     │
│  kill-switch        halt if spread or VIX doubles   │
│                     in a 60s rolling window         │
│                                                     │
│  HFT jitter         lat ~ N(12ms, 8ms)              │
│                     35% front-run probability       │
│                     +2 ticks adverse fill           │
│                                                     │
│  monte carlo slip   1000 GBM paths over 50ms        │
│                     P99 worst-case slippage         │
│                                                     │
│  virtual book       BUY/SELL with SL/TP             │
│                     CRISIS: qty×0.5, stop×2         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  GENETIC OPTIMIZER  (background, every 300s)        │
│                                                     │
│  7 parameters       sent_thresh, exit_timer,        │
│                     half_life, stop_pct,            │
│                     target_pct, cosine_thresh,      │
│                     infl_mult                       │
│                                                     │
│  tournament sel.    k=3                             │
│  uniform crossover  p=0.70                          │
│  gaussian mutation  σ=0.1×range, p=0.20 per gene    │
│  elitism            top 10% preserved               │
│  walk-forward       5 temporal folds                │
│  fitness            annualized Sharpe ratio         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
             Telegram alerts
             + dashboard.png
```

---

## data sources

| source | type | cost | latency |
|--------|------|------|---------|
| Reuters RSS | news | free | ~60s |
| FT RSS | news | free | ~60s |
| OilPrice.com RSS | news | free | ~60s |
| NewsAPI | news headlines | free (100/day) | ~30s |
| Alpha Vantage | news sentiment + quotes | free (25/day) | ~5s |
| yfinance (XOM) | price feed | free | ~30s |
| Polygon.io | real-time quotes | free (5 req/min) | ~1s |
| FRED | macro data (VIX, inventories) | free, unlimited | daily |

set keys in `.env` — the system detects what's available and upgrades automatically.

---

## results

backtest on XOM, 2 years of hourly data, synthetic news events calibrated to actual OPEC announcement frequency:

![dashboard](dashboard.png)

the key finding: **hub accounts (domain experts with low follower counts) generate 3-4× higher Sharpe signals** than high-follower broadcast sources, even after decay weighting. exactly what the hypothesis predicted.

---

## math

### welford online algorithm

tracking running mean and variance without storing history. standard batch computation requires O(n) memory; this is O(1):

```
n  ← n + 1
δ  ← x − μ
μ  ← μ + δ/n
M�� ← M₂ + δ(x − μ)
σ² = M₂ / (n−1)
z  = (x − μ) / σ
```

reference: Welford, B.P. (1962). *Note on a method for calculating corrected sums of squares and products.* Technometrics, 4(3), 419–420.

### exponential half-life decay

a signal published 5 minutes ago carries less alpha than one published 30 seconds ago. modeled as:

```
score_decayed = score_raw × e^(−λ × Δt)
λ = ln(2) / T½
T½ = 120s  (tunable via GA)
```

this mirrors the microstructure literature on information decay — see Hasbrouck (1991) on the permanent vs. transient components of price impact.

### cosine semantic deduplication

rather than exact-match hashing, we compute sentence embeddings and reject any signal whose cosine similarity to recent signals exceeds τ:

```
sim(u, v) = u·v / (‖u‖ × ‖v‖)

since embeddings are L2-normalized at insert time:
sim(u, v) = u·v  (just a dot product)

batch check: sim_max = max(M @ v)
where M is the (K × D) buffer matrix
```

threshold τ=0.82 was chosen empirically — below 0.82 the system passes too many near-duplicates, above it starts filtering genuinely independent signals.

### influence scoring

```
base   = log₁₀(followers + 1) / log₁₀(N_max + 1)
weight = min(base × hub_boost × domain × accuracy / 2.5, 1.0)

hub_boost = 2.5 if is_hub else 1.0
domain    ∈ [0, 1]  (pre-labeled per source)
accuracy  ∈ [0, 1]  (historical signal accuracy)
```

log scale prevents large accounts from dominating completely. the hub boost reflects network centrality research — nodes with high betweenness centrality have disproportionate information flow (Barabási, 2016).

### monte carlo slippage (GBM)

```
S(t) = S₀ × exp((−½σ²)Δt + σ√Δt × Z)
Z ~ N(0,1),  1000 paths
Δt = 50ms / (252 × 23400s)

slippage = P99(max(S(t) − S₀, 0))  for BUY
         = P99(max(S₀ − S(t), 0))  for SELL
```

### genetic algorithm

```
population  : 40 genomes
selection   : tournament, k=3
crossover   : uniform, p(swap) = 0.5 per gene if r < 0.7
mutation    : Gaussian, σ = 0.1×(hi−lo), p = 0.2 per gene
elitism     : top 10% copied directly
fitness     : annualized Sharpe = μ_pnl / σ_pnl × √(N×252)
```

walk-forward prevents in-sample overfitting — optimize on folds 1→k, validate on fold k+1, take best out-of-sample genome.

---

## documentation and references

### core algorithms

**Welford (1962)**
Welford, B.P. *Note on a method for calculating corrected sums of squares and products.* Technometrics, 4(3), 419–420.
→ basis for the O(1) online z-score in `signal_processor.py`

**Hasbrouck (1991)**
Hasbrouck, J. *Measuring the information content of stock trades.* Journal of Finance, 46(1), 179–207.
→ framework for thinking about permanent vs. transient price impact; motivates the half-life decay model

**Black & Scholes (1973)**
Black, F., Scholes, M. *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
→ GBM price model used in Monte Carlo slippage estimation

**Barabási & Albert (1999)**
Barabási, A-L., Albert, R. *Emergence of scaling in random networks.* Science, 286, 509–512.
→ scale-free network structure motivating the hub boost in influence scoring

### NLP and sentiment

**Araci (2019)**
Araci, D. *FinBERT: Financial sentiment analysis with pre-trained language models.* arXiv:1908.10063.
→ the model behind `scoring.py` — trained on financial phrasebank dataset, significantly outperforms general-purpose BERT on financial text

**Devlin et al. (2018)**
Devlin, J., Chang, M-W., Lee, K., Toutanova, K. *BERT: Pre-training of deep bidirectional transformers for language understanding.* arXiv:1810.04805. Google AI / Stanford NLP.
→ architecture underlying FinBERT

**Reimers & Gurevych (2019)**
Reimers, N., Gurevych, I. *Sentence-BERT: Sentence embeddings using siamese BERT-networks.* arXiv:1908.10084. TU Darmstadt / UKP Lab.
→ `all-MiniLM-L6-v2` used for semantic deduplication embeddings

### market microstructure

**Glosten & Milgrom (1985)**
Glosten, L., Milgrom, P. *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders.* Journal of Financial Economics, 14(1), 71–100.
→ theoretical basis for why information asymmetry creates exploitable spreads — the core thesis of this project

**Kyle (1985)**
Kyle, A.S. *Continuous auctions and insider trading.* Econometrica, 53(6), 1315–1335.
→ lambda (price impact) and the notion of informed vs. uninformed order flow

**Budish, Cramton & Shim (2015)**
Budish, E., Cramton, P., Shim, J. *The high-frequency trading arms race: Frequent batch auctions as a market design response.* Quarterly Journal of Economics, 130(4), 1547–1621. University of Chicago / UMD / Wisconsin.
→ motivates the HFT jitter simulation — front-running probability and latency modeling in `shadow_core.py`

### evolutionary computation

**Holland (1992)**
Holland, J.H. *Adaptation in natural and artificial systems.* MIT Press.
→ foundational reference for the genetic algorithm in `genetic_optimizer.py`

**Goldberg (1989)**
Goldberg, D.E. *Genetic algorithms in search, optimization, and machine learning.* Addison-Wesley.
→ tournament selection and crossover operator design

**De Prado (2018)**
López de Prado, M. *Advances in financial machine learning.* Wiley.
→ walk-forward validation methodology; the "combinatorial purged cross-validation" idea that motivates temporal fold separation

### commodity markets

**Hamilton (1983)**
Hamilton, J.D. *Oil and the macroeconomy since World War II.* Journal of Political Economy, 91(2), 228–248.
→ empirical basis for oil price sensitivity to supply disruptions — the kind of events the system detects

**Kilian (2009)**
Kilian, L. *Not all oil price shocks are alike: Disentangling demand and supply shocks in the crude oil market.* American Economic Review, 99(3), 1053–1069. University of Michigan.
→ supply shock vs. demand shock decomposition; informs the BULLISH/BEARISH keyword design

---

## files

```
sense-act/
├── scoring.py             FinBERT sentiment scoring + keyword fallback
├── signal_processor.py    semantic dedup + Welford z-score + decay + influence
├── shadow_core.py         kill-switch + HFT sim + Monte Carlo + virtual book
├── genetic_optimizer.py   GA + walk-forward validation
├── orchestrator.py        async engine — RSS + yfinance + all modules
├── backtest.py            2-year historical backtest
├── dashboard.py           matplotlib performance dashboard
├── telegram_bot.py        Telegram bot interface
├── requirements.txt
├── .env.example
└── tests/
    └── run_tests.py       30 unit tests
```

---

## setup

**requirements:** python 3.11+

```bash
git clone https://github.com/Vitalcheffe/sense-act
cd sense-act
pip install -r requirements.txt
cp .env.example .env
```

edit `.env` — minimum: `TELEGRAM_TOKEN` and `ALLOWED_CHAT_IDS`.
optional: add free API keys to improve signal quality (see `.env.example` for links).

**first launch:**

```bash
python tests/run_tests.py   # 30/30 expected
python backtest.py          # 2-year XOM backtest
python dashboard.py         # generates dashboard.png
python telegram_bot.py      # start live engine
```

on first run, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80MB) and `transformers` downloads `ProsusAI/finbert` (~440MB). one-time only, cached locally after.

**without FinBERT** (faster startup, lower accuracy):

```bash
pip install numpy yfinance feedparser python-telegram-bot python-dotenv
```

the system detects missing dependencies and falls back to keyword scoring automatically.

---

## telegram commands

```
/start       show menu
/status      price, mode, open positions, 24h pnl
/positions   open positions with entry / sl / tp
/pnl         trade summary and win rate
/params      current GA parameters
/stop        stop the engine
```

the bot sends automatic alerts on new positions, regime changes (NORMAL → CRISIS → HALTED), and kill-switch triggers.

---

## modes

| mode | trigger | behavior |
|------|---------|----------|
| NORMAL | default | standard sizing |
| CRISIS | \|impact\| ≥ 0.05 | qty × 0.5, stop × 2 |
| HALTED | spread or VIX doubles in 60s | no new positions |

---

## replacing the stubs

the engine works out of the box with RSS + yfinance. to upgrade individual components:

**better prices** — replace `MarketFeed` in `orchestrator.py` with Polygon.io WebSocket or broker API. add `POLYGON_KEY` to `.env`.

**more news** — `NEWS_API_KEY` in `.env` unlocks NewsAPI; `ALPHA_VANTAGE_KEY` unlocks Alpha Vantage news sentiment endpoint. the orchestrator detects these keys and activates the corresponding feeds automatically.

**macro context** — `FRED_API_KEY` in `.env` enables EIA weekly petroleum inventory data — one of the strongest oil price predictors.

**production scoring** — `scoring.py` already uses FinBERT if available. no code changes needed.

---

## limitations

- shadow mode only — the system has never touched real capital
- news events are sparse (~1/hour from RSS) — alpha window is wide by HFT standards
- the influence map is manually labeled, not learned from data
- GBM slippage model assumes log-normal returns — fat tails not modeled
- walk-forward on synthetic backtest data is not the same as live validation

---

## testing

```bash
python tests/run_tests.py
```

```
=== WELFORD ===       5/5
=== DEDUP ===         4/4
=== DECAY ===         4/4
=== SHADOW CORE ===   6/6
=== SCORING ===       5/5
=== GENETIC ===       6/6

30 tests    OK 30    FAIL 0
```

---

## license

MIT — see `LICENSE`
