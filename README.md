# sense-act

sentiment arbitrage engine for oil market signals — shadow mode research project

![tests](https://img.shields.io/badge/tests-30%2F30-brightgreen) ![python](https://img.shields.io/badge/python-3.11+-blue)

---

## background

i started this after noticing a pattern. a ras tanura pipeline engineer with 200 followers would post something on twitter, and oil futures would move 3 minutes later. reuters would publish the same story 8 minutes after that. by then it was already priced in.

the signal was never in the follower count. it was in the information type.

most existing sentiment systems weight sources by follower count, which is basically modeling noise. i wanted to test whether building explicit domain expertise scoring — combined with proper signal decay, semantic deduplication, and adversarial fill simulation — actually produces something useful.

this is a shadow-mode research engine. no real capital. architecture is production-grade, execution is virtual.

---

## how it works

a news article comes in from RSS or a news API. it gets scored by `ProsusAI/finbert` — a BERT model fine-tuned specifically on financial text, which handles phrases like "maintains output despite pressure" correctly where keyword matching fails. the score is a float in [-1, +1].

before the signal goes anywhere, it gets checked against a rolling buffer of recent embeddings using cosine similarity. this catches semantic duplicates — the Reuters headline and the Bloomberg paraphrase of the same OPEC announcement both computing to a near-identical vector and only one passing through.

surviving signals get a welford z-score to flag statistical anomalies, a half-life decay (signals lose 50% weight every 120s by default), and an influence weight that combines log-follower-count with a manually-labeled domain expertise score. a ras tanura engineer gets a 2.5x hub boost over a broadcast account.

the resulting `impact` score goes to the shadow book, which first runs a kill-switch check (halts if spread or VIX doubles in the last 60s), estimates P99 slippage over a 50ms window using 1000 GBM paths, simulates HFT front-running at 35% probability, then opens a virtual position with stop and target.

in the background, a genetic algorithm tunes 7 parameters — sentiment threshold, half-life, stop/target percentages, cosine threshold, influence multiplier, exit timer — using walk-forward validation on closed trades. it only runs when there are enough trades to validate on, and reports annualized Sharpe as fitness.

---

## results

![dashboard](dashboard.png)

backtest on XOM, 2 years of hourly data. hub accounts with domain expertise generated 3-4× higher Sharpe signals than high-follower broadcast sources. signal decay was significant — anything older than ~4 minutes had near-zero impact in this dataset. the genetic optimizer converged on a sentiment threshold of ~0.12 and a half-life of ~95s.

---

## the math

**welford z-score**

standard variance computation requires keeping all values in memory. welford's 1962 algorithm does it in a single pass with O(1) storage:

```
n  ← n + 1
δ  ← x − μ
μ  ← μ + δ/n
M�� ← M₂ + δ(x − μ)

σ² = M₂ / (n−1)
z  = (x − μ) / σ
```

numerically stable — avoids the catastrophic cancellation in the naive `Σx²/n − (Σx/n)²` formula. reference: Welford (1962), Technometrics 4(3).

**half-life decay**

```
score_decayed = score_raw × exp(−λ × Δt)
λ = ln(2) / T½
```

T½ is a GA parameter, empirically converges around 90-150s. consistent with Hasbrouck (1991) on the permanent component of price impact being absorbed within a few minutes of a trade.

**cosine semantic dedup**

embeddings are L2-normalized at insert time, so cosine similarity reduces to a dot product. the buffer check becomes a single matrix multiplication:

```
sim_max = max(M @ v)   where M is (K × 384)
```

K=500, threshold τ=0.82. below 0.82 too many near-duplicates pass through. above it you start rejecting genuinely independent signals from different sources covering the same event.

**monte carlo slippage**

P99 adverse fill over a 50ms execution horizon using GBM:

```
S(t) = S₀ × exp((−½σ²)Δt + σ√Δt × Z),   Z ~ N(0,1),   1000 paths

σ = (spread/mid) × √(252 × 23400)
Δt = 0.05 / (252 × 23400)

slippage = percentile(max(S(t) − S₀, 0), 99)   for BUY
```

**influence weight**

```
base   = log₁₀(followers + 1) / log₁₀(10_000_001)
weight = min(base × hub_boost × domain × accuracy / 2.5, 1.0)
```

log scale so megaphones don't dominate. hub_boost=2.5 for domain expert accounts, based on betweenness centrality in scale-free networks (Barabási & Albert, 1999).

---

## files

```
scoring.py              FinBERT sentiment, keyword fallback
signal_processor.py     dedup + welford + decay + influence
shadow_core.py          kill-switch + HFT + MC slippage + book
genetic_optimizer.py    GA + walk-forward
orchestrator.py         async engine, RSS + yfinance feeds
backtest.py             2-year historical backtest
dashboard.py            matplotlib performance charts
telegram_bot.py         bot interface
tests/run_tests.py      30 unit tests
```

---

## setup

python 3.11+

```bash
pip install -r requirements.txt
cp .env.example .env
```

only `TELEGRAM_TOKEN` and `ALLOWED_CHAT_IDS` are required. the optional API keys in `.env` upgrade the news feeds automatically when present.

```bash
python tests/run_tests.py   # 30/30
python backtest.py          # 2y XOM backtest
python dashboard.py         # generates dashboard.png
python telegram_bot.py      # live engine + telegram
```

first run downloads `all-MiniLM-L6-v2` (~80MB) and `ProsusAI/finbert` (~440MB), cached after that. without them the system uses keyword scoring, no code changes needed.

---

## telegram

`/start` · `/status` · `/positions` · `/pnl` · `/params` · `/stop`

auto-alerts on new positions, CRISIS/HALTED mode transitions, and kill-switch triggers.

---

## references

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*

Devlin, J. et al. (2018). BERT: Pre-training of deep bidirectional transformers. *arXiv:1810.04805* — Google AI / Stanford NLP

Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *arXiv:1908.10084* — UKP Lab, TU Darmstadt

Welford, B.P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics, 4(3)*

Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance, 46(1)*

Glosten, L., Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics, 14(1)*

Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica, 53(6)*

Budish, E., Cramton, P., Shim, J. (2015). The high-frequency trading arms race. *Quarterly Journal of Economics, 130(4)* — University of Chicago

Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81(3)*

Holland, J.H. (1992). *Adaptation in natural and artificial systems.* MIT Press

López de Prado, M. (2018). *Advances in financial machine learning.* Wiley

Hamilton, J.D. (1983). Oil and the macroeconomy since World War II. *Journal of Political Economy, 91(2)*

Kilian, L. (2009). Not all oil price shocks are alike. *American Economic Review, 99(3)* — University of Michigan

Barabási, A-L., Albert, R. (1999). Emergence of scaling in random networks. *Science, 286*

---

MIT license
