# sense-act

sentiment arbitrage engine for oil market signals.

started this because i kept noticing that follower count is a terrible proxy for signal quality. a ras tanura engineer with 200 followers knows more about pipeline supply than a reuters bot with 2M. built something to test that idea properly.

## how it works

```
text signal
    → keyword scoring + embedding
    → cosine dedup (catches paraphrases, not just exact copies)
    → welford z-score (flags statistically unusual signals)
    → half-life decay (signal loses 50% alpha every 2 min)
    → influence weighting (log followers × domain expertise × hub status)
    → shadow position (virtual, no real money)
        → kill-switch (halts if spread or vix doubles in 60s)
        → HFT jitter sim (35% chance of adverse fill)
        → monte carlo slippage P99 (1000 GBM paths, 50ms horizon)
    → genetic optimizer (tunes 7 params via walk-forward, runs in background)
    → telegram alerts
```

## files

```
sentiment.py          keyword scorer + entry point
signal_processor.py   dedup, welford, decay, influence map
shadow_core.py        kill-switch, HFT sim, MC slippage, virtual book
orchestrator.py       async engine (asyncio.TaskGroup)
genetic_optimizer.py  GA + walk-forward validation
telegram_bot.py       bot interface
```

## setup

python 3.11+

```bash
pip install -r requirements.txt
cp .env.example .env
# add your token from @BotFather and your chat ID from @userinfobot
python telegram_bot.py
```

without telegram:

```bash
python orchestrator.py
```

## telegram commands

```
/start      menu
/status     price, mode, open positions, pnl
/positions  open positions with entry/sl/tp
/pnl        24h summary and win rate
/params     current GA parameters
/stop       stop the engine
```

auto-alerts on new positions, crisis mode, kill-switch triggers.

## replacing the stubs

the engine runs on fake data by default. to use real data:

- `SentimentStub` → ProsusAI/finbert via transformers
- `SignalFeed` → twitter API v2, RSS, websocket
- `MarketFeed` → broker websocket (alpaca, IBKR)

drop API connectors in `./plugins/` with a `plugin_factory()` function, the loader picks them up automatically.

## notes

shadow mode only — no broker integration, no real orders. the genetic optimizer needs ~50 closed trades before walk-forward produces meaningful results. oracle reconciliation (drift detection) needs real tick data to work properly, currently falls back to noise as placeholder.

---

MIT license
