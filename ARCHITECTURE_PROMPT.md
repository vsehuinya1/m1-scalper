You are an expert crypto-market micro-structure engineer.  
Build the complete MVP for a **BTC-USDT perpetual 1-minute 3R scalper** that satisfies **every** requirement below.  
Deliver **runnable Python code** (Python 3.11) plus **unit tests** and **Makefile**.  
Do **not** add features that are not explicitly listed.  
Code must be **deterministic**, **fast** (<2 s for 400 k rows), and **free of look-ahead bias**.

1.  Data
    a.  Input: Binance 1-minute klines (REST endpoint `/fapi/v1/klines?symbol=BTCUSDT&interval=1m`).  
    b.  Store as compressed CSV `data/binance_1m/btcusdt_1m_%Y.csv` with header:  
        `timestamp,open,high,low,close,volume,bid_size,ask_size,vwap`  
        (derive `vwap = sum(quote_asset_volume)/sum(volume)` from kline response).  
    c.  Provide `scripts/download.py` that back-fills from 2022-01-01 to yesterday.

2.  Regime detector (tradeable, no future data)
    a.  Micro-trend: 20-bar EMA of `(close - open)/open`.  
    b.  Volatility: 20-bar EMA of `abs(close - close[1])/close[1]`.  
    c.  Combine into 2-state rule:  
        ```
        regime = "low_vol_up"   if micro_trend > 0 and volatility < 60th-percentile(lookback=60)
        regime = "high_vol_down" if micro_trend < 0 and volatility > 60th-percentile(lookback=60)
        regime = "neutral"      otherwise
        ```
    d.  Percentile uses **rolling** 60-bar window (no future).  
    e.  Export `labels/regime_%Y.csv` with columns `timestamp,regime`.

3.  Strategy logic (deterministic, <25 lines in `strategy/entry.py`)
    ```
    def signal(df):
        # df has cols: timestamp,open,high,low,close,volume,bid_size,ask_size,vwap,regime
        long = (close > vwap) & (bid_size > ask_size*1.8) & (regime != "high_vol_down")
        short = (close < vwap) & (ask_size > bid_size*1.8) & (regime != "low_vol_up")
        range_prev = (high.shift(1) - low.shift(1))
        long &= close <= low.shift(1) + 0.3*range_prev   # limit chase
        short &= close >= high.shift(1) - 0.3*range_prev
        return long, short
    ```
    Stop-loss = structural pivot (low_prev - 2 ticks for long, high_prev + 2 ticks for short).  
    Target = entry ± 3× risk (3R).  
    Tick-size = 0.1 USD for BTC-USDT perpetual.

4.  Risk management (`strategy/risk.py`)
    a.  Fixed cash risk per trade: `risk_usd = equity * 0.002` (0.2 %).  
    b.  Position size `qty = risk_usd / abs(entry - stop)` rounded to **lot-step** 0.001 BTC.  
    c.  Daily risk budget: 5 trades max; if `daily_pnl < -1 %` → halt until next UTC day.  
    d.  Drawdown halt: if `equity / peak_equity < 0.95` → flat, send Telegram alert, require manual reset.

5.  Back-test engine (`backtest/engine.pyx` or `engine.py` with numba)
    a.  Iterate once over 1-minute bars, emit fill on bar close (market order).  
    b.  Commission 0.017 % taker, slippage model `uniform(1, 3) ticks`.  
    c.  Output CSV `results/trades_%Y%m%d.csv` with cols:  
        `timestamp,side,qty,entry,exit,entry_idx,exit_idx,pnl,commission,slippage_ticks,dd_run`  
    d.  Write `results/equity_%Y%m%d.csv`: `timestamp,equity,drawdown`.  
    e.  Speed target: 400 000 rows in <2 s on M1 MacBook Air.

6.  Walk-forward routine (`scripts/walk_forward.py`)
    a.  expanding window: train 3 200 h → validate 400 h → test 400 h.  
    b.  Parameter vector = `{risk_pct, ema_len, percentile_thr, bid_ask_ratio, range_pct}`  
    c.  Optimise **deflated Sharpe** on validation set (grid limited to 162 combinations).  
    d.  Report **only** test-set stats.  
    e.  Append row to `results/walk_forward.csv` every night.

7.  Nightly pipeline (`Makefile` target `nightly`)
    ```
    make download   # new 1m data
    make regime     # new labels
    make backtest   # full history
    make walk       # walk-forward
    make report     # stats.json + equity.png
    make telegram   # send summary
    ```

8.  Telegram notifier (`live/telegram.py`)
    a.  Read `stats.json`, send:  
        ```
        BTC 1m 3R scalper – nightly report
        Trades 24 h: 8 | Win: 5 (62.5 %) | RR: 3.0 | PnL: +0.41 %
        Sharpe: 2.1 | Drawdown: -0.9 % | Status: OK
        ```
    b.  Also push **every paper fill** in real-time when `live/paper.py` is running.

9.  Unit tests (`tests/`)
    a.  `test_engine.py`: 100 pre-computed bars → known equity curve to 8-decimal.  
    b.  `test_risk.py`: 0.2 % risk, 3R target → verify qty calculation.  
    c.  `pytest` must pass in CI (GitHub Actions free tier).

10.  Repo structure
    ```
    m1-scalper/
     ├── data/
     ├── labels/
     ├── results/
     ├── scripts/
     ├── strategy/
     ├── backtest/
     ├── live/
     ├── tests/
     ├── Makefile
     ├── requirements.txt
     └── ARCHITECTURE_PROMPT.md
    ```

11.  Deliverables
    - `make backtest` produces `results/equity_latest.png` and `results/stats.json`  
    - `stats.json` keys:  
      `{"trades":int,"win_rate":float,"mean_r":float,"sharpe":float,"psr":float,"max_dd":float,"in_market_pct":float}`  
    - Code is **black**-formatted, **ruff**-clean, **typed** (PEP 561).

12.  Exclude
    - CVD / OI filters (already removed).  
    - Execution code (only paper trading).  
    - Multi-pair or multi-session logic.  
    - Machine-learning models.

Produce the full code base now.  After I paste this prompt, **do not** ask clarifying questions—implement **exactly** what is specified.
