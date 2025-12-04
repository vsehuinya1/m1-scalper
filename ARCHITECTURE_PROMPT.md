Markdown
# LOCKED PROMPT – LEAN BTC-USDT 1-MINUTE 3R SCALPER MVP
# Production-ready, pure functions, no over-engineering, auto-ci included

## 1. DATA PIPELINE (Parquet, 1-min only)
- REST back-fill → `data/binance_1m/btcusdt_<year>.parquet` (Snappy, monthly partitions)
- WebSocket append → live file `btcusdt_current.parquet`
- Fields: timestamp,open,high,low,close,volume,bid_size,ask_size,quote_asset_volume
- Derived: vwap = quote_asset_volume / volume

## 2. REGIME DETECTOR (pure, vectorised)

def rolling_percentile(series, window, percentile, min_periods=None):
    if min_periods is None:
        min_periods = window
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: np.nanpercentile(x, percentile), raw=True)

def micro_trend(close, open_):
    raw = (close - open_) / open_
    vol = close.pct_change().rolling(5).std()
    trend = raw.rolling(20).mean().fillna(0)          # NaN → 0
    return trend.where(vol < vol.rolling(20).quantile(0.6), 0)

def volume_gate(volume, mult, min_bars=20):
    med = volume.rolling(min_bars, min_periods=min_bars).median()
    above = volume > med * mult
    return above & above.shift(1).fillna(False)       # 2-bar confirmation

def classify(trend, vol_cat, vol_ok):
    return {('up','low',True):  'low_vol_up',
            ('down','high',True):'high_vol_down'}.get(
            (np.sign(trend), vol_cat, vol_ok), 'neutral')
vol_cat = pd.cut(volatility, 3, labels=['low','mid','high'])

3. ENTRY SIGNAL (pure, params injected)
def core_long(close, vwap, bid_size, ask_size, ratio):
    return (close > vwap) & (bid_size > ask_size * ratio)

def core_short(close, vwap, bid_size, ask_size, ratio):
    return (close < vwap) & (ask_size > bid_size * ratio)

def anti_chase(close, prev_range, pct, side):
    if side == 'long':
        return close <= low.shift(1) + pct * prev_range
    else:
        return close >= high.shift(1) - pct * prev_range

def build_signal(df, p):
    prev_r = (df.high.shift(1) - df.low.shift(1))
    core_l = core_long(df.close, df.vwap, df.bid_size, df.ask_size, p['bid_ask_ratio'])
    core_s = core_short(df.close, df.vwap, df.bid_size, df.ask_size, p['bid_ask_ratio'])
    regime_l = (df.regime == "low_vol_up")
    regime_s = (df.regime == "high_vol_down")
    chase_l  = anti_chase(df.close, prev_r, p['range_pct'], 'long')
    chase_s  = anti_chase(df.close, prev_r, p['range_pct'], 'short')
    long  = core_l  & regime_l  & chase_l
    short = core_s  & regime_s  & chase_s
    return long, short
    
4. SCALE-OUT EXITS (trade-splitting, priority enforced)
def make_exit_prices(entry, stop, side):
    risk = abs(entry - stop)
    t2 = entry + 2*risk if side=='long' else entry - 2*risk
    t3 = entry + 3*risk if side=='long' else entry - 3*risk
    trail_trigger = entry + 1*risk if side=='long' else entry - 1*risk
    return t2, t3, trail_trigger   # priority: t2 > t3 > trail
Emit three child legs with parent_id; fill order: 50 % @ t2, 25 % @ t3, 25 % trail (0.5 R offset).

5. RISK & POSITION (vectorised)
def risk_filter(long, short, max_pos, max_daily):
    pos_ok = len(open_trades) < max_pos
    pnl_ok = daily_pnl > -max_daily
    return long & pos_ok & pnl_ok, short & pos_ok & pnl_ok

def position_size(equity, entry, stop):
    risk_usd = equity * 0.002              # 0.2 %
    qty = risk_usd / abs(entry - stop)
    return round(qty, 3)                   # 0.001 BTC lot
    
6. BACKTEST ENGINE (numba, single-pass)
•	Fill on bar close, market order
•	Costs: 0.017 % commission + uniform(1,3) tick slippage
•	Seed=42 for reproducibility
•	Target: 400 k bars <1.5 s on M1 Air

7. WALK-FORWARD (grid search, deterministic)
param_grid = {
  'ema_len':       [15,20,25],
  'percentile_thr':[55,60,65],
  'bid_ask_ratio': [1.6,1.8,2.0],
  'range_pct':     [0.25,0.30,0.35],
  'volume_mult':   [1.1,1.2,1.3]   # 3^5 = 243 combos
}
Train 3 200 h → validate 400 h → test 400 h
Optimise deflated Sharpe on validation, report test only.

8. REALISTIC TARGETS
Win Rate 42-48 % | Avg R 2.8-3.2 | Trades/day 4-9
Sharpe 1.9-2.4 | Max DD 8-12 % | Monthly ret 2.5-4 %

9. REPO LAYOUT

m1-scalper/
├── data/binance_1m/   # parquet
├── labels/            # regime parquet
├── results/           # trades + equity parquet
├── scripts/           # download.py  walk_forward.py
├── strategy/          # entry.py  risk.py  exits.py
├── backtest/          # engine.pyx (numba)
├── live/              # paper.py  telegram.py
├── tests/             # pytest
├── Makefile
├── requirements.txt
└── .github/workflows/ci.yml
10. AUTOMATED NIGHTLY PIPELINE
Makefile:
makefile
Copy
nightly: download regime backtest walk report ci

ci: test
	git add -A
	git diff --cached --quiet || (git commit -m "auto: $$(date +%F-%H%M)" && git push origin main)
GitHub Actions runs same make nightly at 04:00 UTC.
11. CODE QUALITY
•	Type hints + mypy --strict
•	Black / isort / ruff + pre-commit
•	95 % test coverage minimum
•	structlog for logging
12. DELIVERABLES
make backtest → equity.png + stats.json
make walk → optimisation report
make paper → real-time paper trading
make report → Telegram daily summary
 
WORKFLOW RULE (for Cline)
After you generate each file, run make quick, wait for green exit code, then ask me for confirmation before creating the next file. If any check fails, rewrite the file until all tests pass.
CRITICAL: NO OVER-ENGINEERING
No L2, no ML, no external APIs beyond Binance, no adaptive sizing beyond fixed 0.2 %, no complex event engine.
 
LIVE CHECKLIST
Create and keep a file PROGRESS.md in the repo root.
After every successful file (green make quick), append:
•	[x] filename.py – short description – commit hash
•	[ ] next_file.py – short description
Update the checklist in the same commit. Never overwrite existing checks.
Generate the complete codebase now—every file must be runnable with zero placeholders.

<img width="467" height="721" alt="image" src="https://github.com/user-attachments/assets/2e3abfd9-1a8d-453f-bd68-2290be0965c8" />
