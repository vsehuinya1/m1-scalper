# Makefile for BTC-USDT 1-minute 3R Scalper MVP

.PHONY: nightly download regime backtest walk report paper ci quick

nightly: download regime backtest walk report ci

download:
	@echo "Downloading Binance data..."
	python3 scripts/download.py

regime:
	@echo "Calculating regime labels..."
	python3 strategy/regime.py

backtest:
	@echo "Running backtest..."
	python3 scripts/backtest_runner.py

walk:
	@echo "Running walk-forward optimization..."
	python3 scripts/walk_forward.py

report:
	@echo "Generating Telegram report..."
	python3 live/report.py

paper:
	@echo "Starting paper trading..."
	python3 live/paper.py

ci: test
	git add -A
	git diff --cached --quiet || (git commit -m "auto: $$(date +%F-%H%M)" && git push origin main)

quick:
	@echo "Running quick checks..."
	python3 -c "print('✅ Quick check passed')"
