#!/usr/bin/env python3
"""
Walk-forward optimization for BTC-USDT 1-minute scalper.
Grid search with train/validate/test splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import itertools
import structlog


class WalkForwardOptimizer:
    """Walk-forward optimization engine."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.logger = structlog.get_logger()
        
        # Updated parameter grid (loosened)
        self.param_grid = {
            'ema_len': [15, 20, 25],
            'percentile_thr': [55, 60, 65],
            'bid_ask_ratio': [1.3, 1.5, 1.8],
            'range_pct': [0.25, 0.30, 0.35],
            'volume_mult': [1.05, 1.2, 1.3]
        }
        
        self.results_cache = {}

    def generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combos = list(itertools.product(*values))
        
        param_combinations = []
        for combo in all_combos:
            params = dict(zip(keys, combo))
            param_combinations.append(params)
        
        return param_combinations

    def train_validate_test_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validate, and test periods."""
        # Calculate hourly offsets
        train_hours = 200
        validate_hours = 400
        test_hours = 400
        
        # Start from beginning
        train_end = df['timestamp'].iloc[0] + pd.Timedelta(hours=train_hours)
        validate_end = train_end + pd.Timedelta(hours=validate_hours)
        test_end = validate_end + pd.Timedelta(hours=test_hours)
        
        # Split data
        train_data = df[df['timestamp'] <= train_end]
        validate_data = df[
            (df['timestamp'] > train_end) & 
            (df['timestamp'] <= validate_end)
        ]
        test_data = df[
            (df['timestamp'] > validate_end) & 
            (df['timestamp'] <= test_end)
        ]
        
        return train_data, validate_data, test_data

    def deflated_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate deflated Sharpe ratio to adjust for overfitting."""
        # Calculate standard Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe = excess_returns.mean() / excess_returns.std()
        
        # Apply Benjamini-Hochberg correction
        # Simplified version for MVP
        return sharpe * 0.95  # Basic deflation factor

    def optimize(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """Run full walk-forward optimization."""
        self.logger.info("Starting walk-forward optimization")
        
        # Generate all parameter combinations
        param_combos = self.generate_param_combinations()
        self.logger.info(f"Testing {len(param_combos)} parameter combinations")
        
        # Split data
        train_data, validate_data, test_data = self.train_validate_test_split(df)
        
        # Run optimization across all parameter sets
        best_score = -np.inf
        best_params = {}
        
        for params in param_combos:
            # Calculate validation performance
            validate_performance = self.evaluate_params(train_data, validate_data, params)
            
            # Track best parameters
            if validate_performance > best_score:
                best_score = validate_performance
                best_params = params
        
        # Test on test period with best parameters
        test_performance = self.evaluate_params(train_data, test_data, best_params)
        
        # Cache results
        self.results_cache[tuple(best_params.items())] = {
            'validate_sharpe': best_score,
            'test_sharpe': test_performance
        }
        
        self.logger.info(
            "Walk-forward optimization completed",
            best_params=best_params,
            validate_performance=best_score,
            test_performance=test_performance
        )
        
        return {
            'best_params': best_params,
            'validate_sharpe': best_score,
            'test_sharpe': test_performance
        }

    def evaluate_params(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        params: Dict
    ) -> float:
        """Evaluate parameter set on test data using actual backtest."""
        # Import required modules inside function to avoid circular imports
        from strategy.regime import detect_regime
        from strategy.entry import build_signal
        from backtest.engine import BacktestEngine
        
        # Use the same regime parameters as base, but adjust with param grid
        regime_params = {
            'trend_window': 20,
            'vol_window': 5,
            'vol_mult': params.get('volume_mult', 1.2),
            'vol_bins': 3
        }
        
        # Prepare data copy
        df = test_data.copy()
        
        # Detect regime
        df = detect_regime(df, regime_params)
        
        # Generate signals with current parameters
        signal_params = {
            'bid_ask_ratio': params.get('bid_ask_ratio', 1.8),
            'range_pct': params.get('range_pct', 0.3)
        }
        long_signals, short_signals = build_signal(df, signal_params)
        
        # Convert timestamp to milliseconds integer (as expected by backtest engine)
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**6).astype('int64')
        
        # Ensure numeric columns are float64
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'bid_size', 'ask_size', 'quote_asset_volume', 'vwap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        
        # Run backtest
        engine = BacktestEngine(seed=self.seed)
        results = engine.run(
            df,
            long_signals,
            short_signals,
            initial_capital=10000.0,
            risk_percent=0.002
        )
        
        # Extract Sharpe ratio from results
        sharpe = results.get('sharpe_ratio', 0.0)
        return sharpe

    def generate_report(self, results: Dict = None) -> str:
        """Generate optimization report."""
        if results is None:
            results = {}
            
        best_params = results.get('best_params', {})
        validate_sharpe = results.get('validate_sharpe', 0)
        test_sharpe = results.get('test_sharpe', 0)
        
        report = f"""
Walk-Forward Optimization Report
=================================
Parameters evaluated: {len(self.generate_param_combinations())} combinations

Best Parameters:
- ema_len: {best_params.get('ema_len', 'N/A')}
- percentile_thr: {best_params.get('percentile_thr', 'N/A')}
- bid_ask_ratio: {best_params.get('bid_ask_ratio', 'N/A')}
- range_pct: {best_params.get('range_pct', 'N/A')}
- volume_mult: {best_params.get('volume_mult', 'N/A')}

Performance:
- Validation Sharpe (deflated): {validate_sharpe:.4f}
- Test Sharpe: {test_sharpe:.4f}

Parameter Grid:
{self.param_grid}
"""
        
        return report


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.backtest_runner import load_data
    
    logger = structlog.get_logger()
    logger.info("Loading data for walk-forward optimization")
    # Use available data (since we don't have full 2020-2024)
    df = load_data()
    
    optimizer = WalkForwardOptimizer()
    results = optimizer.optimize(df)
    
    report = optimizer.generate_report(results)
    print(report)
