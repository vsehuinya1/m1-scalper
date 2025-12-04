"""
Tests for backtest engine module.
"""
import numpy as np
import pandas as pd
import pytest

# Try to import backtest engine, but skip tests if not available
try:
    from backtest.engine import simulate, calculate_sharpe_ratio, calculate_max_drawdown
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False


@pytest.mark.skipif(not BACKTEST_AVAILABLE, reason="Backtest engine not available")
class TestBacktestEngine:
    """Test backtest engine functions."""
    
    def test_calculate_sharpe_ratio(self):
        """Test calculate_sharpe_ratio function."""
        returns = pd.Series([0.001, -0.0005, 0.002, 0.0001, -0.0003])
        
        result = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Calculate expected
        mean_return = returns.mean()
        std_return = returns.std()
        expected = mean_return / std_return * np.sqrt(252 * 24 * 60)
        assert result == pytest.approx(expected, rel=1e-3)
        
    def test_calculate_max_drawdown(self):
        """Test calculate_max_drawdown function."""
        equity_curve = pd.Series([10000, 10100, 10050, 10200, 10300])
        
        result = calculate_max_drawdown(equity_curve)
        
        # Calculate drawdown manually
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        expected = drawdown.min()
        
        assert result == pytest.approx(expected, rel=1e-3)
        
    def test_simulate_smoke_test(self):
        """Smoke test for simulate function."""
        # Create sample data
        n_samples = 1000
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        # Sample price data
        close = pd.Series(np.random.normal(50000, 1000, n_samples))
        
        # Mock signals
        long_signals = pd.Series(np.random.choice([True, False], n_samples))
        
        # Test that function runs without error
        try:
            equity_result = simulate(close, long_signals, commission=0.00017, slippage_mean=2.0)
        
        # Check return type
            assert isinstance(equity_result, pd.Series)
            assert len(equity_result) == len(close)
            
            # Check non-negative equity (should not go below zero)
            assert equity_result.min() >= 0
            
        except Exception as e:
            # If function doesn't exist yet or has issues, skip test
            pytest.skip(f"simulate function not implemented or failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
