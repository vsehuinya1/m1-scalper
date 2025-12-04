"""
Tests for regime detection module.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.regime import rolling_percentile, micro_trend, volume_gate, detect_regime


class TestRegimeDetection:
    """Test regime detection functions."""
    
    def test_rolling_percentile(self):
        """Test rolling_percentile function."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test with window 3
        result = rolling_percentile(data, window=3, percentile=50)
        expected = pd.Series([np.nan, np.nan, 2, 3, 4, 5, 6, 7, 8, 9])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
        
        # Test with min_periods
        result = rolling_percentile(data, window=5, percentile=25, min_periods=3)
        # With min_periods=3, first two should be NaN (need at least 3 observations)
        assert result.iloc[0:2].isna().all()  # First 2 should be NaN
        assert not pd.isna(result.iloc[2])  # Third should have value
        
    def test_micro_trend(self):
        """Test micro_trend function."""
        close = pd.Series([100, 101, 102, 103, 104, 105])
        open_ = pd.Series([100, 100, 101, 102, 103, 104])
        
        result = micro_trend(close, open_)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)
        assert result.iloc[0] == 0  # First value should be 0 due to vol filter
        
    def test_volume_gate(self):
        """Test volume_gate function."""
        volume = pd.Series([100, 200, 300, 400, 500, 600, 700])
        
        # Test with mult=1.0
        result = volume_gate(volume, mult=1.0, min_bars=3)
        
        # Should be False for first min_bars
        assert not result.iloc[0:3].any()
        
        # Should identify values above rolling median
        # Last values should be True if above median * mult
        
    def test_detect_regime(self):
        """Test detect_regime function."""
        # Create sample data
        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(50000, 1000, n_samples),
            'high': np.random.normal(50100, 1000, n_samples),
            'low': np.random.normal(49900, 1000, n_samples),
            'close': np.random.normal(50000, 1000, n_samples),
            'volume': np.random.normal(100, 10, n_samples),
            'quote_asset_volume': np.random.normal(5000000, 100000, n_samples)
        })
        
        # Calculate VWAP
        df['vwap'] = df['quote_asset_volume'] / df['volume']
        
        # Add bid/ask size columns
        df['bid_size'] = df['volume'] * 0.6
        df['ask_size'] = df['volume'] * 0.4
        
        # Define parameters
        params = {
            'ema_len': 20,
            'percentile_thr': 60,
            'trend_window': 20,
            'vol_window': 5,
            'vol_bins': 3,
            'volume_mult': 1.2
        }
        
        # Test regime detection
        result = detect_regime(df, params)
        
        # Check return type
        assert isinstance(result, pd.DataFrame)
        assert 'regime' in result.columns
        assert 'volatility' in result.columns
        assert 'vol_cat' in result.columns
        assert 'vol_ok' in result.columns
        
        # Check regime values
        valid_regimes = {'low_vol_up', 'high_vol_down', 'neutral'}
        assert all(result['regime'].isin(valid_regimes))
        
        # Check non-empty result
        assert len(result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
