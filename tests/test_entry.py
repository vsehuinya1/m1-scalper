"""
Tests for entry signal module.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.entry import core_long, core_short, anti_chase, build_signal


class TestEntrySignals:
    """Test entry signal functions."""
    
    def test_core_long(self):
        """Test core_long function."""
        close = pd.Series([100, 101, 102, 103, 104])
        vwap = pd.Series([99, 100, 101, 102, 103])
        bid_size = pd.Series([100, 110, 120, 130, 140])
        ask_size = pd.Series([90, 95, 100, 105, 110])
        ratio = 1.1
        
        result = core_long(close, vwap, bid_size, ask_size, ratio)
        
        # Check conditions
        # close > vwap: [True, True, True, True, True]
        # bid_size > ask_size * ratio: calculate
        expected = pd.Series([
            100 > 99 and 100 > 90 * 1.1,   # False (90*1.1=99, 100>99 true but 100>99? 100>99 true)
            101 > 100 and 110 > 95 * 1.1,  # True (95*1.1=104.5, 110>104.5 true)
            102 > 101 and 120 > 100 * 1.1, # True (100*1.1=110, 120>110 true)
            103 > 102 and 130 > 105 * 1.1, # True (105*1.1=115.5, 130>115.5 true)
            104 > 103 and 140 > 110 * 1.1  # True (110*1.1=121, 140>121 true)
        ])
        
        pd.testing.assert_series_equal(result, expected)
        
    def test_core_short(self):
        """Test core_short function."""
        close = pd.Series([100, 99, 98, 97, 96])
        vwap = pd.Series([101, 100, 99, 98, 97])
        bid_size = pd.Series([90, 95, 100, 105, 110])
        ask_size = pd.Series([100, 110, 120, 130, 140])
        ratio = 1.1
        
        result = core_short(close, vwap, bid_size, ask_size, ratio)
        
        # Check conditions
        # close < vwap: [True, True, True, True, True]
        # ask_size > bid_size * ratio: calculate
        expected = pd.Series([
            100 < 101 and 100 > 90 * 1.1,   # False (90*1.1=99, 100>99 true)
            99 < 100 and 110 > 95 * 1.1,    # True (95*1.1=104.5, 110>104.5 true)
            98 < 99 and 120 > 100 * 1.1,    # True (100*1.1=110, 120>110 true)
            97 < 98 and 130 > 105 * 1.1,    # True (105*1.1=115.5, 130>115.5 true)
            96 < 97 and 140 > 110 * 1.1     # True (110*1.1=121, 140>121 true)
        ])
        
        pd.testing.assert_series_equal(result, expected)
        
    def test_anti_chase(self):
        """Test anti_chase function."""
        close = pd.Series([100, 101, 102, 103, 104])
        prev_range = pd.Series([2, 2, 2, 2, 2])
        pct = 0.3
        low = pd.Series([99, 100, 101, 102, 103])
        high = pd.Series([101, 102, 103, 104, 105])
        
        # Test long side
        result_long = anti_chase(close, prev_range, pct, 'long', low=low)
        # Formula: close <= low.shift(1) + pct * prev_range
        # low.shift(1) = [NaN, 99, 100, 101, 102]
        # pct * prev_range = 0.3 * 2 = 0.6
        # low.shift(1) + 0.6 = [NaN, 99.6, 100.6, 101.6, 102.6]
        # close <= that = [False, False, False, False, False] (since close > low.shift(1)+0.6 for all)
        # Let's compute:
        # idx1: 101 <= 99.6? False
        # idx2: 102 <= 100.6? False
        # idx3: 103 <= 101.6? False
        # idx4: 104 <= 102.6? False
        expected_long = pd.Series([False, False, False, False, False])
        
        # Test short side  
        result_short = anti_chase(close, prev_range, pct, 'short', high=high)
        # Formula: close >= high.shift(1) - pct * prev_range
        # high.shift(1) = [NaN, 101, 102, 103, 104]
        # high.shift(1) - 0.6 = [NaN, 100.4, 101.4, 102.4, 103.4]
        # close >= that = [False, True, True, True, True]
        # idx1: 101 >= 100.4? True
        # idx2: 102 >= 101.4? True
        # idx3: 103 >= 102.4? True
        # idx4: 104 >= 103.4? True
        expected_short = pd.Series([False, True, True, True, True])
        
        pd.testing.assert_series_equal(result_long, expected_long)
        pd.testing.assert_series_equal(result_short, expected_short)
        
    def test_build_signal(self):
        """Test build_signal function."""
        # Create sample data
        n_samples = 50
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(50000, 100, n_samples),
            'high': np.random.normal(50100, 100, n_samples),
            'low': np.random.normal(49900, 100, n_samples),
            'close': np.random.normal(50000, 100, n_samples),
            'volume': np.random.normal(100, 10, n_samples),
            'vwap': np.random.normal(50000, 100, n_samples),
            'bid_size': np.random.normal(60, 10, n_samples),
            'ask_size': np.random.normal(40, 10, n_samples),
            'regime': np.random.choice(['low_vol_up', 'high_vol_down', 'neutral'], n_samples)
        })
        
        # Define parameters
        params = {
            'bid_ask_ratio': 1.8,
            'range_pct': 0.3,
            'ema_len': 20,
            'percentile_thr': 60
        }
        
        # Test signal generation
        long_signals, short_signals = build_signal(df, params)
        
        # Check return types
        assert isinstance(long_signals, pd.Series)
        assert isinstance(short_signals, pd.Series)
        assert len(long_signals) == len(df)
        assert len(short_signals) == len(df)
        
        # Check boolean values
        assert long_signals.dtype == bool
        assert short_signals.dtype == bool
        
        # Check that we can't have both long and short signals on same row
        # (though this depends on regime, which can't be both low_vol_up and high_vol_down)
        # This is more of a logical test
        for long_val, short_val in zip(long_signals, short_signals):
            # It's possible to have both False, but not both True
            if long_val and short_val:
                pytest.fail("Row has both long and short signals")
                
        # Test with specific regime conditions
        df_specific = df.copy()
        df_specific['regime'] = 'low_vol_up'
        long_specific, short_specific = build_signal(df_specific, params)
        
        # With low_vol_up regime, we should have some long signals
        # (though they might all be False depending on other conditions)
        assert isinstance(long_specific, pd.Series)
        assert isinstance(short_specific, pd.Series)
        
        # Test with high_vol_down regime
        df_specific['regime'] = 'high_vol_down'
        long_specific, short_specific = build_signal(df_specific, params)
        
        # With high_vol_down regime, we should have some short signals
        assert isinstance(long_specific, pd.Series)
        assert isinstance(short_specific, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
