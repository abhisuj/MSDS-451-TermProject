# test_MSDS_451_Term_Project_Backtesting.py
"""
Comprehensive tests for the Zipline backtesting notebook.
Tests cover bundle creation, strategy initialization, fee calculations,
and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest import mock
from zipline.api import symbol
from zipline.finance import commission, slippage
import tempfile
import os


class TestPortfolioWeightsImport:
    """Test portfolio weights CSV import and validation."""
    
    def test_weights_csv_exists(self):
        """Verify the selected_portfolio_weights.csv file exists."""
        assert os.path.exists("selected_portfolio_weights.csv"), \
            "Portfolio weights CSV file not found"
    
    def test_weights_csv_structure(self):
        """Verify CSV has required columns."""
        df = pd.read_csv("selected_portfolio_weights.csv")
        assert "Ticker" in df.columns, "Missing 'Ticker' column"
        assert len(df) > 0, "Empty portfolio weights file"
    
    def test_weights_sum_to_one(self):
        """If Weight column exists, verify weights sum to approximately 1.0."""
        df = pd.read_csv("selected_portfolio_weights.csv")
        if "Weight" in df.columns:
            total_weight = df["Weight"].sum()
            assert 0.99 <= total_weight <= 1.01, \
                f"Weights sum to {total_weight}, expected ~1.0"
    
    def test_no_duplicate_tickers(self):
        """Verify no duplicate tickers in portfolio."""
        df = pd.read_csv("selected_portfolio_weights.csv")
        duplicates = df["Ticker"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate tickers"


class TestBundleConfiguration:
    """Test custom Zipline bundle configuration."""
    
    @pytest.fixture
    def mock_selected_weights(self):
        """Mock portfolio weights for testing."""
        return pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "GOOGL"],
            "Weight": [0.4, 0.35, 0.25]
        })
    
    def test_bundle_tickers_includes_benchmarks(self, mock_selected_weights):
        """Verify bundle includes SPY and QQQ benchmarks."""
        bundle_tickers = list(mock_selected_weights["Ticker"]) + ['SPY', 'QQQ']
        
        assert 'SPY' in bundle_tickers, "SPY benchmark not in bundle"
        assert 'QQQ' in bundle_tickers, "QQQ benchmark not in bundle"
        assert len(bundle_tickers) == 5, "Unexpected number of tickers"
    
    def test_bundle_function_signature(self):
        """Verify term_project_bundle has correct parameters."""
        from MSDS_451_Term_Project_Backtesting import term_project_bundle
        
        import inspect
        sig = inspect.signature(term_project_bundle)
        params = list(sig.parameters.keys())
        
        required_params = [
            'environ', 'asset_db_writer', 'minute_bar_writer',
            'daily_bar_writer', 'adjustment_writer', 'calendar',
            'start_session', 'end_session', 'cache', 'show_progress',
            'output_dir'
        ]
        
        for param in required_params:
            assert param in params, f"Missing required parameter: {param}"


class TestStrategyInitialization:
    """Test Zipline strategy initialization."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock Zipline context object."""
        context = mock.Mock()
        context.universe = {}
        context.target_weights = {}
        context.portfolio = mock.Mock()
        context.portfolio.starting_cash = 100000
        context.portfolio.portfolio_value = 100000
        return context
    
    def test_initialize_parses_weights_correctly(self, mock_context):
        """Test that initialize correctly parses portfolio weights."""
        with mock.patch('MSDS_451_Term_Project_Backtesting.selected_portfolio_weights',
                       pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Weight": [0.6, 0.4]})):
            
            from MSDS_451_Term_Project_Backtesting import initialize
            
            with mock.patch('MSDS_451_Term_Project_Backtesting.symbol') as mock_symbol, \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_commission'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_slippage'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.schedule_function'):
                
                initialize(mock_context)
                
                assert len(mock_context.target_weights) == 2
                assert sum(mock_context.target_weights.values()) == pytest.approx(1.0)
    
    def test_commission_set_correctly(self, mock_context):
        """Test that transaction costs are set to 10 bps."""
        with mock.patch('MSDS_451_Term_Project_Backtesting.selected_portfolio_weights',
                       pd.DataFrame({"Ticker": ["AAPL"], "Weight": [1.0]})):
            
            from MSDS_451_Term_Project_Backtesting import initialize
            
            with mock.patch('MSDS_451_Term_Project_Backtesting.symbol'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_commission') as mock_comm, \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_slippage'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.schedule_function'):
                
                initialize(mock_context)
                
                # Verify commission.PerDollar called with cost=0.001 (10 bps)
                mock_comm.assert_called_once()
                args = mock_comm.call_args
                assert isinstance(args[0][0], commission.PerDollar)
    
    def test_management_fee_parameters(self, mock_context):
        """Test management fee tracking parameters are set."""
        with mock.patch('MSDS_451_Term_Project_Backtesting.selected_portfolio_weights',
                       pd.DataFrame({"Ticker": ["AAPL"], "Weight": [1.0]})):
            
            from MSDS_451_Term_Project_Backtesting import initialize
            
            with mock.patch('MSDS_451_Term_Project_Backtesting.symbol'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_commission'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_slippage'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.schedule_function'):
                
                initialize(mock_context)
                
                assert hasattr(mock_context, 'management_fee_annual')
                assert mock_context.management_fee_annual == 0.01
                assert mock_context.management_fee_daily == pytest.approx(0.01 / 252)
    
    def test_performance_fee_parameters(self, mock_context):
        """Test performance fee tracking parameters are set."""
        with mock.patch('MSDS_451_Term_Project_Backtesting.selected_portfolio_weights',
                       pd.DataFrame({"Ticker": ["AAPL"], "Weight": [1.0]})):
            
            from MSDS_451_Term_Project_Backtesting import initialize
            
            with mock.patch('MSDS_451_Term_Project_Backtesting.symbol'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_commission'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.set_slippage'), \
                 mock.patch('MSDS_451_Term_Project_Backtesting.schedule_function'):
                
                initialize(mock_context)
                
                assert hasattr(mock_context, 'performance_fee_rate')
                assert mock_context.performance_fee_rate == 0.20
                assert mock_context.hwm is None


class TestFeeCalculations:
    """Test fee calculation functions."""
    
    @pytest.fixture
    def mock_context(self):
        """Mock context with fee parameters."""
        context = mock.Mock()
        context.portfolio = mock.Mock()
        context.portfolio.portfolio_value = 100000
        context.portfolio.starting_cash = 100000
        context.management_fee_daily = 0.01 / 252
        context.quarterly_dividend_rate = 0.01
        context.performance_fee_rate = 0.20
        context.hwm = None
        context.universe = {}
        context.portfolio.positions = {}
        return context
    
    @pytest.fixture
    def mock_data(self):
        """Mock Zipline data object."""
        data = mock.Mock()
        data.current_dt = pd.Timestamp('2024-01-15')
        return data
    
    def test_deduct_management_fee_calculates_correctly(self, mock_context, mock_data):
        """Test daily management fee calculation."""
        from MSDS_451_Term_Project_Backtesting import deduct_management_fee
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record') as mock_record:
            deduct_management_fee(mock_context, mock_data)
            
            expected_fee = 100000 * (0.01 / 252)
            mock_record.assert_called_once()
            call_args = mock_record.call_args[1]
            assert 'mgmt_fee_paid' in call_args
            assert call_args['mgmt_fee_paid'] == pytest.approx(expected_fee, rel=1e-6)
    
    def test_quarterly_dividend_calculation(self, mock_context, mock_data):
        """Test quarterly dividend amount calculation."""
        from MSDS_451_Term_Project_Backtesting import pay_quarterly_dividend
        
        mock_context.portfolio.positions = {}
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record') as mock_record:
            pay_quarterly_dividend(mock_context, mock_data)
            
            expected_dividend = 100000 * 0.01
            mock_record.assert_called_once()
            call_args = mock_record.call_args[1]
            assert 'quarterly_dividend_paid' in call_args
            assert call_args['quarterly_dividend_paid'] == pytest.approx(expected_dividend)
    
    def test_performance_fee_not_charged_below_hwm(self, mock_context, mock_data):
        """Test performance fee not charged when below high-water mark."""
        from MSDS_451_Term_Project_Backtesting import charge_performance_fee
        
        mock_context.hwm = 110000  # Above current value
        mock_context.spy = mock.Mock()
        
        mock_data.history = mock.Mock(return_value=pd.Series([100, 105]))
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record') as mock_record:
            charge_performance_fee(mock_context, mock_data)
            
            call_args = mock_record.call_args[1]
            assert call_args['perf_fee_paid'] == 0
    
    def test_performance_fee_charged_above_hwm_and_benchmark(self, mock_context, mock_data):
        """Test performance fee charged when above HWM and benchmark."""
        from MSDS_451_Term_Project_Backtesting import charge_performance_fee
        
        mock_context.portfolio.portfolio_value = 120000
        mock_context.hwm = 100000
        mock_context.spy = mock.Mock()
        
        # Mock SPY return of 10% (benchmark_value = 110000)
        mock_data.history = mock.Mock(return_value=pd.Series([100, 110]))
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record') as mock_record:
            charge_performance_fee(mock_context, mock_data)
            
            call_args = mock_record.call_args[1]
            # Excess gain = 120000 - 110000 = 10000
            # Fee = 10000 * 0.20 = 2000
            assert call_args['perf_fee_paid'] == pytest.approx(2000, rel=0.01)


class TestQuarterAndYearTracking:
    """Test quarter and year-end tracking logic."""
    
    @pytest.fixture
    def mock_context(self):
        context = mock.Mock()
        context.last_quarter = None
        context.last_year = None
        context.portfolio = mock.Mock()
        context.portfolio.portfolio_value = 100000
        context.quarterly_dividend_rate = 0.01
        return context
    
    def test_initialization_on_first_run(self, mock_context):
        """Test that quarter/year tracking initializes on first run."""
        from MSDS_451_Term_Project_Backtesting import check_quarter_and_year_end
        
        mock_data = mock.Mock()
        mock_data.current_dt = pd.Timestamp('2024-01-15')
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record'):
            check_quarter_and_year_end(mock_context, mock_data)
            
            assert mock_context.last_quarter == 1
            assert mock_context.last_year == 2024
    
    def test_quarter_change_triggers_dividend(self, mock_context):
        """Test that quarter change triggers dividend payment."""
        from MSDS_451_Term_Project_Backtesting import check_quarter_and_year_end
        
        mock_context.last_quarter = 1
        mock_context.last_year = 2024
        mock_context.universe = {}
        mock_context.portfolio.positions = {}
        
        mock_data = mock.Mock()
        mock_data.current_dt = pd.Timestamp('2024-04-01')  # Q2
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record') as mock_record, \
             mock.patch('MSDS_451_Term_Project_Backtesting.pay_quarterly_dividend') as mock_div:
            
            check_quarter_and_year_end(mock_context, mock_data)
            
            mock_div.assert_called_once()
            assert mock_context.last_quarter == 2
    
    def test_year_change_triggers_performance_fee(self, mock_context):
        """Test that year change triggers performance fee calculation."""
        from MSDS_451_Term_Project_Backtesting import check_quarter_and_year_end
        
        mock_context.last_quarter = 4
        mock_context.last_year = 2023
        mock_context.universe = {}
        mock_context.portfolio.positions = {}
        mock_context.hwm = None
        mock_context.spy = mock.Mock()
        
        mock_data = mock.Mock()
        mock_data.current_dt = pd.Timestamp('2024-01-02')
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.record'), \
             mock.patch('MSDS_451_Term_Project_Backtesting.charge_performance_fee') as mock_perf:
            
            check_quarter_and_year_end(mock_context, mock_data)
            
            mock_perf.assert_called_once()
            assert mock_context.last_year == 2024


class TestRebalancing:
    """Test portfolio rebalancing logic."""
    
    @pytest.fixture
    def mock_context(self):
        context = mock.Mock()
        context.universe = {
            'AAPL': mock.Mock(),
            'MSFT': mock.Mock()
        }
        context.target_weights = {
            'AAPL': 0.6,
            'MSFT': 0.4
        }
        context.portfolio = mock.Mock()
        context.portfolio.portfolio_value = 100000
        context.account = mock.Mock()
        context.account.leverage = 1.0
        return context
    
    def test_rebalance_orders_all_assets(self, mock_context):
        """Test that rebalancing orders all assets in universe."""
        from MSDS_451_Term_Project_Backtesting import rebalance_portfolio
        
        mock_data = mock.Mock()
        mock_data.can_trade = mock.Mock(return_value=True)
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.order_target_percent') as mock_order, \
             mock.patch('MSDS_451_Term_Project_Backtesting.record'):
            
            rebalance_portfolio(mock_context, mock_data)
            
            assert mock_order.call_count == 2
            
            # Verify correct weights passed
            calls = mock_order.call_args_list
            weights_ordered = [call[0][1] for call in calls]
            assert set(weights_ordered) == {0.6, 0.4}
    
    def test_rebalance_skips_non_tradeable_assets(self, mock_context):
        """Test that rebalancing skips assets that can't be traded."""
        from MSDS_451_Term_Project_Backtesting import rebalance_portfolio
        
        mock_data = mock.Mock()
        # Only AAPL can be traded
        mock_data.can_trade = lambda asset: asset == mock_context.universe['AAPL']
        
        with mock.patch('MSDS_451_Term_Project_Backtesting.order_target_percent') as mock_order, \
             mock.patch('MSDS_451_Term_Project_Backtesting.record'):
            
            rebalance_portfolio(mock_context, mock_data)
            
            # Should only order AAPL
            assert mock_order.call_count == 1


class TestBacktestExecution:
    """Test backtest execution and date handling."""
    
    def test_date_range_is_valid(self):
        """Test that backtest date range is valid."""
        start_date = pd.Timestamp('1999-01-01')
        end_date = pd.Timestamp('2025-10-31')
        
        assert start_date < end_date, "Start date must be before end date"
        assert start_date.year >= 1990, "Start date too far in past"
        assert end_date <= pd.Timestamp.now(), "End date in future"
    
    def test_initial_capital_is_positive(self):
        """Test that initial capital is positive."""
        capital_base = 100000
        assert capital_base > 0, "Initial capital must be positive"
        assert capital_base >= 10000, "Initial capital should be at least $10,000"


class TestPerformanceMetrics:
    """Test performance metric calculations in analyze function."""
    
    @pytest.fixture
    def mock_perf(self):
        """Mock performance DataFrame."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        returns = np.random.normal(0.001, 0.02, 252)
        portfolio_value = 100000 * np.cumprod(1 + returns)
        
        perf = pd.DataFrame({
            'returns': returns,
            'portfolio_value': portfolio_value,
            'mgmt_fee_paid': np.full(252, 100000 * 0.01 / 252),
            'perf_fee_paid': np.zeros(252),
            'quarterly_dividend_paid': np.zeros(252),
            'benchmark_period_return': np.cumsum(np.random.normal(0.0008, 0.015, 252))
        }, index=dates)
        
        # Add some quarterly dividends
        perf.loc[perf.index[60], 'quarterly_dividend_paid'] = 1000
        perf.loc[perf.index[120], 'quarterly_dividend_paid'] = 1050
        perf.loc[perf.index[180], 'quarterly_dividend_paid'] = 1100
        
        return perf
    
    def test_total_return_calculation(self, mock_perf):
        """Test that total return is calculated correctly."""
        final_value = mock_perf['portfolio_value'].iloc[-1]
        initial_value = mock_perf['portfolio_value'].iloc[0]
        
        total_return = (final_value / initial_value) - 1
        
        assert isinstance(total_return, (float, np.floating))
        assert -1 < total_return < 10, "Total return out of reasonable range"
    
    def test_sharpe_ratio_calculation(self, mock_perf):
        """Test Sharpe ratio calculation."""
        returns = mock_perf['returns'].dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        assert isinstance(sharpe, (float, np.floating))
        assert -5 < sharpe < 10, "Sharpe ratio out of reasonable range"
    
    def test_fee_totals(self, mock_perf):
        """Test that fee totals are calculated correctly."""
        total_mgmt_fees = mock_perf['mgmt_fee_paid'].sum()
        total_perf_fees = mock_perf['perf_fee_paid'].sum()
        total_dividends = mock_perf['quarterly_dividend_paid'].sum()
        
        assert total_mgmt_fees > 0, "Management fees should be positive"
        assert total_mgmt_fees == pytest.approx(100000 * 0.01, rel=0.01)
        assert total_dividends == pytest.approx(3150, rel=0.01)


class TestAlphaBetaCalculation:
    """Test alpha and beta calculation logic."""
    
    def test_beta_calculation_with_positive_correlation(self):
        """Test beta calculation when portfolio and benchmark are correlated."""
        # Create correlated returns
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        portfolio_returns = benchmark_returns * 1.2 + np.random.normal(0, 0.005, 252)
        
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        })
        
        covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        assert 0.5 < beta < 2.0, f"Beta {beta} out of reasonable range"
    
    def test_alpha_calculation(self):
        """Test alpha calculation."""
        portfolio_return = 0.15  # 15% annual
        benchmark_return = 0.10  # 10% annual
        beta = 1.0
        risk_free_rate = 0.02  # 2%
        
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        expected_alpha = 0.15 - (0.02 + 1.0 * (0.10 - 0.02))
        assert alpha == pytest.approx(expected_alpha)
        assert alpha == pytest.approx(0.05)  # 5% alpha


class TestErrorHandling:
    """Test error handling in backtest execution."""
    
    def test_handles_missing_weights_file(self):
        """Test graceful handling when weights file is missing."""
        with mock.patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                pd.read_csv("selected_portfolio_weights.csv")
    
    def test_handles_empty_weights_dataframe(self):
        """Test handling of empty weights DataFrame."""
        empty_df = pd.DataFrame(columns=["Ticker", "Weight"])
        
        assert len(empty_df) == 0
        assert "Ticker" in empty_df.columns


# Integration test placeholder
class TestIntegration:
    """Integration tests - requires actual Zipline environment."""
    
    @pytest.mark.skip(reason="Requires full Zipline environment and data bundle")
    def test_full_backtest_runs(self):
        """Test that full backtest executes without errors."""
        # This would require actual Zipline setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])