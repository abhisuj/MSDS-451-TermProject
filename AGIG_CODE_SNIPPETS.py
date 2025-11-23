# AGIG ETF Optimization - Key Code Snippets for Consolidation
# Quick reference for copying essential code blocks into the consolidated notebook

# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================

# Install dependencies (Cell 2)
"""
pip install yfinance pymoo tensorflow pandas numpy matplotlib seaborn scikit-learn polars TA-Lib
"""

# Imports (Cell 3)
import warnings
import yfinance as yf
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# ==============================================================================
# SECTION 2: CONFIGURATION (Cell 4)
# ==============================================================================

# Portfolio & Data
PORTFOLIO_TICKERS = [
    'AAPL', 'GOOG', 'MSFT', 'NVDA', 'AMZN', 'AMD', 'INTC', 'META',
    'COST', 'PG', 'KO', 'PEP', 'WMT', 'CME', 'AVGO', 'PFE', 'ABBV', 'HD',
    'XOM', 'VDE', 'SCHD', 'VYM', 'VWO', 'VEA', 'GLD', 'SLV', 'FXY', 
    'FDIVX', 'TLT', 'SPLB'
]
BENCHMARK_TICKER = 'SPY'
DATA_PERIOD = '25y'

# Feature Engineering
LOG_RETURN_CLIP = 0.20

# Bi-LSTM Model
LOOKBACK_WINDOW = 60
TRAIN_SPLIT = 0.64
VALIDATION_SPLIT = 0.16
LSTM_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 32

# NSGA-II Optimization - Settings for concentrated portfolios
POPULATION_SIZE = 100  # Smaller population = less diversity = more concentration
N_GENERATIONS = 250    # Fewer generations prevent over-exploration
TARGET_DIVIDEND_YIELD = 0.03  # 3% annual dividend yield
MIN_WEIGHT_PER_ASSET = 0.00   # 0% min - allow excluding weak assets
MAX_WEIGHT_PER_ASSET = 0.15   # 15% max - allow concentration in strong assets
MAX_ASSET_VOLATILITY = 0.90   # 90% maximum individual asset volatility

# ==============================================================================
# SECTION 3: DATA LOADING HELPER FUNCTIONS (Cell 5)
# ==============================================================================

def load_market_data(tickers, benchmark, period):
    """Download and prepare price and dividend data."""
    print(f"Downloading {len(tickers)} portfolio tickers + benchmark...")
    all_tickers = tickers + [benchmark]
    
    # Download price data
    data = yf.download(all_tickers, period=period, progress=False)
    close_data = data['Close']
    
    # Convert to Polars for fast processing
    price_df = pl.from_pandas(close_data.reset_index())
    price_df = price_df.drop_nulls()
    price_df = price_df.fill_null(strategy='forward').fill_null(strategy='backward')
    
    # Convert to Pandas for compatibility
    price_df_pd = price_df.to_pandas().set_index('Date')
    
    print(f"Price data shape: {price_df_pd.shape}")
    print(f"Date range: {price_df_pd.index[0].date()} to {price_df_pd.index[-1].date()}")
    
    # Download dividend data
    print("\nDownloading dividend data...")
    div_data = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            divs = ticker_obj.dividends
            if len(divs) > 0:
                div_data[ticker] = divs
        except:
            pass
    
    dividend_df = pd.DataFrame(div_data)
    dividend_df = dividend_df.reindex(price_df_pd.index, fill_value=0.0)
    
    print(f"Dividend data shape: {dividend_df.shape}")
    
    return price_df_pd, dividend_df, price_df

# ==============================================================================
# SECTION 4: FEATURE ENGINEERING (Cell 7 - Streamlined version)
# ==============================================================================

def create_features_streamlined(price_data_pd, tickers):
    """Generate technical indicators and targets for model training."""
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    # Download OHLCV data
    start_date = price_data_pd.index[0]
    end_date = price_data_pd.index[-1]
    print(f"Downloading OHLCV data from {start_date.date()} to {end_date.date()}...")
    ohlcv_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    ohlcv_data = ohlcv_data.reindex(price_data_pd.index).ffill().bfill()
    
    # Initialize feature DataFrame
    all_features = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # Extract OHLCV for this ticker
        close = ohlcv_data['Close'][ticker]
        high = ohlcv_data['High'][ticker]
        low = ohlcv_data['Low'][ticker]
        volume = ohlcv_data['Volume'][ticker]
        
        # Skip if too much missing data
        if close.isnull().sum() / len(close) > 0.5:
            print(f"  ⚠️ Skipping {ticker} - too much missing data")
            continue
        
        # Calculate technical indicators
        features_dict = {
            'ticker': ticker,
            # Trend
            'SMA_20': ta.SMA(close, timeperiod=20),
            'SMA_50': ta.SMA(close, timeperiod=50),
            'EMA_12': ta.EMA(close, timeperiod=12),
            'EMA_26': ta.EMA(close, timeperiod=26),
            
            # Momentum
            'RSI_14': ta.RSI(close, timeperiod=14),
            'MACD': ta.MACD(close)[0],
            'MACD_signal': ta.MACD(close)[1],
            
            # Volatility
            'ATR_14': ta.ATR(high, low, close, timeperiod=14),
            'BBANDS_upper': ta.BBANDS(close)[0],
            'BBANDS_middle': ta.BBANDS(close)[1],
            'BBANDS_lower': ta.BBANDS(close)[2],
            
            # Volume
            'OBV': ta.OBV(close, volume),
            
            # Oscillators
            'STOCH_K': ta.STOCH(high, low, close)[0],
            'STOCH_D': ta.STOCH(high, low, close)[1],
            
            # Long-term features
            'LT_VOL_252': close.pct_change().rolling(252).std(),
            'PRICE_TO_SMA_252': close / ta.SMA(close, timeperiod=252),
            'PRICE_TO_SMA_500': close / ta.SMA(close, timeperiod=500),
            'VOL_ANOMALY': volume.rolling(60).mean() / volume.rolling(252).mean(),
            
            # Target: Log returns (clipped)
            f'{ticker}_TARGET': np.clip(np.log(close / close.shift(1)), -LOG_RETURN_CLIP, LOG_RETURN_CLIP)
        }
        
        # Create DataFrame for this ticker
        ticker_df = pd.DataFrame(features_dict, index=close.index)
        all_features.append(ticker_df)
    
    # Combine all features
    features_df = pd.concat(all_features, axis=1)
    features_df = features_df.dropna()
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}")
    
    return features_df

# ==============================================================================
# SECTION 5: MODEL ARCHITECTURE (Cell 9)
# ==============================================================================

def build_bilstm_model(input_shape, num_return_targets, num_cov_targets, lstm_units, dropout_rate):
    """Build Bi-LSTM model with two outputs: returns and covariances."""
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(input_layer)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(lstm_units[1], return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(lstm_units[2], return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output branches
    returns_output = Dense(num_return_targets, activation='linear', name='returns')(x)
    cov_output = Dense(num_cov_targets, activation='linear', name='covariances')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=[returns_output, cov_output])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss={'returns': Huber(), 'covariances': Huber()},
        metrics={'returns': 'mse', 'covariances': 'mse'}
    )
    
    return model

# ==============================================================================
# SECTION 6: PORTFOLIO OPTIMIZATION PROBLEM (Cell 15)
# ==============================================================================

class PortfolioOptimizationProblem(Problem):
    """Multi-objective portfolio optimization with 4 objectives and dynamic constraints."""
    
    def __init__(self, mu, dividend_yields, predicted_returns_timeseries,
                 predicted_cov_matrices_time_series, test_dates,
                 min_weight, max_weight, target_dividend,
                 max_portfolio_volatility=None, min_portfolio_return=None):
        
        # Count constraints
        n_constraints = 2  # Always have: weights sum to 1, dividend >= target
        if max_portfolio_volatility is not None:
            n_constraints += 1
        if min_portfolio_return is not None:
            n_constraints += 1
        
        super().__init__(
            n_var=len(mu),
            n_obj=4,  # Return, Dividend, Volatility, CVaR
            n_constr=n_constraints,
            xl=min_weight,
            xu=max_weight
        )
        
        self.mu = mu
        self.dividend_yields = dividend_yields
        self.predicted_returns_timeseries = predicted_returns_timeseries
        self.predicted_cov_matrices_time_series = predicted_cov_matrices_time_series
        self.test_dates = test_dates
        self.target_dividend = target_dividend
        self.max_portfolio_volatility = max_portfolio_volatility
        self.min_portfolio_return = min_portfolio_return
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives and constraints for population."""
        
        # Normalize weights to sum to 1
        weights = x / x.sum(axis=1, keepdims=True)
        
        # Initialize arrays
        n_pop = x.shape[0]
        f1 = np.zeros(n_pop)  # Negative Return
        f2 = np.zeros(n_pop)  # Negative Dividend
        f3 = np.zeros(n_pop)  # Volatility
        f4 = np.zeros(n_pop)  # CVaR
        
        g1 = np.zeros(n_pop)  # Weights constraint
        g2 = np.zeros(n_pop)  # Dividend constraint
        
        if self.max_portfolio_volatility is not None:
            g3 = np.zeros(n_pop)
        if self.min_portfolio_return is not None:
            g_return = np.zeros(n_pop)
        
        # Evaluate each portfolio
        for i, w in enumerate(weights):
            # Use predicted covariance for this time step
            cov_idx = i % len(self.predicted_cov_matrices_time_series)
            current_sigma = self.predicted_cov_matrices_time_series[cov_idx]
            
            # Objective 1: Maximize Return (minimize negative return)
            annualized_return = np.dot(w, self.mu) * 252
            f1[i] = -annualized_return
            
            # Objective 2: Maximize Dividend (minimize negative dividend)
            dividend = np.sum(w * self.dividend_yields)
            f2[i] = -dividend
            
            # Objective 3: Minimize Volatility
            portfolio_var = np.dot(w.T, np.dot(current_sigma, w))
            portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)
            f3[i] = portfolio_vol
            
            # Objective 4: Minimize CVaR (95%)
            returns_ts = self.predicted_returns_timeseries[cov_idx] @ w
            q = np.percentile(returns_ts, 5)
            cvar = np.mean(returns_ts[returns_ts <= q])
            f4[i] = -cvar * 252
            
            # Constraint 1: Weights sum to 1
            g1[i] = np.abs(np.sum(w) - 1.0) - 1e-6
            
            # Constraint 2: Dividend >= target
            g2[i] = self.target_dividend - dividend
            
            # Constraint 3: Volatility <= max (optional)
            if self.max_portfolio_volatility is not None:
                g3[i] = portfolio_vol - self.max_portfolio_volatility
            
            # Constraint 4: Return >= min (optional)
            if self.min_portfolio_return is not None:
                g_return[i] = self.min_portfolio_return - annualized_return
        
        # Set outputs
        out["F"] = np.column_stack([f1, f2, f3, f4])
        
        constraints = [g1, g2]
        if self.max_portfolio_volatility is not None:
            constraints.append(g3)
        if self.min_portfolio_return is not None:
            constraints.append(g_return)
        out["G"] = np.column_stack(constraints)

# ==============================================================================
# SECTION 7: PORTFOLIO SELECTION STRATEGIES (Cell 17)
# ==============================================================================

def select_concentrated_portfolio(pareto_results, asset_tickers):
    """
    Select portfolio using concentration penalty strategy.
    Penalizes portfolios with too many active positions.
    """
    # Add concentration score
    pareto_results['NumActiveAssets'] = (pareto_results[asset_tickers] > 0.01).sum(axis=1)
    pareto_results['ConcentrationScore'] = (
        pareto_results['Sharpe'] - 0.01 * pareto_results['NumActiveAssets']
    )
    
    # Select best concentrated portfolio
    best_idx = pareto_results['ConcentrationScore'].idxmax()
    weights = pareto_results.loc[best_idx, asset_tickers]
    
    # Print info
    print(f"\n{'='*80}")
    print(f"CONCENTRATED PORTFOLIO SELECTION")
    print(f"{'='*80}")
    print(f"Sharpe Ratio: {pareto_results.loc[best_idx, 'Sharpe']:.4f}")
    print(f"Expected Return: {pareto_results.loc[best_idx, 'Return']:.2%}")
    print(f"Volatility: {pareto_results.loc[best_idx, 'Volatility']:.2%}")
    print(f"Dividend Yield: {pareto_results.loc[best_idx, 'Dividend']:.2%}")
    print(f"Active Assets: {int(pareto_results.loc[best_idx, 'NumActiveAssets'])}")
    print(f"Top 5 Concentration: {weights.nlargest(5).sum():.2%}")
    
    return weights

# ==============================================================================
# SECTION 8: BACKTESTING METRICS (Cell 18)
# ==============================================================================

def calculate_backtest_metrics(returns, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics."""
    
    # Basic metrics
    cumulative_return = (1 + returns).prod() - 1
    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    # Max drawdown
    cumulative_wealth = (1 + returns).cumprod()
    peak = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - peak) / peak
    max_drawdown = drawdown.min()
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    return {
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Downside Deviation': downside_deviation
    }

# ==============================================================================
# SECTION 9: RESULTS EXPORT (Cell 20)
# ==============================================================================

def export_results(weights, metrics, pareto_results, asset_tickers):
    """Export portfolio results to CSV files."""
    
    print(f"\n{'='*80}")
    print(f"EXPORTING RESULTS")
    print(f"{'='*80}")
    
    # Export weights
    weights_df = pd.DataFrame({
        'Asset': weights.index,
        'Weight': weights.values
    })
    weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
    weights_df.to_csv('selected_portfolio_weights.csv', index=False)
    print(f"✓ Exported weights to: selected_portfolio_weights.csv")
    
    # Export metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('selected_portfolio_metrics.csv', index=False)
    print(f"✓ Exported metrics to: selected_portfolio_metrics.csv")
    
    # Export Pareto front
    pareto_export = pareto_results[['Return', 'Dividend', 'Volatility', 'CVaR', 'Sharpe']].copy()
    pareto_export.to_csv('pareto_front_solutions.csv', index=False)
    print(f"✓ Exported Pareto front to: pareto_front_solutions.csv")
    
    print(f"{'='*80}\n")

# ==============================================================================
# EXAMPLE USAGE WORKFLOW
# ==============================================================================

"""
# Cell 5: Load Data
price_df_pd, dividend_df_pd, price_df = load_market_data(
    PORTFOLIO_TICKERS, BENCHMARK_TICKER, DATA_PERIOD
)

# Cell 7: Create Features
features_df = create_features_streamlined(price_df_pd, PORTFOLIO_TICKERS)

# Cell 8-10: Prepare Data, Build & Train Model
# [See prototype cells 20-25 for complete data prep and training code]

# Cell 13-14: Generate Predictions
# [See prototype cells 26-27 for prediction code]

# Cell 15-16: Run Optimization
problem = PortfolioOptimizationProblem(...)
algorithm = NSGA2(pop_size=POPULATION_SIZE, eliminate_duplicates=True)
res = minimize(problem, algorithm, ('n_gen', N_GENERATIONS), seed=42)

# Cell 17: Select Portfolio
weights = select_concentrated_portfolio(pareto_results, asset_tickers)

# Cell 18: Backtest
portfolio_returns = (actual_returns * weights).sum(axis=1)
metrics = calculate_backtest_metrics(portfolio_returns)

# Cell 20: Export
export_results(weights, metrics, pareto_results, asset_tickers)
"""
