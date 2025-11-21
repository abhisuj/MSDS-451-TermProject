# Portfolio Optimization Refactoring - User Guide

## Overview
This refactored system provides a clean, configuration-based approach to generate **Aggressive**, **Moderate**, and **Conservative** portfolios using NSGA-II multi-objective optimization.

## 🎯 Key Features

### 1. **Three Pre-Configured Strategies**
- **Aggressive Growth**: High risk, high return (12%+ target, 35% max volatility)
- **Moderate Balanced**: Balanced approach (9%+ target, 22% max volatility)  
- **Conservative Income**: Low risk, income-focused (6%+ target, 15% max volatility)

### 2. **Easy Configuration**
All portfolio parameters are defined in dictionaries:
- Min/max weight per asset
- Target dividend yield
- Max portfolio volatility
- Min portfolio return
- Max individual asset volatility

### 3. **Automated Processing**
Single function call generates all three portfolios and saves results to CSV

## 📁 Files Created

### Core Modules
1. **portfolio_config_refactored.py** - Main refactored optimization engine
   - `PORTFOLIO_PROFILES`: Configuration dictionaries
   - `run_portfolio_optimization()`: Single strategy optimization
   - `generate_all_portfolios()`: Generate all three strategies

2. **run_portfolio_strategies.py** - Complete usage example
   - Shows how to use the refactored module
   - Includes custom strategy example

3. **notebook_cell_snippet.py** - Quick notebook integration
   - Copy-paste code to add to your notebook

## 🚀 How to Use

### Method 1: Add to Existing Notebook

**Copy the content from `notebook_cell_snippet.py` and paste as a new cell** after your current optimization setup (after cell 40).

```python
# Import the refactored module
import sys
sys.path.append('.')
from portfolio_config_refactored import generate_all_portfolios

# Run optimization
weights_df, metrics_df, results = generate_all_portfolios(
    mu=mu,
    avg_annual_yield=avg_annual_yield,
    predicted_simple_returns_filtered=predicted_simple_returns_filtered,
    predicted_cov_matrices_time_series=predicted_cov_matrices_time_series,
    test_dates=test_dates,
    asset_tickers=asset_tickers,
    actual_test_returns=actual_test_returns,
    population_size=POPULATION_SIZE,
    n_generations=N_GENERATIONS
)
```

### Method 2: Run as Standalone Script

```bash
# In your notebook, after data preparation:
%run run_portfolio_strategies.py
```

## 📊 Output Files

### 1. `selected_portfolio_weights.csv`
Portfolio weights for all three strategies:
```
Ticker,Aggressive Growth,Moderate Balanced,Conservative Income
AAPL,0.0523,0.0412,0.0356
MSFT,0.0687,0.0543,0.0423
...
```

### 2. `selected_portfolio_metrics.csv`
Performance metrics for each strategy:
```
Strategy,Expected_Return,Dividend_Yield,Volatility,Sharpe_Ratio,CVaR_95,Assets_Included
Aggressive Growth,0.1245,0.0223,0.2834,0.3684,-0.0456,25
Moderate Balanced,0.0956,0.0367,0.1823,0.4152,-0.0312,22
Conservative Income,0.0678,0.0421,0.1234,0.3875,-0.0189,18
```

## 🎨 Customizing Strategies

### Modify Existing Strategy
Edit `PORTFOLIO_PROFILES` in `portfolio_config_refactored.py`:

```python
PORTFOLIO_PROFILES = {
    'aggressive': {
        'name': 'Aggressive Growth',
        'min_weight_per_asset': 0.00,  # ← Change this
        'max_weight_per_asset': 0.20,  # ← More concentration
        'target_dividend_yield': 0.015,  # ← Lower dividend
        'max_portfolio_volatility': 0.40,  # ← Higher risk
        'min_portfolio_return': 0.15,  # ← Higher target
        'max_asset_volatility': 0.60,  # ← Accept volatility
    },
    # ... other strategies
}
```

### Create Custom Strategy

```python
from portfolio_config_refactored import run_portfolio_optimization

custom_config = {
    'name': 'Tech-Focused',
    'description': 'Technology sector emphasis',
    'min_weight_per_asset': 0.01,
    'max_weight_per_asset': 0.12,
    'target_dividend_yield': 0.02,
    'max_portfolio_volatility': 0.28,
    'min_portfolio_return': 0.11,
    'max_asset_volatility': 0.45,
}

result = run_portfolio_optimization(
    config_profile=custom_config,
    mu=mu,
    avg_annual_yield=avg_annual_yield,
    # ... other parameters
)
```

## 🔧 Configuration Parameters Explained

| Parameter | Description | Aggressive | Moderate | Conservative |
|-----------|-------------|------------|----------|--------------|
| `min_weight_per_asset` | Minimum allocation per asset | 0% | 2% | 3% |
| `max_weight_per_asset` | Maximum allocation per asset | 15% | 10% | 8% |
| `target_dividend_yield` | Minimum portfolio dividend | 2% | 3.5% | 4% |
| `max_portfolio_volatility` | Maximum portfolio volatility | 35% | 22% | 15% |
| `min_portfolio_return` | Minimum expected return | 12% | 9% | 6% |
| `max_asset_volatility` | Maximum individual asset vol | 50% | 35% | 25% |

## 📈 Workflow

```
1. Data Preparation (existing notebook cells)
   ├── BiLSTM predictions
   ├── Dividend data
   └── Covariance matrices

2. Run Refactored Optimization
   ├── Filter assets by volatility (per strategy)
   ├── Create NSGA-II problem (per strategy)
   ├── Optimize (per strategy)
   └── Select best Sharpe ratio

3. Output
   ├── selected_portfolio_weights.csv
   ├── selected_portfolio_metrics.csv
   └── Detailed results dictionary
```

## 🐛 Troubleshooting

### No Valid Solutions Found
- **Issue**: Constraints too strict
- **Fix**: Relax `max_portfolio_volatility` or lower `min_portfolio_return`

### Too Few Assets in Portfolio
- **Issue**: `max_asset_volatility` filtering too many
- **Fix**: Increase threshold or check asset volatility distribution

### Low Sharpe Ratios
- **Issue**: Model predictions pessimistic or constraints conflicting
- **Fix**: Review return blending (20% predicted + 80% actual), adjust constraints

## 📝 Best Practices

1. **Start with default configurations** - They're well-balanced
2. **Check feasibility warnings** - Heed max achievable return warnings
3. **Compare strategies** - Review metrics_df to understand trade-offs
4. **Validate allocations** - Ensure diversification meets your needs
5. **Iterate configurations** - Adjust based on validation results

## 🔄 Migration from Old Code

### Before (Manual Configuration):
```python
MAX_PORTFOLIO_VOLATILITY = 0.50
MIN_PORTFOLIO_RETURN = 0.10
# ... run optimization manually
# ... extract results manually
# ... repeat for each strategy
```

### After (Automated Configuration):
```python
weights_df, metrics_df, results = generate_all_portfolios(
    mu=mu, avg_annual_yield=avg_annual_yield, ...
)
# All three strategies generated automatically!
```

## 💡 Tips

- **Aggressive**: Good for long time horizons, high risk tolerance
- **Moderate**: Balanced for most investors, good starting point
- **Conservative**: Capital preservation, income generation, near retirement
- **Custom**: Experiment with configurations for specific goals

## 📧 Questions?

Review the code comments in:
1. `portfolio_config_refactored.py` - Well-documented functions
2. `run_portfolio_strategies.py` - Complete usage examples
3. Your notebook cells - Integrated seamlessly

---

**Ready to generate your portfolios? Run the notebook cell with the snippet!** 🚀
