# Adaptive Global Income and Growth (AGIG) ETF
## AI-Driven Multi-Objective Portfolio Optimization Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI-Powered](https://img.shields.io/badge/AI-Powered-brightgreen.svg)](https://github.com/abhisuj/MSDS-451-TermProject)

> **Note**: This project documentation and code were developed with assistance from GitHub Copilot and Claude AI to accelerate development, improve code quality, and enhance documentation comprehensiveness.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Investment Philosophy](#investment-philosophy)
- [Technical Architecture](#technical-architecture)
- [Getting Started](#getting-started)
- [Usage Instructions](#usage-instructions)
- [Strategy Configuration](#strategy-configuration)
- [Portfolio Strategies](#portfolio-strategies)
- [Performance Metrics](#performance-metrics)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [References](#references)

---

## 🎯 Overview

This research project proposes the development of a robust, adaptive investment framework, the **Adaptive Global Income and Growth (AGIG) ETF**, designed to manage portfolios dynamically over time, moving beyond simple, single-period decisions (Markowitz, 1952). The fundamental concept is to make cutting-edge quantitative research—typically reserved for large hedge funds—available to all personal investors in a low-cost ETF structure.

### Key Innovation: AI at Every Step

The starting point of this research is the Modern Portfolio Theory (MPT), pioneered by Markowitz (1952), which established the principle of maximizing return for a given level of risk. However, MPT assumes static relationships and needs to be extended to adjust to changing market conditions and times.

The AGIG ETF overcomes these limitations by integrating **Artificial Intelligence (AI)** and **Operations Research (OR)** throughout the entire investment pipeline:

**AI-Driven Components:**
1. **Predictive AI**: Bidirectional LSTM neural networks forecast asset returns and risk
2. **Optimization AI**: NSGA-II evolutionary algorithm discovers optimal portfolio allocations
3. **Feature Engineering AI**: 20+ technical indicators automatically computed per asset
4. **Risk Management AI**: CVaR optimization with tail-risk prediction
5. **Rebalancing AI**: Dynamic monthly adjustments based on market conditions

The fund's objective is to provide personalized financial planning for all age groups:
- **High Growth** for younger investors seeking capital appreciation
- **Balanced Income & Growth** for middle-aged investors (***RECOMMENDED***)
- **Controlled Income** for retirees (consistent withdrawals via the 4% rule) with dynamic risk control

---

## 💡 Investment Philosophy

### Core Principles

1. **Dynamic Adaptation**: AI-driven predictions replace static historical assumptions
2. **Multi-Objective Optimization**: Simultaneously balance return, income, volatility, and tail risk
3. **Concentrated High-Conviction**: 30 positions with top 10 holdings representing 71.5% of portfolio
4. **Risk-Managed Growth**: CVaR optimization for downside protection
5. **Transparent & Explainable**: Full methodology disclosure

### Performance Highlights (25-Year Backtest: 1999-2024)

| Metric | Hybrid Strategy | S&P 500 (SPY) |
|--------|----------------|---------------|
| Annualized Return | 12.32% | ~10.5% |
| Sharpe Ratio | 0.88 | ~0.65 |
| Dividend Yield | 3.50% | ~1.5% |
| Max Drawdown | -18.5% | -34.0% |
| Alpha | +1.80% | 0.00% |
| Beta | 0.68 | 1.00 |

---

## 🏗️ Technical Architecture

### AI Technology Stack

The AGIG ETF leverages multiple AI/ML technologies working in concert:

**Deep Learning (Prediction)**
- TensorFlow/Keras framework
- Bidirectional LSTM architecture
- 25 years of training data
- 60-day temporal sequences

**Evolutionary Algorithms (Optimization)**
- pymoo library with NSGA-II
- Multi-objective genetic algorithm
- Pareto-efficient frontier generation
- 100 population × 250 generations

**Feature Engineering (Intelligence)**
- TA-Lib technical indicators
- Custom long-term features
- Automated anomaly detection
- Volume and volatility analysis

**Risk AI (Protection)**
- CVaR tail-risk modeling
- Covariance matrix prediction
- Drawdown forecasting
- Correlation monitoring

### 1. Bidirectional LSTM Neural Network (AI Prediction Engine)
- **Purpose**: Predict next-day returns and covariance matrices for 30 assets
- **Architecture**: Multi-layer Bi-LSTM (128→64→32 units)
- **Input**: 60-day lookback window with 20+ technical indicators per asset
- **Outputs**: 
  - Predicted returns (30 assets)
  - Predicted covariance matrix (465 pairwise relationships)
- **Training**: 25 years of historical data with 64/16/20 train/validation/test split

### 2. NSGA-II Multi-Objective Optimization
- **Algorithm**: Non-dominated Sorting Genetic Algorithm II
- **Objectives** (4 competing goals):
  1. Maximize Expected Return (annualized)
  2. Maximize Dividend Yield (target ≥3%)
  3. Minimize Portfolio Volatility
  4. Minimize CVaR (95% tail risk)
- **Constraints**:
  - Weights sum to 100%
  - Dividend yield ≥ 3.0%
  - Individual asset weight: 0% - 15%
  - Population size: 100 portfolios per generation
  - Generations: 250

### 3. Portfolio Selection Strategies
Six strategies for selecting optimal portfolio from Pareto frontier:
1. **Highest Sharpe Ratio** (classic risk-adjusted return)
2. **Highest Expected Return** (aggressive growth)
3. **Minimum Volatility** (conservative income)
4. **Concentrated Portfolio** (penalty for over-diversification) ***RECOMMENDED***
5. **Sparse Portfolio** (L1 regularization, fewer holdings)
6. **Risk-Adjusted with Concentration Penalty** (hybrid approach)

---

## 🚀 Getting Started

### Prerequisites

**Required Software:**
- Python 3.8 or higher
- Jupyter Notebook or VS Code with Jupyter extension
- Git (for cloning repository)

**Hardware Recommendations:**
- RAM: 16GB minimum (32GB recommended for large datasets)
- CPU: Multi-core processor (8+ cores recommended)
- GPU: Optional but accelerates Bi-LSTM training (CUDA-compatible NVIDIA GPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/abhisuj/MSDS-451-TermProject.git
cd MSDS-451-TermProject

# Create virtual environment (optional but recommended)
python -m venv agig_env
source agig_env/bin/activate  # On Windows: agig_env\Scripts\activate

# Install dependencies
pip install ipykernel yfinance pymoo pandas numpy matplotlib seaborn scikit-learn tensorflow keras talib-binary TA-Lib polars
```

### Quick Start

```bash
# Open the prototype notebook
jupyter notebook MSDS_451_TermProject_prototype.ipynb

# Or use VS Code
code MSDS_451_TermProject_prototype.ipynb
```

---

## 📖 Usage Instructions

### Workflow Overview

The AGIG ETF implementation consists of two primary notebooks:

1. **`MSDS_451_TermProject_prototype.ipynb`** - Complete pipeline for portfolio optimization
2. **`MSDS-451-Term-Project-Backtesting.ipynb`** - Advanced backtesting and Monte Carlo simulation

### Step-by-Step Guide

#### **Phase 1: Portfolio Optimization (Prototype Notebook)**

**Cell Execution Sequence:**

1. **Cells 1-2**: Install dependencies and import libraries
   ```python
   # Run these cells first to set up environment
   ```

2. **Cell 3**: Load market data (30 assets + SPY benchmark, 25 years)
   ```python
   # Downloads price data using yfinance
   # Includes: Tech, Consumer, Energy, International, Alternatives, Fixed Income
   ```

3. **Cell 4-5**: Calculate returns and cumulative performance
   ```python
   # Generates daily returns and cumulative returns for visualization
   ```

4. **Cells 6-15**: Data exploration, correlation analysis, and technical indicators
   ```python
   # Visualize historical performance
   # Calculate correlation matrices
   # Analyze sector relationships
   ```

5. **Cell 16**: **⚙️ CONFIGURATION PARAMETERS** (***IMPORTANT - MODIFY HERE***)
   ```python
   # This cell contains all strategy parameters
   # See "Strategy Configuration" section below for detailed settings
   ```

6. **Cells 17-19**: Feature engineering with TA-Lib
   ```python
   # Creates 20+ technical indicators per asset
   # Generates training features for Bi-LSTM model
   ```

7. **Cells 20-27**: Bi-LSTM model training and prediction
   ```python
   # Builds, trains, and validates neural network
   # Generates return and covariance predictions
   # Saves model to: bilstm_best_model.keras
   ```

8. **Cells 28-36**: Dividend data processing
   ```python
   # Downloads dividend history for income optimization
   # Calculates average annual dividend yields
   ```

9. **Cell 37**: Portfolio optimization problem definition
   ```python
   # Defines PortfolioOptimizationProblem class with 4 objectives
   ```

10. **Cell 38**: Data preparation for optimization
    ```python
    # Blends historical and predicted returns
    # Filters assets by volatility and weight constraints
    ```

11. **Cell 39**: **NSGA-II OPTIMIZATION EXECUTION**
    ```python
    # Runs multi-objective optimization (250 generations)
    # Generates Pareto-efficient frontier
    # Visualizes optimal portfolios
    # Runtime: ~5-10 minutes depending on hardware
    ```

12. **Cell 40**: **PORTFOLIO SELECTION** (6 strategies compared)
    ```python
    # Compares all 6 selection strategies
    # Displays holdings, metrics, and concentration
    # Exports selected portfolio weights to CSV
    ```

13. **Cell 41**: Historical backtesting
    ```python
    # Evaluates portfolio performance on test set (2019-2024)
    # Calculates Sharpe, Sortino, max drawdown
    ```

14. **Cell 42**: Export results
    ```python
    # Saves portfolio weights, metrics, and returns to CSV files
    # Files: selected_portfolio_weights.csv, selected_portfolio_metrics.csv
    ```

**Expected Runtime (Prototype Notebook):** 40-60 minutes for complete execution

**Key Outputs:**
- `bilstm_best_model.keras` - Trained neural network model
- `selected_portfolio_weights.csv` - Optimized portfolio allocations
- `selected_portfolio_metrics.csv` - Performance metrics
- `selected_portfolio_historical_cumulative_returns.csv` - Return time series

---

#### **Phase 2: Advanced Backtesting (Backtesting Notebook)**

**Purpose:** Evaluate portfolio performance with realistic fees, transaction costs, and Monte Carlo uncertainty analysis.

**Cell Execution Sequence:**

1. **Cells 1-2**: Install dependencies
   ```python
   # Includes Zipline, PyFolio for institutional-grade backtesting
   ```

2. **Cell 3**: Load portfolio weights from Phase 1
   ```python
   # Imports: highest_sharpe_portfolio_weights.csv or selected_portfolio_weights.csv
   ```

3. **Cells 4-6**: Create custom Zipline data bundle
   ```python
   # Registers 'term-project-bundle' with 25 years of data
   # Runtime: 5-10 minutes (one-time setup)
   ```

4. **Cells 7-8**: Define backtesting algorithm with fees
   ```python
   # Management Fee: 1.0% annual
   # Performance Fee: 20% on excess returns above SPY
   # Transaction Costs: 10 bps per trade
   # Rebalancing: Monthly
   ```

5. **Cell 9**: **MONTE CARLO SIMULATION** (10,000 scenarios)
   ```python
   # Generates probabilistic return distributions
   # Accounts for fees, dividends, rebalancing costs
   # Runtime: 15-20 minutes
   ```

6. **Cells 10-11**: Calculate comprehensive metrics
   ```python
   # Sharpe, Sortino, Calmar ratios
   # Alpha, Beta vs SPY
   # Information Ratio
   # Maximum Drawdown, Recovery Time
   # VaR/CVaR at multiple confidence levels
   ```

7. **Cells 12-13**: Export results and generate summary
   ```python
   # Saves backtest results to CSV
   # Creates performance summary report
   ```

**Expected Runtime (Backtesting Notebook):** 30-45 minutes

**Key Outputs:**
- `highest_sharpe_portfolio_performance_comparison.csv` - Strategy comparison
- `backtest_returns_with_fees.csv` - Fee-adjusted return series
- `monte_carlo_simulation_results.csv` - Probabilistic scenarios

---

## ⚙️ Strategy Configuration

### Cell 16: Configuration Parameters (Prototype Notebook)

To test different portfolio strategies, modify the parameters in **Cell 16**:

```python
# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Portfolio & Data Settings
PORTFOLIO_TICKERS = [...]  # 30 assets (do not modify unless changing universe)
BENCHMARK_TICKER = 'SPY'
DATA_PERIOD = '25y'

# Feature Engineering
LOG_RETURN_CLIP = 0.20  # Clip extreme returns at ±20%

# Bi-LSTM Model Configuration
LOOKBACK_WINDOW = 60      # 60-day sequences for time-series prediction
TRAIN_SPLIT = 0.64        # 64% training data (1999-2014)
VALIDATION_SPLIT = 0.16   # 16% validation data (2014-2018)
LSTM_UNITS = [128, 64, 32]  # Layer sizes (do not modify)
DROPOUT_RATE = 0.2        # Regularization (do not modify)
EPOCHS = 100              # Training epochs with early stopping
BATCH_SIZE = 32

# ============================================================================
# NSGA-II OPTIMIZATION PARAMETERS - MODIFY FOR DIFFERENT STRATEGIES
# ============================================================================

# --- AGGRESSIVE GROWTH STRATEGY ---
# Objective: Maximize returns, accept higher volatility
# Suitable for: Ages 18-35, 20+ year horizon
# POPULATION_SIZE = 150
# N_GENERATIONS = 300
# TARGET_DIVIDEND_YIELD = 0.02         # 2% (lower priority)
# MIN_WEIGHT_PER_ASSET = 0.00          # Allow full exclusion
# MAX_WEIGHT_PER_ASSET = 0.20          # 20% max (higher concentration)
# MAX_ASSET_VOLATILITY = 1.00          # 100% (accept high-volatility assets)

# --- HYBRID BALANCED STRATEGY (RECOMMENDED) ---
# Objective: Balance growth, income, and risk
# Suitable for: Ages 35-65, 10-20 year horizon
POPULATION_SIZE = 100                  # Moderate diversity
N_GENERATIONS = 250                    # Balanced exploration
TARGET_DIVIDEND_YIELD = 0.03           # 3% target yield
MIN_WEIGHT_PER_ASSET = 0.00            # Allow exclusion of weak assets
MAX_WEIGHT_PER_ASSET = 0.15            # 15% max (concentrated but diversified)
MAX_ASSET_VOLATILITY = 0.90            # 90% (moderate risk tolerance)

# --- CONSERVATIVE INCOME STRATEGY ---
# Objective: Minimize volatility, maximize dividends
# Suitable for: Ages 65+, retirees, <10 year horizon
# POPULATION_SIZE = 80
# N_GENERATIONS = 200
# TARGET_DIVIDEND_YIELD = 0.04         # 4% high yield requirement
# MIN_WEIGHT_PER_ASSET = 0.02          # Minimum 2% (force diversification)
# MAX_WEIGHT_PER_ASSET = 0.12          # 12% max (prevent over-concentration)
# MAX_ASSET_VOLATILITY = 0.60          # 60% (exclude high-volatility assets)
```

### How to Switch Strategies

**To Test Aggressive Growth:**
1. Open Cell 16 in the prototype notebook
2. Comment out the "HYBRID BALANCED" section (add `#` before each line)
3. Uncomment the "AGGRESSIVE GROWTH" section (remove `#`)
4. Run Cell 16 and continue with Cells 37-42 (skip re-running Cells 1-36)

**To Test Conservative Income:**
1. Open Cell 16 in the prototype notebook
2. Comment out the "HYBRID BALANCED" section
3. Uncomment the "CONSERVATIVE INCOME" section
4. Run Cell 16 and continue with Cells 37-42

**To Test Custom Settings:**
1. Copy one of the strategy configurations
2. Modify parameters based on your objectives:
   - ↑ `POPULATION_SIZE` + ↑ `N_GENERATIONS` = More diversified, smoother portfolios
   - ↓ `POPULATION_SIZE` + ↓ `N_GENERATIONS` = More concentrated, aggressive portfolios
   - ↑ `MAX_WEIGHT_PER_ASSET` = Allow more concentration in top holdings
   - ↑ `TARGET_DIVIDEND_YIELD` = Prioritize income over growth
   - ↓ `MAX_ASSET_VOLATILITY` = Exclude risky assets

---

## 📊 Portfolio Strategies

### Strategy Comparison Matrix

| Strategy | Sharpe Ratio | Annual Return | Volatility | Dividend Yield | Active Assets | Suitable For |
|----------|-------------|---------------|------------|----------------|---------------|--------------|
| **Aggressive Growth** | 3.15 | 23.59% | 6.85% | 2.61% | 28 | Ages 18-35 |
| **Hybrid Balanced** ⭐ | 0.88 | 12.32% | 11.74% | 3.50% | 30 | Ages 35-65 |
| **Conservative Income** | 0.69 | 9.80% | 11.23% | 3.43% | 14 | Ages 65+ |

### Selection Strategy Options (Cell 40)

The prototype notebook implements **6 portfolio selection strategies** from the Pareto frontier:

1. **Strategy 1: Highest Sharpe Ratio**
   - Classic risk-adjusted return optimization
   - Best for: Traditional mean-variance investors

2. **Strategy 2: Highest Expected Return**
   - Maximizes predicted returns (ignores risk)
   - Best for: Aggressive investors with high risk tolerance

3. **Strategy 3: Minimum Volatility**
   - Minimizes portfolio standard deviation
   - Best for: Conservative investors prioritizing stability

4. **Strategy 4: Concentrated Portfolio** ⭐ ***RECOMMENDED***
   - Concentration Score = Sharpe - 0.01 × NumAssets
   - Penalizes over-diversification
   - Best for: High-conviction active management

5. **Strategy 5: Sparse Portfolio**
   - Uses L1 regularization to minimize number of holdings
   - Best for: Simplicity, easier rebalancing

6. **Strategy 6: Risk-Adjusted Concentration**
   - Balances Sharpe ratio with concentration penalty
   - Best for: Hybrid approach between strategies 1 and 4

**Default Recommendation:** Strategy 4 (Concentrated) provides the best balance of risk-adjusted returns and focused capital allocation.

---

## 📈 Performance Metrics

### Key Metrics Calculated

**Return Metrics:**
- Cumulative Return
- Annualized Return (CAGR)
- Annualized Volatility (Standard Deviation)
- Downside Deviation (negative returns only)

**Risk-Adjusted Metrics:**
- Sharpe Ratio = (Return - Risk-Free Rate) / Volatility
- Sortino Ratio = (Return - Risk-Free Rate) / Downside Deviation
- Calmar Ratio = Annualized Return / Max Drawdown
- Information Ratio = Active Return / Tracking Error

**Risk Metrics:**
- Maximum Drawdown (peak-to-trough decline)
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Beta (market sensitivity vs SPY)
- Alpha (excess return vs SPY)

**Income Metrics:**
- Dividend Yield (annual)
- Total Return (price appreciation + dividends)

---

## 📁 File Structure

```
MSDS-451-TermProject/
├── MSDS_451_TermProject_prototype.ipynb       # Main optimization pipeline
├── MSDS-451-Term-Project-Backtesting.ipynb    # Advanced backtesting
├── AGIG_ETF_Optimization.ipynb                # Consolidated notebook (in progress)
├── AGIG_ETF_Investor_Prospectus.md            # Investor documentation
├── AGIG_CONSOLIDATION_GUIDE.md                # Notebook consolidation guide
├── AGIG_CODE_SNIPPETS.py                      # Reusable code snippets
├── PORTFOLIO_CONCENTRATION_GUIDE.md           # Concentration strategies
├── README.md                                  # This file
│
├── Output Files (Generated):
│   ├── bilstm_best_model.keras                        # Trained Bi-LSTM model
│   ├── selected_portfolio_weights.csv                 # Portfolio allocations
│   ├── selected_portfolio_metrics.csv                 # Performance metrics
│   ├── selected_portfolio_historical_cumulative_returns.csv
│   ├── highest_sharpe_portfolio_weights.csv           # Sharpe-optimized weights
│   ├── highest_sharpe_portfolio_weights_comparison.csv
│   ├── highest_sharpe_portfolio_performance_comparison.csv
│   ├── efficient_frontier_portfolios.csv              # Pareto frontier solutions
│   └── backtest_results/                              # Backtesting outputs
│
└── utils/
    ├── create_bundle.py                       # Zipline bundle creation utility
    └── __pycache__/
```

---

## 🔧 Dependencies

### Required Python Packages

```
# Core Data & Computation
numpy>=1.21.0
pandas>=1.3.0
polars>=0.15.0

# Machine Learning & AI
tensorflow>=2.8.0
keras>=2.8.0
scikit-learn>=1.0.0

# Financial Data
yfinance>=0.2.0

# Technical Analysis
TA-Lib>=0.4.24
talib-binary>=0.4.19  # Alternative for Windows

# Optimization
pymoo>=0.6.0

# Backtesting & Performance
zipline-reloaded>=3.0.0
pyfolio-reloaded>=0.9.5

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Jupyter
ipykernel>=6.0.0
```

### Installation Notes

**For Windows Users (TA-Lib installation):**
```bash
# Option 1: Use pre-compiled binary
pip install TA-Lib-binary

# Option 2: Install from wheel file
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp38‑cp38‑win_amd64.whl
```

**For Mac/Linux Users:**
```bash
# Install TA-Lib C library first
# Mac:
brew install ta-lib

# Ubuntu/Debian:
sudo apt-get install ta-lib

# Then install Python wrapper:
pip install TA-Lib
```

---

## 🎓 Methodology & Theory

### 1. Modern Portfolio Theory (MPT) Extensions

**Markowitz (1952) Foundation:**
- Mean-variance optimization
- Efficient frontier
- Risk-return tradeoff

**AGIG Enhancements:**
- **Dynamic Predictions**: Replace historical mean/covariance with AI forecasts
- **Multi-Period**: Continuous rebalancing vs single-period optimization
- **Multi-Objective**: 4 objectives vs single Sharpe ratio
- **Tail Risk**: CVaR optimization vs variance minimization

### 2. Deep Learning for Financial Forecasting

**Bidirectional LSTM Advantages:**
- Captures long-term dependencies (60-day sequences)
- Bidirectional processing (past + future context)
- Handles non-linear relationships
- Adapts to regime changes

**Feature Engineering:**
- Trend indicators (SMA, EMA, MACD)
- Momentum oscillators (RSI, Stochastic)
- Volatility measures (ATR, Bollinger Bands)
- Volume analysis (OBV, anomaly detection)
- Long-term features (252-day, 500-day metrics)

### 3. Multi-Objective Evolutionary Algorithms

**NSGA-II Key Concepts:**
- **Pareto Dominance**: Solution A dominates B if A is better in ≥1 objective and not worse in others
- **Non-Dominated Sorting**: Ranks solutions into Pareto fronts
- **Crowding Distance**: Maintains diversity in population
- **Elitism**: Best solutions always survive

**Why NSGA-II for Portfolio Optimization:**
- No need to specify objective weights a priori
- Generates multiple optimal solutions (Pareto frontier)
- Handles conflicting objectives naturally
- Robust to non-convex problems

---

## 🔬 Research Applications

### Academic Use Cases

1. **Financial Engineering Courses**
   - Portfolio optimization algorithms
   - Multi-objective decision making
   - AI in finance applications

2. **Machine Learning Projects**
   - Time-series forecasting with LSTMs
   - Feature engineering for financial data
   - Model validation and backtesting

3. **Operations Research**
   - Evolutionary algorithms
   - Constraint satisfaction problems
   - Pareto efficiency analysis

### Industry Applications

1. **Asset Management**
   - Robo-advisor development
   - Quantitative portfolio construction
   - Risk management systems

2. **Fintech Startups**
   - Retail investment platforms
   - Automated rebalancing services
   - Personalized wealth management

3. **Institutional Investors**
   - Pension fund optimization
   - Endowment portfolio management
   - Multi-strategy fund allocation

---

## 🤝 Contributing

### AI-Assisted Development

This project was developed with assistance from:
- **GitHub Copilot**: Code completion, function generation, debugging assistance
- **Claude AI (Anthropic)**: Documentation writing, architecture design, code review
- **Human Oversight**: All AI-generated content reviewed and validated by project maintainers

Contributions are welcome! Areas for improvement:

1. **Model Enhancements**
   - Transformer architectures for time-series
   - Ensemble methods (LSTM + GRU + Attention)
   - Alternative loss functions
   - GPT-based sentiment analysis integration

2. **Optimization Algorithms**
   - MOEA/D (decomposition-based)
   - SPEA2 (strength Pareto)
   - Custom constraint handling

3. **Feature Engineering**
   - Sentiment analysis from news
   - Macroeconomic indicators
   - Alternative data sources

4. **Backtesting Improvements**
   - Live paper trading integration
   - More realistic transaction costs
   - Tax-aware rebalancing

**To Contribute:**
```bash
# Fork the repository
git checkout -b feature/your-feature-name
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Submit pull request
```

---

## 📚 References

### Academic Papers

#### Foundation & Theory

1. **Markowitz, H. (1952).** "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91. https://doi.org/10.2307/2975974
   - Seminal paper establishing Modern Portfolio Theory and mean-variance optimization

2. **Markowitz, H. (1991).** "Foundations of Portfolio Theory." *The Journal of Finance*, 46(2), 466-477. https://doi.org/10.2307/2328831
   - Theoretical extensions and refinements to MPT framework

3. **Hochreiter, S., & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
   - Foundation of LSTM architecture for sequence modeling

#### Machine Learning & Optimization

4. **Ban, G., El Karoui, N., & Lim, A. (2018).** "Machine Learning and Portfolio Optimization." *Management Science*, 64(3), 1136-1154. https://www.jstor.org/stable/48748004
   - Integration of ML with portfolio optimization, Performance-Based Regularization (PBR) for covariance estimation

5. **Deb, K., et al. (2002).** "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
   - Complete NSGA-II algorithm specification and theoretical foundations

6. **Kaucic, M., Moradi, M., & Mirzazadeh, M. (2019).** "Portfolio optimization by improved NSGA-II and SPEA 2 based on different risk measures." *Financial Innovation*, 5(1), 1-28. https://doi.org/10.1186/s40854-019-0140-6
   - Application of NSGA-II to portfolio problems with CVaR, Sharpe ratio, and multiple risk measures

7. **Gaurav, A., Baishnab, K., & Singh, P. K. (2025).** "Intelligent ESG portfolio optimization: A multi-objective AI-driven framework for sustainable investments in the Indian stock market." *Sustainable Futures*, 9. https://doi.org/10.1016/j.sftr.2025.100832
   - Multi-objective AI frameworks, Bi-LSTM performance validation for portfolio optimization

#### Multi-Period & Advanced Methods

8. **Liu, Y., Zhang, W., & Xu, W. (2012).** "Fuzzy multi-period portfolio selection optimization models using multiple criteria." *Automatica*, 48(12), 3042-3053. https://doi.org/10.1016/j.automatica.2012.08.036
   - Multi-period portfolio optimization with fuzzy logic and multiple criteria

9. **Ren, X., Sun, R., Jiang, Z., Stefanidis, A., Liu, H., & Su, J. (2025).** "Time series is not enough: Financial Transformer Reinforcement Learning for portfolio management." *Neurocomputing*, 647. https://doi.org/10.1016/j.neucom.2025.130451
   - Advanced Transformer + RL architectures for portfolio management (FTRL methodology)

### Books

- **Grinold, R. C., & Kahn, R. N. (2000).** *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk.* McGraw-Hill.
  - Industry-standard reference for active portfolio management

- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
  - Modern ML techniques applied to finance, including backtesting and feature engineering

- **Deb, K. (2001).** *Multi-Objective Optimization Using Evolutionary Algorithms.* Wiley.
  - Comprehensive treatment of multi-objective optimization theory and algorithms

### Industry Resources & Guides

- **Chen, J. (2025).** "Conditional Value at Risk (CVaR): Expert Guide, Uses, and Formula." *Investopedia*. https://www.investopedia.com/terms/c/conditional_value_at_risk.asp
  - Practical guide to CVaR calculation and interpretation for tail risk management

- **Waterworth, K. (2025).** "2 Data Center REITs to Consider in 2025." *The Motley Fool*. https://www.fool.com/investing/stock-market/market-sectors/real-estate-investing/reit/data-center-reit/
  - Alternative asset analysis: data center REITs and real estate investment trends

### Software & Libraries

- **TensorFlow/Keras**: https://www.tensorflow.org/
  - Deep learning framework for Bi-LSTM implementation

- **pymoo**: https://pymoo.org/
  - Multi-objective optimization library with NSGA-II, MOEA/D, and other algorithms
  - **NSGA-II Documentation**: "NSGA-II: Non-dominated Sorting Genetic Algorithm." https://pymoo.org/algorithms/moo/nsga2.html

- **Zipline Reloaded**: https://github.com/quantopian/zipline
  - Event-driven backtesting engine with realistic transaction costs and slippage

- **PyFolio Reloaded**: https://github.com/quantopian/pyfolio
  - Portfolio performance and risk analysis library

- **TA-Lib**: https://ta-lib.org/
  - Technical analysis library for indicator calculation (RSI, MACD, Bollinger Bands, etc.)

---

## 📞 Contact & Support

**Project Maintainer:** MSDS-451 Research Team  
**Institution:** Northwestern University - Masters in Data Science  
**Course:** MSDS 451 - Deep Learning  
**GitHub:** https://github.com/abhisuj/MSDS-451-TermProject

**For Questions:**
- Open an issue on GitHub
- Email: [Your institutional email]

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

**IMPORTANT INVESTMENT DISCLOSURE:**

This software is for **educational and research purposes only**. It does not constitute investment advice, financial advice, trading advice, or any other sort of advice. The AGIG ETF is a **hypothetical research concept** and is not a registered investment product.

**AI Usage Disclosure:**

This project and its documentation were developed with assistance from AI tools (GitHub Copilot, Claude AI). While AI assistance accelerated development and improved code quality, all outputs have been reviewed, validated, and tested by human developers. AI-generated code and strategies should not be used for actual trading without thorough independent verification.

**Key Disclaimers:**

1. **No Investment Recommendation**: Nothing in this repository should be construed as a recommendation to buy, sell, or hold any security.

2. **Past Performance**: Historical backtesting results do not guarantee future performance. Markets can behave differently than historical patterns.

3. **Model Risk**: Machine learning models can fail unpredictably, especially during unprecedented market conditions.

4. **Loss of Capital**: All investments involve risk of loss. You can lose some or all of your invested capital.

5. **No Registration**: This is not a registered investment advisor, broker-dealer, or fund. Do not use this for actual investment decisions without consulting qualified professionals.

6. **Regulatory Compliance**: If you adapt this code for commercial use, ensure compliance with all applicable securities regulations (SEC, FINRA, etc.).

7. **No Warranty**: This software is provided "as is" without warranty of any kind, express or implied.

**Before making any investment decisions, consult with qualified financial advisors who understand your individual circumstances, goals, and risk tolerance.**

---

## 🎯 Project Status

**Version:** 1.0  
**Last Updated:** November 23, 2025  
**Status:** Active Research Project

**Completed:**
- ✅ Bi-LSTM model implementation
- ✅ NSGA-II multi-objective optimization
- ✅ 6 portfolio selection strategies
- ✅ 25-year historical backtesting
- ✅ Monte Carlo simulation framework
- ✅ Comprehensive performance metrics
- ✅ Investor prospectus documentation

**In Progress:**
- 🔄 Notebook consolidation (AGIG_ETF_Optimization.ipynb)
- 🔄 Real-time prediction API
- 🔄 Interactive dashboard (Streamlit/Dash)

**Future Enhancements:**
- 📋 Transformer-based architectures
- 📋 Sentiment analysis integration
- 📋 Tax-loss harvesting optimization
- 📋 ESG (Environmental, Social, Governance) constraints
- 📋 Live paper trading integration

---

**Thank you for your interest in the Adaptive Global Income and Growth (AGIG) ETF research project!**

*Making institutional-grade quantitative investment management accessible to all.*
