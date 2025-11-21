# ============================================================================
# PORTFOLIO STRATEGY RUNNER - Usage Example
# ============================================================================
# This script demonstrates how to use the refactored portfolio optimization
# Run this in your notebook after all data preparation steps are complete
# ============================================================================

# Import the refactored module
from portfolio_config_refactored import generate_all_portfolios, run_portfolio_optimization, PORTFOLIO_PROFILES

# ============================================================================
# STEP 1: Generate all three portfolio strategies
# ============================================================================
print("Running portfolio optimization for Aggressive, Moderate, and Conservative strategies...")

weights_df, metrics_df, detailed_results = generate_all_portfolios(
    mu=mu,  # From notebook: blended expected returns
    avg_annual_yield=avg_annual_yield,  # From notebook: dividend yields
    predicted_simple_returns_filtered=predicted_simple_returns_filtered,  # From notebook
    predicted_cov_matrices_time_series=predicted_cov_matrices_time_series,  # From notebook
    test_dates=test_dates,  # From notebook
    asset_tickers=asset_tickers,  # From notebook
    actual_test_returns=actual_test_returns,  # From notebook
    population_size=POPULATION_SIZE,  # From notebook global
    n_generations=N_GENERATIONS  # From notebook global
)

# ============================================================================
# STEP 2: Display results
# ============================================================================
print("\n" + "="*80)
print("PORTFOLIO WEIGHTS (Top 10 allocations per strategy)")
print("="*80)

for col in weights_df.columns:
    print(f"\n{col}:")
    top_weights = weights_df[col][weights_df[col] > 0.01].sort_values(ascending=False).head(10)
    for ticker, weight in top_weights.items():
        print(f"  {ticker:6s}: {weight:6.2%}")

# ============================================================================
# STEP 3: Optional - Run individual strategy with custom parameters
# ============================================================================
print("\n" + "="*80)
print("CUSTOM STRATEGY EXAMPLE")
print("="*80)

# Example: Create a custom ultra-conservative strategy
custom_config = {
    'name': 'Ultra-Conservative',
    'description': 'Maximum safety, capital preservation',
    'min_weight_per_asset': 0.04,
    'max_weight_per_asset': 0.06,
    'target_dividend_yield': 0.045,
    'max_portfolio_volatility': 0.12,
    'min_portfolio_return': 0.04,
    'max_asset_volatility': 0.20,
}

try:
    custom_result = run_portfolio_optimization(
        config_profile=custom_config,
        mu=mu,
        avg_annual_yield=avg_annual_yield,
        predicted_simple_returns_filtered=predicted_simple_returns_filtered,
        predicted_cov_matrices_time_series=predicted_cov_matrices_time_series,
        test_dates=test_dates,
        asset_tickers=asset_tickers,
        actual_test_returns=actual_test_returns,
        population_size=100,
        n_generations=100,
        verbose=False
    )
    
    print("\nCustom Portfolio Top 10 Allocations:")
    top_custom = custom_result['weights'][custom_result['weights'] > 0.01].sort_values(ascending=False).head(10)
    for ticker, weight in top_custom.items():
        print(f"  {ticker:6s}: {weight:6.2%}")
        
except Exception as e:
    print(f"Custom strategy failed: {e}")

print("\n✅ Portfolio generation complete!")
print("📁 Files saved:")
print("   - selected_portfolio_weights.csv")
print("   - selected_portfolio_metrics.csv")
