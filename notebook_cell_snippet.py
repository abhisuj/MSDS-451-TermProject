# ============================================================================
# ADD THIS CELL TO YOUR NOTEBOOK AFTER THE OPTIMIZATION SETUP
# Cell Title: "Generate Aggressive, Moderate, and Conservative Portfolios"
# ============================================================================

# Import the refactored portfolio optimization module
import sys
sys.path.append('.')  # Add current directory to path
from portfolio_config_refactored import generate_all_portfolios, PORTFOLIO_PROFILES

# Run optimization for all three strategies
print("\n🚀 Generating Portfolio Strategies...")
print("This will create: Aggressive, Moderate, and Conservative portfolios")
print(f"Using: {POPULATION_SIZE} population size, {N_GENERATIONS} generations\n")

weights_df, metrics_df, detailed_results = generate_all_portfolios(
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

# Display portfolio weights (non-zero allocations only)
print("\n" + "="*80)
print("PORTFOLIO ALLOCATIONS (showing allocations > 1%)")
print("="*80)

for strategy in weights_df.columns:
    print(f"\n📊 {strategy}:")
    active_weights = weights_df[strategy][weights_df[strategy] > 0.01].sort_values(ascending=False)
    print(f"   Active positions: {len(active_weights)}")
    print(f"   Largest holdings:")
    for ticker, weight in active_weights.head(5).items():
        print(f"      {ticker:8s}: {weight:7.2%}")

print("\n✅ Complete! Check files:")
print("   📄 selected_portfolio_weights.csv")
print("   📄 selected_portfolio_metrics.csv")
