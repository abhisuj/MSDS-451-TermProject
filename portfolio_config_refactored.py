# ============================================================================
# REFACTORED NSGA-II PORTFOLIO OPTIMIZATION - CONFIGURATION-BASED APPROACH
# ============================================================================
# This module provides a clean, configurable approach to generating
# Aggressive, Moderate, and Conservative portfolios using NSGA-II optimization
# ============================================================================

import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Portfolio Configuration Profiles
PORTFOLIO_PROFILES = {
    'aggressive': {
        'name': 'Aggressive Growth',
        'description': 'High risk, high return strategy',
        'min_weight_per_asset': 0.00,  # Allow zero weights
        'max_weight_per_asset': 0.15,  # Allow 15% max concentration
        'target_dividend_yield': 0.02,  # Lower dividend requirement (2%)
        'max_portfolio_volatility': 0.35,  # Accept higher volatility (35%)
        'min_portfolio_return': 0.12,  # Target 12%+ returns
        'max_asset_volatility': 0.50,  # Allow volatile assets (50%)
    },
    'moderate': {
        'name': 'Moderate Balanced',
        'description': 'Balanced risk-reward strategy',
        'min_weight_per_asset': 0.02,  # Minimum 2% allocation
        'max_weight_per_asset': 0.10,  # 10% max concentration
        'target_dividend_yield': 0.035,  # Moderate dividend (3.5%)
        'max_portfolio_volatility': 0.22,  # Moderate volatility (22%)
        'min_portfolio_return': 0.09,  # Target 9%+ returns
        'max_asset_volatility': 0.35,  # Moderate asset volatility (35%)
    },
    'conservative': {
        'name': 'Conservative Income',
        'description': 'Low risk, income-focused strategy',
        'min_weight_per_asset': 0.03,  # Ensure diversification (3% min)
        'max_weight_per_asset': 0.08,  # Lower concentration (8% max)
        'target_dividend_yield': 0.04,  # Higher dividend requirement (4%)
        'max_portfolio_volatility': 0.15,  # Low volatility (15%)
        'min_portfolio_return': 0.06,  # Lower return target (6%)
        'max_asset_volatility': 0.25,  # Only stable assets (25%)
    }
}


def run_portfolio_optimization(config_profile, mu, avg_annual_yield, predicted_simple_returns_filtered,
                               predicted_cov_matrices_time_series, test_dates, asset_tickers,
                               actual_test_returns, population_size=100, n_generations=100, verbose=False):
    """
    Runs NSGA-II optimization for a given portfolio configuration profile.
    
    Parameters:
    -----------
    config_profile : dict
        Configuration dictionary containing portfolio parameters
    mu : np.ndarray
        Expected returns (daily, blended)
    avg_annual_yield : np.ndarray
        Average annual dividend yields
    predicted_simple_returns_filtered : np.ndarray
        Predicted returns time series (filtered)
    predicted_cov_matrices_time_series : list
        List of predicted covariance matrices
    test_dates : pd.DatetimeIndex
        Test period dates
    asset_tickers : pd.Index
        Asset ticker symbols
    actual_test_returns : pd.DataFrame
        Actual test period returns for volatility filtering
    population_size : int
        NSGA-II population size
    n_generations : int
        Number of generations
    verbose : bool
        Print detailed output
        
    Returns:
    --------
    dict : Results containing weights, metrics, and configuration
    """
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {config_profile['name'].upper()}")
    print(f"Description: {config_profile['description']}")
    print(f"{'='*80}")
    
    # Step 1: Filter assets by volatility
    individual_asset_volatilities = actual_test_returns.std().values * np.sqrt(252)
    max_asset_vol = config_profile['max_asset_volatility']
    low_vol_mask = individual_asset_volatilities <= max_asset_vol
    
    assets_removed = (~low_vol_mask).sum()
    if verbose or assets_removed > 0:
        print(f"\n📊 Asset Filtering (volatility <= {max_asset_vol*100:.0f}%):")
        print(f"   Assets removed: {assets_removed} / {len(asset_tickers)}")
        if assets_removed > 0:
            removed_assets = asset_tickers[~low_vol_mask]
            removed_vols = individual_asset_volatilities[~low_vol_mask]
            for asset, vol in zip(removed_assets, removed_vols):
                print(f"      - {asset}: {vol:.2%}")
    
    # Apply filter
    filtered_tickers = asset_tickers[low_vol_mask]
    filtered_mu = mu[low_vol_mask]
    filtered_yields = avg_annual_yield[low_vol_mask]
    filtered_indices = np.where(low_vol_mask)[0]
    filtered_returns = predicted_simple_returns_filtered[:, filtered_indices]
    filtered_cov_matrices = [
        cov_matrix[np.ix_(filtered_indices, filtered_indices)]
        for cov_matrix in predicted_cov_matrices_time_series
    ]
    
    print(f"   Portfolio assets: {len(filtered_tickers)}")
    
    # Step 2: Create optimization problem
    from __main__ import PortfolioOptimizationProblem  # Import from notebook namespace
    
    problem = PortfolioOptimizationProblem(
        mu=filtered_mu,
        dividend_yields=filtered_yields,
        predicted_returns_timeseries=filtered_returns,
        predicted_cov_matrices_time_series=filtered_cov_matrices,
        test_dates=test_dates,
        min_weight=config_profile['min_weight_per_asset'],
        max_weight=config_profile['max_weight_per_asset'],
        target_dividend=config_profile['target_dividend_yield'],
        max_portfolio_volatility=config_profile['max_portfolio_volatility'],
        min_portfolio_return=config_profile['min_portfolio_return']
    )
    
    # Step 3: Run optimization
    algorithm = NSGA2(pop_size=population_size)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=42,  # Fixed seed for reproducibility
        verbose=verbose
    )
    
    # Step 4: Extract results
    if res.X is None or res.F is None:
        # Use population if Pareto front is empty
        pareto_weights = res.pop.get("X")
        pareto_objectives = res.pop.get("F")
    else:
        pareto_weights = res.X
        pareto_objectives = res.F
    
    # Normalize and clean
    weight_sums = pareto_weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums == 0, 1, weight_sums)
    pareto_weights = pareto_weights / weight_sums
    
    valid_mask = ~np.isnan(pareto_weights).any(axis=1) & ~np.isnan(pareto_objectives).any(axis=1)
    pareto_weights = pareto_weights[valid_mask]
    pareto_objectives = pareto_objectives[valid_mask]
    
    if len(pareto_weights) == 0:
        raise ValueError(f"No valid solutions found for {config_profile['name']}")
    
    # Step 5: Calculate metrics
    pareto_df = pd.DataFrame({
        'Return': -pareto_objectives[:, 0],
        'Dividend': -pareto_objectives[:, 1],
        'Volatility': pareto_objectives[:, 2],
        'CVaR_95': -pareto_objectives[:, 3]
    })
    
    pareto_df['Sharpe'] = np.where(
        pareto_df['Volatility'] > 0,
        (pareto_df['Return'] - 0.02) / pareto_df['Volatility'],
        0
    )
    
    # Step 6: Select best Sharpe ratio portfolio
    best_idx = pareto_df['Sharpe'].idxmax()
    best_weights = pareto_weights[best_idx]
    best_metrics = pareto_df.loc[best_idx]
    
    # Create full weight vector (including filtered-out assets with 0 weight)
    full_weights = np.zeros(len(asset_tickers))
    full_weights[low_vol_mask] = best_weights
    
    print(f"\n✅ Optimization Complete:")
    print(f"   Solutions found: {len(pareto_weights)}")
    print(f"   Best Sharpe Ratio: {best_metrics['Sharpe']:.4f}")
    print(f"   Expected Return: {best_metrics['Return']:.2%}")
    print(f"   Dividend Yield: {best_metrics['Dividend']:.2%}")
    print(f"   Volatility: {best_metrics['Volatility']:.2%}")
    print(f"   CVaR (95%): {best_metrics['CVaR_95']:.2%}")
    
    # Return results
    return {
        'profile_name': config_profile['name'],
        'weights': pd.Series(full_weights, index=asset_tickers),
        'metrics': best_metrics.to_dict(),
        'config': config_profile,
        'pareto_solutions': len(pareto_weights),
        'assets_included': len(filtered_tickers),
        'assets_filtered': assets_removed
    }


def generate_all_portfolios(mu, avg_annual_yield, predicted_simple_returns_filtered,
                           predicted_cov_matrices_time_series, test_dates, asset_tickers,
                           actual_test_returns, population_size=100, n_generations=100):
    """
    Generates Aggressive, Moderate, and Conservative portfolios and saves to CSV.
    
    Returns:
    --------
    pd.DataFrame : DataFrame with portfolio weights for all three strategies
    """
    
    print("\n" + "="*80)
    print("GENERATING PORTFOLIO STRATEGIES: AGGRESSIVE, MODERATE, CONSERVATIVE")
    print("="*80)
    
    results = {}
    all_weights = {}
    
    for profile_name in ['aggressive', 'moderate', 'conservative']:
        config = PORTFOLIO_PROFILES[profile_name]
        
        try:
            result = run_portfolio_optimization(
                config_profile=config,
                mu=mu,
                avg_annual_yield=avg_annual_yield,
                predicted_simple_returns_filtered=predicted_simple_returns_filtered,
                predicted_cov_matrices_time_series=predicted_cov_matrices_time_series,
                test_dates=test_dates,
                asset_tickers=asset_tickers,
                actual_test_returns=actual_test_returns,
                population_size=population_size,
                n_generations=n_generations,
                verbose=False
            )
            
            results[profile_name] = result
            all_weights[config['name']] = result['weights']
            
        except Exception as e:
            print(f"\n❌ ERROR in {config['name']}: {str(e)}")
            results[profile_name] = None
    
    # Create combined DataFrame
    weights_df = pd.DataFrame(all_weights)
    weights_df.index.name = 'Ticker'
    
    # Add summary metrics as additional rows
    summary_rows = {}
    for profile_name, result in results.items():
        if result:
            config_name = result['profile_name']
            summary_rows[config_name] = {
                'Expected_Return': result['metrics']['Return'],
                'Dividend_Yield': result['metrics']['Dividend'],
                'Volatility': result['metrics']['Volatility'],
                'Sharpe_Ratio': result['metrics']['Sharpe'],
                'CVaR_95': result['metrics']['CVaR_95'],
                'Assets_Included': result['assets_included'],
                'Pareto_Solutions': result['pareto_solutions']
            }
    
    # Save to CSV
    output_file = 'selected_portfolio_weights.csv'
    weights_df.to_csv(output_file)
    print(f"\n💾 Portfolio weights saved to: {output_file}")
    
    # Save summary metrics
    summary_df = pd.DataFrame(summary_rows).T
    summary_file = 'selected_portfolio_metrics.csv'
    summary_df.to_csv(summary_file)
    print(f"💾 Portfolio metrics saved to: {summary_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("PORTFOLIO COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string())
    
    return weights_df, summary_df, results
