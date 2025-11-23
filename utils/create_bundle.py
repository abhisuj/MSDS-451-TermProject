# File: create_bundle.py
# Description: Creates a custom Zipline data bundle using yfinance.

import pandas as pd
import yfinance as yf
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

# --- 1. Define the Tickers and Time Range ---
# Define the universe for your ETF project.
# Importantly, this includes SPY, QQQ, and IEF, which were missing from the Quandl bundle.
TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'JPM', 'V', 'JNJ', 'WMT', 'XOM', # Core stocks
    'SPY',  # S&P 500 ETF (Benchmark)
    'QQQ',  # Nasdaq 100 ETF
    'IEF',  # 7-10 Year Treasury Bond ETF (Risk-off asset)
    'GLD',  # Gold ETF
    'VNQ',  # Real Estate ETF
    'AVGO', 'NFLX', 'TSLA', 'NVDA', 'DIS', 'PYPL', 'ADBE', 'CMCSA', 'PEP', 'KO',
    'CME', 'INTC', 'CSCO', 'ABT', 'CRM', 'T', 'VZ', 'CVX', 'PFE', 'MRK',
    'AMD', 'SQ', 'UBER', 'LYFT', 'ZM', 'TWTR', 'SNAP', 'BABA', 'TCEHY', 'NIO',
]

# Define the date range for the data you want to ingest.
# Let's go from the start of 2015 to the present.
START_DATE = '2015-03-01'
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')


# --- 2. Define the Ingestion Function ---
# This function will be called by `zipline ingest`.
def yfinance_bundle(tickers, start_date, end_date):
    """
    An ingestion function to download data from Yahoo Finance
    and return a generator of (symbol, dataframe) tuples.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")

    # Download the data from yfinance
    # We use 'auto_adjust=False' to get the raw prices and handle adjustments ourselves
    # if needed, but for simplicity, we'll use the 'Adj Close' for price calculations.
    # yfinance returns a multi-index DataFrame.
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False, # Set to False to get 'Adj Close'
        group_by='ticker'
    )

    # Loop through each ticker and yield a formatted DataFrame
    for ticker in tickers:
        df = data[ticker].copy()

        # The Zipline writer expects specific column names
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividend',
            'Stock Splits': 'split',
        }, inplace=True)

        # Use 'Adj Close' for price calculations in Zipline. Zipline's default
        # 'price' column is 'close', but we can specify to use 'adjusted_close'.
        # For simplicity, we can just overwrite 'close' with 'Adj Close'.
        # A more advanced approach would be to keep both.
        if 'Adj Close' in df.columns:
            df['close'] = df['Adj Close']

        # Drop the original 'Adj Close' if it exists, as we've copied it
        df.drop(columns=['Adj Close'], inplace=True, errors='ignore')

        # Ensure the index is a timezone-aware DatetimeIndex (UTC)
        df.index = df.index.tz_localize('UTC')

        # Drop rows with missing data
        df.dropna(inplace=True)

        # Yield the ticker and its data
        if not df.empty:
            yield ticker, df


# --- 3. Register the Bundle with Zipline ---
# We use a lambda function to pass our arguments to the ingestion function.
# The register function expects a callable that it can invoke. By using a lambda,
# we create a new, argument-free function that, when called by Zipline,
# executes our yfinance_bundle function with the correct parameters.
register(
    'etf-bundle',  # This is the name of our new bundle
    lambda *args, **kwargs: yfinance_bundle(
        tickers=TICKERS, start_date=START_DATE, end_date=END_DATE
    ),
    calendar_name='NYSE',  # The trading calendar to use
)

if __name__ == '__main__':
    print("="*50)
    print("Custom Bundle Registration Script")
    print("="*50)
    print("This script registers a custom Zipline bundle named 'etf-bundle'.")
    print("To ingest the data, run the following command in your terminal:")
    print("\n    zipline ingest -b etf-bundle\n")
    print("After ingestion, you can run your Zipline algorithm with this bundle by using:")
    print("\n    zipline run -f your_algo.py --bundle etf-bundle --start 2016-01-01 --end 2023-01-01\n")
