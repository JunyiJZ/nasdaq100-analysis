import pandas as pd
import pandas_ta as ta
import os

def generate_features():
    """
    Loads the cleaned data, calculates a comprehensive set of technical indicators
    and features, and saves the result to a new CSV file.
    """
    # --- Configuration ---
    BASE_DIR = os.getcwd()
    DATA_FILE = os.path.join(BASE_DIR, 'data_quality_report.csv')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'features_technical.csv')

    # --- Load Data ---
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Error: The input file was not found at {DATA_FILE}")
        print("Please ensure 'data_quality_report.csv' exists in the same directory.")
        return
    print("Data loaded successfully.")

    # Sort data by ticker and date to ensure correct calculations for time-series indicators
    df.sort_values(by=['ticker', 'Date'], inplace=True)

    # --- Feature Engineering (Milestone 2 & 3) ---
    print("Starting feature engineering...")

    # We will group by 'ticker' to apply the indicators to each stock individually.
    # This is crucial to prevent data from one stock leaking into the calculations for another.
    grouped = df.groupby('ticker')

    all_features = []
    total_tickers = len(grouped)
    current_ticker = 0

    for name, group in grouped:
        current_ticker += 1
        print(f"  ({current_ticker}/{total_tickers}) Calculating features for {name}...")
        
        # Make a copy to avoid pandas' SettingWithCopyWarning
        group_copy = group.copy()
        
        # --- Use pandas_ta to calculate indicators ---
        # The `append=True` argument adds the new columns directly to the group_copy DataFrame.
        
        # 1. Bollinger Bands (BBL, BBM, BBU)
        group_copy.ta.bbands(length=20, append=True)
        
        # 2. Relative Strength Index (RSI)
        group_copy.ta.rsi(length=14, append=True)
        
        # 3. Moving Average Convergence Divergence (MACD)
        group_copy.ta.macd(fast=12, slow=26, append=True)
        
        # 4. Average True Range (ATR) - for volatility
        group_copy.ta.atr(length=14, append=True)
        
        # 5. Simple Moving Average (SMA)
        group_copy.ta.sma(length=20, append=True)
        
        # 6. Exponential Moving Average (EMA)
        group_copy.ta.ema(length=50, append=True)
        
        # 7. Historical Volatility (rolling standard deviation of returns)
        # First, calculate daily returns
        group_copy['returns_1d'] = group_copy['Close'].pct_change(1)
        # Then, calculate annualized rolling volatility
        group_copy['volatility_20d'] = group_copy['returns_1d'].rolling(window=20).std() * (252**0.5)
        
        # 8. Multi-day returns
        group_copy['returns_3d'] = group_copy['Close'].pct_change(3)
        group_copy['returns_5d'] = group_copy['Close'].pct_change(5)

        all_features.append(group_copy)

    # Combine all the processed groups back into a single DataFrame
    df_features = pd.concat(all_features)

    print("Feature engineering complete.")

    # --- Clean up and Save ---
    # The indicators create NaN values for the initial periods (e.g., the first 19 days for a 20-day SMA).
    # We must drop these rows as they cannot be used for modeling.
    initial_rows = len(df_features)
    df_features.dropna(inplace=True)
    final_rows = len(df_features)
    print(f"Dropped {initial_rows - final_rows} rows with NaN values. Shape is now: {df_features.shape}")

    # Save the final DataFrame with all features
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved features to {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_features()