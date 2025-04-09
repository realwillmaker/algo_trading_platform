import pandas as pd
import yfinance as yf # Using yfinance for demo
# import alpaca_trade_api as tradeapi # Example for Alpaca
# from fredapi import Fred # Example for FRED
import logging
import time
from datetime import datetime

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_yfinance_data(tickers, start_date, end_date):
    """Fetches historical OHLCV data using yfinance."""
    logging.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date} using yfinance...")
    all_data = {}
    max_tickers_per_call = 100 # yfinance might handle many, but good practice to batch
    delay_between_batches = 1 # seconds

    for i in range(0, len(tickers), max_tickers_per_call):
        batch_tickers = tickers[i:i+max_tickers_per_call]
        try:
            # Interval='1d' for daily data
            data = yf.download(batch_tickers, start=start_date, end=end_date, interval='1d', group_by='ticker', auto_adjust=True, threads=True)

            if not data.empty:
                 if len(batch_tickers) == 1: # yfinance returns DataFrame directly if only one ticker
                     ticker = batch_tickers[0]
                     if not data.empty:
                         all_data[ticker] = data[config.MARKET_DATA_COLS] # Select needed columns
                 else: # Multi-ticker download returns MultiIndex DataFrame
                     for ticker in batch_tickers:
                         if ticker in data and not data[ticker].dropna(how='all').empty:
                             all_data[ticker] = data[ticker][config.MARKET_DATA_COLS].copy() # Select needed columns
                         else:
                             logging.warning(f"No data returned for ticker: {ticker}")
            else:
                logging.warning(f"No data returned for batch starting with {batch_tickers[0]}")

            logging.info(f"Fetched batch {i//max_tickers_per_call + 1}/{(len(tickers)-1)//max_tickers_per_call + 1}")
            time.sleep(delay_between_batches) # Be nice to the API

        except Exception as e:
            logging.error(f"Error fetching batch starting with {batch_tickers[0]}: {e}")
            time.sleep(delay_between_batches * 2) # Longer delay on error

    logging.info(f"Finished fetching data for {len(all_data)} tickers.")
    return all_data


def fetch_macro_data(macro_map, start_date, end_date):
    """Fetches macro data (e.g., VIX from yfinance, FRED data)."""
    macro_data = {}
    yf_tickers = []
    # fred_codes = [] # Uncomment if using FRED

    # --- Using yfinance for VIX example ---
    vix_ticker = macro_map.get('VIX')
    if vix_ticker:
        yf_tickers.append(vix_ticker)

    if yf_tickers:
        try:
            logging.info(f"Fetching macro data (VIX) using yfinance: {yf_tickers}")
            data = yf.download(yf_tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True)
            if not data.empty:
                 if len(yf_tickers) == 1:
                      macro_data['VIX'] = data[['Close']].rename(columns={'Close': 'VIX'})
                 # Add logic if fetching multiple macro tickers from yfinance
            else:
                 logging.warning(f"No macro data returned from yfinance for {yf_tickers}")
        except Exception as e:
            logging.error(f"Error fetching macro data from yfinance: {e}")

    # --- Add FRED data fetching here if needed ---
    # fred_api_key = os.getenv('FRED_API_KEY')
    # if fred_api_key and fred_codes:
    #     fred = Fred(api_key=fred_api_key)
    #     logging.info(f"Fetching macro data from FRED: {fred_codes}")
    #     for code in fred_codes:
    #         try:
    #             series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
    #             macro_data[code] = series.to_frame(name=code)
    #             time.sleep(0.5) # Be nice to FRED API
    #         except Exception as e:
    #             logging.error(f"Error fetching FRED series {code}: {e}")

    # Combine all macro data into a single dataframe, forward fill missing values
    if macro_data:
        combined_macro = pd.concat(macro_data.values(), axis=1)
        # Ensure daily frequency and forward fill
        date_range = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
        combined_macro = combined_macro.reindex(date_range).ffill()
        return combined_macro
    else:
        return pd.DataFrame()


def fetch_and_save_all_data(tickers, start_date, end_date):
    """Fetches stock and macro data and saves to files."""
    # Fetch Stock Data
    stock_data = fetch_yfinance_data(tickers, start_date, end_date)
    saved_count = 0
    for ticker, df in stock_data.items():
        if df is not None and not df.empty:
             # Simple check for obviously bad data (e.g., all zeros)
             if (df['Close'] > 0).any():
                 utils.save_data_to_file(df.dropna(), ticker) # Drop rows with any NaN before saving
                 saved_count += 1
             else:
                 logging.warning(f"Skipping save for {ticker} due to potentially bad data (e.g., all zero closes).")
        else:
             logging.warning(f"No data fetched or empty dataframe for {ticker}, not saving.")
    logging.info(f"Saved data for {saved_count} stock tickers.")

    # Fetch Macro Data
    macro_df = fetch_macro_data(config.MACRO_FEATURES, start_date, end_date)
    if not macro_df.empty:
        utils.save_data_to_file(macro_df, "_macro_data") # Save with a special name
        logging.info("Saved macro data.")
    else:
        logging.warning("No macro data was fetched or generated.")


if __name__ == "__main__":
    logging.info("--- Starting Data Fetching Process ---")
    sp500_tickers = utils.get_sp500_tickers()

    # Determine overall date range needed (training start to "yesterday")
    fetch_start = config.START_DATE
    fetch_end = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    fetch_and_save_all_data(sp500_tickers, fetch_start, fetch_end)
    logging.info("--- Data Fetching Process Completed ---")
