import pandas as pd
import yfinance as yf
# Import YFRateLimitError if yfinance exposes it directly, otherwise catch general Exception
# from yfinance.utils import YFRateLimitError # Check if this path is correct or exists
import logging
import time
from datetime import datetime

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_yfinance_data(tickers, start_date, end_date):
    """Fetches historical OHLCV data using yfinance with delays and retries."""
    logging.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date} using yfinance...")
    all_data = {}
    # Use config for batch size and delays
    max_tickers_per_call = config.YFINANCE_BATCH_SIZE
    delay_between_batches = config.YFINANCE_DELAY_PER_BATCH
    max_retries = config.YFINANCE_MAX_RETRIES
    retry_delay = config.YFINANCE_RETRY_DELAY

    num_batches = (len(tickers) - 1) // max_tickers_per_call + 1

    for i in range(0, len(tickers), max_tickers_per_call):
        batch_tickers = tickers[i:i+max_tickers_per_call]
        batch_num = i // max_tickers_per_call + 1
        logging.info(f"Fetching batch {batch_num}/{num_batches} ({len(batch_tickers)} tickers starting with {batch_tickers[0]})")

        for attempt in range(max_retries):
            try:
                # Interval='1d' for daily data
                data = yf.download(
                    batch_tickers,
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    group_by='ticker',
                    auto_adjust=True,
                    threads=True, # Keep threading enabled
                    progress=False # Disable progress bar within batches for cleaner logs
                )

                if not data.empty:
                     if len(batch_tickers) == 1:
                         ticker = batch_tickers[0]
                         # Check if yf returns empty df even for single ticker success but no data
                         if isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                             all_data[ticker] = data[config.MARKET_DATA_COLS].copy()
                         else:
                             logging.warning(f"No data returned for single ticker: {ticker}")
                     else: # Multi-ticker download
                         for ticker in batch_tickers:
                             # Check if ticker column exists and has non-NaN data
                             if ticker in data.columns.get_level_values(0) and not data[ticker].dropna(how='all').empty:
                                 all_data[ticker] = data[ticker][config.MARKET_DATA_COLS].copy()
                             else:
                                 # Log only if not already logged as failed download by yfinance
                                 if ticker not in yf.shared._ERRORS:
                                      logging.warning(f"No data returned or empty data for ticker: {ticker}")
                # Add specific check for yfinance errors dictionary
                if yf.shared._ERRORS:
                     failed_tickers_in_batch = {t:e for t,e in yf.shared._ERRORS.items() if t in batch_tickers}
                     if failed_tickers_in_batch:
                          logging.error(f"Batch {batch_num} had {len(failed_tickers_in_batch)} failed downloads: {failed_tickers_in_batch}")
                          # Check if the failure was rate limiting
                          is_rate_limited = any('RateLimitError' in str(e) for e in failed_tickers_in_batch.values())
                          if is_rate_limited and attempt < max_retries - 1:
                               raise Exception("RateLimitErrorTrigger") # Raise exception to trigger retry logic

                logging.info(f"Fetched batch {batch_num}/{num_batches} successfully on attempt {attempt+1}.")
                yf.shared._ERRORS = {} # Clear errors after successful batch processing
                break # Exit retry loop on success

            except Exception as e:
                # Check if the exception is likely a rate limit error based on known yfinance behavior or our trigger
                is_rate_limit_error = 'RateLimitError' in str(e) or str(e) == "RateLimitErrorTrigger"

                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected for batch {batch_num} on attempt {attempt+1}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Error fetching batch {batch_num} starting with {batch_tickers[0]} on attempt {attempt+1}: {e}", exc_info=not is_rate_limit_error)
                    # Don't retry if it's the last attempt or not a rate limit error
                    yf.shared._ERRORS = {} # Clear errors before next batch
                    break # Exit retry loop

        # Delay before fetching the next batch
        if i + max_tickers_per_call < len(tickers): # Don't sleep after the last batch
             logging.debug(f"Waiting {delay_between_batches}s before next batch...")
             time.sleep(delay_between_batches)

    logging.info(f"Finished fetching data. Successfully retrieved data for {len(all_data)} tickers.")
    return all_data


def fetch_macro_data(macro_map, start_date, end_date):
    """Fetches macro data (e.g., VIX from yfinance, FRED data) with retries."""
    macro_data = {}
    yf_tickers = []
    # fred_codes = [] # Uncomment if using FRED
    max_retries = config.YFINANCE_MAX_RETRIES
    retry_delay = config.YFINANCE_RETRY_DELAY

    # --- Using yfinance for VIX example ---
    vix_ticker = macro_map.get('VIX')
    if vix_ticker:
        yf_tickers.append(vix_ticker)

    if yf_tickers:
        logging.info(f"Fetching macro data using yfinance: {yf_tickers}")
        for attempt in range(max_retries):
            try:
                data = yf.download(yf_tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)

                if not data.empty:
                    if len(yf_tickers) == 1:
                         # Check if data is Series (single ticker, non-empty) or DataFrame (multi-ticker or empty)
                         if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                              macro_data['VIX'] = data[['Close']].rename(columns={'Close': 'VIX'})
                         else: # Handle case where single ticker returns no data or error
                              logging.warning(f"No valid data returned for single macro ticker {yf_tickers[0]}")
                    # Add logic if fetching multiple macro tickers from yfinance
                elif yf.shared._ERRORS:
                     logging.error(f"Failed downloads for macro tickers: {yf.shared._ERRORS}")
                     is_rate_limited = any('RateLimitError' in str(e) for e in yf.shared._ERRORS.values())
                     if is_rate_limited and attempt < max_retries - 1:
                          raise Exception("RateLimitErrorTrigger") # Trigger retry
                else:
                     logging.warning(f"No macro data returned from yfinance for {yf_tickers}, and no specific errors reported.")


                yf.shared._ERRORS = {} # Clear errors
                logging.info(f"Successfully fetched macro data on attempt {attempt+1}")
                break # Exit retry loop on success

            except Exception as e:
                is_rate_limit_error = 'RateLimitError' in str(e) or str(e) == "RateLimitErrorTrigger"
                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected for macro data on attempt {attempt+1}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Error fetching macro data from yfinance on attempt {attempt+1}: {e}", exc_info=not is_rate_limit_error)
                    yf.shared._ERRORS = {} # Clear errors
                    break # Exit retry loop

    # --- Add FRED data fetching here if needed (add similar retry logic) ---
    # ...

    # Combine all macro data into a single dataframe, forward fill missing values
    if macro_data:
        # Combine potentially multiple macro sources if implemented
        combined_macro = pd.concat(macro_data.values(), axis=1)
        # Ensure daily frequency and forward fill
        # Use business day frequency for market data alignment
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        combined_macro = combined_macro.reindex(date_range).ffill() # Reindex and fill NaNs
        return combined_macro
    else:
        return pd.DataFrame()


def fetch_and_save_all_data(tickers, start_date, end_date):
    """Fetches stock and macro data and saves to files."""
    # Fetch Stock Data
    stock_data = fetch_yfinance_data(tickers, start_date, end_date)
    saved_count = 0
    valid_tickers_downloaded = list(stock_data.keys()) # Tickers we actually got data for

    # Filter the original ticker list to only those successfully downloaded
    # This ensures feature engineering doesn't try to load files for failed downloads
    tickers_to_process = [t for t in tickers if t in valid_tickers_downloaded]

    for ticker in tickers_to_process:
        df = stock_data.get(ticker) # Use .get for safety, though it should exist
        if df is not None and not df.empty:
             # Simple check for obviously bad data (e.g., all zeros)
             if 'Close' in df.columns and (df['Close'] > 1e-6).any(): # Check if Close exists and has positive values
                 # Drop rows with any NaN values *before* saving
                 df_cleaned = df.dropna(how='any')
                 if not df_cleaned.empty:
                      utils.save_data_to_file(df_cleaned, ticker)
                      saved_count += 1
                 else:
                      logging.warning(f"Skipping save for {ticker}, all rows contained NaN after dropna().")

             else:
                 logging.warning(f"Skipping save for {ticker} due to potentially bad data (e.g., zero closes or missing Close column).")
        # No need for 'else' here, already filtered by tickers_to_process

    logging.info(f"Saved data for {saved_count} stock tickers.")

    # Fetch Macro Data
    macro_df = fetch_macro_data(config.MACRO_FEATURES, start_date, end_date)
    if macro_df is not None and not macro_df.empty:
        utils.save_data_to_file(macro_df.dropna(), "_macro_data") # Save with a special name, drop NaNs
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
