import pandas as pd
import yfinance as yf
# Import YFRateLimitError if yfinance exposes it directly, otherwise catch general Exception
# from yfinance.utils import YFRateLimitError # Check if this path is correct or exists
import logging
import time
from datetime import datetime, timedelta # Make sure timedelta is imported

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
                             # Ensure required columns exist before trying to access
                             if all(col in data.columns for col in config.MARKET_DATA_COLS):
                                 all_data[ticker] = data[config.MARKET_DATA_COLS].copy()
                             else:
                                 logging.warning(f"Required columns missing for single ticker {ticker}. Columns found: {data.columns.tolist()}")
                         else:
                             # Check yfinance errors explicitly for the single ticker case
                             if ticker not in yf.shared._ERRORS:
                                  logging.warning(f"No data returned or empty dataframe for single ticker: {ticker}")
                     else: # Multi-ticker download
                         for ticker in batch_tickers:
                             # Check if ticker column exists and has non-NaN data
                             if ticker in data.columns.get_level_values(0) and not data[ticker].dropna(how='all').empty:
                                  # Ensure required columns exist in the sub-dataframe
                                  ticker_df = data[ticker]
                                  if all(col in ticker_df.columns for col in config.MARKET_DATA_COLS):
                                      all_data[ticker] = ticker_df[config.MARKET_DATA_COLS].copy()
                                  else:
                                      logging.warning(f"Required columns missing for ticker {ticker} in multi-download. Columns found: {ticker_df.columns.tolist()}")

                             else:
                                 # Log only if not already logged as failed download by yfinance
                                 if ticker not in yf.shared._ERRORS:
                                      logging.warning(f"No data returned or empty data for ticker: {ticker}")
                # Add specific check for yfinance errors dictionary
                if yf.shared._ERRORS:
                     # Filter errors specific to the current batch
                     failed_tickers_in_batch = {t:e for t,e in yf.shared._ERRORS.items() if t in batch_tickers}
                     if failed_tickers_in_batch:
                          logging.error(f"Batch {batch_num} had {len(failed_tickers_in_batch)} failed downloads: {failed_tickers_in_batch}")
                          # Check if the failure was rate limiting to trigger retry
                          is_rate_limited = any('RateLimitError' in str(e) or 'Too Many Requests' in str(e) for e in failed_tickers_in_batch.values())
                          if is_rate_limited and attempt < max_retries - 1:
                               raise Exception("RateLimitErrorTrigger") # Raise exception to trigger retry logic

                logging.info(f"Fetched batch {batch_num}/{num_batches} successfully on attempt {attempt+1}.")
                yf.shared._ERRORS = {} # Clear errors after successful batch processing
                break # Exit retry loop on success

            except Exception as e:
                # Check if the exception is likely a rate limit error based on known yfinance behavior or our trigger
                is_rate_limit_error = 'RateLimitError' in str(e) or 'Too Many Requests' in str(e) or str(e) == "RateLimitErrorTrigger"

                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected for batch {batch_num} on attempt {attempt+1}. Retrying in {retry_delay}s...")
                    yf.shared._ERRORS = {} # Clear errors before retry
                    time.sleep(retry_delay)
                else:
                    # Log the actual error if it's not a retryable rate limit or if retries exhausted
                    logging.error(f"Error fetching batch {batch_num} starting with {batch_tickers[0]} on attempt {attempt+1}: {e}", exc_info=not is_rate_limit_error)
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
    macro_data_dict = {} # Use a dictionary to store individual series first
    yf_tickers = []
    max_retries = config.YFINANCE_MAX_RETRIES
    retry_delay = config.YFINANCE_RETRY_DELAY

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
                         single_ticker = yf_tickers[0]
                         # Check if data is Series or DataFrame and has 'Close'
                         if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                              # Select the 'Close' column Series
                              vix_series = data['Close']
                              # Assign the desired name attribute
                              vix_series.name = 'VIX'
                              macro_data_dict['VIX'] = vix_series # Store the named Series
                         elif isinstance(data, pd.Series) and data.name.upper() == 'CLOSE': # Check if it's already a Close series
                              vix_series = data
                              vix_series.name = 'VIX'
                              macro_data_dict['VIX'] = vix_series
                         else:
                              logging.warning(f"No valid 'Close' data structure returned for single macro ticker {single_ticker}. Type: {type(data)}")
                    # Add logic here if fetching multiple yfinance macro tickers
                elif yf.shared._ERRORS:
                     logging.error(f"Failed downloads for macro tickers: {yf.shared._ERRORS}")
                     is_rate_limited = any('RateLimitError' in str(e) or 'Too Many Requests' in str(e) for e in yf.shared._ERRORS.values())
                     if is_rate_limited and attempt < max_retries - 1:
                          raise Exception("RateLimitErrorTrigger") # Trigger retry
                else:
                     logging.warning(f"No macro data returned from yfinance for {yf_tickers}, and no specific errors reported.")

                yf.shared._ERRORS = {} # Clear errors after successful attempt
                logging.info(f"Successfully fetched yfinance macro data on attempt {attempt+1}")
                break # Exit retry loop on success

            except Exception as e:
                is_rate_limit_error = 'RateLimitError' in str(e) or 'Too Many Requests' in str(e) or str(e) == "RateLimitErrorTrigger"
                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected for macro data on attempt {attempt+1}. Retrying in {retry_delay}s...")
                    yf.shared._ERRORS = {} # Clear errors before retry
                    time.sleep(retry_delay)
                else:
                    # Log actual error if not retryable or retries exhausted
                    logging.error(f"Error fetching macro data from yfinance on attempt {attempt+1}: {e}", exc_info=not is_rate_limit_error)
                    yf.shared._ERRORS = {} # Clear errors before breaking
                    break # Exit retry loop

    # --- Add FRED data fetching here if needed ---
    # fred_api_key = os.getenv('FRED_API_KEY') ... etc ...
    # Store results in macro_data_dict['FRED_CODE'] = series (apply similar retry logic)

    # Combine all macro data into a single dataframe
    if macro_data_dict:
        # Concatenate the Series/DataFrames stored in the dictionary
        combined_macro = pd.concat(macro_data_dict.values(), axis=1) # Keys are now Series names

        # If concat resulted in MultiIndex columns (less likely now but possible if values were DataFrames), flatten
        if isinstance(combined_macro.columns, pd.MultiIndex):
             logging.warning("Combined macro data has MultiIndex columns after concat. Flattening.")
             combined_macro.columns = combined_macro.columns.get_level_values(-1) # Often the actual name is last level
             combined_macro = combined_macro.loc[:,~combined_macro.columns.duplicated()]

        logging.info(f"Combined Macro Columns Before Reindex: {combined_macro.columns.tolist()}")

        # Ensure daily frequency and forward fill using Business Day frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        combined_macro = combined_macro.reindex(date_range).ffill()

        # Final check before returning
        if isinstance(combined_macro.columns, pd.MultiIndex):
             logging.error("FATAL: Macro data STILL has MultiIndex columns after processing!")
             return pd.DataFrame()
        if isinstance(combined_macro.index, pd.MultiIndex):
             logging.error("FATAL: Macro data has MultiIndex rows after processing!")
             return pd.DataFrame()

        logging.info(f"Processed Macro Data Columns: {combined_macro.columns.tolist()}")
        return combined_macro
    else:
        logging.info("No macro data series were successfully fetched or processed.")
        return pd.DataFrame()


def fetch_and_save_all_data(tickers, start_date, end_date):
    """Fetches stock and macro data and saves to files."""
    # Fetch Stock Data
    stock_data = fetch_yfinance_data(tickers, start_date, end_date)
    saved_count = 0
    valid_tickers_downloaded = list(stock_data.keys())

    # Filter the original ticker list to only those successfully downloaded
    tickers_to_process = [t for t in tickers if t in valid_tickers_downloaded]

    for ticker in tickers_to_process:
        df = stock_data.get(ticker)
        if df is not None and not df.empty:
             # Check for 'Close' column and positive values
             if 'Close' in df.columns and (df['Close'] > 1e-6).any():
                 # Drop rows with any NaN values *before* saving
                 df_cleaned = df.dropna(how='any')
                 if not df_cleaned.empty:
                      utils.save_data_to_file(df_cleaned, ticker)
                      saved_count += 1
                 else:
                      logging.warning(f"Skipping save for {ticker}, all rows contained NaN after dropna().")
             else:
                 logging.warning(f"Skipping save for {ticker} due to potentially bad data (e.g., zero closes or missing Close column).")

    logging.info(f"Saved data for {saved_count} stock tickers.")

    # Fetch Macro Data (already processed and cleaned)
    macro_df = fetch_macro_data(config.MACRO_FEATURES, start_date, end_date)
    if macro_df is not None and not macro_df.empty:
        # Save the potentially NaN-dropped (from reindex/ffill) DataFrame
        utils.save_data_to_file(macro_df.dropna(how='all'), "_macro_data") # Drop rows that might be all NaN after ffill
        logging.info("Saved macro data.")
    else:
        logging.warning("No final macro data DataFrame to save.")


if __name__ == "__main__":
    logging.info("--- Starting Data Fetching Process ---")
    # Add a small delay before starting, sometimes helps with initial rate limits
    time.sleep(2)
    sp500_tickers = utils.get_sp500_tickers()

    # Determine overall date range needed (training start to "yesterday")
    fetch_start = config.START_DATE
    # Ensure end_date is calculated correctly relative to 'now'
    fetch_end = datetime.now().strftime('%Y-%m-%d')
    logging.info(f"Data fetch range: {fetch_start} to {fetch_end}")


    fetch_and_save_all_data(sp500_tickers, fetch_start, fetch_end)
    logging.info("--- Data Fetching Process Completed ---")
