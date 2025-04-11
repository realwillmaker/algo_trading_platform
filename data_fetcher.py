# data_fetcher.py (Full Content)

import pandas as pd
import yfinance as yf
from fredapi import Fred # <--- Import FRED API wrapper
import logging
import time
from datetime import datetime, timedelta
import os #<--- Ensure os is imported

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_yfinance_data(tickers, start_date, end_date):
    """Fetches historical OHLCV data using yfinance with delays and retries."""
    logging.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date} using yfinance...")
    all_data = {}
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
                data = yf.download(batch_tickers, start=start_date, end=end_date, interval='1d',
                                   group_by='ticker', auto_adjust=True, threads=True, progress=False)

                if not data.empty:
                     if len(batch_tickers) == 1:
                         ticker = batch_tickers[0]
                         if isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                             if all(col in data.columns for col in config.MARKET_DATA_COLS):
                                 all_data[ticker] = data[config.MARKET_DATA_COLS].copy()
                             else: logging.warning(f"Required columns missing for single {ticker}. Cols: {data.columns.tolist()}")
                         else:
                              if ticker not in yf.shared._ERRORS: logging.warning(f"No data/empty df for single {ticker}")
                     else: # Multi-ticker
                         for ticker in batch_tickers:
                             if ticker in data.columns.get_level_values(0) and not data[ticker].dropna(how='all').empty:
                                  ticker_df = data[ticker]
                                  if all(col in ticker_df.columns for col in config.MARKET_DATA_COLS):
                                      all_data[ticker] = ticker_df[config.MARKET_DATA_COLS].copy()
                                  else: logging.warning(f"Required columns missing for {ticker} in multi. Cols: {ticker_df.columns.tolist()}")
                             else:
                                 if ticker not in yf.shared._ERRORS: logging.warning(f"No data/empty data for {ticker} in multi")

                if yf.shared._ERRORS:
                     failed_tickers_in_batch = {t:e for t,e in yf.shared._ERRORS.items() if t in batch_tickers}
                     if failed_tickers_in_batch:
                          logging.error(f"Batch {batch_num} had {len(failed_tickers_in_batch)} yfinance failed downloads: {failed_tickers_in_batch}")
                          is_rate_limited = any('RateLimitError' in str(e) or 'Too Many Requests' in str(e) for e in failed_tickers_in_batch.values())
                          if is_rate_limited and attempt < max_retries - 1: raise Exception("RateLimitErrorTrigger")

                logging.info(f"Fetched batch {batch_num}/{num_batches} successfully on attempt {attempt+1}.")
                yf.shared._ERRORS = {}
                break # Exit retry loop on success

            except Exception as e:
                is_rate_limit_error = 'RateLimitError' in str(e) or 'Too Many Requests' in str(e) or str(e) == "RateLimitErrorTrigger"
                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected (yf batch {batch_num}, attempt {attempt+1}). Retrying in {retry_delay}s...")
                    yf.shared._ERRORS = {}
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Error fetching yf batch {batch_num} (attempt {attempt+1}): {e}", exc_info=not is_rate_limit_error)
                    yf.shared._ERRORS = {}
                    break # Exit retry loop

        if i + max_tickers_per_call < len(tickers):
             logging.debug(f"Waiting {delay_between_batches}s before next yf batch...")
             time.sleep(delay_between_batches)

    logging.info(f"Finished fetching yfinance stock/index data. Successfully retrieved data for {len(all_data)} tickers.")
    return all_data


def fetch_macro_data(macro_map, start_date, end_date):
    """Fetches macro data from yfinance (e.g., VIX) and FRED."""
    logging.info("Fetching macro data...")
    macro_data_dict = {}
    max_retries = config.YFINANCE_MAX_RETRIES # Reuse retry config for simplicity
    retry_delay = config.YFINANCE_RETRY_DELAY
    fred_delay = config.FRED_DELAY

    # --- Separate sources ---
    yf_tickers_map = {k: v for k, v in macro_map.items() if v.startswith('^')} # Simple check for yf tickers
    fred_codes_map = {k: v for k, v in macro_map.items() if not v.startswith('^')}

    # --- Fetch yfinance Macro Data (e.g., VIX) ---
    if yf_tickers_map:
        yf_ticker_list = list(yf_tickers_map.values())
        yf_keys = list(yf_tickers_map.keys())
        logging.info(f"Fetching macro data using yfinance: {yf_ticker_list}")
        for attempt in range(max_retries):
            try:
                data = yf.download(yf_ticker_list, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
                if not data.empty:
                    if len(yf_ticker_list) == 1:
                        key_name = yf_keys[0] # The name we want in our dict (e.g., 'VIX')
                        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                            series = data['Close']
                            series.name = key_name # Assign name
                            macro_data_dict[key_name] = series
                        else: logging.warning(f"No valid 'Close' data returned for yf macro ticker {yf_ticker_list[0]}")
                    else:
                        # Handle multiple yf macro tickers if needed
                        logging.warning("Multi-ticker yfinance macro fetching not fully implemented yet.")
                        pass # Implement similar logic as single ticker for each key/value pair

                elif yf.shared._ERRORS:
                     logging.error(f"Failed downloads for yf macro tickers: {yf.shared._ERRORS}")
                     is_rate_limited = any('RateLimitError' in str(e) or 'Too Many Requests' in str(e) for e in yf.shared._ERRORS.values())
                     if is_rate_limited and attempt < max_retries - 1: raise Exception("RateLimitErrorTrigger")
                else: logging.warning(f"No data returned from yfinance for {yf_ticker_list}")

                yf.shared._ERRORS = {}
                logging.info(f"Successfully fetched yfinance macro data on attempt {attempt+1}")
                break
            except Exception as e:
                is_rate_limit_error = 'RateLimitError' in str(e) or 'Too Many Requests' in str(e) or str(e) == "RateLimitErrorTrigger"
                if is_rate_limit_error and attempt < max_retries - 1:
                    logging.warning(f"Rate limit suspected for yf macro (attempt {attempt+1}). Retrying in {retry_delay}s...")
                    yf.shared._ERRORS = {}
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Error fetching yf macro (attempt {attempt+1}): {e}", exc_info=not is_rate_limit_error)
                    yf.shared._ERRORS = {}
                    break

    # --- Fetch FRED Macro Data ---
    if fred_codes_map:
        fred_api_key = config.FRED_API_KEY
        if not fred_api_key:
            logging.warning("FRED API key not found in config or .env. Skipping FRED data fetching.")
        else:
            try:
                fred = Fred(api_key=fred_api_key)
                logging.info(f"Fetching macro data from FRED: {list(fred_codes_map.values())}")
                for key_name, fred_code in fred_codes_map.items():
                    for attempt in range(max_retries): # Simple retry for FRED too
                         try:
                              series = fred.get_series(fred_code, observation_start=start_date, observation_end=end_date)
                              if not series.empty:
                                   series.name = key_name # Assign the desired internal name
                                   macro_data_dict[key_name] = series
                                   logging.debug(f"Fetched FRED series {fred_code} ({key_name}) successfully.")
                                   time.sleep(fred_delay) # Delay between FRED calls
                                   break # Success, break inner retry loop
                              else:
                                   logging.warning(f"No data returned for FRED series {fred_code} ({key_name}).")
                                   break # Don't retry if empty
                         except Exception as e_fred:
                              logging.error(f"Error fetching FRED series {fred_code} ({key_name}) on attempt {attempt+1}: {e_fred}")
                              if attempt < max_retries - 1:
                                   logging.info(f"Retrying FRED fetch for {fred_code} in {retry_delay}s...")
                                   time.sleep(retry_delay)
                              else:
                                   logging.error(f"Max retries reached for FRED series {fred_code}.")
                                   break # Failed after retries
            except Exception as e_fred_init:
                 logging.error(f"Failed to initialize FRED client or fetch data: {e_fred_init}", exc_info=True)


    # --- Combine and Process All Macro Data ---
    if macro_data_dict:
        logging.info(f"Combining {len(macro_data_dict)} fetched macro series...")
        # Concatenate all collected Series/DataFrames
        try:
            combined_macro = pd.concat(macro_data_dict.values(), axis=1)
            # Keys should align with Series names assigned above
            combined_macro.columns = list(macro_data_dict.keys()) # Ensure columns match keys
        except Exception as concat_err:
             logging.error(f"Error concatenating macro data: {concat_err}", exc_info=True)
             return pd.DataFrame()

        logging.info(f"Combined Macro Columns Before Processing: {combined_macro.columns.tolist()}")

        # Ensure daily frequency (business days) and forward fill
        # Use the full fetch range for reindexing before potential NaNs at start
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        combined_macro = combined_macro.reindex(full_date_range)

        # Forward fill NaNs (common for FRED data released less frequently than daily)
        initial_nans = combined_macro.isnull().sum().sum()
        combined_macro.ffill(inplace=True)
        nans_after_ffill = combined_macro.isnull().sum().sum()
        logging.info(f"Forward filled {initial_nans - nans_after_ffill} NaNs in macro data.")

        # Optional: Handle remaining NaNs at the beginning (e.g., backfill or drop)
        if nans_after_ffill > 0:
             logging.warning(f"{nans_after_ffill} NaNs remain in macro data after ffill (likely at start). Backfilling.")
             combined_macro.bfill(inplace=True) # Backfill to handle initial NaNs
             if combined_macro.isnull().any().any():
                  logging.error("NaNs still persist after bfill! Dropping rows with NaNs.")
                  combined_macro.dropna(inplace=True)

        # Final check for multi-index (shouldn't happen)
        if isinstance(combined_macro.columns, pd.MultiIndex):
             logging.error("FATAL: Macro data STILL has MultiIndex columns after processing!")
             return pd.DataFrame()
        if isinstance(combined_macro.index, pd.MultiIndex):
             logging.error("FATAL: Macro data has MultiIndex rows after processing!")
             return pd.DataFrame()

        logging.info(f"Processed Macro Data Columns: {combined_macro.columns.tolist()}")
        return combined_macro
    else:
        logging.warning("No macro data series were successfully fetched or processed.")
        return pd.DataFrame()


def fetch_and_save_all_data(tickers, start_date, end_date):
    """Fetches stock and macro data and saves to files."""
    # Fetch Stock Data
    stock_data = fetch_yfinance_data(tickers, start_date, end_date)
    saved_count = 0
    valid_tickers_downloaded = list(stock_data.keys())
    tickers_to_process = [t for t in tickers if t in valid_tickers_downloaded]

    for ticker in tickers_to_process:
        df = stock_data.get(ticker)
        if df is not None and not df.empty:
             if 'Close' in df.columns and (df['Close'] > 1e-6).any():
                 df_cleaned = df.dropna(how='any')
                 if not df_cleaned.empty:
                      utils.save_data_to_file(df_cleaned, ticker)
                      saved_count += 1
                 else: logging.warning(f"Skipping save for {ticker}, all rows NaN after dropna().")
             else: logging.warning(f"Skipping save for {ticker} due to bad data.")
    logging.info(f"Saved data for {saved_count} stock tickers.")

    # Fetch Macro Data (already processed and potentially filled)
    macro_df = fetch_macro_data(config.MACRO_FEATURES, start_date, end_date)
    if macro_df is not None and not macro_df.empty:
        # Save the processed DataFrame
        utils.save_data_to_file(macro_df.dropna(how='all'), "_macro_data") # Drop rows if all NaN after ffill/bfill
        logging.info("Saved macro data.")
    else:
        logging.warning("No final macro data DataFrame to save.")


if __name__ == "__main__":
    logging.info("--- Starting Data Fetching Process ---")
    time.sleep(1) # Small delay
    sp500_tickers = utils.get_sp500_tickers()
    fetch_start = config.START_DATE
    fetch_end = (datetime.now() - timedelta(days=0)).strftime('%Y-%m-%d') # Fetch including today for yfinance
    logging.info(f"Data fetch range: {fetch_start} to {fetch_end} (yf end date is exclusive)")

    fetch_and_save_all_data(sp500_tickers, fetch_start, fetch_end)
    logging.info("--- Data Fetching Process Completed ---")
