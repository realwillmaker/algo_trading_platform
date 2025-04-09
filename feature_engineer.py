import pandas as pd
import numpy as np
import pandas_ta as ta # Import pandas-ta
import logging

import config
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_technical_indicators(df, indicators_config=config.TECHNICAL_INDICATORS):
    """Calculates technical indicators using pandas-ta."""
    if df is None or df.empty:
        return df

    # Create a copy to avoid modifying the original DataFrame unexpectedly
    df_ta = df.copy()

    # pandas-ta works best with lowercase column names
    df_ta.columns = [col.lower() for col in df_ta.columns]

    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_ta.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_ta.columns]
        logging.warning(f"DataFrame missing required columns for pandas-ta: {missing}. Skipping TA calculation.")
        return df

    # Create a list of dictionaries, where each dict represents an indicator
    strategy_list = []
    for name, params in indicators_config.items():
        # The dictionary itself should contain 'kind' and its parameters
        indicator_dict = params.copy() # Contains 'kind' and other params like 'length'
        strategy_list.append(indicator_dict)


    # Create and apply the strategy if the list is not empty
    if strategy_list:
        strategy = ta.Strategy(
            name="Custom TA Strategy",
            ta=strategy_list # Pass the list of indicator dictionaries
        )
        try:
            # Append indicators to the DataFrame
            df_ta.ta.strategy(strategy, append=True)
            logging.debug(f"Applied pandas-ta strategy. New columns: {df_ta.columns}")
        except Exception as e:
            logging.error(f"Error applying pandas-ta strategy: {e}", exc_info=True)
            return df # Return original df on error
    else:
        logging.warning("No indicators defined in strategy_list. Skipping TA calculation.")
        return df # Return original df if no indicators


    # Merge the calculated indicators back into the original DataFrame
    original_cols = list(df.columns)
    # Identify new columns added by pandas-ta (case-insensitive comparison)
    new_cols = [col for col in df_ta.columns if col.lower() not in [c.lower() for c in original_cols]]

    df_result = df.copy()
    for col in new_cols:
        # Add the new column to the result DataFrame, preserving original casing if it existed somehow
        # but typically pandas-ta columns are new (e.g., SMA_10, MACD_12_26_9)
        df_result[col] = df_ta[col]

    return df_result


def add_fundamental_features(df, ticker):
     """Placeholder - No changes needed here"""
     # In a real system, load point-in-time fundamental data here and merge based on date.
     logging.debug(f"Fundamental feature addition skipped/placeholder for {ticker}.")
     return df


def preprocess_data_for_ticker(ticker, macro_data):
    """Loads, adds features, and merges macro data for a single ticker."""
    df = utils.load_data_from_file(ticker)
    if df is None or df.empty:
        logging.warning(f"No data loaded for {ticker}, skipping feature engineering.")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)

    # Calculate Technical Indicators
    df = calculate_technical_indicators(df)
    if df is None: # Check if TA calculation failed
         logging.error(f"Technical indicator calculation failed for {ticker}.")
         return None

    # Add Fundamental Features (Placeholder)
    df = add_fundamental_features(df, ticker)

    # Calculate Log Return
    if 'Close' in df.columns:
        # Ensure Close is numeric before calculating log return
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        # Replace inf/-inf that can result from log(0) or log(x/0)
        df['log_return'].replace([np.inf, -np.inf], 0, inplace=True) # Replace infinities with 0
    else:
        logging.warning(f"'Close' column not found or not numeric for {ticker}. Cannot compute log_return.")
        df['log_return'] = 0.0 # Assign default value if Close is missing


    # --- Merge Macro Data ---
    if macro_data is not None and not macro_data.empty:
        if not isinstance(macro_data.index, pd.DatetimeIndex):
             logging.warning(f"[{ticker}] Converting macro_data index to datetime.")
             macro_data.index = pd.to_datetime(macro_data.index)

        # --- START DEBUGGING & SAFETY NET ---
        logging.debug(f"[{ticker}] Pre-Merge df Index Type: {type(df.index)}, Levels: {df.index.nlevels}, Is MultiIndex: {isinstance(df.index, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge df Columns Type: {type(df.columns)}, Levels: {df.columns.nlevels}, Is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Index Type: {type(macro_data.index)}, Levels: {macro_data.index.nlevels}, Is MultiIndex: {isinstance(macro_data.index, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Columns Type: {type(macro_data.columns)}, Levels: {macro_data.columns.nlevels}, Is MultiIndex: {isinstance(macro_data.columns, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Columns: {macro_data.columns.tolist()}")

        # Safety Net: Flatten columns AGAIN just in case
        macro_data_to_merge = macro_data.copy() # Work on a copy
        if isinstance(macro_data_to_merge.columns, pd.MultiIndex):
            logging.warning(f"[{ticker}] Macro data STILL has MultiIndex columns just before merge! Flattening again.")
            original_macro_cols = macro_data_to_merge.columns
            try:
                macro_data_to_merge.columns = macro_data_to_merge.columns.get_level_values(0) # Try first level
                macro_data_to_merge = macro_data_to_merge.loc[:,~macro_data_to_merge.columns.duplicated()]
                logging.info(f"[{ticker}] Flattened macro cols: {macro_data_to_merge.columns.tolist()}")
            except Exception as flatten_err:
                 logging.error(f"[{ticker}] Failed to flatten macro columns ({original_macro_cols}): {flatten_err}. Skipping merge.", exc_info=True)
                 macro_data_to_merge = pd.DataFrame() # Prevent merge if flattening failed

        # Check index compatibility
        if isinstance(macro_data_to_merge.index, pd.MultiIndex):
            logging.error(f"[{ticker}] Macro data has MultiIndex rows just before merge! Cannot merge.")
        elif df.index.nlevels != macro_data_to_merge.index.nlevels:
            logging.error(f"[{ticker}] Mismatched index levels before merge! df: {df.index.nlevels}, macro: {macro_data_to_merge.index.nlevels}. Cannot merge.")
        elif not macro_data_to_merge.empty: # Only merge if macro_data is not empty after checks
            # Perform the merge
            try:
                initial_df_cols = set(df.columns)
                df = pd.merge(df, macro_data_to_merge, left_index=True, right_index=True, how='left', suffixes=('', '_macro'))
                merged_cols = set(macro_data_to_merge.columns) & set(df.columns) # Columns actually added/present
                logging.debug(f"[{ticker}] Successfully merged macro columns: {list(merged_cols)}")

                # Forward fill the merged columns (ensure they exist in df)
                cols_to_fill = [col for col in merged_cols if col in df.columns]
                if cols_to_fill:
                     df[cols_to_fill] = df[cols_to_fill].ffill()

            except pd.errors.MergeError as me:
                logging.error(f"[{ticker}] pd.merge failed EVEN AFTER CHECKS: {me}", exc_info=True)
                logging.error(f"[{ticker}] FAILED MERGE df Index: {type(df.index)}, Levels: {df.index.nlevels}, Columns: {df.columns.tolist()}")
                logging.error(f"[{ticker}] FAILED MERGE macro_data Index: {type(macro_data_to_merge.index)}, Levels: {macro_data_to_merge.index.nlevels}, Columns: {macro_data_to_merge.columns.tolist()}")
                pass # Continue without macro features for this ticker
            except Exception as e:
                logging.error(f"[{ticker}] Unexpected error during merge or ffill: {e}", exc_info=True)
                pass
        # --- END DEBUGGING & SAFETY NET ---


    # --- Determine indicator columns dynamically for NaN check ---
    indicator_cols = []
    temp_df_for_cols = None # Clear previous dummy df
    try:
        dummy_strategy_list = [p.copy() for p in config.TECHNICAL_INDICATORS.values()]
        if dummy_strategy_list: # Only proceed if indicators are defined
            dummy_strategy = ta.Strategy(name="Dummy", ta=dummy_strategy_list)
            # Use slightly more data for dummy calculation robustness
            dummy_df = pd.DataFrame({
                'open': [10, 11], 'high': [12, 12], 'low': [9, 10],
                'close': [11, 11.5], 'volume': [100, 110]
            }, index=pd.to_datetime(['2023-01-01', '2023-01-02'])) # Add index
            dummy_df.ta.strategy(dummy_strategy, append=True)
            indicator_cols = [col for col in dummy_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            logging.debug(f"Dynamically determined indicator columns: {indicator_cols}")
        else:
            logging.debug("No technical indicators defined in config, skipping dynamic column determination.")
    except Exception as e:
        logging.warning(f"Could not dynamically determine indicator columns: {e}. Falling back.")
        indicator_cols = [] # Ensure it's empty on error


    # Define columns to check for NaNs before dropping
    nan_check_cols = []
    if 'log_return' in df.columns:
        nan_check_cols.append('log_return')
    nan_check_cols.extend(indicator_cols)

    # Ensure only existing columns are in the check list
    nan_check_cols = [col for col in nan_check_cols if col in df.columns]

    if nan_check_cols:
        initial_rows = len(df)
        # Drop rows where *any* of the critical columns (log_return, indicators) are NaN
        df = df.dropna(subset=nan_check_cols, how='any')
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
             logging.debug(f"Dropped {dropped_rows} rows with NaNs in check columns ({nan_check_cols}) for {ticker}.")
    else:
         logging.warning(f"No specific columns found for NaN check in {ticker} (log_return or indicators missing/failed). Applying dropna(how='any') broadly.")
         initial_rows = len(df)
         df = df.dropna(how='any') # Fallback: drop rows if any column has NaN
         dropped_rows = initial_rows - len(df)
         if dropped_rows > 0:
             logging.debug(f"Dropped {dropped_rows} rows with NaNs (broad check) for {ticker}.")


    # Final check for required columns needed by environment
    required_env_cols = ['Open', 'Close'] # Check original case
    if not all(col in df.columns for col in required_env_cols):
         missing_env_cols = [col for col in required_env_cols if col not in df.columns]
         logging.error(f"Required env columns {missing_env_cols} missing for {ticker} after processing.")
         return None

    # Ensure DataFrame is not empty after all processing
    if df.empty:
        logging.warning(f"DataFrame for {ticker} became empty after processing (NaN dropping).")
        return None

    return df


def create_feature_dataset(tickers, start_date, end_date):
    """Creates the final feature dataset for all tickers."""
    logging.info("--- Starting Feature Engineering Process ---")
    all_features = {}
    macro_data = utils.load_data_from_file("_macro_data") # Load macro data

    if macro_data is not None:
        macro_data.index = pd.to_datetime(macro_data.index)
        logging.info(f"Loaded macro data. Shape: {macro_data.shape}, Index Type: {type(macro_data.index)}, Columns: {macro_data.columns}")

        # --- FIX: Check for and handle MultiIndex columns ---
        if isinstance(macro_data.columns, pd.MultiIndex):
            logging.warning("Macro data columns are MultiIndex. Attempting to flatten.")
            try:
                 # Prefer using the last level if names might be nested (like from yfinance)
                 macro_data.columns = macro_data.columns.get_level_values(-1)
                 # Remove duplicate columns if flattening created them
                 macro_data = macro_data.loc[:,~macro_data.columns.duplicated()]
                 logging.info(f"Flattened macro data columns: {macro_data.columns}")
            except Exception as e:
                 logging.error(f"Failed to flatten MultiIndex columns for macro data: {e}. Proceeding without macro data.", exc_info=True)
                 macro_data = pd.DataFrame() # Reset if flattening fails

        # Check if index is also MultiIndex (less likely but possible)
        if isinstance(macro_data.index, pd.MultiIndex):
             logging.error("Macro data has MultiIndex on rows. This is unexpected and likely an error in data saving/loading. Proceeding without macro data.")
             macro_data = pd.DataFrame()

        # Drop any potential all-NaN columns that might result from loading/flattening
        macro_data = macro_data.dropna(axis=1, how='all')

        if macro_data.empty:
             logging.warning("Macro data became empty after processing MultiIndex or NaNs. Proceeding without macro features.")

    else:
        logging.warning("Macro data file not found or empty. Proceeding without macro features.")
        macro_data = pd.DataFrame() # Use empty DataFrame to allow processing to continue


    # --- Process each ticker ---
    feature_count = 0
    processed_tickers = []
    for ticker in tickers:
        logging.debug(f"Processing features for {ticker}...")
        # Pass the processed (or empty) macro_data DataFrame
        processed_df = preprocess_data_for_ticker(ticker, macro_data)

        if processed_df is not None and not processed_df.empty:
            # Filter date range AFTER feature calculation to avoid lookahead bias
            # Ensure index is timezone-naive for comparison if necessary
            if processed_df.index.tz is not None:
                processed_df.index = processed_df.index.tz_localize(None)

            processed_df_filtered = processed_df.loc[start_date:end_date].copy() # Use .copy()

            if not processed_df_filtered.empty:
                 # Identify feature columns (excluding OHLCV and log_return)
                 # Use original case for OHLCV as they are needed by the env
                 exclude_cols_lower = ['open', 'high', 'low', 'close', 'volume', 'log_return']
                 feature_cols = [col for col in processed_df_filtered.columns
                                 if col.lower() not in exclude_cols_lower
                                 and col not in ['Open','High','Low','Close','Volume','log_return']] # Also check original case

                 # Ensure all identified feature columns are numeric
                 processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')

                 # --- Check for NaNs after coercion ---
                 nan_check = processed_df_filtered[feature_cols].isnull()
                 if nan_check.any().any():
                      nan_cols_dict = nan_check.any() # Series indicating which columns have NaNs
                      nan_col_names = nan_cols_dict[nan_cols_dict].index.tolist()
                      logging.warning(f"NaNs found in feature columns for {ticker} after to_numeric coercion: {nan_col_names}. Applying forward fill (ffill).")

                      # --- Use forward fill instead of filling with 0 ---
                      processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].ffill()

                      # --- Optional: Check for remaining NaNs (at the beginning) and fill with 0 ---
                      nan_check_after_ffill = processed_df_filtered[feature_cols].isnull()
                      if nan_check_after_ffill.any().any():
                           remaining_nan_cols = nan_check_after_ffill.any()
                           remaining_nan_names = remaining_nan_cols[remaining_nan_cols].index.tolist()
                           logging.warning(f"NaNs still present for {ticker} after ffill (likely at start): {remaining_nan_names}. Filling these specific ones with 0.")
                           processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].fillna(0) # Fallback fill with 0

                 # Final check if any NaNs remain in features (should not happen ideally)
                 if processed_df_filtered[feature_cols].isnull().any().any():
                      logging.error(f"FATAL: NaNs remain in feature columns for {ticker} after all filling attempts. Skipping ticker.")
                      continue # Skip this ticker


                 all_features[ticker] = processed_df_filtered
                 feature_count += 1
                 processed_tickers.append(ticker)
                 logging.debug(f"Finished features for {ticker}, shape: {processed_df_filtered.shape}")
            else:
                 logging.warning(f"No data remaining for {ticker} after date filtering {start_date} to {end_date}.")
        else:
             logging.warning(f"Skipping {ticker} due to processing issues or lack of data in preprocess_data_for_ticker.")


    if not all_features:
         logging.error("No features were generated for ANY ticker. Exiting.")
         return None # Return None if no features generated

    logging.info(f"--- Feature Engineering Completed for {feature_count} tickers ---")
    # Return only the features for tickers that were successfully processed
    return {ticker: df for ticker, df in all_features.items() if ticker in processed_tickers}


if __name__ == "__main__":
    # Example Usage
    logging.info("--- Running Feature Engineering Standalone ---")
    sp500_tickers = utils.get_sp500_tickers()
    # Exclude benchmark/macro from the list of stocks to generate features for
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    # Generate features for the training period
    features_dict = create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)

    if features_dict:
        # Safely get the first ticker key
        sample_ticker = next(iter(features_dict), None)
        if sample_ticker:
            print(f"\nSample Features for {sample_ticker} (using pandas-ta):")
            # Display first few rows and column names/types
            print(features_dict[sample_ticker].head())
            print("\nFeatures Summary:")
            print(f"Shape for {sample_ticker}: {features_dict[sample_ticker].shape}")
            print(f"Columns for {sample_ticker}: {features_dict[sample_ticker].columns.tolist()}")
            print("\nData Types:")
            print(features_dict[sample_ticker].dtypes)
        else:
             print("\nNo features dictionary generated.")
    else:
        print("\nFeature generation failed or produced no results.")
    logging.info("--- Feature Engineering Standalone Run Finished ---")
