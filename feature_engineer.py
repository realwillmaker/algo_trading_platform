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
        logging.warning("Input DataFrame to calculate_technical_indicators is None or empty.")
        return df # Return original potentially None df

    # Create a copy to avoid modifying the original DataFrame unexpectedly
    df_ta = df.copy()

    # pandas-ta works best with lowercase column names
    df_ta.columns = [col.lower() for col in df_ta.columns]

    # Check if required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_ta.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_ta.columns]
        logging.warning(f"DataFrame missing required columns for pandas-ta: {missing}. Skipping TA calculation.")
        # Return original DataFrame (with original casing) if prereqs not met
        return df

    # Create a list of dictionaries, where each dict represents an indicator
    strategy_list = []
    for name, params in indicators_config.items():
        indicator_dict = params.copy()
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
        # Return original df if no indicators defined, no changes made
        return df

    # Merge the calculated indicators back into the original DataFrame
    original_cols = list(df.columns)
    # Identify new columns added by pandas-ta (case-insensitive comparison)
    new_cols = [col for col in df_ta.columns if col.lower() not in [c.lower() for c in original_cols]]

    df_result = df.copy()
    for col in new_cols:
        # Add the new column to the result DataFrame
        df_result[col] = df_ta[col]

    return df_result


def add_fundamental_features(df, ticker):
     """Placeholder - Add fundamental feature logic here."""
     # In a real system, load point-in-time fundamental data here and merge based on date.
     logging.debug(f"Fundamental feature addition skipped/placeholder for {ticker}.")
     # Return the DataFrame unmodified for now
     if df is None: return None # Handle None input
     return df


def preprocess_data_for_ticker(ticker, macro_data):
    """Loads, adds features, and merges macro data for a single ticker."""
    logging.debug(f"Starting preprocessing for {ticker}")
    df = utils.load_data_from_file(ticker)
    if df is None or df.empty:
        logging.warning(f"No data loaded for {ticker}, skipping feature engineering.")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
         logging.debug(f"Converting index to DatetimeIndex for {ticker}")
         df.index = pd.to_datetime(df.index)

    # Calculate Technical Indicators
    logging.debug(f"Calculating technical indicators for {ticker}")
    df = calculate_technical_indicators(df)
    if df is None: # Check if TA calculation failed or returned None
         logging.error(f"Technical indicator calculation failed for {ticker}.")
         return None

    # Add Fundamental Features (Placeholder)
    logging.debug(f"Adding fundamental features (placeholder) for {ticker}")
    df = add_fundamental_features(df, ticker)
    if df is None: return None # Check if fundamentals processing failed

    # Calculate Log Return
    logging.debug(f"Calculating log return for {ticker}")
    if 'Close' in df.columns:
        # Ensure Close is numeric before calculating log return
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isnull().any():
             logging.warning(f"NaNs found in 'Close' column for {ticker} after coercion. Log returns might be affected.")
             # Optionally fill NaNs in Close before log return calculation
             # df['Close'] = df['Close'].ffill().bfill() # Example: forward then backward fill
        # Calculate log return, handle potential division by zero or log(0)
        df['log_return'] = np.log(df['Close'].pct_change() + 1).fillna(0) # Use pct_change() + 1 for robustness
        # Replace inf/-inf that can result from edge cases
        df['log_return'].replace([np.inf, -np.inf], 0, inplace=True)
    else:
        logging.warning(f"'Close' column not found or not numeric for {ticker}. Cannot compute log_return. Setting to 0.")
        df['log_return'] = 0.0 # Assign default value

    # --- Merge Macro Data ---
    logging.debug(f"Attempting to merge macro data for {ticker}")
    if macro_data is not None and not macro_data.empty:
        # Ensure indices are compatible before merge
        if not isinstance(macro_data.index, pd.DatetimeIndex):
             logging.warning(f"[{ticker}] Converting macro_data index to datetime.")
             macro_data.index = pd.to_datetime(macro_data.index)

        # Ensure indices have the same timezone status (e.g., both naive)
        if df.index.tz is not None and macro_data.index.tz is None:
             macro_data.index = macro_data.index.tz_localize(df.index.tz)
        elif df.index.tz is None and macro_data.index.tz is not None:
             macro_data.index = macro_data.index.tz_localize(None)
        elif df.index.tz != macro_data.index.tz:
             logging.warning(f"[{ticker}] Timezone mismatch between df ({df.index.tz}) and macro ({macro_data.index.tz}). Localizing macro data to None.")
             macro_data.index = macro_data.index.tz_localize(None) # Default to naive if mismatch


        # --- START DEBUGGING & SAFETY NET ---
        logging.debug(f"[{ticker}] Pre-Merge df Index Type: {type(df.index)}, Levels: {df.index.nlevels}, Is MultiIndex: {isinstance(df.index, pd.MultiIndex)}, TZ: {df.index.tz}")
        logging.debug(f"[{ticker}] Pre-Merge df Columns Type: {type(df.columns)}, Levels: {df.columns.nlevels}, Is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Index Type: {type(macro_data.index)}, Levels: {macro_data.index.nlevels}, Is MultiIndex: {isinstance(macro_data.index, pd.MultiIndex)}, TZ: {macro_data.index.tz}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Columns Type: {type(macro_data.columns)}, Levels: {macro_data.columns.nlevels}, Is MultiIndex: {isinstance(macro_data.columns, pd.MultiIndex)}")
        logging.debug(f"[{ticker}] Pre-Merge macro_data Columns: {macro_data.columns.tolist()}")

        # Safety Net: Flatten columns AGAIN just in case
        macro_data_to_merge = macro_data.copy()
        if isinstance(macro_data_to_merge.columns, pd.MultiIndex):
            logging.warning(f"[{ticker}] Macro data STILL has MultiIndex columns just before merge! Flattening again.")
            original_macro_cols = macro_data_to_merge.columns
            try:
                macro_data_to_merge.columns = macro_data_to_merge.columns.get_level_values(-1) # Try last level
                macro_data_to_merge = macro_data_to_merge.loc[:,~macro_data_to_merge.columns.duplicated()]
                logging.info(f"[{ticker}] Flattened macro cols: {macro_data_to_merge.columns.tolist()}")
            except Exception as flatten_err:
                 logging.error(f"[{ticker}] Failed to flatten macro columns ({original_macro_cols}): {flatten_err}. Skipping merge.", exc_info=True)
                 macro_data_to_merge = pd.DataFrame()


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
                merged_cols = set(macro_data_to_merge.columns) & set(df.columns)
                logging.debug(f"[{ticker}] Successfully merged macro columns: {list(merged_cols)}")

                # Forward fill the merged columns
                cols_to_fill = [col for col in merged_cols if col in df.columns]
                if cols_to_fill:
                     logging.debug(f"[{ticker}] Forward filling merged columns: {cols_to_fill}")
                     df[cols_to_fill] = df[cols_to_fill].ffill()

            except pd.errors.MergeError as me:
                logging.error(f"[{ticker}] pd.merge failed EVEN AFTER CHECKS: {me}", exc_info=True)
                logging.error(f"[{ticker}] FAILED MERGE df Index: {type(df.index)}, Levels: {df.index.nlevels}, Columns: {df.columns.tolist()}")
                logging.error(f"[{ticker}] FAILED MERGE macro_data Index: {type(macro_data_to_merge.index)}, Levels: {macro_data_to_merge.index.nlevels}, Columns: {macro_data_to_merge.columns.tolist()}")
                pass # Continue without macro features for this ticker
            except Exception as e:
                logging.error(f"[{ticker}] Unexpected error during merge or ffill: {e}", exc_info=True)
                pass
        else:
             logging.debug(f"[{ticker}] Macro data empty after checks, skipping merge.")
        # --- END DEBUGGING & SAFETY NET ---
    else:
        logging.debug(f"[{ticker}] No macro data provided or macro data is empty, skipping merge.")


    # --- Determine indicator columns dynamically for NaN check ---
    logging.debug(f"[{ticker}] Determining indicator columns for NaN check.")
    indicator_cols = []
    try:
        dummy_strategy_list = [p.copy() for p in config.TECHNICAL_INDICATORS.values()]
        if dummy_strategy_list:
            dummy_strategy = ta.Strategy(name="Dummy", ta=dummy_strategy_list)
            dummy_df = pd.DataFrame({'open': [10, 11], 'high': [12, 12], 'low': [9, 10],'close': [11, 11.5], 'volume': [100, 110]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
            dummy_df.ta.strategy(dummy_strategy, append=True)
            indicator_cols = [col for col in dummy_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            logging.debug(f"[{ticker}] Dynamically determined indicator columns: {indicator_cols}")
        else:
            logging.debug(f"[{ticker}] No technical indicators defined in config.")
    except Exception as e:
        logging.warning(f"[{ticker}] Could not dynamically determine indicator columns: {e}. Falling back.")
        indicator_cols = []


    # Define columns to check for NaNs before dropping
    nan_check_cols = []
    if 'log_return' in df.columns: nan_check_cols.append('log_return')
    nan_check_cols.extend(indicator_cols)
    nan_check_cols = [col for col in nan_check_cols if col in df.columns] # Ensure cols exist

    # --- Drop initial rows with NaNs ---
    if nan_check_cols:
        initial_rows = len(df)
        df.dropna(subset=nan_check_cols, how='any', inplace=True) # Use inplace=True
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
             logging.debug(f"[{ticker}] Dropped {dropped_rows} initial rows with NaNs in check columns ({nan_check_cols}).")
    else:
         logging.warning(f"[{ticker}] No specific columns found for NaN check. Applying dropna(how='any') broadly.")
         initial_rows = len(df)
         df.dropna(how='any', inplace=True)
         dropped_rows = initial_rows - len(df)
         if dropped_rows > 0:
             logging.debug(f"[{ticker}] Dropped {dropped_rows} rows with NaNs (broad check).")


    # --- Define final feature columns ---
    logging.debug(f"[{ticker}] Selecting final feature columns.")
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    final_feature_cols = [col for col in df.columns if col not in exclude_cols]
    logging.debug(f"[{ticker}] Final feature columns identified: {final_feature_cols}")

    # Select only these columns + required OHLC for env
    required_env_cols = ['Open', 'Close'] # Columns needed by env logic
    # Check if required env cols exist before selecting
    missing_env_cols = [col for col in required_env_cols if col not in df.columns]
    if missing_env_cols:
         logging.error(f"[{ticker}] Required env columns {missing_env_cols} missing before final selection. Cannot proceed.")
         return None

    # Ensure final feature cols also exist
    missing_feature_cols = [col for col in final_feature_cols if col not in df.columns]
    if missing_feature_cols:
         logging.warning(f"[{ticker}] Identified feature columns {missing_feature_cols} missing from DataFrame. Adjusting final_feature_cols.")
         final_feature_cols = [col for col in final_feature_cols if col in df.columns]


    # Create final DataFrame
    df_final = df[required_env_cols + final_feature_cols].copy()

    # Final check for required env columns after selection
    if not all(col in df_final.columns for col in required_env_cols):
         logging.error(f"[{ticker}] Required env columns missing AFTER FINAL SELECTION. This should not happen.")
         return None

    # Ensure DataFrame is not empty after all processing
    if df_final.empty:
        logging.warning(f"DataFrame for {ticker} became empty after final processing/selection.")
        return None

    logging.debug(f"[{ticker}] Preprocessing complete. Final shape: {df_final.shape}")
    return df_final


def create_feature_dataset(tickers, start_date, end_date):
    """Creates the final feature dataset for all tickers."""
    logging.info("--- Starting Feature Engineering Process ---")
    all_features = {}
    macro_data = utils.load_data_from_file("_macro_data")

    if macro_data is not None:
        macro_data.index = pd.to_datetime(macro_data.index)
        logging.info(f"Loaded macro data. Shape: {macro_data.shape}, Index Type: {type(macro_data.index)}, Columns: {macro_data.columns}")

        if isinstance(macro_data.columns, pd.MultiIndex):
            logging.warning("Macro data columns are MultiIndex. Attempting to flatten.")
            try:
                 macro_data.columns = macro_data.columns.get_level_values(-1)
                 macro_data = macro_data.loc[:,~macro_data.columns.duplicated()]
                 logging.info(f"Flattened macro data columns: {macro_data.columns}")
            except Exception as e:
                 logging.error(f"Failed to flatten MultiIndex columns for macro data: {e}. Proceeding without macro data.", exc_info=True)
                 macro_data = pd.DataFrame()

        if isinstance(macro_data.index, pd.MultiIndex):
             logging.error("Macro data has MultiIndex on rows. Proceeding without macro data.")
             macro_data = pd.DataFrame()

        macro_data = macro_data.dropna(axis=1, how='all')
        if macro_data.empty:
             logging.warning("Macro data became empty after processing. Proceeding without macro features.")
    else:
        logging.warning("Macro data file not found or empty. Proceeding without macro features.")
        macro_data = pd.DataFrame()


    # --- Process each ticker ---
    feature_count = 0
    processed_tickers = []
    for ticker in tickers:
        processed_df = preprocess_data_for_ticker(ticker, macro_data)

        if processed_df is not None and not processed_df.empty:
            # Filter date range AFTER feature calculation
            try:
                # Ensure index is timezone-naive for comparison if necessary
                if processed_df.index.tz is not None:
                    processed_df.index = processed_df.index.tz_localize(None)

                processed_df_filtered = processed_df.loc[start_date:end_date].copy() # Use .copy()
            except KeyError:
                 logging.warning(f"Date range slicing ({start_date} to {end_date}) failed for {ticker}. Maybe no data in range?")
                 continue # Skip this ticker if no data in desired range
            except Exception as e:
                 logging.error(f"Error slicing date range for {ticker}: {e}", exc_info=True)
                 continue


            if not processed_df_filtered.empty:
                 # Identify feature columns again after date filtering (should be same as before)
                 exclude_cols_lower = ['open', 'high', 'low', 'close', 'volume', 'log_return']
                 feature_cols = [col for col in processed_df_filtered.columns
                                 if col.lower() not in exclude_cols_lower
                                 and col not in ['Open','High','Low','Close','Volume','log_return']]

                 # Ensure all identified feature columns are numeric
                 processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')

                 # --- Check for NaNs after coercion ---
                 nan_check = processed_df_filtered[feature_cols].isnull()
                 if nan_check.any().any():
                      nan_cols_dict = nan_check.any()
                      nan_col_names = nan_cols_dict[nan_cols_dict].index.tolist()
                      logging.warning(f"NaNs found in feature columns for {ticker} after to_numeric coercion: {nan_col_names}. Applying forward fill (ffill).")

                      processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].ffill()

                      # --- Optional: Check for remaining NaNs and fill with 0 ---
                      nan_check_after_ffill = processed_df_filtered[feature_cols].isnull()
                      if nan_check_after_ffill.any().any():
                           remaining_nan_cols = nan_check_after_ffill.any()
                           remaining_nan_names = remaining_nan_cols[remaining_nan_cols].index.tolist()
                           logging.warning(f"NaNs still present for {ticker} after ffill (likely at start): {remaining_nan_names}. Filling these specific ones with 0.")
                           processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].fillna(0)

                 # Final check if any NaNs remain in features
                 if processed_df_filtered[feature_cols].isnull().any().any():
                      logging.error(f"FATAL: NaNs remain in feature columns for {ticker} after all filling attempts. Skipping ticker.")
                      continue

                 all_features[ticker] = processed_df_filtered
                 feature_count += 1
                 processed_tickers.append(ticker)
                 logging.debug(f"Finished features for {ticker}, final shape for env: {processed_df_filtered.shape}")
            else:
                 logging.warning(f"No data remaining for {ticker} after date filtering {start_date} to {end_date}.")
        else:
             # Already logged in preprocess_data_for_ticker if failed there
             pass


    if not all_features:
         logging.error("No features were generated for ANY ticker within the specified date range.")
         return None

    logging.info(f"--- Feature Engineering Completed for {feature_count} tickers ---")
    # Return only the features for tickers that were successfully processed fully
    return {ticker: df for ticker, df in all_features.items() if ticker in processed_tickers}


if __name__ == "__main__":
    # Example Usage
    logging.info("--- Running Feature Engineering Standalone ---")
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    features_dict = create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)

    if features_dict:
        sample_ticker = next(iter(features_dict), None)
        if sample_ticker:
            print(f"\nSample Features for {sample_ticker} (using pandas-ta):")
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
