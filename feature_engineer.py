# feature_engineer.py (Full Content with Verification Block)

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import os # Import os for file path joining in main block if needed

import config
import utils

# Ensure logging is configured when running standalone
# If run via other scripts, their logging setup might take precedence
if __name__ == "__main__":
     log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
     log_file_fe = "feature_engineer_standalone.log"
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s', handlers=[logging.FileHandler(log_file_fe), logging.StreamHandler()])
else:
    # Use logger configured by the calling script
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_technical_indicators(df, indicators_config=config.TECHNICAL_INDICATORS):
    """Calculates technical indicators using pandas-ta."""
    if df is None or df.empty:
        logging.warning("Input DataFrame to calculate_technical_indicators is None or empty.")
        return df

    df_ta = df.copy()
    df_ta.columns = [col.lower() for col in df_ta.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_ta.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_ta.columns]
        logging.warning(f"DataFrame missing {missing} for pandas-ta. Skipping TA.")
        return df

    strategy_list = [params.copy() for params in indicators_config.values()]

    if strategy_list:
        strategy = ta.Strategy(name="Custom TA Strategy", ta=strategy_list)
        try:
            df_ta.ta.strategy(strategy, append=True)
            logging.debug(f"Applied pandas-ta strategy. New columns: {df_ta.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error applying pandas-ta strategy: {e}", exc_info=True)
            return df
    else:
        logging.warning("No indicators in config. Skipping TA.")
        return df

    # Merge results back
    original_cols_lower = [c.lower() for c in df.columns]
    new_cols = [col for col in df_ta.columns if col.lower() not in original_cols_lower]
    df_result = df.copy()
    for col in new_cols:
        df_result[col] = df_ta[col] # Use pandas-ta generated names
    return df_result


def add_fundamental_features(df, ticker):
     """Placeholder - Add fundamental feature logic here."""
     logging.debug(f"Fundamental feature addition skipped/placeholder for {ticker}.")
     if df is None: return None
     return df


def preprocess_data_for_ticker(ticker, macro_data):
    """Loads, adds features, and merges macro data for a single ticker."""
    logging.debug(f"Starting preprocessing for {ticker}")
    df = utils.load_data_from_file(ticker)
    if df is None or df.empty: logging.warning(f"No data for {ticker}."); return None

    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True) # Crucial for TA/time series logic

    # Calculate Technical Indicators
    logging.debug(f"Calculating technical indicators for {ticker}")
    df = calculate_technical_indicators(df)
    if df is None: logging.error(f"TA calculation failed for {ticker}."); return None

    # Add Fundamental Features (Placeholder)
    df = add_fundamental_features(df, ticker)
    if df is None: return None

    # Calculate Log Return
    logging.debug(f"Calculating log return for {ticker}")
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isnull().any(): logging.warning(f"NaNs in 'Close' for {ticker} after coercion.")
        df['log_return'] = np.log(df['Close'].pct_change() + 1).fillna(0)
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], 0) # Assign back
    else:
        logging.warning(f"No 'Close' column for {ticker}. Log return set to 0.")
        df['log_return'] = 0.0

    # Merge Macro Data
    logging.debug(f"Attempting to merge {len(macro_data.columns) if macro_data is not None else 0} macro cols for {ticker}")
    if macro_data is not None and not macro_data.empty:
        if not isinstance(macro_data.index, pd.DatetimeIndex): macro_data.index = pd.to_datetime(macro_data.index)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if macro_data.index.tz is not None: macro_data.index = macro_data.index.tz_localize(None)

        macro_data_to_merge = macro_data.copy()
        if isinstance(macro_data_to_merge.columns, pd.MultiIndex):
            try:
                 macro_data_to_merge.columns = macro_data_to_merge.columns.get_level_values(-1)
                 macro_data_to_merge = macro_data_to_merge.loc[:,~macro_data_to_merge.columns.duplicated()]
            except Exception: macro_data_to_merge = pd.DataFrame()

        if not macro_data_to_merge.empty and df.index.nlevels == macro_data_to_merge.index.nlevels and not isinstance(macro_data_to_merge.index, pd.MultiIndex):
            try:
                df = pd.merge(df, macro_data_to_merge, left_index=True, right_index=True, how='left', suffixes=('', '_macro'))
                merged_cols = [col for col in macro_data_to_merge.columns if col in df.columns]
                if merged_cols:
                     logging.debug(f"[{ticker}] Forward filling merged macro columns: {merged_cols}")
                     df[merged_cols] = df[merged_cols].ffill() # Assign back
            except Exception as e: logging.error(f"[{ticker}] Error during merge/ffill: {e}", exc_info=True)
        else: logging.debug(f"[{ticker}] Macro data incompatible or empty, skipping merge.")
    else: logging.debug(f"[{ticker}] No macro data provided, skipping merge.")


    # Determine indicator/macro columns for NaN check
    logging.debug(f"[{ticker}] Determining columns for NaN check.")
    indicator_and_macro_cols = []
    try:
        dummy_strategy_list = [p.copy() for p in config.TECHNICAL_INDICATORS.values()]
        if dummy_strategy_list:
            dummy_strategy = ta.Strategy(name="Dummy", ta=dummy_strategy_list)
            dummy_df = pd.DataFrame({'open': [10, 11], 'high': [12, 12], 'low': [9, 10],'close': [11, 11.5], 'volume': [100, 110]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
            dummy_df.ta.strategy(dummy_strategy, append=True)
            indicator_cols_dynamic = [col for col in dummy_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            indicator_and_macro_cols.extend(indicator_cols_dynamic)
            logging.debug(f"[{ticker}] Dynamically determined TA columns: {indicator_cols_dynamic}")
    except Exception as e: logging.warning(f"[{ticker}] Could not dynamically determine TA columns: {e}")

    macro_cols_expected = list(config.MACRO_FEATURES.keys())
    indicator_and_macro_cols.extend(macro_cols_expected)
    indicator_and_macro_cols = list(set(indicator_and_macro_cols))

    nan_check_cols = []
    if 'log_return' in df.columns: nan_check_cols.append('log_return')
    nan_check_cols.extend([col for col in indicator_and_macro_cols if col in df.columns])

    # Drop initial rows with NaNs
    if nan_check_cols:
        initial_rows = len(df)
        df = df.dropna(subset=nan_check_cols, how='any') # Assign back
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0: logging.debug(f"[{ticker}] Dropped {dropped_rows} initial rows with NaNs.")
    else:
         logging.warning(f"[{ticker}] No specific columns for NaN check. Applying broad dropna.")
         initial_rows = len(df); df = df.dropna(how='any'); dropped_rows = initial_rows - len(df)
         if dropped_rows > 0: logging.debug(f"[{ticker}] Dropped {dropped_rows} rows (broad check).")

    # Define final feature columns (excluding base OHLCV)
    logging.debug(f"[{ticker}] Selecting final feature columns.")
    exclude_cols_base = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Optionally exclude log_return if not used as a feature for the model state
    # exclude_cols_base.append('log_return')
    final_feature_cols = [col for col in df.columns if col not in exclude_cols_base]
    logging.debug(f"[{ticker}] Final feature columns identified: {final_feature_cols}")

    required_env_cols = ['Open', 'Close']
    missing_env_cols = [col for col in required_env_cols if col not in df.columns]
    if missing_env_cols: logging.error(f"[{ticker}] Required env columns {missing_env_cols} missing."); return None
    missing_feature_cols = [col for col in final_feature_cols if col not in df.columns]
    if missing_feature_cols: logging.warning(f"[{ticker}] Features missing: {missing_feature_cols}. Adjusting."); final_feature_cols = [c for c in final_feature_cols if c in df.columns]

    # Create final DataFrame
    all_needed_cols = required_env_cols + final_feature_cols
    df_final = df[all_needed_cols].copy()

    if df_final.empty: logging.warning(f"DF for {ticker} empty after final processing."); return None
    # Check final features for NaNs again (e.g., if macro ffill didn't cover everything)
    if df_final[final_feature_cols].isnull().any().any():
        logging.warning(f"NaNs still present in final features for {ticker} before return. Applying final ffill/bfill/fillna(0).")
        df_final[final_feature_cols] = df_final[final_feature_cols].ffill().bfill().fillna(0) # Chain fills

    logging.debug(f"[{ticker}] Preprocessing complete. Final shape: {df_final.shape}")
    return df_final


def create_feature_dataset(tickers, start_date, end_date):
    """Creates the final feature dataset for all tickers."""
    logging.info("--- Starting Feature Engineering Process ---")
    all_features = {}
    macro_data = utils.load_data_from_file("_macro_data")

    # Process Macro Data (flattening, NaN checks)
    if macro_data is not None and not macro_data.empty:
        macro_data.index = pd.to_datetime(macro_data.index)
        logging.info(f"Loaded macro data. Shape: {macro_data.shape}, Columns: {macro_data.columns.tolist()}")
        if isinstance(macro_data.columns, pd.MultiIndex):
            try:
                 macro_data.columns = macro_data.columns.get_level_values(-1) # Try last level
                 macro_data = macro_data.loc[:,~macro_data.columns.duplicated()]
                 logging.info(f"Flattened macro data columns: {macro_data.columns.tolist()}")
            except Exception as e: logging.error(f"Failed to flatten macro: {e}"); macro_data = pd.DataFrame()
        if isinstance(macro_data.index, pd.MultiIndex): logging.error("Macro data has MultiIndex rows."); macro_data = pd.DataFrame()
        macro_data = macro_data.dropna(axis=1, how='all')
        if macro_data.empty: logging.warning("Macro data empty after processing.")
    else:
        logging.warning("Macro data file not found or empty.")
        macro_data = pd.DataFrame()

    # Process each ticker
    feature_count = 0
    processed_tickers = []
    for ticker in tickers:
        processed_df = preprocess_data_for_ticker(ticker, macro_data)

        if processed_df is not None and not processed_df.empty:
            try: # Slice date range
                if processed_df.index.tz is not None: processed_df.index = processed_df.index.tz_localize(None)
                processed_df_filtered = processed_df.loc[start_date:end_date].copy()
            except Exception as e: logging.warning(f"Date slice failed {ticker}: {e}"); continue

            if not processed_df_filtered.empty:
                 # Identify feature columns (all except Open, Close)
                 feature_cols = [col for col in processed_df_filtered.columns if col not in ['Open', 'Close']]

                 # Ensure numeric and handle NaNs after coercion
                 if feature_cols: # Only apply if feature columns exist
                     processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')
                     nan_check = processed_df_filtered[feature_cols].isnull()
                     if nan_check.any().any():
                          nan_cols_dict = nan_check.any(); nan_col_names = nan_cols_dict[nan_cols_dict].index.tolist()
                          logging.warning(f"NaNs found for {ticker} after coercion: {nan_col_names}. Ffilling.")
                          processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].ffill() # Assign back
                          nan_check_after = processed_df_filtered[feature_cols].isnull()
                          if nan_check_after.any().any():
                               remaining_nan_cols=nan_check_after.any(); remaining_nan_names=remaining_nan_cols[remaining_nan_cols].index.tolist()
                               logging.warning(f"NaNs still present for {ticker} after ffill: {remaining_nan_names}. Filling with 0.")
                               processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].fillna(0) # Assign back

                 # Final NaN check in all columns (including Open/Close if coerced earlier)
                 if processed_df_filtered.isnull().any().any():
                      final_nan_cols = processed_df_filtered.isnull().any()
                      logging.error(f"FATAL: NaNs remain in df for {ticker} after all processing: {final_nan_cols[final_nan_cols].index.tolist()}. Skipping.")
                      continue

                 all_features[ticker] = processed_df_filtered
                 feature_count += 1
                 processed_tickers.append(ticker)
                 logging.debug(f"Finished features for {ticker}, final shape: {processed_df_filtered.shape}")
            else: logging.warning(f"No data for {ticker} after date filtering.")
        # else: already logged in preprocess

    if not all_features: logging.error("No features generated for ANY ticker."); return None
    logging.info(f"--- Feature Engineering Completed for {feature_count} tickers ---")
    return {ticker: df for ticker, df in all_features.items() if ticker in processed_tickers}


# ==============================================================
# ============ STANDALONE VERIFICATION BLOCK ===================
# ==============================================================
if __name__ == "__main__":
    logging.info("--- Running Feature Engineering Standalone for Verification ---")
    # Make sure logging is set up for standalone execution
    log_formatter_sa = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    logger_sa = logging.getLogger() # Get root logger
    logger_sa.setLevel(logging.DEBUG) # Set level for verification
    if not logger_sa.handlers: # Add handlers if not already configured by import
         logger_sa.addHandler(logging.StreamHandler()) # Log to console
         # Optionally add file handler too
         # logger_sa.addHandler(logging.FileHandler("feature_engineer_verify.log"))


    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    # Use a smaller date range for faster verification
    verify_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    verify_start_date = (pd.to_datetime(verify_end_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    logging.info(f"Generating features for verification range: {verify_start_date} to {verify_end_date}")

    # Need data for this range + lookback buffer
    verify_fetch_start = (pd.to_datetime(verify_start_date) - pd.Timedelta(days=config.LOOKBACK_WINDOW + 45)).strftime('%Y-%m-%d')
    logging.info("Fetching data for verification range...")
    # Ensure data exists - run data_fetcher if necessary for this range
    # For simplicity, assume data_fetcher ran recently covering this period
    # data_fetcher.fetch_and_save_all_data(stock_tickers, verify_fetch_start, verify_end_date)

    features_dict = create_feature_dataset(stock_tickers, verify_start_date, verify_end_date)

    if not features_dict:
        print("\nFeature generation failed or produced no results for verification.")
        exit()

    # --- Verification Steps ---
    print("\n" + "="*30)
    print(" Feature Verification")
    print(f" Generated features for {len(features_dict)} tickers.")
    print("="*30)

    sample_tickers_to_check = ['AAPL', 'MSFT', 'JPM', 'AMT'] # Add tickers you want to inspect
    expected_macro_indicators = list(config.MACRO_FEATURES.keys()) # Get expected macro names

    # Determine expected pandas-ta columns dynamically
    expected_ta_output_cols = []
    try:
        dummy_strategy_list_ver = [p.copy() for p in config.TECHNICAL_INDICATORS.values()]
        if dummy_strategy_list_ver:
            dummy_strategy_ver = ta.Strategy(name="DummyVerify", ta=dummy_strategy_list_ver)
            dummy_df_ver = pd.DataFrame({'open': [10, 11], 'high': [12, 12], 'low': [9, 10],'close': [11, 11.5], 'volume': [100, 110]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
            dummy_df_ver.ta.strategy(dummy_strategy_ver, append=True)
            expected_ta_output_cols = [col for col in dummy_df_ver.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            logging.info(f"Verification: Expecting TA columns like: {expected_ta_output_cols[:5]}...") # Show sample
    except Exception as e:
        logging.warning(f"Verification: Could not dynamically determine expected TA columns: {e}")
        # Fallback: Manually list expected columns if dynamic check fails
        # expected_ta_output_cols = ['SMA_10', 'SMA_50', 'RSI_14', ...]


    for sample_ticker in sample_tickers_to_check:
        if sample_ticker in features_dict:
            print(f"\n--- Verifying Ticker: {sample_ticker} ---")
            df_sample = features_dict[sample_ticker]

            # 1. Check Columns
            print("\n[Check 1: Columns]")
            actual_columns = df_sample.columns.tolist()
            print(f"Columns present ({len(actual_columns)}): {actual_columns}")
            missing_tech = [col for col in expected_ta_output_cols if col not in actual_columns]
            missing_macro = [col for col in expected_macro_indicators if col not in actual_columns]
            if not missing_tech: print("  > Expected technical indicators found.")
            else: print(f"  > !! WARNING !! Missing technical indicators: {missing_tech}")
            if not missing_macro: print("  > Expected macro indicators found.")
            else: print(f"  > !! WARNING !! Missing macro indicators: {missing_macro}")
            if 'log_return' not in actual_columns: print(" > !! WARNING !! Missing log_return column.")

            # 2. Check Data Types
            print("\n[Check 2: Data Types]")
            print(df_sample.dtypes.to_string())

            # 3. Check for NaNs
            print("\n[Check 3: NaN Values]")
            nan_counts = df_sample.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if nan_cols.empty: print("  > No NaN values found.")
            else: print(f"  > !! WARNING !! NaN values found:\n{nan_cols.to_string()}")

            # 4. Check Sample Values
            print("\n[Check 4: Sample Values (Last 5 Rows)]")
            pd.set_option('display.width', 1000) # Adjust display width for wide dataframes
            print(df_sample.tail().to_string())

        else:
            print(f"\n--- Ticker {sample_ticker} not found in features_dict ---")

    print("\n" + "="*30)
    print(" Verification Complete")
    print("="*30)

    logging.info("--- Feature Engineering Standalone Verification Finished ---")
