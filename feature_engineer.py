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

    df_ta = df.copy()
    df_ta.columns = [col.lower() for col in df_ta.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_ta.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_ta.columns]
        logging.warning(f"DataFrame missing required columns for pandas-ta: {missing}. Skipping TA calculation.")
        return df

    strategy_list = []
    for name, params in indicators_config.items():
        indicator_dict = params.copy()
        strategy_list.append(indicator_dict)

    if strategy_list:
        strategy = ta.Strategy(name="Custom TA Strategy", ta=strategy_list)
        try:
            df_ta.ta.strategy(strategy, append=True)
            logging.debug(f"Applied pandas-ta strategy. New columns: {df_ta.columns}")
        except Exception as e:
            logging.error(f"Error applying pandas-ta strategy: {e}", exc_info=True)
            return df
    else:
        logging.warning("No indicators defined in strategy_list. Skipping TA calculation.")
        return df

    original_cols = list(df.columns)
    new_cols = [col for col in df_ta.columns if col.lower() not in [c.lower() for c in original_cols]]
    df_result = df.copy()
    for col in new_cols:
        df_result[col] = df_ta[col]
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
    if df is None or df.empty:
        logging.warning(f"No data loaded for {ticker}, skipping.")
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
         logging.debug(f"Converting index to DatetimeIndex for {ticker}")
         df.index = pd.to_datetime(df.index)

    # Calculate Technical Indicators
    logging.debug(f"Calculating technical indicators for {ticker}")
    df = calculate_technical_indicators(df)
    if df is None:
         logging.error(f"Technical indicator calculation failed for {ticker}.")
         return None

    # Add Fundamental Features (Placeholder)
    logging.debug(f"Adding fundamental features (placeholder) for {ticker}")
    df = add_fundamental_features(df, ticker)
    if df is None: return None

    # Calculate Log Return
    logging.debug(f"Calculating log return for {ticker}")
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isnull().any():
             logging.warning(f"NaNs found in 'Close' column for {ticker} after coercion. Log returns might be affected.")
        # Use pct_change() + 1 for robustness against zero prices
        df['log_return'] = np.log(df['Close'].pct_change() + 1).fillna(0)
        # --- CORRECTED LINE: Assign back instead of using inplace=True ---
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], 0)
        # ----------------------------------------------------------------
    else:
        logging.warning(f"'Close' column not found or not numeric for {ticker}. Cannot compute log_return. Setting to 0.")
        df['log_return'] = 0.0

    # Merge Macro Data
    logging.debug(f"Attempting to merge macro data for {ticker}")
    if macro_data is not None and not macro_data.empty:
        # --- Start Merge Logic (ensure indices compatible, flatten, merge, ffill) ---
        if not isinstance(macro_data.index, pd.DatetimeIndex): macro_data.index = pd.to_datetime(macro_data.index)
        if df.index.tz != macro_data.index.tz: macro_data.index = macro_data.index.tz_localize(None); df.index = df.index.tz_localize(None) # Ensure both naive
        macro_data_to_merge = macro_data.copy()
        if isinstance(macro_data_to_merge.columns, pd.MultiIndex):
            try:
                macro_data_to_merge.columns = macro_data_to_merge.columns.get_level_values(-1)
                macro_data_to_merge = macro_data_to_merge.loc[:,~macro_data_to_merge.columns.duplicated()]
            except Exception: macro_data_to_merge = pd.DataFrame()
        if isinstance(macro_data_to_merge.index, pd.MultiIndex) or df.index.nlevels != macro_data_to_merge.index.nlevels: pass # Skip merge if issues
        elif not macro_data_to_merge.empty:
            try:
                df = pd.merge(df, macro_data_to_merge, left_index=True, right_index=True, how='left', suffixes=('', '_macro'))
                merged_cols = set(macro_data_to_merge.columns) & set(df.columns)
                cols_to_fill = [col for col in merged_cols if col in df.columns]
                if cols_to_fill: df[cols_to_fill] = df[cols_to_fill].ffill() # Use direct assignment for ffill result
            except Exception as e: logging.error(f"[{ticker}] Error during merge/ffill: {e}", exc_info=True)
        # --- End Merge Logic ---
    else:
        logging.debug(f"[{ticker}] No macro data provided or empty, skipping merge.")


    # Determine indicator columns for NaN check
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
    except Exception as e: logging.warning(f"[{ticker}] Could not dynamically determine indicator columns: {e}")

    # Drop initial rows with NaNs
    nan_check_cols = []
    if 'log_return' in df.columns: nan_check_cols.append('log_return')
    nan_check_cols.extend(indicator_cols)
    nan_check_cols = [col for col in nan_check_cols if col in df.columns]
    if nan_check_cols:
        initial_rows = len(df)
        # --- Use direct assignment instead of inplace=True for dropna ---
        df = df.dropna(subset=nan_check_cols, how='any')
        # ---------------------------------------------------------------
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0: logging.debug(f"[{ticker}] Dropped {dropped_rows} initial rows with NaNs.")
    else:
         logging.warning(f"[{ticker}] No specific columns for NaN check. Applying dropna(how='any').")
         initial_rows = len(df)
         # --- Use direct assignment instead of inplace=True for dropna ---
         df = df.dropna(how='any')
         # ---------------------------------------------------------------
         dropped_rows = initial_rows - len(df)
         if dropped_rows > 0: logging.debug(f"[{ticker}] Dropped {dropped_rows} rows (broad check).")


    # Define final feature columns
    logging.debug(f"[{ticker}] Selecting final feature columns.")
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']
    final_feature_cols = [col for col in df.columns if col not in exclude_cols]
    logging.debug(f"[{ticker}] Final feature columns identified: {final_feature_cols}")
    required_env_cols = ['Open', 'Close']
    missing_env_cols = [col for col in required_env_cols if col not in df.columns]
    if missing_env_cols:
         logging.error(f"[{ticker}] Required env columns {missing_env_cols} missing before final selection.")
         return None
    missing_feature_cols = [col for col in final_feature_cols if col not in df.columns]
    if missing_feature_cols:
         logging.warning(f"[{ticker}] Feature columns {missing_feature_cols} missing. Adjusting.")
         final_feature_cols = [col for col in final_feature_cols if col in df.columns]

    # Create final DataFrame
    all_needed_cols = required_env_cols + final_feature_cols
    df_final = df[all_needed_cols].copy() # Select needed cols

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

    # --- Process Macro Data ---
    if macro_data is not None:
        macro_data.index = pd.to_datetime(macro_data.index)
        logging.info(f"Loaded macro data. Shape: {macro_data.shape}, Columns: {macro_data.columns}")
        if isinstance(macro_data.columns, pd.MultiIndex):
            try:
                 macro_data.columns = macro_data.columns.get_level_values(-1)
                 macro_data = macro_data.loc[:,~macro_data.columns.duplicated()]
                 logging.info(f"Flattened macro data columns: {macro_data.columns}")
            except Exception as e: macro_data = pd.DataFrame() # Reset if error
        if isinstance(macro_data.index, pd.MultiIndex): macro_data = pd.DataFrame()
        macro_data = macro_data.dropna(axis=1, how='all')
        if macro_data.empty: logging.warning("Macro data empty after processing.")
    else:
        logging.warning("Macro data file not found or empty.")
        macro_data = pd.DataFrame()
    # --- End Macro Processing ---


    # --- Process each ticker ---
    feature_count = 0
    processed_tickers = []
    for ticker in tickers:
        processed_df = preprocess_data_for_ticker(ticker, macro_data) # Returns df with specific columns

        if processed_df is not None and not processed_df.empty:
            try:
                if processed_df.index.tz is not None: processed_df.index = processed_df.index.tz_localize(None)
                processed_df_filtered = processed_df.loc[start_date:end_date].copy()
            except Exception as e:
                 logging.warning(f"Date range slicing ({start_date} to {end_date}) failed for {ticker}: {e}")
                 continue

            if not processed_df_filtered.empty:
                 # Identify feature columns (all except Open, Close)
                 feature_cols = [col for col in processed_df_filtered.columns if col not in ['Open', 'Close']]

                 # Convert features to numeric, coerce errors
                 processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce')

                 # Check for NaNs after coercion and fill
                 nan_check = processed_df_filtered[feature_cols].isnull()
                 if nan_check.any().any():
                      nan_cols_dict = nan_check.any()
                      nan_col_names = nan_cols_dict[nan_cols_dict].index.tolist()
                      logging.warning(f"NaNs found for {ticker} after coercion: {nan_col_names}. Applying ffill.")
                      # --- Use direct assignment for ffill result ---
                      processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].ffill()
                      # --- Check again and fill remaining with 0 ---
                      nan_check_after_ffill = processed_df_filtered[feature_cols].isnull()
                      if nan_check_after_ffill.any().any():
                           remaining_nan_cols = nan_check_after_ffill.any()
                           remaining_nan_names = remaining_nan_cols[remaining_nan_cols].index.tolist()
                           logging.warning(f"NaNs still present for {ticker} after ffill: {remaining_nan_names}. Filling with 0.")
                           # --- Use direct assignment for fillna result ---
                           processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].fillna(0)

                 # Final NaN check
                 if processed_df_filtered[feature_cols].isnull().any().any():
                      logging.error(f"FATAL: NaNs remain in features for {ticker}. Skipping.")
                      continue

                 all_features[ticker] = processed_df_filtered
                 feature_count += 1
                 processed_tickers.append(ticker)
                 logging.debug(f"Finished features for {ticker}, final shape: {processed_df_filtered.shape}")
            else:
                 logging.warning(f"No data for {ticker} after date filtering.")
        else:
             pass # Already logged in preprocess_data_for_ticker

    if not all_features:
         logging.error("No features generated for ANY ticker.")
         return None

    logging.info(f"--- Feature Engineering Completed for {feature_count} tickers ---")
    return {ticker: df for ticker, df in all_features.items() if ticker in processed_tickers}


if __name__ == "__main__":
    # Example Usage & Verification
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Ensure logging is configured for standalone run
    logging.info("--- Running Feature Engineering Standalone for Verification ---")
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    # Use a smaller date range for faster verification if needed
    verify_start_date = (pd.to_datetime(config.END_DATE_TRAIN) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    verify_end_date = config.END_DATE_TRAIN
    logging.info(f"Generating features for verification range: {verify_start_date} to {verify_end_date}")

    features_dict = create_feature_dataset(stock_tickers, verify_start_date, verify_end_date)

    if not features_dict:
        print("\nFeature generation failed or produced no results.")
        exit()

    # --- Verification Steps ---
    print("\n" + "="*30)
    print(" Feature Verification")
    print(f" Generated features for {len(features_dict)} tickers.")
    print("="*30)

    # Choose a sample ticker (or loop through a few)
    sample_tickers_to_check = ['AAPL', 'MSFT', 'JPM'] # Add tickers you want to inspect
    for sample_ticker in sample_tickers_to_check:
        if sample_ticker in features_dict:
            print(f"\n--- Verifying Ticker: {sample_ticker} ---")
            df_sample = features_dict[sample_ticker]

            # 1. Check Columns: Do expected indicator columns exist?
            print("\n[Check 1: Columns]")
            actual_columns = df_sample.columns.tolist()
            print(f"Columns present ({len(actual_columns)}): {actual_columns}")
            # Define expected technical indicator columns based on config and pandas-ta naming
            # Note: pandas-ta MACD adds 3 cols (_12_26_9, h_12_26_9, s_12_26_9), BBands adds 5 (L, M, U, B, P)
            expected_tech_indicators = list(config.TECHNICAL_INDICATORS.keys()) # From config
            # Manually list expected output columns from pandas-ta for the config
            expected_output_cols = ['SMA_10', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'ATRr_14', 'OBV'] # Adjust based on your EXACT config and pandas-ta version output
            expected_macro_indicators = [k for k,v in config.MACRO_FEATURES.items() if v == '^VIX'] # Currently just 'VIX' if using ^VIX
            missing_tech = [col for col in expected_output_cols if col not in actual_columns]
            missing_macro = [col for col in expected_macro_indicators if col not in actual_columns]
            if not missing_tech: print("  > Expected technical indicators found.")
            else: print(f"  > !! WARNING !! Missing technical indicators: {missing_tech}")
            if not missing_macro: print("  > Expected macro indicators found.")
            else: print(f"  > !! WARNING !! Missing macro indicators: {missing_macro}")

            # 2. Check Data Types: Are indicators numeric?
            print("\n[Check 2: Data Types]")
            print(df_sample.dtypes.to_string())
            # Look for float64 or int64 for indicator columns

            # 3. Check for NaNs: Should be minimal after processing
            print("\n[Check 3: NaN Values]")
            nan_counts = df_sample.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if nan_cols.empty:
                print("  > No NaN values found in the final features for this date range.")
            else:
                print("  > !! WARNING !! NaN values found:")
                print(nan_cols.to_string())

            # 4. Check Sample Values: Look at recent data
            print("\n[Check 4: Sample Values (Last 5 Rows)]")
            print(df_sample.tail().to_string())
            # Manually inspect if values look reasonable (e.g., RSI range, SMA values near Close, VIX values)

        else:
            print(f"\n--- Ticker {sample_ticker} not found in features_dict ---")

    print("\n" + "="*30)
    print(" Verification Complete")
    print("="*30)

    logging.info("--- Feature Engineering Standalone Verification Finished ---")
