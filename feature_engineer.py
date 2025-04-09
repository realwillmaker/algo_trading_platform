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

    # --- CORRECTED STRATEGY BUILDING ---
    # Create a list of dictionaries, where each dict represents an indicator
    strategy_list = []
    for name, params in indicators_config.items():
        # The dictionary itself should contain 'kind' and its parameters
        indicator_dict = params.copy() # Contains 'kind' and other params like 'length'
        strategy_list.append(indicator_dict)
    # --- END CORRECTION ---


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
    new_cols = [col for col in df_ta.columns if col.lower() not in [c.lower() for c in original_cols]]

    df_result = df.copy()
    for col in new_cols:
        df_result[col] = df_ta[col]

    return df_result


# --- Keep the rest of feature_engineer.py the same ---
def add_fundamental_features(df, ticker):
     """Placeholder - No changes needed here"""
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

    df = calculate_technical_indicators(df) # Uses updated function

    df = add_fundamental_features(df, ticker)

    if 'Close' in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    else:
        logging.warning(f"'Close' column not found in DataFrame for {ticker} after TA calculation. Cannot compute log_return.")

    if macro_data is not None and not macro_data.empty:
         if not isinstance(macro_data.index, pd.DatetimeIndex):
             macro_data.index = pd.to_datetime(macro_data.index)
         df = pd.merge(df, macro_data, left_index=True, right_index=True, how='left')
         macro_cols = macro_data.columns
         cols_to_fill = [col for col in macro_cols if col in df.columns]
         if cols_to_fill:
              df[cols_to_fill] = df[cols_to_fill].ffill()

    # --- Determine indicator columns dynamically for NaN check ---
    indicator_cols = []
    try:
        # Create a dummy strategy object to get expected column names
        dummy_strategy_list = [p.copy() for p in config.TECHNICAL_INDICATORS.values()]
        dummy_strategy = ta.Strategy(name="Dummy", ta=dummy_strategy_list)
        # Apply to a minimal dummy DataFrame to see generated names
        dummy_df = pd.DataFrame({
            'open': [1, 1], 'high': [1, 1], 'low': [1, 1],
            'close': [1, 1], 'volume': [1, 1]
        })
        dummy_df.ta.strategy(dummy_strategy, append=True)
        indicator_cols = [col for col in dummy_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        logging.debug(f"Dynamically determined indicator columns: {indicator_cols}")
    except Exception as e:
        logging.warning(f"Could not dynamically determine indicator columns: {e}. Falling back to broad NaN check.")

    if not indicator_cols: # Fallback if dynamic determination failed
         nan_check_cols = ['log_return'] # Check at least log return if available
         if 'log_return' not in df.columns: nan_check_cols = []
    else:
         nan_check_cols = ['log_return'] + indicator_cols
         nan_check_cols = [col for col in nan_check_cols if col in df.columns] # Ensure cols exist

    if nan_check_cols:
        initial_rows = len(df)
        df = df.dropna(subset=nan_check_cols, how='any')
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
             logging.debug(f"Dropped {dropped_rows} rows with NaNs in check columns ({nan_check_cols}) for {ticker}.")
    else:
         logging.warning(f"No specific columns found for NaN check in {ticker}. Applying dropna(how='any') broadly.")
         initial_rows = len(df)
         df = df.dropna(how='any')
         dropped_rows = initial_rows - len(df)
         if dropped_rows > 0:
             logging.debug(f"Dropped {dropped_rows} rows with NaNs (broad check) for {ticker}.")


    if 'Open' not in df.columns or 'Close' not in df.columns:
         logging.error(f"Required columns 'Open' or 'Close' missing for {ticker} after processing. Check TA function.")
         return None

    return df

# --- create_feature_dataset function remains the same ---
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
            # Example: Keep only the first level (assuming it's the feature name like 'VIX')
            macro_data.columns = macro_data.columns.get_level_values(0)
            # Remove duplicate columns if flattening created them (e.g., multiple 'Close')
            macro_data = macro_data.loc[:,~macro_data.columns.duplicated()]
            logging.info(f"Flattened macro data columns: {macro_data.columns}")
        # --- END FIX ---

        # Check if index is also MultiIndex (less likely but possible)
        if isinstance(macro_data.index, pd.MultiIndex):
             logging.warning("Macro data index is MultiIndex. This is unexpected. Attempting to reset index.")
             # This might lose date information if not handled carefully
             # Depending on structure, might need .reset_index(level=...) or other logic
             # For now, log a strong warning as this indicates a deeper issue
             logging.error("Macro data has MultiIndex on rows. Merge will likely fail or be incorrect. Please check _macro_data.parquet structure.")
             # Set to empty DataFrame to prevent merge errors, but features will be missing
             macro_data = pd.DataFrame()


        # Drop any potential all-NaN columns that might result from loading/flattening
        macro_data = macro_data.dropna(axis=1, how='all')

        if macro_data.empty:
             logging.warning("Macro data became empty after processing MultiIndex or NaNs. Proceeding without macro features.")

    else:
        # Handle case where macro data file doesn't exist or is empty from the start
        logging.warning("Macro data file not found or empty. Proceeding without macro features.")
        macro_data = pd.DataFrame() # Use empty DataFrame to allow processing to continue


    # --- Rest of the function remains the same ---
    for ticker in tickers:
        logging.debug(f"Processing features for {ticker}...")
        # Pass the processed (or empty) macro_data DataFrame
        processed_df = preprocess_data_for_ticker(ticker, macro_data)
        if processed_df is not None and not processed_df.empty:
            processed_df_filtered = processed_df.loc[start_date:end_date].copy()
            if not processed_df_filtered.empty:
                 feature_cols = [col for col in processed_df_filtered.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']]
                 for col in feature_cols:
                      processed_df_filtered[col] = pd.to_numeric(processed_df_filtered[col], errors='coerce')
                 if processed_df_filtered[feature_cols].isnull().any().any():
                      nan_cols = processed_df_filtered[feature_cols].isnull().any()
                      logging.warning(f"NaNs found in feature columns for {ticker} after to_numeric coercion: {nan_cols[nan_cols].index.tolist()}. Filling with 0.")
                      processed_df_filtered[feature_cols] = processed_df_filtered[feature_cols].fillna(0)

                 all_features[ticker] = processed_df_filtered
                 logging.debug(f"Finished features for {ticker}, shape: {processed_df_filtered.shape}")
            else:
                 logging.warning(f"No data remaining for {ticker} after date filtering {start_date} to {end_date}.")
        else:
             logging.warning(f"Skipping {ticker} due to processing issues or lack of data.")


    if not all_features:
         logging.error("No features were generated for any ticker. Exiting.")
         return None

    logging.info(f"--- Feature Engineering Completed for {len(all_features)} tickers ---")
    return all_features


    if not all_features:
         logging.error("No features were generated for any ticker. Exiting.")
         return None

    logging.info(f"--- Feature Engineering Completed for {len(all_features)} tickers ---")
    return all_features # Return dictionary


if __name__ == "__main__":
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]

    features_dict = create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)

    if features_dict:
        sample_ticker = next(iter(features_dict), None) # Get first available ticker safely
        if sample_ticker:
            print(f"\nSample Features for {sample_ticker} (using pandas-ta):")
            print(features_dict[sample_ticker].head())
            print("\nFeatures Summary:")
            print(f"Shape for {sample_ticker}: {features_dict[sample_ticker].shape}")
            print(f"Columns for {sample_ticker}: {features_dict[sample_ticker].columns.tolist()}")
        else:
             print("\nNo features dictionary generated.")
