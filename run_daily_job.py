import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, A2C # Import your algo
import logging
import os
import time
from datetime import datetime, timedelta
import json # Import json

import config
import utils # Imports load_portfolio_state, save_portfolio_state, get_market_caps
import data_fetcher
import feature_engineer
import portfolio_manager
# No longer need schwab_executor

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
log_file = "run_daily_job.log" # Use a specific log file name

# File Handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO) # Log INFO level and above to file

# Stream Handler (Console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO) # Log INFO level and above to console

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Set root logger to DEBUG to capture all levels
# Remove default basicConfig handlers if they exist to avoid duplicate logs
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# --------------------


def get_latest_features(tickers):
    """Generates features using data up to the latest available day."""
    logging.info("Generating latest features...")
    # --- INCREASE BUFFER SLIGHTLY ---
    buffer_days = 45 # Increased from 35, adjust if needed
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=config.LOOKBACK_WINDOW + buffer_days)).strftime('%Y-%m-%d')
    logging.info(f"Feature calculation range for job: {start_date} to {end_date}")
    # -----------------------------

    # Generate features for the required range
    features = feature_engineer.create_feature_dataset(tickers, start_date, end_date)
    if not features:
         raise RuntimeError("Failed to generate features for latest data (create_feature_dataset returned None or empty).")

    latest_features_dict = {}
    latest_common_date = None
    common_dates = None
    valid_tickers = sorted(list(features.keys())) # Tickers where feature gen succeeded

    if not valid_tickers:
        raise RuntimeError("Feature generation succeeded but produced an empty dictionary.")

    # --- Find common dates across generated features ---
    logging.debug(f"Finding common dates across {len(valid_tickers)} tickers...")
    for ticker in valid_tickers:
        df_ticker = features[ticker]
        if df_ticker is None or df_ticker.empty:
             logging.warning(f"Feature dictionary contains None or empty DataFrame for {ticker}. Skipping for common date calculation.")
             continue # Skip this ticker if data is bad

        # Ensure index is DatetimeIndex
        if not isinstance(df_ticker.index, pd.DatetimeIndex):
             logging.warning(f"Converting index to DatetimeIndex for {ticker} during common date check.")
             try:
                 df_ticker.index = pd.to_datetime(df_ticker.index)
             except Exception as e:
                 logging.error(f"Failed to convert index for {ticker}: {e}. Skipping this ticker.")
                 continue

        # Ensure timezone naive for comparison
        if df_ticker.index.tz is not None:
             df_ticker.index = df_ticker.index.tz_localize(None)

        dates = df_ticker.index
        if common_dates is None: common_dates = dates
        else: common_dates = common_dates.intersection(dates)

        # Check if intersection became empty - means no common dates found
        if common_dates is not None and common_dates.empty:
             logging.error(f"Common date intersection became empty after processing ticker {ticker}.")
             break # No need to check further


    if common_dates is None or common_dates.empty:
        raise RuntimeError("No common dates found in the latest feature set across tickers.")
    # -----------------------------------------------

    latest_common_date = max(common_dates)
    # --- ADD LOGGING ---
    logging.info(f"Latest common date with features determined as: {latest_common_date.strftime('%Y-%m-%d')}")
    # -------------------

    # Check how recent the data is
    if latest_common_date.date() < (datetime.now() - timedelta(days=4)).date(): # Increased tolerance
         logging.warning(f"Latest feature date {latest_common_date.strftime('%Y-%m-%d')} seems old. Data might be stale.")

    # Define the specific lookback window range ending on the common date
    lookback_start_date = latest_common_date - pd.Timedelta(days=config.LOOKBACK_WINDOW - 1)
    # --- ADD LOGGING ---
    logging.info(f"Required lookback range for final state: {lookback_start_date.strftime('%Y-%m-%d')} to {latest_common_date.strftime('%Y-%m-%d')}")
    # -------------------

    processed_features_for_state = {}
    final_valid_tickers = [] # Tickers that meet the final lookback length requirement

    # --- Check lookback window length for each ticker ---
    logging.debug("Checking final lookback window length for each ticker...")
    for ticker in valid_tickers:
         # Skip if features dict doesn't contain the ticker (shouldn't happen if valid_tickers is from keys)
         if ticker not in features: continue

         ticker_features_full = features[ticker]
         if ticker_features_full is None or ticker_features_full.empty: continue # Skip if empty df sneaked through

         # Slice the specific lookback window
         try:
              # Ensure index is datetime and timezone naive again before slicing
              if not isinstance(ticker_features_full.index, pd.DatetimeIndex):
                  ticker_features_full.index = pd.to_datetime(ticker_features_full.index)
              if ticker_features_full.index.tz is not None:
                  ticker_features_full.index = ticker_features_full.index.tz_localize(None)

              # Slice using the calculated start/end dates
              ticker_features_window = ticker_features_full.loc[lookback_start_date:latest_common_date]
              actual_rows = len(ticker_features_window)

              # --- ADD DEBUG LOG ---
              # Log details for potentially problematic tickers or just first/last few
              if ticker in ['ZBRA', 'ZTS', valid_tickers[0], valid_tickers[-1]]: # Example logging targets
                  actual_start = ticker_features_window.index.min().strftime('%Y-%m-%d') if actual_rows > 0 else "N/A"
                  actual_end = ticker_features_window.index.max().strftime('%Y-%m-%d') if actual_rows > 0 else "N/A"
                  logging.debug(f"Checking {ticker}: Range {actual_start} to {actual_end}, Rows={actual_rows}, Required={config.LOOKBACK_WINDOW}")
              # --- END DEBUG LOG ---

              # Check if exactly the required number of rows are present
              if actual_rows == config.LOOKBACK_WINDOW:
                  # Select feature columns (all except Open, Close)
                  feature_cols = [col for col in ticker_features_window.columns if col not in ['Open', 'Close']]
                  # Final check for NaNs in the selected window slice's features
                  if ticker_features_window[feature_cols].isnull().any().any():
                       nan_cols = ticker_features_window[feature_cols].isnull().any()
                       logging.warning(f"NaNs found within the final lookback window's features for {ticker}: {nan_cols[nan_cols].index.tolist()}. Excluding.")
                       continue # Skip if NaNs exist in the final slice's features

                  processed_features_for_state[ticker] = ticker_features_window[feature_cols]
                  final_valid_tickers.append(ticker)
              else:
                  # Warning already includes row count
                  logging.warning(f"Ticker {ticker} does not have enough data ({actual_rows} rows) for the lookback window ending {latest_common_date.strftime('%Y-%m-%d')}. Excluding from prediction.")

         except KeyError:
              # This might happen if lookback_start_date or latest_common_date is not in the index after feature processing
               logging.warning(f"KeyError slicing lookback window for {ticker} (Dates: {lookback_start_date} - {latest_common_date}). Maybe data gap? Excluding.")
               continue
         except Exception as slice_err:
              logging.warning(f"Error slicing lookback window for {ticker}: {slice_err}. Excluding.", exc_info=True) # Log traceback for unexpected errors
              continue # Skip ticker if slicing fails

    # --- Final Check ---
    if not final_valid_tickers:
         # Provide more context in the error
         logging.error(f"Failed to find *any* tickers with exactly {config.LOOKBACK_WINDOW} contiguous days of data ending on {latest_common_date.strftime('%Y-%m-%d')}.")
         raise RuntimeError("No tickers had sufficient continuous data for the final lookback window.")

    logging.info(f"Features ready for state construction for {len(final_valid_tickers)} tickers.")
    return processed_features_for_state, final_valid_tickers, latest_common_date


def construct_observation_state(feature_window_dict, current_weights_dict, ordered_tickers):
     """Constructs the flattened observation state for the RL model."""
     # Ensure inputs are valid
     if not ordered_tickers: raise ValueError("Ordered tickers list is empty.")
     if not feature_window_dict: raise ValueError("Feature dictionary is empty.")
     first_ticker = ordered_tickers[0]
     if first_ticker not in feature_window_dict or feature_window_dict[first_ticker].empty:
          raise ValueError(f"Feature data missing/empty for first ticker {first_ticker}.")

     num_stocks = len(ordered_tickers)
     # Use shape from the first ticker's data - assumes consistency checked elsewhere
     num_features_per_stock = feature_window_dict[first_ticker].shape[1]
     lookback_window = config.LOOKBACK_WINDOW # Use config directly for consistency

     state_feature_dim = lookback_window * num_features_per_stock * num_stocks
     state_weight_dim = num_stocks
     obs_dim = state_feature_dim + state_weight_dim

     # Pre-allocate observation array for efficiency
     observation = np.zeros(obs_dim, dtype=np.float32)

     # --- Populate Feature Part ---
     feature_part = observation[:state_feature_dim] # Get a view
     idx = 0
     for i in range(lookback_window):
         for ticker in ordered_tickers:
             try:
                 # iloc is generally faster if index is guaranteed sequential within window
                 feature_vector = feature_window_dict[ticker].iloc[i].values
                 expected_len = num_features_per_stock
                 actual_len = len(feature_vector)

                 if actual_len == expected_len:
                      feature_part[idx : idx + expected_len] = feature_vector
                 else: # Fallback padding (should be rare if preprocessing is correct)
                     logging.warning(f"Shape mismatch {ticker} day {i}: Expected {expected_len}, got {actual_len}. Padding.")
                     padded_vector = np.zeros(expected_len, dtype=np.float32)
                     len_to_copy = min(actual_len, expected_len)
                     padded_vector[:len_to_copy] = feature_vector[:len_to_copy]
                     feature_part[idx : idx + expected_len] = padded_vector

             except (IndexError, KeyError) as e:
                 logging.warning(f"Error accessing features {ticker} day {i}: {e}. Using zeros.")
                 # Zeros are already there due to initialization
             except Exception as e:
                  logging.error(f"Unexpected error getting features {ticker} day {i}: {e}. Using zeros.", exc_info=True)
             idx += num_features_per_stock
     # --- End Feature Population ---


     # --- Populate Weight Part ---
     current_weights = np.array([current_weights_dict.get(ticker, 0.0) for ticker in ordered_tickers], dtype=np.float32)
     if len(current_weights) != state_weight_dim:
          logging.error(f"Weight dimension mismatch! Expected {state_weight_dim}, got {len(current_weights)}. Check weight dict.")
          # Pad weights if needed
          padded_weights = np.zeros(state_weight_dim, dtype=np.float32)
          len_to_copy = min(len(current_weights), state_weight_dim)
          padded_weights[:len_to_copy] = current_weights[:len_to_copy]
          current_weights = padded_weights

     observation[state_feature_dim:] = current_weights # Assign weights to the end
     # --- End Weight Population ---


     # Final check should ideally not be needed if dimensions calculated correctly
     if idx != state_feature_dim:
          logging.critical(f"CRITICAL: Index mismatch after feature population! idx={idx}, expected={state_feature_dim}")
          raise RuntimeError("Observation construction failed: Feature index mismatch.")

     return observation # Already float32


def main():
    logging.info("====== Starting Daily Rebalance Job (Manual Execution Focus) ======")
    run_start_time = time.time()
    planned_orders = []
    latest_feature_date = None # Initialize
    current_total_value = 0.0
    current_cash = 0.0
    current_holdings = {}

    try:
        # === Phase 1: Load State, Get Data, Predict, Calculate Trades ===
        logging.info("--- Phase 1: Load State, Get Data, Predict, Calculate Trades ---")

        # 1. Load Current Portfolio State
        portfolio_state = utils.load_portfolio_state()
        current_cash = portfolio_state.get('cash', config.INITIAL_CAPITAL) # Use get with default
        current_holdings = portfolio_state.get('positions', {}) # Use get with default
        logging.info(f"Loaded state - Cash: ${current_cash:,.2f}, Holdings: {len(current_holdings)} stocks")

        # 2. Update S&P 500 List & Fetch Latest Market Data
        logging.info("Fetching S&P 500 list and latest market data...")
        sp500_tickers = utils.get_sp500_tickers()
        # Fetch data for S&P stocks + any stocks currently held but not in S&P500 anymore
        stock_tickers_to_fetch = list(set(sp500_tickers) | set(current_holdings.keys()))
        fetch_end = datetime.now().strftime('%Y-%m-%d') # Include today for yfinance to get yesterday's adjusted close
        fetch_start = (datetime.now() - timedelta(days=config.LOOKBACK_WINDOW + 45)).strftime('%Y-%m-%d') # Match buffer in get_latest_features
        data_fetcher.fetch_and_save_all_data(stock_tickers_to_fetch, fetch_start, fetch_end)

        # 3. Feature Engineering for Latest Data
        logging.info("Calculating latest features...")
        # Generate features for the S&P 500 universe
        stock_tickers_for_features = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
        # get_latest_features now returns tickers that passed the final length check
        model_input_features, model_tickers, latest_feature_date = get_latest_features(stock_tickers_for_features)
        logging.info(f"Features prepared for {len(model_tickers)} tickers up to {latest_feature_date.strftime('%Y-%m-%d')}.")


        # 4. Load RL Model
        # Determine EXACTLY which tickers the model expects based on the training run logic
        # Needs MAX_TICKERS_MODEL_WAS_TRAINED_ON to be accurate
        MAX_TICKERS_MODEL_WAS_TRAINED_ON = 100 # *** Match this to the value used in train_rl.py ***
        selected_model_tickers = [] # Tickers the model *should* have been trained on

        # We need the list of tickers *before* the length check in get_latest_features
        # Let's assume 'model_tickers' from get_latest_features might be a subset of what model expects
        # Re-run selection logic based on market cap using the 'valid_tickers' from feature gen start
        # This requires slight modification of get_latest_features or re-running parts...
        # --- Simpler approach for now: Assume model uses the tickers from get_latest_features ---
        # --- This assumes the training selection resulted in the same set OR ---
        # --- that the model can handle slight variations if trained robustly ---
        # --- A more robust solution saves the exact training ticker list ---
        if len(model_tickers) > MAX_TICKERS_MODEL_WAS_TRAINED_ON:
             logging.warning(f"get_latest_features returned more tickers ({len(model_tickers)}) than model trained on ({MAX_TICKERS_MODEL_WAS_TRAINED_ON}). Using subset.")
             # Re-select based on market cap from this list if needed, or just slice
             market_caps = utils.get_market_caps(model_tickers)
             ticker_cap_list = [(t, market_caps.get(t, 0)) for t in model_tickers]
             ticker_cap_list.sort(key=lambda item: item[1], reverse=True)
             selected_model_tickers = sorted([item[0] for item in ticker_cap_list[:MAX_TICKERS_MODEL_WAS_TRAINED_ON]])
        elif len(model_tickers) == 0:
             raise RuntimeError("No tickers available after feature processing and lookback check.")
        else:
            selected_model_tickers = sorted(model_tickers) # Use the ones available

        if not selected_model_tickers:
             raise RuntimeError("Could not determine the tickers the model requires.")

        logging.info(f"Model will use input for {len(selected_model_tickers)} tickers: {selected_model_tickers[:5]}...")

        # Filter features dict for the exact selected tickers
        model_features = {t: model_input_features[t] for t in selected_model_tickers if t in model_input_features}
        if len(model_features) != len(selected_model_tickers):
             missing = [t for t in selected_model_tickers if t not in model_features]
             raise RuntimeError(f"Features missing for selected model tickers: {missing}")

        # Load the model
        logging.info(f"Loading RL model: {config.MODEL_FILENAME}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(config.MODEL_FILENAME): raise FileNotFoundError(f"Model file not found: {config.MODEL_FILENAME}")
        # Load model based on config
        model_class = {"PPO": PPO, "SAC": SAC, "A2C": A2C}.get(config.RL_ALGORITHM)
        if model_class is None: raise ValueError(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}")
        model = model_class.load(config.MODEL_FILENAME, device=device)


        # 5. Calculate Current Portfolio Value & Weights for Observation
        logging.info("Calculating current portfolio value and weights...")
        current_total_value = current_cash
        current_prices_dict = {}
        # Need prices for selected model tickers + any other holdings
        tickers_for_value_calc = list(set(selected_model_tickers) | set(current_holdings.keys()))

        logging.debug(f"Fetching prices for {len(tickers_for_value_calc)} tickers for value calc...")
        for ticker in tickers_for_value_calc:
             price_data = utils.load_data_from_file(ticker)
             if price_data is not None:
                  # Ensure index is datetime and timezone naive
                  if not isinstance(price_data.index, pd.DatetimeIndex): price_data.index = pd.to_datetime(price_data.index)
                  if price_data.index.tz is not None: price_data.index = price_data.index.tz_localize(None)
                  # Get price for the latest common feature date
                  if latest_feature_date in price_data.index:
                       price = price_data.loc[latest_feature_date]['Close']
                       if pd.notna(price) and price > 0:
                            current_prices_dict[ticker] = price
                       else: logging.warning(f"Price is NaN or zero for {ticker} on {latest_feature_date}.")
                  else: logging.warning(f"Latest feature date {latest_feature_date} not in price data index for {ticker}.")
             else: logging.warning(f"Price data file not found for {ticker}.")


        logging.debug(f"Prices fetched for value calc ({len(current_prices_dict)} tickers)")

        # Calculate total value using fetched prices
        stock_value_total = 0
        valid_holdings = {} # Recalculate holdings based on available prices
        for ticker, shares in current_holdings.items():
            price = current_prices_dict.get(ticker)
            if price is not None: # Check if price was successfully fetched
                 stock_value_total += shares * price
                 valid_holdings[ticker] = shares # Only include holdings we could value
            else:
                 logging.warning(f"Could not get price for held ticker {ticker}. Excluding from value calculation but keeping in state.")
                 valid_holdings[ticker] = shares # Keep it in state, just can't value it now

        current_total_value = current_cash + stock_value_total
        logging.info(f"Calculated current total value (based on available prices): ${current_total_value:,.2f}")

        # Calculate weights ONLY for the tickers the model needs input for
        current_weights_dict = {}
        if current_total_value > 1e-6:
             for ticker in selected_model_tickers:
                  shares = current_holdings.get(ticker, 0) # Use original holdings here
                  price = current_prices_dict.get(ticker) # Use fetched price
                  if price is not None: # Only calculate weight if price is valid
                       current_weights_dict[ticker] = (shares * price) / current_total_value
                  else:
                       current_weights_dict[ticker] = 0.0 # Assign 0 weight if price missing for model ticker
        else:
             current_weights_dict = {ticker: 0.0 for ticker in selected_model_tickers}

        logging.debug(f"Current weights for observation state ({len(current_weights_dict)} tickers)")


        # 6. Construct Observation State
        logging.info("Constructing observation state for RL model...")
        # Pass features and weights only for the selected model tickers
        observation = construct_observation_state(model_features, current_weights_dict, selected_model_tickers)


        # 7. Predict Action (Target Weights for NEXT DAY)
        logging.info("Predicting action with RL model...")
        action, _states = model.predict(observation, deterministic=True)
        # Action corresponds to selected_model_tickers order

        # 8. Calculate Target Orders
        logging.info("Calculating target orders based on prediction...")
        # Use prices fetched earlier for order calculation
        prices_for_order_calc = {t: p for t, p in current_prices_dict.items() if pd.notna(p) and p > 0}

        planned_orders = portfolio_manager.calculate_target_orders(
            current_holdings=current_holdings, # Pass the original loaded holdings
            current_cash=current_cash,
            total_portfolio_value=current_total_value, # Use value calculated from available prices
            target_weights=action,
            stock_tickers=selected_model_tickers, # List corresponding to action array
            current_prices=prices_for_order_calc # Pass dict of valid prices
        )

        logging.info(f"Calculation complete. Found {len(planned_orders)} potential trades.")
        logging.info("--- Phase 1 Completed ---")

    except Exception as e:
        logging.critical(f"!!! CRITICAL ERROR in Phase 1: {e}", exc_info=True)
        exit(1) # Exit if decision making fails


    # === Phase 2: REMOVED (No Execution) ===


    # === Phase 3: Reporting Trades for Manual Execution ===
    logging.info("--- Phase 3: Reporting Planned Trades ---")
    print("\n" + "="*30)
    print(" PLANNED REBALANCE TRADES")
    print(f" Calculation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if latest_feature_date: print(f" Based on Data Up To: {latest_feature_date.strftime('%Y-%m-%d')}")
    print(f" Current Portfolio Value Used: ${current_total_value:,.2f}")
    print(f" Current Cash Used: ${current_cash:,.2f}")
    print("-"*30)

    buys = sorted([o for o in planned_orders if o['shares'] > 0], key=lambda x: x['ticker'])
    sells = sorted([o for o in planned_orders if o['shares'] < 0], key=lambda x: x['ticker'])

    if not buys and not sells:
        print("  No trades required for rebalance.")
    else:
        if sells:
            print("SELL ORDERS:")
            for order in sells:
                print(f"  - SELL {abs(order['shares'])} {order['ticker']}")
        if buys:
            print("\nBUY ORDERS:")
            for order in buys:
                print(f"  - BUY {order['shares']} {order['ticker']}")

    print("="*30 + "\n")

    # --- Save the state THAT WAS USED for this calculation ---
    # This reflects the portfolio *before* the manual trades are placed
    utils.save_portfolio_state({'cash': current_cash, 'positions': current_holdings})

    run_end_time = time.time()
    logging.info(f"====== Daily Rebalance Job (Manual) Completed in {(run_end_time - run_start_time):.2f} seconds ======")


if __name__ == "__main__":
    main()
