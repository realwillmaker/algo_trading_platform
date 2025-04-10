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
import utils # Imports load/save state, get_market_caps
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
    """
    Generates features using data up to the latest available day.
    Selects the last N rows for the final feature window.
    """
    logging.info("Generating latest features...")
    buffer_days = 45 # Buffer for feature calculation
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=config.LOOKBACK_WINDOW + buffer_days)).strftime('%Y-%m-%d')
    logging.info(f"Feature calculation range for job: {start_date} to {end_date}")

    # Generate features for the required range
    features = feature_engineer.create_feature_dataset(tickers, start_date, end_date)
    if not features:
         raise RuntimeError("Failed to generate features for latest data (create_feature_dataset returned None or empty).")

    # Find the latest date present across *any* ticker
    latest_date_overall = None
    valid_tickers_initial = [] # Tickers with features generated
    for ticker, df_ticker in features.items():
        if df_ticker is not None and not df_ticker.empty:
            if not isinstance(df_ticker.index, pd.DatetimeIndex):
                 try: df_ticker.index = pd.to_datetime(df_ticker.index)
                 except Exception: continue
            if df_ticker.index.tz is not None: df_ticker.index = df_ticker.index.tz_localize(None)

            valid_tickers_initial.append(ticker)
            current_max_date = df_ticker.index.max()
            if latest_date_overall is None or current_max_date > latest_date_overall:
                latest_date_overall = current_max_date

    if latest_date_overall is None:
         raise RuntimeError("Could not determine any latest date from generated features.")

    latest_feature_date = latest_date_overall # Use the max date found as the reference date
    logging.info(f"Latest date found across any feature set: {latest_feature_date.strftime('%Y-%m-%d')}")

    # Check how recent the data is
    if latest_feature_date.date() < (datetime.now() - timedelta(days=4)).date():
         logging.warning(f"Latest feature date {latest_feature_date.strftime('%Y-%m-%d')} seems old. Data might be stale.")


    processed_features_for_state = {}
    final_valid_tickers = [] # Tickers that meet the final lookback length requirement

    # --- Check lookback window by taking the LAST N rows ---
    logging.debug(f"Selecting last {config.LOOKBACK_WINDOW} rows for each ticker ending near {latest_feature_date.strftime('%Y-%m-%d')}...")
    for ticker in valid_tickers_initial:
         if ticker not in features: continue
         ticker_features_full = features[ticker]
         if ticker_features_full is None or ticker_features_full.empty: continue

         # Ensure data is sorted by date ascending
         ticker_features_full.sort_index(inplace=True)

         # Check if there are enough rows *in total*
         if len(ticker_features_full) >= config.LOOKBACK_WINDOW:
             # Select the last LOOKBACK_WINDOW rows
             ticker_features_window = ticker_features_full.iloc[-config.LOOKBACK_WINDOW:]

             actual_rows = len(ticker_features_window)
             if actual_rows == config.LOOKBACK_WINDOW: # Check length after slicing
                 # Select feature columns (all except Open, Close)
                 feature_cols = [col for col in ticker_features_window.columns if col not in ['Open', 'Close']]
                 # Check for NaNs in the selected window slice's features
                 if ticker_features_window[feature_cols].isnull().any().any():
                      nan_cols = ticker_features_window[feature_cols].isnull().any()
                      logging.warning(f"NaNs found within the final lookback window's features for {ticker}: {nan_cols[nan_cols].index.tolist()}. Excluding.")
                      continue # Skip if NaNs exist

                 processed_features_for_state[ticker] = ticker_features_window[feature_cols]
                 final_valid_tickers.append(ticker)
                 # Log date range for debug
                 actual_start = ticker_features_window.index.min().strftime('%Y-%m-%d')
                 actual_end = ticker_features_window.index.max().strftime('%Y-%m-%d')
                 logging.debug(f"Selected window for {ticker}: {actual_start} to {actual_end} ({actual_rows} rows)")

             else: # Should not happen if len check above works
                  logging.warning(f"Ticker {ticker} - iloc slicing failed to return {config.LOOKBACK_WINDOW} rows, got {actual_rows}. Excluding.")
         else:
              logging.warning(f"Ticker {ticker} only has {len(ticker_features_full)} total rows, less than lookback {config.LOOKBACK_WINDOW}. Excluding.")

    # --- Final Check ---
    if not final_valid_tickers:
         logging.error(f"Failed to find *any* tickers with at least {config.LOOKBACK_WINDOW} data points.")
         raise RuntimeError("No tickers had sufficient data points for the lookback window.")

    logging.info(f"Features ready for state construction for {len(final_valid_tickers)} tickers.")
    # Return dict of DFs {ticker: feature_df_slice}, list of included tickers, and reference date
    return processed_features_for_state, sorted(final_valid_tickers), latest_feature_date # Return sorted list


def construct_observation_state(feature_window_dict, current_weights_dict, ordered_tickers):
    """Constructs the flattened observation state for the RL model."""
    if not ordered_tickers: raise ValueError("Ordered tickers list is empty.")
    if not feature_window_dict: raise ValueError("Feature dictionary is empty.")
    first_ticker = ordered_tickers[0]
    if first_ticker not in feature_window_dict or feature_window_dict[first_ticker].empty:
          raise ValueError(f"Feature data missing/empty for first ticker {first_ticker}.")

    num_stocks = len(ordered_tickers)
    num_features_per_stock = feature_window_dict[first_ticker].shape[1]
    lookback_window = config.LOOKBACK_WINDOW

    state_feature_dim = lookback_window * num_features_per_stock * num_stocks
    state_weight_dim = num_stocks
    obs_dim = state_feature_dim + state_weight_dim

    observation = np.zeros(obs_dim, dtype=np.float32)

    # Populate Feature Part
    feature_part = observation[:state_feature_dim]
    idx = 0
    for i in range(lookback_window):
        for ticker in ordered_tickers:
            try:
                feature_vector = feature_window_dict[ticker].iloc[i].values
                expected_len = num_features_per_stock
                actual_len = len(feature_vector)
                if actual_len == expected_len:
                    feature_part[idx : idx + expected_len] = feature_vector
                else:
                    logging.warning(f"Padding shape mismatch {ticker} day {i}. Expected {expected_len}, got {actual_len}.")
                    padded_vector = np.zeros(expected_len, dtype=np.float32)
                    len_to_copy = min(actual_len, expected_len)
                    padded_vector[:len_to_copy] = feature_vector[:len_to_copy]
                    feature_part[idx : idx + expected_len] = padded_vector
            except (IndexError, KeyError) as e:
                logging.warning(f"Error accessing features {ticker} day {i}: {e}. Using zeros.")
            except Exception as e:
                logging.error(f"Unexpected error getting features {ticker} day {i}: {e}. Using zeros.", exc_info=True)
            idx += num_features_per_stock

    if idx != state_feature_dim:
        raise RuntimeError(f"Observation construction failed: Feature index mismatch. idx={idx}, expected={state_feature_dim}")


    # Populate Weight Part
    current_weights = np.array([current_weights_dict.get(ticker, 0.0) for ticker in ordered_tickers], dtype=np.float32)
    if len(current_weights) != state_weight_dim:
        logging.error(f"Weight dimension mismatch! Expected {state_weight_dim}, got {len(current_weights)}. Padding.")
        padded_weights = np.zeros(state_weight_dim, dtype=np.float32)
        len_to_copy = min(len(current_weights), state_weight_dim)
        padded_weights[:len_to_copy] = current_weights[:len_to_copy]
        current_weights = padded_weights
    observation[state_feature_dim:] = current_weights

    return observation


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

        # 1. Load Current Portfolio State from file
        portfolio_state = utils.load_portfolio_state()
        current_cash = portfolio_state.get('cash', config.INITIAL_CAPITAL)
        current_holdings = portfolio_state.get('positions', {})
        logging.info(f"Loaded state - Cash: ${current_cash:,.2f}, Holdings: {len(current_holdings)} stocks")

        # 2. Update S&P 500 List & Fetch Latest Market Data
        logging.info("Fetching S&P 500 list and latest market data...")
        sp500_tickers = utils.get_sp500_tickers()
        stock_tickers_to_fetch = list(set(sp500_tickers) | set(current_holdings.keys()))
        fetch_end = datetime.now().strftime('%Y-%m-%d')
        fetch_start = (datetime.now() - timedelta(days=config.LOOKBACK_WINDOW + 45)).strftime('%Y-%m-%d')
        data_fetcher.fetch_and_save_all_data(stock_tickers_to_fetch, fetch_start, fetch_end)

        # 3. Feature Engineering for Latest Data
        logging.info("Calculating latest features...")
        stock_tickers_for_features = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
        # model_input_features: dict {ticker: df_slice} for tickers passing length check
        # model_tickers: list of tickers included in model_input_features
        # latest_feature_date: reference date used
        model_input_features, model_tickers, latest_feature_date = get_latest_features(stock_tickers_for_features)
        logging.info(f"Features prepared for {len(model_tickers)} tickers up to {latest_feature_date.strftime('%Y-%m-%d')}.")


        # 4. Load RL Model & Determine Exact Tickers Used
        # Re-run selection logic based on market cap to match training precisely
        MAX_TICKERS_MODEL_WAS_TRAINED_ON = 100 # *** Match training ***
        selected_model_tickers = [] # Tickers the model requires input for

        if not model_tickers: # Check if get_latest_features returned any valid tickers
             raise RuntimeError("No tickers available after feature processing and lookback check.")

        # Select based on market cap from the tickers that passed the lookback check
        if len(model_tickers) <= MAX_TICKERS_MODEL_WAS_TRAINED_ON:
             selected_model_tickers = sorted(model_tickers)
             logging.info(f"Using all {len(selected_model_tickers)} available tickers as it's <= model training limit.")
        else:
             logging.info(f"Selecting top {MAX_TICKERS_MODEL_WAS_TRAINED_ON} from {len(model_tickers)} based on market cap for model input...")
             # Need utils.get_market_caps function available!
             try:
                 market_caps = utils.get_market_caps(model_tickers)
             except AttributeError:
                 logging.error("utils module does not have get_market_caps. Make sure it's defined in utils.py")
                 raise # Re-raise the error
             except Exception as e:
                 logging.error(f"Error getting market caps: {e}", exc_info=True)
                 raise # Re-raise other errors

             ticker_cap_list = [(t, market_caps.get(t, 0)) for t in model_tickers]
             ticker_cap_list.sort(key=lambda item: item[1], reverse=True)
             selected_model_tickers = sorted([item[0] for item in ticker_cap_list[:MAX_TICKERS_MODEL_WAS_TRAINED_ON]])

        if not selected_model_tickers:
             raise RuntimeError("Could not select model tickers based on market cap.")
        logging.info(f"Model requires input for {len(selected_model_tickers)} tickers: {selected_model_tickers[:5]}...")

        # Filter features dict for the EXACT selected tickers
        model_features = {t: model_input_features[t] for t in selected_model_tickers if t in model_input_features}
        if len(model_features) != len(selected_model_tickers):
             missing = [t for t in selected_model_tickers if t not in model_features]
             # This indicates an internal logic error if it happens
             raise RuntimeError(f"INTERNAL ERROR: Features missing for selected model tickers: {missing}")

        # Load the model
        logging.info(f"Loading RL model: {config.MODEL_FILENAME}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(config.MODEL_FILENAME): raise FileNotFoundError(f"Model file not found: {config.MODEL_FILENAME}")
        model_class = {"PPO": PPO, "SAC": SAC, "A2C": A2C}.get(config.RL_ALGORITHM)
        if model_class is None: raise ValueError(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}")
        model = model_class.load(config.MODEL_FILENAME, device=device)
        # Optional: Check model's expected observation space if possible
        try:
             expected_model_obs_shape = model.observation_space.shape
             logging.info(f"Loaded model expects observation shape: {expected_model_obs_shape}")
             # Compare with calculation? Needs careful implementation based on selected_model_tickers length
             # expected_calc_obs_dim = config.LOOKBACK_WINDOW * len(model_features[selected_model_tickers[0]].columns) * len(selected_model_tickers) + len(selected_model_tickers)
             # if expected_model_obs_shape[0] != expected_calc_obs_dim:
             #      logging.warning(f"Model's expected obs shape {expected_model_obs_shape[0]} differs from calculation {expected_calc_obs_dim}")
        except Exception as e:
             logging.warning(f"Could not verify model's observation space: {e}")


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
                  if not isinstance(price_data.index, pd.DatetimeIndex): price_data.index = pd.to_datetime(price_data.index)
                  if price_data.index.tz is not None: price_data.index = price_data.index.tz_localize(None)
                  if latest_feature_date in price_data.index:
                       price = price_data.loc[latest_feature_date]['Close']
                       if pd.notna(price) and price > 0: current_prices_dict[ticker] = price
                       # else: logging.debug(f"Price NaN/zero for {ticker} on {latest_feature_date}") # Too verbose maybe
                  # else: logging.debug(f"Date {latest_feature_date} not in index for {ticker}")
             # else: logging.debug(f"No price data file found for {ticker}")

        logging.debug(f"Prices fetched for value calc ({len(current_prices_dict)} tickers)")

        # Calculate total value using fetched prices
        stock_value_total = 0
        for ticker, shares in current_holdings.items():
            price = current_prices_dict.get(ticker)
            if price is not None: stock_value_total += shares * price
            else: logging.warning(f"Could not get price for held {ticker} to calculate total value.")

        current_total_value = current_cash + stock_value_total
        logging.info(f"Calculated current total value (based on available prices): ${current_total_value:,.2f}")

        # Calculate weights ONLY for the tickers the model needs input for
        current_weights_dict = {}
        if current_total_value > 1e-6:
             for ticker in selected_model_tickers:
                  shares = current_holdings.get(ticker, 0)
                  price = current_prices_dict.get(ticker)
                  if price is not None: current_weights_dict[ticker] = (shares * price) / current_total_value
                  else: current_weights_dict[ticker] = 0.0
        else: current_weights_dict = {ticker: 0.0 for ticker in selected_model_tickers}
        logging.debug(f"Current weights for observation: {len(current_weights_dict)} tickers")


        # 6. Construct Observation State
        logging.info("Constructing observation state for RL model...")
        # Pass features and weights only for the selected model tickers
        observation = construct_observation_state(model_features, current_weights_dict, selected_model_tickers)
        logging.info(f"Observation state constructed with shape: {observation.shape}")


        # 7. Predict Action (Target Weights for NEXT DAY)
        logging.info("Predicting action with RL model...")
        action, _states = model.predict(observation, deterministic=True)
        # Action corresponds to selected_model_tickers order


        # 8. Calculate Target Orders
        logging.info("Calculating target orders based on prediction...")
        prices_for_order_calc = {t: p for t, p in current_prices_dict.items() if pd.notna(p) and p > 0}
        if not prices_for_order_calc:
             logging.warning("No valid prices available for order calculation. Skipping trade calculation.")
             planned_orders = []
        else:
             planned_orders = portfolio_manager.calculate_target_orders(
                 current_holdings=current_holdings,
                 current_cash=current_cash,
                 total_portfolio_value=current_total_value,
                 target_weights=action,
                 stock_tickers=selected_model_tickers, # List corresponding to action array
                 current_prices=prices_for_order_calc
             )

        logging.info(f"Calculation complete. Found {len(planned_orders)} potential trades.")

        # --- Save Planned Orders ---
        orders_file = "planned_orders.json"
        logging.info(f"Saving {len(planned_orders)} planned orders to {orders_file}")
        try:
            # Convert numpy types if necessary for JSON serialization
            serializable_orders = []
            for order in planned_orders:
                 serializable_orders.append({
                      "ticker": order["ticker"],
                      "shares": int(order["shares"]) # Ensure shares are standard int
                 })
            with open(orders_file, 'w') as f:
                 json.dump(serializable_orders, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save planned orders: {e}", exc_info=True)
        # -------------------------

        logging.info("--- Phase 1 Completed ---")

    except Exception as e:
        logging.critical(f"!!! CRITICAL ERROR in Phase 1: {e}", exc_info=True)
        # Optionally try to save state even on error?
        # utils.save_portfolio_state({'cash': current_cash, 'positions': current_holdings})
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
    # Ensure keys are strings and values are basic types for JSON
    save_state = {
        'cash': float(current_cash),
        'positions': {str(k): int(v) for k, v in current_holdings.items()}
    }
    utils.save_portfolio_state(save_state)

    run_end_time = time.time()
    logging.info(f"====== Daily Rebalance Job (Manual) Completed in {(run_end_time - run_start_time):.2f} seconds ======")


if __name__ == "__main__":
    main()
