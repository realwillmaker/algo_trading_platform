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
import utils # Imports load_portfolio_state, save_portfolio_state
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
file_handler.setLevel(logging.INFO)

# Stream Handler (Console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# --------------------


def get_latest_features(tickers):
    """Generates features using data up to the latest available day."""
    # (Function remains the same as before - calculates features)
    logging.info("Generating latest features...")
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=config.LOOKBACK_WINDOW + 35)).strftime('%Y-%m-%d') # Adjusted buffer slightly
    logging.info(f"Feature calculation range for job: {start_date} to {end_date}")

    features = feature_engineer.create_feature_dataset(tickers, start_date, end_date)
    if not features:
         raise RuntimeError("Failed to generate features for latest data.")

    latest_features_dict = {}
    latest_common_date = None
    common_dates = None
    valid_tickers = sorted(list(features.keys())) # Get tickers where features were generated

    for ticker in valid_tickers:
        dates = features[ticker].index
        if common_dates is None: common_dates = dates
        else: common_dates = common_dates.intersection(dates)

    if common_dates is None or common_dates.empty:
        raise RuntimeError("No common dates found in the latest feature set across tickers.")

    latest_common_date = max(common_dates)
    logging.info(f"Latest common date with features: {latest_common_date.strftime('%Y-%m-%d')}")

    if latest_common_date.date() < (datetime.now() - timedelta(days=4)).date(): # Increased tolerance slightly
         logging.warning(f"Latest feature date {latest_common_date.strftime('%Y-%m-%d')} seems old. Data might be stale.")

    lookback_start_date = latest_common_date - pd.Timedelta(days=config.LOOKBACK_WINDOW - 1)
    processed_features_for_state = {}
    final_valid_tickers = [] # Tickers with enough lookback data

    for ticker in valid_tickers:
         # Select lookback window ending on the latest common date
         ticker_features_full = features[ticker]
         # Ensure index is datetime
         if not isinstance(ticker_features_full.index, pd.DatetimeIndex):
              ticker_features_full.index = pd.to_datetime(ticker_features_full.index)
         # Slice the window
         ticker_features_window = ticker_features_full.loc[lookback_start_date:latest_common_date]

         if len(ticker_features_window) == config.LOOKBACK_WINDOW:
              # Select only feature columns (all except Open, Close)
              feature_cols = [col for col in ticker_features_window.columns if col not in ['Open', 'Close']]
              processed_features_for_state[ticker] = ticker_features_window[feature_cols]
              final_valid_tickers.append(ticker)
         else:
              logging.warning(f"Ticker {ticker} does not have enough data ({len(ticker_features_window)} rows) for the lookback window ending {latest_common_date}. Excluding from prediction.")

    if not final_valid_tickers:
         raise RuntimeError("No tickers had sufficient data for the lookback window.")

    logging.info(f"Features ready for state construction for {len(final_valid_tickers)} tickers.")
    return processed_features_for_state, final_valid_tickers, latest_common_date


def construct_observation_state(feature_window_dict, current_weights_dict, ordered_tickers):
     """Constructs the flattened observation state for the RL model."""
     # (Function remains the same as before - constructs state)
     if not ordered_tickers:
          raise ValueError("Ordered tickers list cannot be empty for state construction.")
     first_ticker = ordered_tickers[0]
     if first_ticker not in feature_window_dict or feature_window_dict[first_ticker].empty:
          raise ValueError(f"Feature data missing or empty for first ticker {first_ticker}.")

     num_features_per_stock = feature_window_dict[first_ticker].shape[1]
     lookback_window = feature_window_dict[first_ticker].shape[0]
     num_stocks = len(ordered_tickers)

     # --- Robust Feature Stacking ---
     historical_features_flat = np.zeros(lookback_window * num_features_per_stock * num_stocks, dtype=np.float32)
     idx = 0
     for i in range(lookback_window):
         for ticker in ordered_tickers:
             try:
                 # Assume keys match dates in window slice
                 feature_vector = feature_window_dict[ticker].iloc[i].values
                 expected_len = num_features_per_stock
                 actual_len = len(feature_vector)
                 if actual_len == expected_len:
                      historical_features_flat[idx : idx + expected_len] = feature_vector
                 else:
                     logging.warning(f"Shape mismatch for {ticker} day {i}. Expected {expected_len}, got {actual_len}. Padding.")
                     # Pad or truncate if necessary (shouldn't happen with preprocessing fixes)
                     padded_vector = np.zeros(expected_len)
                     len_to_copy = min(actual_len, expected_len)
                     padded_vector[:len_to_copy] = feature_vector[:len_to_copy]
                     historical_features_flat[idx : idx + expected_len] = padded_vector

             except (IndexError, KeyError) as e:
                 logging.warning(f"Error getting features for {ticker} day {i}: {e}. Using zeros.")
                 # Zeros are already there due to initialization
                 pass # Keep zeros
             idx += num_features_per_stock
     # --- End Feature Stacking ---


     # Get current weights in the correct order, default to 0.0 if ticker missing
     current_weights = np.array([current_weights_dict.get(ticker, 0.0) for ticker in ordered_tickers], dtype=np.float32)

     # Ensure weights array has correct length
     if len(current_weights) != num_stocks:
          logging.error(f"Weight dimension mismatch! Expected {num_stocks}, got {len(current_weights)}. Check weight dict.")
          # Pad weights if needed, although this indicates a deeper issue
          padded_weights = np.zeros(num_stocks, dtype=np.float32)
          len_to_copy = min(len(current_weights), num_stocks)
          padded_weights[:len_to_copy] = current_weights[:len_to_copy]
          current_weights = padded_weights


     observation = np.concatenate([historical_features_flat, current_weights])
     # Final shape check
     expected_obs_dim = lookback_window * num_features_per_stock * num_stocks + num_stocks
     if observation.shape[0] != expected_obs_dim:
          logging.critical(f"CRITICAL: Final observation shape {observation.shape[0]} != expected {expected_obs_dim}!")
          # Decide how to handle: raise error? return zeros?
          raise RuntimeError("Observation shape mismatch during final construction.")

     return observation.astype(np.float32)


def main():
    logging.info("====== Starting Daily Rebalance Job (Manual Execution Focus) ======")
    run_start_time = time.time()
    planned_orders = [] # Initialize empty list

    try:
        # === Phase 1: Data, State Loading, Prediction, Order Calculation ===
        logging.info("--- Phase 1: Load State, Get Data, Predict, Calculate Trades ---")

        # 1. Load Current Portfolio State
        # This replaces the call to schwab_executor.get_account_info
        portfolio_state = utils.load_portfolio_state()
        current_cash = portfolio_state['cash']
        current_holdings = portfolio_state['positions'] # {ticker: shares}
        logging.info(f"Loaded state - Cash: ${current_cash:,.2f}, Holdings: {len(current_holdings)} stocks")

        # 2. Update S&P 500 List & Fetch Latest Market Data
        logging.info("Fetching S&P 500 list and latest market data...")
        sp500_tickers = utils.get_sp500_tickers()
        stock_tickers_to_fetch = list(set(sp500_tickers) | set(current_holdings.keys())) # Fetch for S&P + current holdings
        fetch_end = datetime.now().strftime('%Y-%m-%d')
        fetch_start = (datetime.now() - timedelta(days=config.LOOKBACK_WINDOW + 35)).strftime('%Y-%m-%d')
        data_fetcher.fetch_and_save_all_data(stock_tickers_to_fetch, fetch_start, fetch_end)

        # 3. Feature Engineering for Latest Data
        logging.info("Calculating latest features...")
        stock_tickers_for_features = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
        latest_features, valid_tickers, latest_feature_date = get_latest_features(stock_tickers_for_features)
        logging.info(f"Features generated for {len(valid_tickers)} tickers up to {latest_feature_date.strftime('%Y-%m-%d')}.")


        # 4. Load RL Model
        # Determine which tickers the model expects based on training run
        # *** IMPORTANT: This assumes the same MAX_TICKERS and selection logic as training ***
        MAX_TICKERS_MODEL_WAS_TRAINED_ON = 100 # Needs to match training!
        model_tickers = []
        if len(valid_tickers) <= MAX_TICKERS_MODEL_WAS_TRAINED_ON:
             model_tickers = sorted(valid_tickers) # Use all if fewer than limit
        else:
             # Re-run selection logic to find the exact tickers model uses
             logging.info(f"Selecting top {MAX_TICKERS_MODEL_WAS_TRAINED_ON} from {len(valid_tickers)} based on market cap for model input...")
             market_caps = utils.get_market_caps(valid_tickers) # Use function from utils if moved there
             ticker_cap_list = [(t, market_caps.get(t, 0)) for t in valid_tickers]
             ticker_cap_list.sort(key=lambda item: item[1], reverse=True)
             model_tickers = sorted([item[0] for item in ticker_cap_list[:MAX_TICKERS_MODEL_WAS_TRAINED_ON]])

        if not model_tickers:
             raise RuntimeError("Could not determine the tickers the model requires.")
        logging.info(f"Model expects input for {len(model_tickers)} tickers.")

        # Filter features dict for the model
        model_features = {t: latest_features[t] for t in model_tickers if t in latest_features}
        if len(model_features) != len(model_tickers):
             missing = [t for t in model_tickers if t not in model_features]
             raise RuntimeError(f"Features missing for required model tickers: {missing}")


        # Load the model
        logging.info(f"Loading RL model: {config.MODEL_FILENAME}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(config.MODEL_FILENAME): raise FileNotFoundError(f"Model file not found: {config.MODEL_FILENAME}")
        if config.RL_ALGORITHM == "PPO": model = PPO.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "SAC": model = SAC.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "A2C": model = A2C.load(config.MODEL_FILENAME, device=device)
        else: raise ValueError(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}")


        # 5. Calculate Current Portfolio Value & Weights for Observation
        logging.info("Calculating current portfolio value and weights...")
        current_total_value = current_cash
        current_prices_dict = {} # Prices at latest_feature_date close
        tickers_for_value_calc = list(set(model_tickers) | set(current_holdings.keys())) # Need prices for model input & current holdings

        for ticker in tickers_for_value_calc:
             price_data = utils.load_data_from_file(ticker)
             if price_data is not None and latest_feature_date in price_data.index:
                  current_prices_dict[ticker] = price_data.loc[latest_feature_date]['Close']

        logging.debug(f"Prices fetched for value calc ({len(current_prices_dict)} tickers)")

        current_weights_dict = {} # Weights of the stocks the *model* expects
        stock_value_total = 0
        for ticker, shares in current_holdings.items():
            price = current_prices_dict.get(ticker)
            if price is not None and price > 0:
                 stock_value_total += shares * price
            elif ticker in model_tickers: # Only warn if it's a ticker the model cares about
                 logging.warning(f"Could not get price for held ticker {ticker} needed for value calculation.")

        current_total_value += stock_value_total
        logging.info(f"Calculated current total value: ${current_total_value:,.2f}")

        if current_total_value > 1e-6:
             for ticker in model_tickers: # Calculate weights ONLY for tickers model uses
                  shares = current_holdings.get(ticker, 0)
                  price = current_prices_dict.get(ticker)
                  if price is not None and price > 0:
                       current_weights_dict[ticker] = (shares * price) / current_total_value
                  else:
                       current_weights_dict[ticker] = 0.0 # Assign 0 weight if price unavailable
        else:
             current_weights_dict = {ticker: 0.0 for ticker in model_tickers}

        logging.debug(f"Current weights for observation state ({len(current_weights_dict)} tickers)")


        # 6. Construct Observation State
        logging.info("Constructing observation state for RL model...")
        # Pass features and weights only for the tickers the model was trained on
        observation = construct_observation_state(model_features, current_weights_dict, model_tickers)


        # 7. Predict Action (Target Weights for NEXT DAY)
        logging.info("Predicting action with RL model...")
        action, _states = model.predict(observation, deterministic=True)
        # Action is an array of target weights corresponding to model_tickers order
        target_weights_dict = dict(zip(model_tickers, action))


        # 8. Calculate Target Orders (Considering ALL current holdings and target weights)
        logging.info("Calculating target orders based on prediction...")
        # Use prices from latest_feature_date for the calculation
        prices_for_order_calc = {t: current_prices_dict[t] for t in tickers_for_value_calc if t in current_prices_dict}

        # Use the portfolio manager to calculate trades needed
        # It needs the full current state (all holdings) and compares against target weights
        # for the stocks the model provides targets for. It implicitly handles stocks
        # currently held but not in the model's target (they should ideally be sold).
        planned_orders = portfolio_manager.calculate_target_orders(
            current_holdings=current_holdings, # All current holdings
            current_cash=current_cash,
            total_portfolio_value=current_total_value,
            # Target weights dict might only contain model_tickers, need to handle this in portfolio_manager?
            # Let's modify calculate_target_orders slightly if needed, or assume it handles partial target dicts
            # For now, assume calculate_target_orders takes the NP array and the corresponding tickers list
            target_weights=action, # Pass the raw action array
            stock_tickers=model_tickers, # Pass the list corresponding to the action array
            current_prices=prices_for_order_calc
        )

        logging.info(f"Calculation complete. Found {len(planned_orders)} potential trades.")
        logging.info("--- Phase 1 Completed ---")

    except Exception as e:
        logging.critical(f"!!! CRITICAL ERROR in Phase 1 (Decision Making): {e}", exc_info=True)
        # Exit prevents moving to execution phase
        exit(1)


    # === Phase 2: REMOVED (No Execution) ===


    # === Phase 3: Reporting Trades for Manual Execution ===
    logging.info("--- Phase 3: Reporting Planned Trades ---")
    print("\n" + "="*30)
    print(" PLANNED REBALANCE TRADES")
    print(f" Calculation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Based on Data Up To: {latest_feature_date.strftime('%Y-%m-%d')}")
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
    # This allows the next day's run to know the starting point *before* these trades
    utils.save_portfolio_state({'cash': current_cash, 'positions': current_holdings})

    run_end_time = time.time()
    logging.info(f"====== Daily Rebalance Job (Manual) Completed in {(run_end_time - run_start_time):.2f} seconds ======")


if __name__ == "__main__":
    main()
