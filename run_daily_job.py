import pandas as pd
import numpy as np
import torch # Import torch
from stable_baselines3 import PPO, SAC, A2C # Import your algo
import logging
import os
import time
from datetime import datetime, timedelta

import config
import utils
import data_fetcher
import feature_engineer
import portfolio_manager
import schwab_executor # Uses the placeholder executor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
                    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()])

# ... (get_latest_features function remains the same) ...
# ... (construct_observation_state function remains the same) ...

def main():
    logging.info("====== Starting Daily Rebalance Job ======")
    run_start_time = time.time()

    # --- Determine device for loading model ---
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA available. Loading model onto GPU for prediction.")
    else:
        device = 'cpu'
        logging.info("CUDA not available. Loading model onto CPU for prediction.")
    # -----------------------------------------

    # === Phase 1: Post-Market Close Actions (Decision Making) ===
    logging.info("--- Phase 1: Post-Market Analysis & Decision ---")
    try:
        # 1. Update S&P 500 List (Same)
        logging.info("Updating S&P 500 ticker list...")
        sp500_tickers = utils.get_sp500_tickers()
        stock_tickers = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
        logging.info(f"Using {len(stock_tickers)} S&P 500 stock tickers.")

        # 2. Fetch Latest Market Data (Same)
        logging.info("Fetching latest market data (yesterday's close)...")
        fetch_end = datetime.now().strftime('%Y-%m-%d')
        fetch_start = (datetime.now() - timedelta(days=config.LOOKBACK_WINDOW + 30)).strftime('%Y-%m-%d')
        data_fetcher.fetch_and_save_all_data(sp500_tickers, fetch_start, fetch_end)

        # 3. Feature Engineering for Latest Data (Same)
        logging.info("Calculating latest features...")
        latest_features, valid_tickers, latest_feature_date = get_latest_features(stock_tickers)
        logging.info(f"Features generated for {len(valid_tickers)} tickers up to {latest_feature_date.strftime('%Y-%m-%d')}.")


        # 4. Load RL Model - Specify Device!
        logging.info(f"Loading RL model: {config.MODEL_FILENAME} onto device: {device}")
        if not os.path.exists(config.MODEL_FILENAME):
             raise FileNotFoundError(f"Model file not found: {config.MODEL_FILENAME}")

        # Load the appropriate model class based on config, specifying device
        if config.RL_ALGORITHM == "PPO": model = PPO.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "SAC": model = SAC.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "A2C": model = A2C.load(config.MODEL_FILENAME, device=device)
        else: raise ValueError(f"Unsupported RL Algorithm for loading: {config.RL_ALGORITHM}")
        logging.info(f"Model loaded successfully onto device: {model.device}")


        # 5. Get Current Portfolio State (Same)
        logging.info("Fetching current portfolio state from broker...")
        schwab_client = None # Force using fake data from get_account_info
        account_info = schwab_executor.get_account_info(schwab_client)
        if account_info is None: raise RuntimeError("Failed to get account information.")
        current_cash = account_info['cash']
        current_holdings = account_info['positions']
        current_total_value = account_info['value']
        # Calculate current weights (Same logic)
        current_weights_dict = {}
        stock_value_total = 0
        logging.info(f"Fetching latest close prices for weight calculation ({latest_feature_date.strftime('%Y-%m-%d')})...")
        current_prices_dict = {}
        for ticker in valid_tickers:
             price_data = utils.load_data_from_file(ticker)
             if price_data is not None and latest_feature_date in price_data.index:
                  current_prices_dict[ticker] = price_data.loc[latest_feature_date]['Close']
        for ticker, shares in current_holdings.items():
            price = current_prices_dict.get(ticker)
            if price is not None and price > 0: stock_value_total += shares * price
        if stock_value_total > 0 : current_total_value = current_cash + stock_value_total
        if current_total_value > 0:
             for ticker in valid_tickers:
                  shares = current_holdings.get(ticker, 0); price = current_prices_dict.get(ticker)
                  current_weights_dict[ticker] = (shares * price) / current_total_value if price is not None and price > 0 else 0.0
        else: current_weights_dict = {ticker: 0.0 for ticker in valid_tickers}


        # 6. Construct Observation State (Same)
        logging.info("Constructing observation state for RL model...")
        observation = construct_observation_state(latest_features, current_weights_dict, valid_tickers)


        # 7. Predict Action (Model predict will run on the specified device)
        logging.info("Predicting action with RL model...")
        action, _states = model.predict(observation, deterministic=True)
        target_weights = action


        # 8. Calculate Target Orders (Same)
        logging.info("Calculating target orders...")
        prices_for_order_calc = {t: current_prices_dict[t] for t in valid_tickers if t in current_prices_dict}
        planned_orders = portfolio_manager.calculate_target_orders(
            current_holdings=current_holdings, current_cash=current_cash,
            total_portfolio_value=current_total_value, target_weights=target_weights,
            stock_tickers=valid_tickers, current_prices=prices_for_order_calc )

        # 9. STORE PLANNED ORDERS SECURELY (Same)
        orders_file = "planned_orders.json"
        logging.info(f"Saving {len(planned_orders)} planned orders to {orders_file}")
        pd.DataFrame(planned_orders).to_json(orders_file, indent=4)

        logging.info("--- Phase 1 Completed ---")

    except Exception as e:
        logging.critical(f"!!! CRITICAL ERROR in Phase 1 (Decision Making): {e}", exc_info=True)
        exit(1)


    # === Phase 2: Pre-Market Open Actions (Execution) === (Same logic, no model interaction here)
    logging.info("--- Phase 2: Pre-Market Execution ---")
    try:
        orders_file = "planned_orders.json"
        logging.info(f"Loading planned orders from {orders_file}")
        if not os.path.exists(orders_file):
             logging.warning("Planned orders file not found. Nothing to execute.")
             exit(0)
        orders_to_execute = pd.read_json(orders_file).to_dict('records')
        os.remove(orders_file)

        if not orders_to_execute:
             logging.info("No orders planned. Execution phase complete.")
        else:
             logging.info(f"Executing {len(orders_to_execute)} orders...")
             schwab_client = None # Force simulation
             sells = [o for o in orders_to_execute if o['shares'] < 0]
             buys = [o for o in orders_to_execute if o['shares'] > 0]
             executed_sells = schwab_executor.execute_orders(schwab_client, sells)
             logging.info(f"Executed {len(executed_sells)} SELL orders (or simulated).")
             executed_buys = schwab_executor.execute_orders(schwab_client, buys)
             logging.info(f"Executed {len(executed_buys)} BUY orders (or simulated).")

        logging.info("--- Phase 2 Completed ---")

    except Exception as e:
        logging.critical(f"!!! CRITICAL ERROR in Phase 2 (Execution): {e}", exc_info=True)
        exit(1)


    # === Phase 3: Post-Execution Reporting & Final Output === (Same logic, no model interaction)
    logging.info("--- Phase 3: Reporting & Final State ---")
    # ... (rest of reporting logic is unchanged) ...
    try:
         logging.info("Fetching final portfolio state post-execution...")
         schwab_client = None
         final_account_info = schwab_executor.get_account_info(schwab_client)

         if final_account_info is None:
              logging.error("Could not fetch final account state.")
              final_total_value, final_cash, final_holdings_list = "N/A", "N/A", []
         else:
              final_total_value = final_account_info['value']
              final_cash = final_account_info['cash']
              final_holdings = final_account_info['positions']
              logging.info("Fetching latest close prices for final NLV...")
              final_prices = {}
              for ticker in final_holdings.keys():
                   price_data = utils.load_data_from_file(ticker)
                   if price_data is not None and latest_feature_date in price_data.index:
                       final_prices[ticker] = price_data.loc[latest_feature_date]['Close']
              final_holdings_list = []
              calculated_stock_value = 0
              for ticker, shares in final_holdings.items():
                  price = final_prices.get(ticker)
                  nlv = (shares * price) if price is not None else 0
                  final_holdings_list.append({'ticker': ticker, 'shares': shares, 'nlv': max(nlv, 0.0)})
                  if nlv > 0: calculated_stock_value += nlv
              if isinstance(final_cash, (int, float)):
                  final_total_value = final_cash + calculated_stock_value
                  logging.info(f"Final calculated portfolio value: ${final_total_value:,.2f}")

         # Generate Output (Same)
         print("\n====== FINAL PORTFOLIO STATE ======")
         # ... print state ...
         print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
         print(f"Total Portfolio Value: ${final_total_value:,.2f}" if isinstance(final_total_value, (int, float)) else f"Total Portfolio Value: {final_total_value}")
         print(f"Cash on Hand: ${final_cash:,.2f}" if isinstance(final_cash, (int, float)) else f"Cash on Hand: {final_cash}")
         print("\nHoldings:")
         if final_holdings_list:
              final_holdings_list.sort(key=lambda x: x['nlv'], reverse=True)
              for item in final_holdings_list:
                  if item['shares'] > 0: print(f"  - {item['ticker']}: {item['shares']} shares, NLV: ${item['nlv']:,.2f}")
         else: print("  No holdings.")
         print("=================================\n")

    except Exception as e:
         logging.error(f"Error in Phase 3 (Reporting): {e}", exc_info=True)


    run_end_time = time.time()
    logging.info(f"====== Daily Rebalance Job Completed in {(run_end_time - run_start_time):.2f} seconds ======")


if __name__ == "__main__":
    main()
