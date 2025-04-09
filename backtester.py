import pandas as pd
import numpy as np
import quantstats as qs
import torch # Import torch
import logging
import os
from datetime import datetime # Import datetime directly for timestamp
from stable_baselines3 import PPO, SAC, A2C # Import your algo

import config
import utils
import feature_engineer
from trading_env import StockTradingEnv
# ... (rest of imports)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... (run_backtest function remains the same internal logic) ...
# ... (generate_tear_sheet function remains the same) ...


if __name__ == "__main__":
    logging.info("--- Running Backtester ---")

    # --- Determine device for loading model ---
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("CUDA available. Loading model onto GPU for backtesting.")
    else:
        device = 'cpu'
        logging.info("CUDA not available. Loading model onto CPU for backtesting.")
    # -----------------------------------------


    # 1. Load Trained Model - Specify the device!
    if not os.path.exists(config.MODEL_FILENAME):
        logging.error(f"Model file not found: {config.MODEL_FILENAME}. Train the model first.")
        exit()

    logging.info(f"Loading trained model: {config.MODEL_FILENAME} onto device: {device}")
    # Pass the device argument to the load method
    if config.RL_ALGORITHM == "PPO":
         model = PPO.load(config.MODEL_FILENAME, device=device)
    elif config.RL_ALGORITHM == "SAC":
         model = SAC.load(config.MODEL_FILENAME, device=device)
    elif config.RL_ALGORITHM == "A2C":
         model = A2C.load(config.MODEL_FILENAME, device=device)
    else:
         logging.error(f"Unsupported RL Algorithm for loading: {config.RL_ALGORITHM}")
         exit()
    logging.info(f"Model loaded successfully onto device: {model.device}")

    # 2. Load/Prepare Data for the Backtest Period (Same as before)
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
    logging.info("Loading/Generating Features for Backtesting Period...")
    feature_start = (pd.to_datetime(config.START_DATE) - pd.Timedelta(days=config.LOOKBACK_WINDOW + 5)).strftime('%Y-%m-%d')
    features = feature_engineer.create_feature_dataset(stock_tickers, feature_start, config.END_DATE_BACKTEST)
    if not features: exit()
    valid_tickers = list(features.keys())
    if not valid_tickers: exit()
    logging.info(f"Backtesting with {len(valid_tickers)} tickers.")


    # 3. Run Backtest Simulation (Same internal logic, but model runs on specified device)
    backtest_start_date = (pd.to_datetime(config.END_DATE_TRAIN) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    backtest_end_date = config.END_DATE_BACKTEST

    returns, portfolio_history, holdings_df, trades_df = run_backtest(
        model=model, # Pass the loaded model
        features_dict=features,
        stock_tickers=valid_tickers,
        start_date=backtest_start_date,
        end_date=backtest_end_date,
        initial_capital=config.INITIAL_CAPITAL
    )

    # 4. Generate Report (Same as before)
    if returns is not None:
        generate_tear_sheet(returns)
        if holdings_df is not None:
             holdings_df.to_csv(os.path.join(config.REPORTS_DIR, "backtest_holdings.csv"))
        if trades_df is not None and not trades_df.empty:
             trades_df.to_csv(os.path.join(config.REPORTS_DIR, "backtest_trades.csv"))

    logging.info("--- Backtester Finished ---")
