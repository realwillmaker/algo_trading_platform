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
from trading_env import StockTradingEnv # Use the env for simulation logic
# from portfolio_manager import calculate_target_orders # Not strictly needed for backtest run, but useful for debugging orders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================
# ================== BACKTESTING FUNCTIONS =====================
# ==============================================================

def run_backtest(model, features_dict, stock_tickers, start_date, end_date, initial_capital):
    """Runs a backtest using a trained model and historical data."""
    logging.info(f"--- Starting Backtest from {start_date} to {end_date} ---")

    # 1. Create a single backtesting environment instance
    # Ensure the environment uses the date range for backtesting
    # Note: The environment itself needs the full feature dictionary for the selected tickers
    backtest_env_features = {ticker: features_dict[ticker] for ticker in stock_tickers if ticker in features_dict}
    if len(backtest_env_features) != len(stock_tickers):
         logging.warning(f"Mismatch between requested tickers ({len(stock_tickers)}) and features found ({len(backtest_env_features)}). Using available features.")
         stock_tickers = list(backtest_env_features.keys()) # Update tickers to only those with features

    if not stock_tickers:
         logging.error("No tickers with features available for backtest environment.")
         return None, None, None, None

    try:
        backtest_env = StockTradingEnv(features_dict=backtest_env_features, stock_tickers=stock_tickers,
                                       initial_capital=initial_capital,
                                       lookback_window=config.LOOKBACK_WINDOW,
                                       commission=config.COMMISSION_PER_SHARE,
                                       slippage=config.SLIPPAGE_PERCENT)
    except ValueError as e:
         logging.error(f"Error creating StockTradingEnv for backtest: {e}. Check date ranges and lookback window.", exc_info=True)
         return None, None, None, None


    # Filter dates within the env to match the backtest period requested
    original_env_dates = backtest_env.dates # Keep original for indexing prices/features if needed
    backtest_env.dates = sorted([d for d in original_env_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date])

    if not backtest_env.dates or len(backtest_env.dates) <= backtest_env.lookback_window:
         logging.error(f"Not enough common data ({len(backtest_env.dates)} days) within the backtest range {start_date} - {end_date} after lookback ({config.LOOKBACK_WINDOW}).")
         return None, None, None, None

    backtest_env.start_step = backtest_env.lookback_window
    backtest_env.end_step = len(backtest_env.dates) - 1

    logging.info(f"Backtest will run for {backtest_env.end_step - backtest_env.start_step + 1} steps.")

    # 2. Run the simulation loop
    obs, info = backtest_env.reset()
    daily_portfolio_values = [initial_capital] # Start with initial capital
    # Find the date corresponding to the start of the simulation (just before first step)
    sim_start_date_idx = original_env_dates.index(backtest_env.dates[backtest_env.start_step -1])
    daily_dates = [original_env_dates[sim_start_date_idx]]

    all_trades = []
    all_holdings = []


    done = False
    truncated = False
    current_step_index = backtest_env.start_step # Track index within filtered backtest_env.dates
    while not done and not truncated:
        # Get action from the trained model (deterministic for evaluation)
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment using its internal logic which advances its current_step
        try:
             obs, reward, terminated, truncated, info = backtest_env.step(action)
             done = terminated # Use terminated flag from env
        except Exception as e:
             logging.error(f"Error during environment step {current_step_index} (Date: {backtest_env.dates[current_step_index]}): {e}", exc_info=True)
             # Decide how to handle: break, log, etc. Let's break for safety.
             break


        # Record results for this step (after portfolio value is updated for end of day T+1)
        # The date in 'info' corresponds to the end of the day trades were valued
        step_date = info.get('date')
        if step_date is None:
             logging.error("Missing 'date' in environment info dictionary. Cannot record history.")
             break # Stop if essential info is missing

        daily_dates.append(step_date)
        daily_portfolio_values.append(info.get('portfolio_value', np.nan)) # Record NaN if missing
        all_trades.extend(info.get('trades', [])) # Get trades executed in this step
        all_holdings.append({
             'date': step_date,
             **(info.get('stock_shares', {})), # Safely get holdings dict
             'cash': info.get('cash', np.nan),
             'portfolio_value': info.get('portfolio_value', np.nan)
             })

        current_step_index += 1 # Manually track index in filtered dates list

        # Check if loop should terminate based on env flags
        if done or truncated:
            logging.info(f"Loop terminated. Done={done}, Truncated={truncated}")
            break
        # Add safety break based on index exceeding bounds (shouldn't happen if done/truncated work)
        if current_step_index > backtest_env.end_step:
             logging.warning("Loop index exceeded end_step unexpectedly. Breaking.")
             break

    # Ensure environment is closed
    backtest_env.close()

    # 3. Process Results
    if not daily_portfolio_values or len(daily_portfolio_values) <= 1:
         logging.error("Backtest finished with no or insufficient portfolio history recorded.")
         return None, None, None, None

    try:
        portfolio_history = pd.Series(daily_portfolio_values, index=pd.to_datetime(daily_dates))
        # Handle potential duplicate dates if env somehow repeats a step/date
        portfolio_history = portfolio_history[~portfolio_history.index.duplicated(keep='last')]
        returns = portfolio_history.pct_change().dropna()
        # Ensure returns don't contain NaNs or Infs that break quantstats
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
    except Exception as e:
        logging.error(f"Error processing portfolio history into returns series: {e}", exc_info=True)
        return None, None, None, None


    holdings_df = pd.DataFrame(all_holdings)
    if not holdings_df.empty:
        holdings_df = holdings_df.set_index('date')

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
         trades_df['date'] = pd.to_datetime(trades_df['date'])


    logging.info(f"--- Backtest Completed ---")
    logging.info(f"Initial Value: ${initial_capital:,.2f}")
    final_val = portfolio_history.iloc[-1]
    if pd.notna(final_val):
        logging.info(f"Final Value:   ${final_val:,.2f}")
        logging.info(f"Total Return:  {((final_val / initial_capital) - 1) * 100:.2f}%")
    else:
        logging.warning("Final portfolio value is NaN.")


    return returns, portfolio_history, holdings_df, trades_df


def generate_tear_sheet(returns, benchmark_ticker=config.BENCHMARK_TICKER, output_dir=config.REPORTS_DIR):
    """Generates a QuantStats tear sheet."""
    if returns is None or returns.empty:
        logging.warning("No returns data to generate tear sheet.")
        return

    logging.info("Generating QuantStats tear sheet...")
    try:
        qs.extend_pandas() # Extend pandas with qs methods

        # Ensure returns index is timezone naive datetime index
        if isinstance(returns.index, pd.DatetimeIndex):
             if returns.index.tz is not None:
                 logging.debug("Converting returns index to timezone naive for QuantStats.")
                 returns.index = returns.index.tz_localize(None)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"tearsheet_{timestamp}.html")
        os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

        # Generate HTML report
        qs.reports.html(returns, benchmark=benchmark_ticker,
                        output=filename, title='RL Strategy Backtest',
                        download_benchmark=(benchmark_ticker is not None)) # Explicitly allow benchmark download

        logging.info(f"Tear sheet saved to {filename}")
        print(f"\nTear sheet generated: {filename}") # Also print to console

    except ImportError as ie:
         logging.error(f"ImportError generating tear sheet: {ie}. Did you install IPython? (`pip install ipython`)")
         print("\nError generating tear sheet: Missing IPython dependency? Run `pip install ipython`")
    except Exception as e:
         logging.error(f"Failed to generate QuantStats tear sheet: {e}", exc_info=True)
         print(f"\nError generating tear sheet: {e}")


# ==============================================================
# ================== MAIN EXECUTION BLOCK ======================
# ==============================================================

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
        exit(1)

    logging.info(f"Loading trained model: {config.MODEL_FILENAME} onto device: {device}")
    try:
        # Pass the device argument to the load method
        if config.RL_ALGORITHM == "PPO":
             model = PPO.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "SAC":
             model = SAC.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "A2C":
             model = A2C.load(config.MODEL_FILENAME, device=device)
        else:
             raise ValueError(f"Unsupported RL Algorithm for loading: {config.RL_ALGORITHM}")
        logging.info(f"Model loaded successfully onto device: {model.device}")
    except Exception as e:
        logging.error(f"Failed to load model {config.MODEL_FILENAME}: {e}", exc_info=True)
        exit(1)


    # 2. Load/Prepare Data for the Backtest Period
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]

    # Need data covering START_DATE to END_DATE_BACKTEST for features
    # Calculate required start date including lookback buffer
    logging.info("Loading/Generating Features for Backtesting Period...")
    try:
         # Calculate feature start date carefully to ensure enough history for lookback
         backtest_start_dt = pd.to_datetime(config.END_DATE_TRAIN) + pd.Timedelta(days=1)
         feature_calc_start_dt = backtest_start_dt - pd.Timedelta(days=config.LOOKBACK_WINDOW + 30) # Add buffer
         feature_calc_start_str = feature_calc_start_dt.strftime('%Y-%m-%d')
         backtest_end_str = config.END_DATE_BACKTEST # Use configured end date
         logging.info(f"Feature calculation range for backtest: {feature_calc_start_str} to {backtest_end_str}")

         features = feature_engineer.create_feature_dataset(stock_tickers, feature_calc_start_str, backtest_end_str)
    except Exception as e:
         logging.error(f"Error during feature generation for backtest: {e}", exc_info=True)
         exit(1)


    if not features:
         logging.error("Feature generation failed for backtest period. Cannot backtest.")
         exit(1)

    # Filter tickers based on successful feature generation FOR THE BACKTEST PERIOD
    valid_tickers_backtest = sorted(list(features.keys()))
    if not valid_tickers_backtest:
         logging.error("No valid tickers after feature generation for backtest period.")
         exit(1)
    logging.info(f"Backtesting with {len(valid_tickers_backtest)} tickers that have features in the backtest range.")
    # Note: This list might differ from the training tickers if some stocks lack recent data


    # 3. Run Backtest Simulation
    backtest_start_date_str = backtest_start_dt.strftime('%Y-%m-%d')

    # Call the run_backtest function (which should be defined above)
    returns, portfolio_history, holdings_df, trades_df = run_backtest(
        model=model,
        features_dict=features, # Pass features generated for backtest period
        stock_tickers=valid_tickers_backtest, # Use tickers valid for backtest
        start_date=backtest_start_date_str, # Start date after training
        end_date=backtest_end_str,      # End date for backtest
        initial_capital=config.INITIAL_CAPITAL
    )

    # 4. Generate Report
    if returns is not None:
        generate_tear_sheet(returns) # Call function defined above
        # Optional: Save holdings and trades to CSV
        output_dir = config.REPORTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        if holdings_df is not None and not holdings_df.empty:
             holdings_path = os.path.join(output_dir, "backtest_holdings.csv")
             holdings_df.to_csv(holdings_path)
             logging.info(f"Backtest holdings saved to {holdings_path}")
        if trades_df is not None and not trades_df.empty:
             trades_path = os.path.join(output_dir, "backtest_trades.csv")
             trades_df.to_csv(trades_path, index=False)
             logging.info(f"Backtest trades saved to {trades_path}")
    else:
        logging.error("Backtest did not produce valid returns. Tear sheet cannot be generated.")


    logging.info("--- Backtester Finished ---")
