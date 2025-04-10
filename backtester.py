import pandas as pd
import numpy as np
import quantstats as qs
import torch # Import torch
import yfinance as yf # Need yfinance for market cap fetch
import logging
import os
import time # Need time for delay
from datetime import datetime # Import datetime directly for timestamp
from stable_baselines3 import PPO, SAC, A2C # Import your algo

import config
import utils
import feature_engineer
from trading_env import StockTradingEnv # Use the env for simulation logic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Copy Market Cap Function from train_rl.py ---
def get_market_caps(tickers, delay=config.MARKET_CAP_FETCH_DELAY):
    """
    Fetches current market capitalization for a list of tickers using yfinance.

    Args:
        tickers (list): List of stock ticker symbols.
        delay (float): Seconds to wait between API calls to avoid rate limits.

    Returns:
        dict: Dictionary mapping ticker symbols to their market cap (or 0 if unavailable/error).
    """
    logging.info(f"Fetching market caps for {len(tickers)} tickers (for backtest selection)...")
    market_caps = {}
    count = 0
    total = len(tickers)
    for ticker in tickers:
        count += 1
        logging.debug(f"Fetching market cap for {ticker} ({count}/{total})")
        try:
            # Check if ticker symbol seems valid before fetching
            if not isinstance(ticker, str) or not ticker or ticker.startswith("^"): # Basic check
                 logging.warning(f"Skipping potentially invalid ticker for market cap: {ticker}")
                 market_caps[ticker] = 0
                 continue

            stock_info = yf.Ticker(ticker).info
            cap = stock_info.get('marketCap', 0) # Get marketCap, default to 0 if missing
            if cap is None: # Sometimes 'marketCap' key exists but value is None
                 cap = 0
            market_caps[ticker] = int(cap) # Store as int
            time.sleep(delay) # Wait to avoid rate limiting
        except Exception as e:
            # More specific error logging could be useful here
            logging.warning(f"Could not fetch info/market cap for {ticker}: {e}")
            market_caps[ticker] = 0 # Assign 0 on error
    logging.info(f"Finished fetching market caps. Found caps for {sum(1 for cap in market_caps.values() if cap > 0)} tickers.")
    return market_caps
# -----------------------------------------------


# ==============================================================
# ================== BACKTESTING FUNCTIONS =====================
# ==============================================================

def run_backtest(model, features_dict, stock_tickers, start_date, end_date, initial_capital):
    """Runs a backtest using a trained model and historical data."""
    logging.info(f"--- Starting Backtest from {start_date} to {end_date} ---")

    # 1. Create environment using ONLY the provided stock_tickers
    # Ensure features_dict only contains data for stock_tickers
    backtest_env_features = {ticker: features_dict[ticker] for ticker in stock_tickers if ticker in features_dict}
    if len(backtest_env_features) != len(stock_tickers):
         missing_tickers = [t for t in stock_tickers if t not in backtest_env_features]
         logging.warning(f"Mismatch between requested tickers ({len(stock_tickers)}) and features found ({len(backtest_env_features)}). Missing features for: {missing_tickers}. Using available features.")
         stock_tickers = list(backtest_env_features.keys()) # Update tickers to only those with features

    if not stock_tickers:
         logging.error("No tickers with features available for backtest environment.")
         return None, None, None, None

    logging.info(f"Initializing backtest environment with {len(stock_tickers)} tickers.")
    try:
        # Use the filtered features and the SPECIFIC list of stock_tickers
        backtest_env = StockTradingEnv(features_dict=backtest_env_features, stock_tickers=stock_tickers,
                                       initial_capital=initial_capital,
                                       lookback_window=config.LOOKBACK_WINDOW,
                                       commission=config.COMMISSION_PER_SHARE,
                                       slippage=config.SLIPPAGE_PERCENT)
    except ValueError as e:
         logging.error(f"Error creating StockTradingEnv for backtest: {e}. Check date ranges and lookback window.", exc_info=True)
         return None, None, None, None
    except Exception as e:
         logging.error(f"Unexpected error creating StockTradingEnv for backtest: {e}", exc_info=True)
         return None, None, None, None


    # Filter dates within the env to match the backtest period requested
    original_env_dates = backtest_env.dates # Keep original for indexing prices/features if needed
    backtest_env.dates = sorted([d for d in original_env_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date])

    if not backtest_env.dates or len(backtest_env.dates) <= backtest_env.lookback_window:
         logging.error(f"Not enough common data ({len(backtest_env.dates)} days) within the backtest range {start_date} - {end_date} after lookback ({config.LOOKBACK_WINDOW}).")
         return None, None, None, None

    # Adjust start/end steps based on the *filtered* dates list
    backtest_env.start_step = backtest_env.lookback_window
    backtest_env.end_step = len(backtest_env.dates) - 1 # Last valid index for valuation

    # Number of simulation steps = number of decisions made
    num_sim_steps = backtest_env.end_step - backtest_env.start_step + 1
    if num_sim_steps <= 0:
        logging.error(f"Calculated number of simulation steps is zero or negative ({num_sim_steps}). Check date range and lookback.")
        return None, None, None, None

    logging.info(f"Backtest will run for {num_sim_steps} simulation steps.")

    # 2. Run the simulation loop
    try:
        obs, info = backtest_env.reset()
    except Exception as e:
         logging.error(f"Error during environment reset: {e}", exc_info=True)
         return None, None, None, None

    daily_portfolio_values = [initial_capital]
    # Ensure original_env_dates is a list or similar indexable structure
    original_env_dates_list = list(original_env_dates)
    try:
        # Find the date corresponding to the state *before* the first step
        sim_start_date = backtest_env.dates[backtest_env.start_step - 1]
        sim_start_date_idx_in_original = original_env_dates_list.index(sim_start_date)
        daily_dates = [original_env_dates_list[sim_start_date_idx_in_original]]
    except (ValueError, IndexError) as e:
         logging.error(f"Error finding backtest start date {backtest_env.dates[backtest_env.start_step - 1]} in original dates: {e}.")
         return None, None, None, None

    all_trades = []
    all_holdings = []
    done = False
    truncated = False
    # Use range for clearer loop control over simulation steps
    for step_num in range(num_sim_steps):
        current_env_step_index = backtest_env.current_step # Step index used internally by env

        # Get action from the trained model
        try:
             # Ensure obs matches expected shape before prediction
             expected_shape = model.observation_space.shape
             if not isinstance(obs, np.ndarray):
                  logging.error(f"Observation is not a numpy array! Type: {type(obs)}. Step: {step_num}")
                  break
             if obs.shape != expected_shape:
                  logging.error(f"Observation shape mismatch before predict! Expected {expected_shape}, got {obs.shape}. Step: {step_num}")
                  break
             action, _states = model.predict(obs, deterministic=True)
        except ValueError as ve:
             logging.error(f"ValueError during model.predict (likely shape mismatch): {ve}. Obs shape: {obs.shape}. Step: {step_num}", exc_info=True)
             break
        except Exception as e:
             logging.error(f"Error during model.predict: {e}. Step: {step_num}", exc_info=True)
             break

        # Step the environment
        try:
             # Environment's step advances its internal current_step
             obs, reward, terminated, truncated, info = backtest_env.step(action)
             done = terminated
        except Exception as e:
             step_date_info = backtest_env.dates[current_env_step_index] if current_env_step_index < len(backtest_env.dates) else "N/A"
             logging.error(f"Error during environment step {current_env_step_index} (Sim Step {step_num}, Date: {step_date_info}): {e}", exc_info=True)
             break

        # Record results for this step
        step_date = info.get('date')
        if step_date is None:
             logging.error(f"Missing 'date' in environment info dictionary at sim step {step_num}. Cannot record history.")
             break
        daily_dates.append(step_date)
        daily_portfolio_values.append(info.get('portfolio_value', np.nan))
        all_trades.extend(info.get('trades', []))
        all_holdings.append({'date': step_date,**(info.get('stock_shares', {})),'cash': info.get('cash', np.nan),'portfolio_value': info.get('portfolio_value', np.nan)})

        # Check for termination flags from the environment
        if done or truncated:
            logging.info(f"Loop terminated by environment flag at sim step {step_num}. Done={done}, Truncated={truncated}")
            break

    # Loop finished or broke
    logging.info(f"Simulation loop finished after {step_num+1} steps.")
    backtest_env.close() # Ensure env is closed

    # 3. Process Results
    if not daily_portfolio_values or len(daily_portfolio_values) <= 1:
         logging.error("Backtest finished with no or insufficient portfolio history recorded.")
         return None, None, None, None
    try:
        portfolio_history = pd.Series(daily_portfolio_values, index=pd.to_datetime(daily_dates))
        portfolio_history = portfolio_history[~portfolio_history.index.duplicated(keep='last')] # Ensure unique index
        returns = portfolio_history.pct_change().dropna()
        # Ensure returns are clean for quantstats
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
    except Exception as e:
        logging.error(f"Error processing portfolio history into returns series: {e}", exc_info=True)
        return None, None, None, None


    holdings_df = pd.DataFrame(all_holdings)
    if not holdings_df.empty:
        try:
             holdings_df['date'] = pd.to_datetime(holdings_df['date'])
             holdings_df = holdings_df.set_index('date')
        except Exception as e:
             logging.error(f"Error processing holdings_df index: {e}")


    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
         try:
             trades_df['date'] = pd.to_datetime(trades_df['date'])
         except Exception as e:
              logging.error(f"Error processing trades_df date: {e}")


    logging.info(f"--- Backtest Results Processing Completed ---")
    final_val = portfolio_history.iloc[-1]
    if pd.notna(final_val):
        logging.info(f"Initial Value: ${initial_capital:,.2f}")
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device} for backtesting.")

    # 1. Load Trained Model
    if not os.path.exists(config.MODEL_FILENAME):
        logging.error(f"Model file not found: {config.MODEL_FILENAME}. Train the model first.")
        exit(1)

    logging.info(f"Loading trained model: {config.MODEL_FILENAME} onto device: {device}")
    try:
        if config.RL_ALGORITHM == "PPO": model = PPO.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "SAC": model = SAC.load(config.MODEL_FILENAME, device=device)
        elif config.RL_ALGORITHM == "A2C": model = A2C.load(config.MODEL_FILENAME, device=device)
        else: raise ValueError(f"Unsupported RL Algorithm for loading: {config.RL_ALGORITHM}")
        logging.info(f"Model loaded successfully onto device: {model.device}")
    except Exception as e:
        logging.error(f"Failed to load model {config.MODEL_FILENAME}: {e}", exc_info=True)
        exit(1)


    # 2. Load/Prepare Data for the Backtest Period
    sp500_tickers_all = utils.get_sp500_tickers()
    stock_tickers_all = [t for t in sp500_tickers_all if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
    logging.info("Loading/Generating Features for Backtesting Period...")
    try:
         backtest_start_dt = pd.to_datetime(config.END_DATE_TRAIN) + pd.Timedelta(days=1)
         feature_calc_start_dt = backtest_start_dt - pd.Timedelta(days=config.LOOKBACK_WINDOW + 30) # Include buffer for features
         feature_calc_start_str = feature_calc_start_dt.strftime('%Y-%m-%d')
         backtest_end_str = config.END_DATE_BACKTEST
         logging.info(f"Feature calculation range for backtest: {feature_calc_start_str} to {backtest_end_str}")
         features_all = feature_engineer.create_feature_dataset(stock_tickers_all, feature_calc_start_str, backtest_end_str)
    except Exception as e:
         logging.error(f"Error during feature generation for backtest: {e}", exc_info=True)
         exit(1)


    if not features_all:
         logging.error("Feature generation failed or produced no results for backtest period.")
         exit(1)

    # Get tickers with features available *in the backtest range*
    valid_tickers_backtest = sorted(list(features_all.keys()))
    if not valid_tickers_backtest:
         logging.error("No valid tickers found after feature generation for backtest period.")
         exit(1)
    logging.info(f"{len(valid_tickers_backtest)} tickers have features in the backtest range.")


    # --- Determine the ACTUAL tickers the loaded model was trained on ---
    # Make sure this matches the value used during the actual training run
    MAX_TICKERS_MODEL_WAS_TRAINED_ON = 100 # *** CHECK THIS VALUE matches train_rl.py ***

    training_tickers = [] # This list will hold the tickers used by the model
    if len(valid_tickers_backtest) == 0:
         logging.error("No tickers available in backtest period after feature generation.")
         exit(1)
    elif len(valid_tickers_backtest) <= MAX_TICKERS_MODEL_WAS_TRAINED_ON:
         logging.warning(f"Fewer tickers available in backtest ({len(valid_tickers_backtest)}) than model was trained on ({MAX_TICKERS_MODEL_WAS_TRAINED_ON}). Using all available tickers for backtest, but results might be skewed if model expected more.")
         # Use the tickers available in backtest, model might underperform if trained on more/different ones
         training_tickers = valid_tickers_backtest
    else:
        # Re-select top N based on current market cap to mimic training selection
        logging.info(f"Selecting top {MAX_TICKERS_MODEL_WAS_TRAINED_ON} tickers from backtest-valid list based on market cap to match training...")
        market_caps_backtest = get_market_caps(valid_tickers_backtest)
        ticker_cap_list_backtest = [(ticker, market_caps_backtest.get(ticker, 0)) for ticker in valid_tickers_backtest]
        ticker_cap_list_backtest.sort(key=lambda item: item[1], reverse=True)
        training_tickers = [item[0] for item in ticker_cap_list_backtest[:MAX_TICKERS_MODEL_WAS_TRAINED_ON]]
        logging.info(f"Selected {len(training_tickers)} tickers for backtest environment based on market cap.")

    if not training_tickers:
         logging.error("Could not determine or select the tickers the model was trained on.")
         exit(1)

    # Filter features dictionary to ONLY include those for the selected training tickers
    backtest_features = {ticker: features_all[ticker] for ticker in training_tickers if ticker in features_all}

    # Final check if features are actually available for the selected tickers
    if len(backtest_features) != len(training_tickers):
         missing_tickers_final = [t for t in training_tickers if t not in backtest_features]
         logging.warning(f"Some selected training tickers ({missing_tickers_final}) were missing features in the final backtest feature set. Removing them.")
         training_tickers = list(backtest_features.keys()) # Update list again
         if not training_tickers:
              logging.error("No features available for any of the selected training tickers in the backtest period.")
              exit(1)
    # ----------------------------------------------------------------------


    # 3. Run Backtest Simulation using the correctly selected tickers and their features
    backtest_start_date_str = backtest_start_dt.strftime('%Y-%m-%d')
    logging.info(f"Running backtest simulation with {len(training_tickers)} tickers matching model training...")

    returns, portfolio_history, holdings_df, trades_df = run_backtest(
        model=model,
        features_dict=backtest_features,   # Pass features for the TRAINING tickers
        stock_tickers=training_tickers,    # Pass the list of TRAINING tickers
        start_date=backtest_start_date_str,
        end_date=backtest_end_str,
        initial_capital=config.INITIAL_CAPITAL
    )

    # 4. Generate Report
    if returns is not None:
        generate_tear_sheet(returns)
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
