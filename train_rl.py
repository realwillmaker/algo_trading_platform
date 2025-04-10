import pandas as pd
import numpy as np
import torch # Import torch
import yfinance as yf # Import yfinance
from stable_baselines3 import PPO, SAC, A2C # Choose your algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed # For seeding envs if needed
import logging
import os
import time

import config
import utils
import feature_engineer
from trading_env import StockTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_env(rank, seed=0, features_dict=None, stock_tickers=None):
    """Utility function for multiprocessed envs."""
    def _init():
        env_features = {ticker: features_dict[ticker] for ticker in stock_tickers if ticker in features_dict}
        env = StockTradingEnv(features_dict=env_features, stock_tickers=stock_tickers,
                              lookback_window=config.LOOKBACK_WINDOW)
        # Optional: Set seed for this specific environment instance for greater reproducibility
        # env.reset(seed=seed + rank)
        return env
    # Consider setting global seed if needed, though SB3 handles it via model's seed
    # if seed is not None:
    #    set_random_seed(seed)
    return _init

# ==============================================================================
# =========================== MAIN TRAINING SCRIPT ===========================
# ==============================================================================
if __name__ == "__main__":
    logging.info("--- Starting RL Agent Training ---")
    start_time = time.time()

    # --- GPU Check ---
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        logging.info(f"CUDA is available! Found {gpu_count} GPU(s). Using GPU {current_device}: {gpu_name}")
        device = 'cuda'
    else:
        logging.warning("CUDA not available. Training will run on CPU (might be slow).")
        device = 'cpu'
    # ---------------

    # 1. Load/Prepare Data (Same as before)
    sp500_tickers = utils.get_sp500_tickers()
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    logging.info("Loading/Generating Features for Training Period...")
    features = feature_engineer.create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)

    if not features:
         logging.error("Feature generation failed or produced no results. Cannot train.")
         exit(1)

    valid_tickers = sorted(list(features.keys())) # Tickers with successful features
    if not valid_tickers:
         logging.error("No valid tickers after feature generation. Cannot train.")
         exit(1)
    logging.info(f"Feature generation successful for {len(valid_tickers)} tickers initially.")


    # --- Option 1 Modification: Limit tickers based on Market Cap ---
    MAX_TICKERS_FOR_TRAINING = 100 # Configurable: Set max number of stocks for training
    final_training_tickers = []

    if len(valid_tickers) <= MAX_TICKERS_FOR_TRAINING:
        # Use all valid tickers if fewer than or equal to the limit
        logging.info(f"Using all {len(valid_tickers)} valid tickers as it's less than/equal to the limit ({MAX_TICKERS_FOR_TRAINING}).")
        final_training_tickers = valid_tickers
    else:
        logging.warning(f"Selecting top {MAX_TICKERS_FOR_TRAINING} tickers from {len(valid_tickers)} based on current market cap.")

        # Fetch market caps for all valid tickers
        #market_caps = get_market_caps(valid_tickers) # Returns dict {ticker: cap_value_or_0}
        market_caps = utils.get_market_caps(valid_tickers) # NEW (using definition from utils)

        # Create a list of tuples (ticker, market_cap)
        ticker_cap_list = [(ticker, market_caps.get(ticker, 0)) for ticker in valid_tickers]

        # Sort the list by market cap in descending order (highest first)
        ticker_cap_list.sort(key=lambda item: item[1], reverse=True)

        # Select the top N tickers from the sorted list
        final_training_tickers = [item[0] for item in ticker_cap_list[:MAX_TICKERS_FOR_TRAINING]]

        logging.info(f"Selected top {len(final_training_tickers)} tickers based on market cap. Example: {final_training_tickers[:10]}")
        # Log tickers with zero market cap if any were included (shouldn't be if MAX_TICKERS is less than total)
        zero_cap_selected = [item[0] for item in ticker_cap_list[:MAX_TICKERS_FOR_TRAINING] if item[1] <= 0]
        if zero_cap_selected:
             logging.warning(f"Tickers selected with zero/missing market cap: {zero_cap_selected}")
    # ----------------------------------------------------------------------

    if not final_training_tickers:
         logging.error("No tickers selected for training after filtering. Exiting.")
         exit(1)

    logging.info(f"Using {len(final_training_tickers)} tickers for training environment.")
    # Filter the features dictionary to include only the selected tickers for the env
    training_features = {ticker: features[ticker] for ticker in final_training_tickers}


    # 2. Create Environment(s)
    logging.info(f"Creating training environment(s)... N_ENVS={config.N_ENVS}")
    env_fns = [make_env(i, features_dict=training_features, stock_tickers=final_training_tickers) for i in range(config.N_ENVS)]

    if config.N_ENVS > 1:
        if device == 'cuda':
             logging.warning(f"Using SubprocVecEnv (N_ENVS={config.N_ENVS}) with GPU. Monitor memory usage.")
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)


    # 3. Setup RL Agent
    tensorboard_log_dir = "./tensorboard_logs/"
    logging.info(f"Setting up RL Agent ({config.RL_ALGORITHM}) on device: {device}")
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    try:
        if config.RL_ALGORITHM == "PPO":
             model_params = config.PPO_PARAMS
             model = PPO('MlpPolicy', env, verbose=1,
                         device=device,
                         tensorboard_log=tensorboard_log_dir, **model_params)
        elif config.RL_ALGORITHM == "SAC":
             model = SAC('MlpPolicy', env, verbose=1,
                         device=device,
                         tensorboard_log=tensorboard_log_dir)
        elif config.RL_ALGORITHM == "A2C":
             model = A2C('MlpPolicy', env, verbose=1,
                         device=device,
                         tensorboard_log=tensorboard_log_dir)
        else:
             raise ValueError(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}") # Raise error for unsupported algo

    except Exception as e:
         logging.error(f"Error initializing RL model: {e}", exc_info=True)
         env.close()
         exit(1)

    logging.info(f"RL Algorithm: {config.RL_ALGORITHM}")
    logging.info(f"Model Policy Architecture: {model.policy}")
    try:
        logging.info(f"Model running on device: {model.device}")
    except AttributeError:
        logging.warning("Could not verify model.device attribute.")


    # 4. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.TRAIN_TIMESTEPS // 10, 10000),
        save_path=config.MODELS_DIR,
        name_prefix=f"rl_model_{config.RL_ALGORITHM.lower()}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 5. Train the Agent
    logging.info(f"Starting training for {config.TRAIN_TIMESTEPS} timesteps...")
    try:
         model.learn(
             total_timesteps=config.TRAIN_TIMESTEPS,
             callback=checkpoint_callback,
             tb_log_name=f"{config.RL_ALGORITHM}_{int(time.time())}",
             reset_num_timesteps=False
             )
         # --- Save final model after successful training ---
         logging.info(f"Training finished. Saving final model to {config.MODEL_FILENAME}")
         model.save(config.MODEL_FILENAME)
         training_successful = True

    except Exception as e:
         logging.error(f"Error during training: {e}", exc_info=True)
         training_successful = False
         # Attempt to save intermediate model on error
         try:
              intermediate_save_path = os.path.join(config.MODELS_DIR, f"rl_model_{config.RL_ALGORITHM.lower()}_error_save.zip")
              model.save(intermediate_save_path)
              logging.info(f"Attempted to save model on error to: {intermediate_save_path}")
         except Exception as save_e:
              logging.error(f"Could not save model on error: {save_e}")
    finally:
         logging.info("Closing training environment(s)...")
         env.close()

    end_time = time.time()
    logging.info(f"--- RL Agent Training process {( 'Completed' if training_successful else 'Aborted' )} in {(end_time - start_time)/60:.2f} minutes ---")

    # --- Final Messages ---
    if training_successful:
        print("\nTraining finished successfully.")
        print(f"Model saved as: {config.MODEL_FILENAME}")
        print("Next steps suggestion:")
        print("1. Evaluate the model using backtester.py")
        print(f"2. Analyze TensorBoard logs: `tensorboard --logdir {tensorboard_log_dir}`")
        print("3. Fine-tune hyperparameters (e.g., PPO_PARAMS in config.py) and retrain if necessary.")
    else:
        print("\nTraining aborted due to an error.")
        print("Check the log file for details.")
        exit(1) # Exit with error status
