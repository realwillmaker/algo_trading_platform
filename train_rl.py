import pandas as pd
import numpy as np
import torch # Import torch
from stable_baselines3 import PPO, SAC, A2C # Choose your algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
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
        env = StockTradingEnv(features_dict=features_dict, stock_tickers=stock_tickers,
                              lookback_window=config.LOOKBACK_WINDOW)
        return env
    return _init

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
    stock_tickers = [t for t in sp500_tickers if t not in config.MACRO_FEATURES.values() and t != config.BENCHMARK_TICKER]
    logging.info("Loading/Generating Features for Training...")
    features = feature_engineer.create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)
    if not features: exit()
    valid_tickers = list(features.keys())
    if not valid_tickers: exit()
    logging.info(f"Training with {len(valid_tickers)} tickers.")

    # 2. Create Environment(s)
    # NOTE: Using SubprocVecEnv with GPU can sometimes lead to CUDA initialization issues
    # or high memory usage if each process tries to grab GPU memory.
    # Start with DummyVecEnv (N_ENVS=1) for simpler GPU debugging if you encounter issues.
    logging.info(f"Creating training environment(s)... N_ENVS={config.N_ENVS}")
    if config.N_ENVS > 1 and device == 'cuda':
         logging.warning(f"Using SubprocVecEnv (N_ENVS={config.N_ENVS}) with GPU. Monitor memory usage. Consider starting with N_ENVS=1 for debugging.")
         env = SubprocVecEnv([make_env(i, features_dict=features, stock_tickers=valid_tickers) for i in range(config.N_ENVS)])
    else:
         # Use DummyVecEnv if N_ENVS=1 or if on CPU
         env = DummyVecEnv([lambda: StockTradingEnv(features_dict=features, stock_tickers=valid_tickers)])


    # 3. Setup RL Agent - Specify the device!
    tensorboard_log_dir = "./tensorboard_logs/"
    logging.info(f"Setting up RL Agent on device: {device}")
    if config.RL_ALGORITHM == "PPO":
         model_params = config.PPO_PARAMS
         model = PPO('MlpPolicy', env, verbose=1,
                     device=device, # <<< SPECIFY GPU or CPU
                     tensorboard_log=tensorboard_log_dir, **model_params)
    elif config.RL_ALGORITHM == "SAC":
         model = SAC('MlpPolicy', env, verbose=1,
                     device=device, # <<< SPECIFY GPU or CPU
                     tensorboard_log=tensorboard_log_dir)
    elif config.RL_ALGORITHM == "A2C":
         model = A2C('MlpPolicy', env, verbose=1,
                     device=device, # <<< SPECIFY GPU or CPU
                      tensorboard_log=tensorboard_log_dir)
    else:
         logging.error(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}")
         env.close()
         exit()

    logging.info(f"Using RL Algorithm: {config.RL_ALGORITHM}")
    logging.info(f"Model Policy Architecture: {model.policy}")
    logging.info(f"Model running on device: {model.device}") # Confirm device

    # 4. Setup Callbacks (Same as before)
    checkpoint_callback = CheckpointCallback(save_freq=max(config.TRAIN_TIMESTEPS // 10, 10000),
                                              save_path=config.MODELS_DIR,
                                              name_prefix=f"rl_model_{config.RL_ALGORITHM.lower()}")

    # 5. Train the Agent (Same as before, but now runs on specified device)
    logging.info(f"Starting training for {config.TRAIN_TIMESTEPS} timesteps...")
    try:
         model.learn(total_timesteps=config.TRAIN_TIMESTEPS, callback=checkpoint_callback,
                     tb_log_name=f"{config.RL_ALGORITHM}_{int(time.time())}")
    except Exception as e:
         logging.error(f"Error during training: {e}", exc_info=True)
    finally:
         env.close()

    # 6. Save the Final Model (Same as before)
    logging.info(f"Saving final trained model to {config.MODEL_FILENAME}")
    model.save(config.MODEL_FILENAME)

    end_time = time.time()
    logging.info(f"--- RL Agent Training Completed in {(end_time - start_time)/60:.2f} minutes ---")

    print("\nTraining finished.")
    print(f"Model saved as: {config.MODEL_FILENAME}")
    # ... rest of suggestions ...
