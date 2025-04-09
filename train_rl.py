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
    """
    Utility function for multiprocessed envs.
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG (Note: Seeding might need adjustment for full reproducibility in multiprocessing)
    :param features_dict: Dictionary of feature dataframes
    :param stock_tickers: List of tickers to use in this env instance
    """
    def _init():
        # Pass only the required features and tickers to the environment instance
        env_features = {ticker: features_dict[ticker] for ticker in stock_tickers if ticker in features_dict}
        env = StockTradingEnv(features_dict=env_features, stock_tickers=stock_tickers,
                              lookback_window=config.LOOKBACK_WINDOW)
        # env.seed(seed + rank) # Deprecated in Gymnasium
        # Use env.reset(seed=seed + rank) if stricter seeding per env instance is needed,
        # but SB3 handles seeding generally well with set_random_seed.
        return env
    # set_global_seeds(seed) # Deprecated
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

    # 1. Load/Prepare Data
    sp500_tickers = utils.get_sp500_tickers()
    # Exclude benchmark/macro from the list of stocks to generate features for
    stock_tickers = [t for t in sp500_tickers
                     if t not in config.MACRO_FEATURES.values()
                     and t != config.BENCHMARK_TICKER]

    logging.info("Loading/Generating Features for Training Period...")
    # Generate features for ALL potentially tradable stocks first
    features = feature_engineer.create_feature_dataset(stock_tickers, config.START_DATE, config.END_DATE_TRAIN)

    if not features:
         logging.error("Feature generation failed or produced no results. Cannot train.")
         exit(1) # Use non-zero exit code for errors

    # Get the list of tickers for which features were successfully generated
    valid_tickers = sorted(list(features.keys())) # Sort for consistency
    if not valid_tickers:
         logging.error("No valid tickers after feature generation. Cannot train.")
         exit(1)
    logging.info(f"Feature generation successful for {len(valid_tickers)} tickers initially.")


    # --- Option 1 Implementation: Limit the number of tickers for training ---
    MAX_TICKERS_FOR_TRAINING = 100 # Configurable: Set max number of stocks for training (e.g., 50, 100)
    if len(valid_tickers) > MAX_TICKERS_FOR_TRAINING:
        logging.warning(f"Limiting training tickers from {len(valid_tickers)} to {MAX_TICKERS_FOR_TRAINING} based on config/stability.")
        # Simple approach: take the first N alphabetically after sorting
        final_training_tickers = valid_tickers[:MAX_TICKERS_FOR_TRAINING]
        # More advanced: Sort by data length, market cap (if available), etc. before slicing
    else:
        # Use all tickers if fewer than the limit were successfully processed
        final_training_tickers = valid_tickers
    # ----------------------------------------------------------------------


    logging.info(f"Using {len(final_training_tickers)} tickers for training environment.")
    # Filter the features dictionary to include only the selected tickers for the env
    training_features = {ticker: features[ticker] for ticker in final_training_tickers}


    # 2. Create Environment(s)
    # Pass the filtered features and the final list of tickers
    logging.info(f"Creating training environment(s)... N_ENVS={config.N_ENVS}")
    # Use a list comprehension to create environment functions
    env_fns = [make_env(i, features_dict=training_features, stock_tickers=final_training_tickers) for i in range(config.N_ENVS)]

    if config.N_ENVS > 1:
        # Consider SubprocVecEnv if N_ENVS > 1, especially on multi-core CPUs
        # Be mindful of potential GPU memory issues if using SubprocVecEnv with device='cuda'
        if device == 'cuda':
             logging.warning(f"Using SubprocVecEnv (N_ENVS={config.N_ENVS}) with GPU. Monitor memory usage.")
        env = SubprocVecEnv(env_fns)
    else:
        # Use DummyVecEnv if N_ENVS=1 or for simpler debugging
        env = DummyVecEnv(env_fns)


    # 3. Setup RL Agent - Specify the device!
    tensorboard_log_dir = "./tensorboard_logs/"
    logging.info(f"Setting up RL Agent ({config.RL_ALGORITHM}) on device: {device}")
    # Ensure the directory exists
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    try:
        if config.RL_ALGORITHM == "PPO":
             model_params = config.PPO_PARAMS
             model = PPO('MlpPolicy', env, verbose=1,
                         device=device, # <<< SPECIFY GPU or CPU
                         tensorboard_log=tensorboard_log_dir, **model_params)
        elif config.RL_ALGORITHM == "SAC":
             # Add SAC params to config if needed
             model = SAC('MlpPolicy', env, verbose=1,
                         device=device, # <<< SPECIFY GPU or CPU
                         tensorboard_log=tensorboard_log_dir)
        elif config.RL_ALGORITHM == "A2C":
             # Add A2C params to config if needed
             model = A2C('MlpPolicy', env, verbose=1,
                         device=device, # <<< SPECIFY GPU or CPU
                         tensorboard_log=tensorboard_log_dir)
        else:
             logging.error(f"Unsupported RL Algorithm: {config.RL_ALGORITHM}")
             env.close()
             exit(1)
    except Exception as e:
         logging.error(f"Error initializing RL model: {e}", exc_info=True)
         env.close()
         exit(1)

    logging.info(f"RL Algorithm: {config.RL_ALGORITHM}")
    logging.info(f"Model Policy Architecture: {model.policy}")
    # Verify the device the model is actually on
    try:
        logging.info(f"Model running on device: {model.device}")
    except AttributeError:
        logging.warning("Could not verify model.device attribute.")


    # 4. Setup Callbacks (Optional but Recommended)
    # Save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.TRAIN_TIMESTEPS // 10, 10000), # Save roughly 10 times or every 10k steps
        save_path=config.MODELS_DIR,
        name_prefix=f"rl_model_{config.RL_ALGORITHM.lower()}",
        save_replay_buffer=True, # Important for off-policy algorithms like SAC
        save_vecnormalize=True, # Save running mean/std if using VecNormalize wrapper
    )

    # 5. Train the Agent
    logging.info(f"Starting training for {config.TRAIN_TIMESTEPS} timesteps...")
    try:
         # The learn method handles interaction with the environment(s)
         model.learn(
             total_timesteps=config.TRAIN_TIMESTEPS,
             callback=checkpoint_callback,
             tb_log_name=f"{config.RL_ALGORITHM}_{int(time.time())}", # Unique name for TensorBoard run
             reset_num_timesteps=False # Continue timestep count if loading a model
             )
    except Exception as e:
         logging.error(f"Error during training: {e}", exc_info=True)
         # Attempt to save intermediate model even on error?
         try:
              intermediate_save_path = os.path.join(config.MODELS_DIR, f"rl_model_{config.RL_ALGORITHM.lower()}_error_save.zip")
              model.save(intermediate_save_path)
              logging.info(f"Attempted to save model on error to: {intermediate_save_path}")
         except Exception as save_e:
              logging.error(f"Could not save model on error: {save_e}")
    finally:
         # Make sure to close the environments to release resources
         logging.info("Closing training environment(s)...")
         env.close()


    # 6. Save the Final Model (If training completed without error)
    # This might be redundant if the last CheckpointCallback saved, but good practice.
    if 'model' in locals(): # Check if model was successfully initialized
        try:
             logging.info(f"Saving final trained model to {config.MODEL_FILENAME}")
             model.save(config.MODEL_FILENAME)
        except Exception as e:
             logging.error(f"Failed to save final model: {e}")

    end_time = time.time()
    logging.info(f"--- RL Agent Training Completed in {(end_time - start_time)/60:.2f} minutes ---")

    # Optional: Suggest next steps
    print("\nTraining finished.")
    print(f"Model saved as: {config.MODEL_FILENAME}")
    print("Next steps suggestion:")
    print("1. Evaluate the model using backtester.py")
    print(f"2. Analyze TensorBoard logs: `tensorboard --logdir {tensorboard_log_dir}`")
    print("3. Fine-tune hyperparameters (e.g., PPO_PARAMS in config.py) and retrain if necessary.")
