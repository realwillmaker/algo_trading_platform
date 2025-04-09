import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- General ---
START_DATE = "2018-01-01"
END_DATE_TRAIN = "2022-12-31"
END_DATE_BACKTEST = pd.Timestamp.now().strftime('%Y-%m-%d')
INITIAL_CAPITAL = 1_000_000.00
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
LOG_FILE = "trading_log.log"

# --- Data ---
DATA_PROVIDER = "yfinance"
MARKET_DATA_COLS = ['Open', 'High', 'Low', 'Close', 'Volume'] # Keep original case for loading
MACRO_FEATURES = {
    'VIX': '^VIX',
    # 'FEDFUNDS': 'FEDFUNDS',
}

# Add near the Data or General section
# --- yfinance Fetching Params ---
YFINANCE_BATCH_SIZE = 100  # How many tickers to fetch in one yf.download call
YFINANCE_DELAY_PER_BATCH = 5 # Seconds to wait between batches (increase from 1)
YFINANCE_MAX_RETRIES = 3     # Max attempts for a failed batch
YFINANCE_RETRY_DELAY = 15    # Seconds to wait before retrying a failed batch
MARKET_CAP_FETCH_DELAY = 0.5 # Seconds to wait between yf.Ticker(t).info calls

# --- Features ---
# Use pandas-ta indicator names (lowercase) and arguments
TECHNICAL_INDICATORS = {
    # Key names (e.g., 'SMA_10') are used for selecting features later if needed,
    # but pandas-ta generates its own column names (e.g., SMA_10, MACD_12_26_9)
    'sma_10': {'kind': 'sma', 'length': 10},
    'sma_50': {'kind': 'sma', 'length': 50},
    'rsi_14': {'kind': 'rsi', 'length': 14},
    'macd_12_26_9': {'kind': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
    # Add more pandas-ta indicators as needed: e.g. bbands, atr, obv
    'bbands_20_2': {'kind': 'bbands', 'length': 20, 'std': 2},
    'atr_14': {'kind': 'atr', 'length': 14},
    'obv': {'kind': 'obv'},
}
LOOKBACK_WINDOW = 60

# --- RL Agent ---
RL_ALGORITHM = "PPO"
MODEL_FILENAME = f"{MODELS_DIR}/rl_model_{RL_ALGORITHM}.zip"
TRAIN_TIMESTEPS = 100_000
N_ENVS = 4
REWARD_STRATEGY = 'log_return'
PPO_PARAMS = {
    'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'learning_rate': 3e-4,
    'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0,
    'vf_coef': 0.5, 'max_grad_norm': 0.5,
    'policy_kwargs': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
}

# --- Portfolio & Execution ---
TARGET_RISK_FREE_RATE = 0.01
COMMISSION_PER_SHARE = 0.00
SLIPPAGE_PERCENT = 0.0005
MAX_POSITION_WEIGHT = 0.10
MIN_POSITION_WEIGHT = 0.00
REBALANCE_THRESHOLD = 0.01
ORDER_EXECUTION_TIME = 'open'

# --- Broker API (Placeholders) ---
# ... (same as before) ...

# --- Reporting ---
BENCHMARK_TICKER = "SPY"

# --- Environment Setup ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
