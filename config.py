# config.py (Updates marked)
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
LOG_FILE = "trading_log.log" # General log file
PORTFOLIO_STATE_FILE = "portfolio_state.json" # For manual trading state

# --- Data ---
DATA_PROVIDER = "yfinance"
MARKET_DATA_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

# --- UPDATED MACRO FEATURES ---
# Key: Name used internally (e.g., in features dataframe)
# Value: Ticker for yfinance OR Code for FRED
MACRO_FEATURES = {
    'VIX': '^VIX', # From yfinance
    'T10Y': 'DGS10', # 10-Year Treasury Constant Maturity Rate from FRED
    'T2Y': 'DGS2',   # 2-Year Treasury Constant Maturity Rate from FRED
    # 'T10Y2Y': 'T10Y2Y', # 10Y-2Y Spread - FRED calculates this directly
    'HY_SPREAD': 'BAMLH0A0HYM2EY' # ICE BofA US High Yield Index Option-Adjusted Spread
}
# ------------------------------

# --- Features ---
# --- UPDATED TECHNICAL INDICATORS ---
# Use pandas-ta indicator names (lowercase) and arguments
TECHNICAL_INDICATORS = {
    'sma_10': {'kind': 'sma', 'length': 10},
    'sma_50': {'kind': 'sma', 'length': 50},
    'rsi_14': {'kind': 'rsi', 'length': 14},
    'macd_12_26_9': {'kind': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
    'bbands_20_2': {'kind': 'bbands', 'length': 20, 'std': 2},
    'atr_14': {'kind': 'atr', 'length': 14}, # ATR needed for ADX
    'obv': {'kind': 'obv'},
    # --- New Indicators ---
    'stoch_14_3_3': {'kind': 'stoch', 'k': 14, 'd': 3, 'smooth_k': 3}, # Stochastic %K, %D
    'cci_20': {'kind': 'cci', 'length': 20}, # Commodity Channel Index
    'roc_10': {'kind': 'roc', 'length': 10}, # Rate of Change 10-day
    'roc_30': {'kind': 'roc', 'length': 30}, # Rate of Change 30-day
    'adx_14': {'kind': 'adx', 'length': 14}, # ADX, +DI, -DI
    'cmf_20': {'kind': 'cmf', 'length': 20}, # Chaikin Money Flow
    # --- End New ---
}
# ---------------------------------
LOOKBACK_WINDOW = 60 # Keep lookback window the same for now

# --- RL Agent ---
RL_ALGORITHM = "PPO"
MODEL_FILENAME = f"{MODELS_DIR}/rl_model_{RL_ALGORITHM}.zip"
TRAIN_TIMESTEPS = 200_000 # <-- INCREASED TIMESTEPS
N_ENVS = 4 # Keep as 4 for parallel processing
REWARD_STRATEGY = 'log_return' # Keep simple reward for now
PPO_PARAMS = {
    'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'learning_rate': 3e-4,
    'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.0,
    'vf_coef': 0.5, 'max_grad_norm': 0.5,
    'policy_kwargs': dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]) # Keep network same initially
}

# --- Portfolio & Execution ---
TARGET_RISK_FREE_RATE = 0.01
COMMISSION_PER_SHARE = 0.00
SLIPPAGE_PERCENT = 0.0005
MAX_POSITION_WEIGHT = 0.10
MIN_POSITION_WEIGHT = 0.00
REBALANCE_THRESHOLD = 0.01
ORDER_EXECUTION_TIME = 'open'

# --- yfinance Fetching Params ---
YFINANCE_BATCH_SIZE = 100
YFINANCE_DELAY_PER_BATCH = 5
YFINANCE_MAX_RETRIES = 3
YFINANCE_RETRY_DELAY = 15

# --- FRED API ---
# Ensure FRED_API_KEY is set in your .env file
FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_DELAY = 0.5 # Delay between FRED API calls

# --- Market Cap Fetching Params ---
MARKET_CAP_FETCH_DELAY = 0.5

# --- Reporting ---
BENCHMARK_TICKER = "SPY"

# --- Environment Setup ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
