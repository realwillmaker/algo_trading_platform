import gymnasium as gym # Use Gymnasium (updated Gym)
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for Reinforcement Learning using Gymnasium.

    Handles a portfolio of multiple stocks and aims to maximize portfolio value.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, features_dict, stock_tickers, initial_capital=config.INITIAL_CAPITAL,
                 lookback_window=config.LOOKBACK_WINDOW, commission=config.COMMISSION_PER_SHARE,
                 slippage=config.SLIPPAGE_PERCENT, reward_strategy=config.REWARD_STRATEGY):
        super().__init__()

        self.features_dict = features_dict
        self.stock_tickers = stock_tickers
        self.num_stocks = len(stock_tickers)
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.commission = commission
        self.slippage = slippage
        self.reward_strategy = reward_strategy

        # Find common date range across all tickers AFTER feature engineering
        common_dates = None
        for ticker in self.stock_tickers:
            if ticker in self.features_dict:
                dates = self.features_dict[ticker].index
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
            else:
                raise ValueError(f"Ticker {ticker} not found in features_dict during env init.")

        if common_dates is None or len(common_dates) < self.lookback_window + 1:
             raise ValueError("Not enough common dates or data across tickers for the environment.")

        self.dates = sorted(list(common_dates))
        self.start_step = self.lookback_window
        self.end_step = len(self.dates) - 1 # Allow one step for final reward calc

        # Prepare data structure for faster lookups during steps
        self.prices = self._prepare_price_data()
        self.features = self._prepare_feature_data()

        # Define action space: Portfolio weights (num_stocks for weights + 1 for cash, maybe?)
        # Option 1: Weights for stocks only, cash is implicit (1 - sum(stock_weights))
        # Action space: Desired weights for each stock [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Define observation space: Lookback window of features for all stocks + portfolio state
        # Shape: (lookback_window, num_features * num_stocks + num_stocks + 1) -> Or flatten
        # Simpler: Flattened features for lookback window + current weights
        # Example: (lookback_window, num_features_per_stock) for ONE stock + current weights
        # Let's use flattened features for all stocks over the window + current weights
        self.num_features_per_stock = self.features[self.stock_tickers[0]].shape[1]
        obs_shape = (self.lookback_window, self.num_features_per_stock * self.num_stocks) # History
        obs_shape = obs_shape[0] * obs_shape[1] # Flatten history
        obs_shape += self.num_stocks # Add space for current weights
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Initialize state variables
        self.reset()

    def _prepare_price_data(self):
        """Creates a DataFrame with 'Open' and 'Close' prices for all stocks."""
        price_data = {}
        for ticker in self.stock_tickers:
            # Use Open for trading simulation, Close for valuation
            price_data[f"{ticker}_Open"] = self.features_dict[ticker]['Open']
            price_data[f"{ticker}_Close"] = self.features_dict[ticker]['Close']
        prices_df = pd.DataFrame(price_data).loc[self.dates] # Align by common dates
        return prices_df

    def _prepare_feature_data(self):
        """Extracts and aligns feature columns from the dictionary."""
        aligned_features = {}
        for ticker in self.stock_tickers:
            # Select only feature columns (exclude OHLCV, returns if not used as features)
            # Assuming all columns EXCEPT OHLCV and 'log_return' are features
            feature_cols = [col for col in self.features_dict[ticker].columns
                            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'log_return']]
            aligned_features[ticker] = self.features_dict[ticker][feature_cols].loc[self.dates]
        return aligned_features


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gymnasium requires handling seed

        self.current_step = self.start_step
        self.cash = self.initial_capital
        self.stock_shares = np.zeros(self.num_stocks, dtype=int)
        self.stock_value = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32) # Stock weights only
        self.portfolio_return_history = [0.0] # Track daily returns for Sharpe etc.

        self.trades = [] # Optional: track trades made

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self):
        """Constructs the observation state."""
        # Get feature history for the lookback window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step # Exclusive index for slicing

        # Stack features for all stocks horizontally for each day in the window
        feature_frames = []
        for i in range(start_idx, end_idx):
            daily_features = []
            current_date = self.dates[i]
            for ticker in self.stock_tickers:
                 # Handle potential missing dates within a ticker's data if not perfectly aligned previously
                 if current_date in self.features[ticker].index:
                      daily_features.append(self.features[ticker].loc[current_date].values)
                 else:
                      # Handle missing feature data (e.g., use zeros or previous day's data)
                      num_feats = self.num_features_per_stock
                      daily_features.append(np.zeros(num_feats)) # Simple imputation
            feature_frames.append(np.concatenate(daily_features))

        historical_features = np.array(feature_frames).flatten()

        # Combine historical features and current portfolio weights
        observation = np.concatenate([historical_features, self.current_weights])

        return observation.astype(np.float32)


    def _get_info(self):
        """Returns supplementary information about the environment state."""
        return {
            "step": self.current_step,
            "date": self.dates[self.current_step],
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_value": self.stock_value.sum(),
            "stock_shares": dict(zip(self.stock_tickers, self.stock_shares)),
            "current_weights": dict(zip(self.stock_tickers, self.current_weights)),
            "trades": self.trades # Trades executed in the *previous* step
        }

    def step(self, action):
        """Executes one time step within the environment."""
        self.trades = [] # Clear trades from previous step
        current_date = self.dates[self.current_step]
        previous_portfolio_value = self.portfolio_value

        # 1. Normalize Action (ensure weights sum to <= 1)
        # The agent outputs desired stock weights. Cash is implicit.
        # Using Softmax is a common way to ensure weights sum to 1,
        # but let's clip and normalize for simplicity here.
        action = np.clip(action, 0, 1)
        total_weight = np.sum(action)
        if total_weight > 1:
            target_weights = action / total_weight # Normalize to sum to 1
        else:
            target_weights = action # Allow holding cash

        # 2. Determine Target Holdings based on Weights and Current Portfolio Value
        target_stock_values = target_weights * previous_portfolio_value

        # 3. Get Prices for Trading (Use next day's Open)
        # We use data up to day T (current_step) to decide action for day T+1
        # Trading happens at open of T+1. Valuation happens at close of T+1.
        next_step_idx = self.current_step + 1
        if next_step_idx >= len(self.dates): # End of data
             # Handle episode end: Calculate final value and return
             terminated = True
             reward = self._calculate_reward(previous_portfolio_value, self.portfolio_value) # Final reward might be 0 or based on last step
             observation = self._get_observation() # Get obs for the last state
             info = self._get_info()
             info['message'] = "End of data reached."
             return observation, reward, terminated, False, info # False for truncated

        next_open_prices = self.prices.iloc[next_step_idx][[f"{ticker}_Open" for ticker in self.stock_tickers]].values
        next_close_prices = self.prices.iloc[next_step_idx][[f"{ticker}_Close" for ticker in self.stock_tickers]].values

        # Handle potential NaN prices (e.g., missing data for next day)
        if np.isnan(next_open_prices).any() or np.isnan(next_close_prices).any():
             logging.warning(f"NaN prices encountered on {self.dates[next_step_idx]}. Ending episode early.")
             # Option 1: End the episode
             terminated = True
             reward = -10 # Penalize for bad data / inability to trade
             observation = self._get_observation()
             info = self._get_info()
             info['error'] = "NaN prices encountered."
             # Use last valid portfolio value for final calculation? Or zero reward?
             # self.portfolio_value stays same as previous step if can't trade/value
             reward = self._calculate_reward(previous_portfolio_value, previous_portfolio_value)

             return observation, reward, terminated, False, info # False for truncated
             # Option 2: Try to hold positions (more complex logic)


        # 4. Calculate Target Shares (Integer Shares)
        # Avoid division by zero if open price is somehow zero
        safe_open_prices = np.where(next_open_prices > 0, next_open_prices, np.inf)
        target_shares = np.floor(target_stock_values / safe_open_prices).astype(int)

        # 5. Determine Trades Needed
        shares_to_trade = target_shares - self.stock_shares

        # 6. Simulate Execution (Sell first, then Buy)
        cash_change = 0

        # --- Simulate Sells ---
        sell_tickers_idx = np.where(shares_to_trade < 0)[0]
        for idx in sell_tickers_idx:
            shares_to_sell = -shares_to_trade[idx]
            ticker = self.stock_tickers[idx]
            sell_price = next_open_prices[idx] * (1 - self.slippage) # Apply slippage
            proceeds = shares_to_sell * sell_price
            commission_cost = shares_to_sell * self.commission
            cash_change += (proceeds - commission_cost)
            self.stock_shares[idx] -= shares_to_sell
            self.trades.append({'date': current_date, 'ticker': ticker, 'action': 'SELL',
                                'shares': shares_to_sell, 'price': sell_price, 'commission': commission_cost})

        self.cash += cash_change
        cash_after_sells = self.cash

        # --- Simulate Buys ---
        buy_tickers_idx = np.where(shares_to_trade > 0)[0]
        cost_of_buys = 0
        buy_orders = [] # Store potential buy orders {idx: shares}

        for idx in buy_tickers_idx:
            shares_to_buy = shares_to_trade[idx]
            ticker = self.stock_tickers[idx]
            buy_price = next_open_prices[idx] * (1 + self.slippage) # Apply slippage
            cost = shares_to_buy * buy_price
            commission_cost = shares_to_buy * self.commission
            total_cost = cost + commission_cost
            buy_orders.append({'idx': idx, 'ticker': ticker, 'shares': shares_to_buy,
                               'price': buy_price, 'cost': total_cost, 'commission': commission_cost})

        # Sort buys (optional, e.g., by target weight or cost) - not essential for simulation
        # buy_orders.sort(key=lambda x: x['cost'], reverse=True)

        # Check affordability and execute buys
        for order in buy_orders:
            if self.cash >= order['cost']:
                self.cash -= order['cost']
                self.stock_shares[order['idx']] += order['shares']
                self.trades.append({'date': current_date, 'ticker': order['ticker'], 'action': 'BUY',
                                    'shares': order['shares'], 'price': order['price'], 'commission': order['commission']})
            else:
                # Cannot afford this buy order, could partially fill or skip
                logging.debug(f"Cannot afford to buy {order['shares']} of {order['ticker']} on {current_date}. Need {order['cost']:.2f}, have {self.cash:.2f}")
                # Simple approach: Skip the buy if cannot afford full amount
                pass # Shares remain unchanged for this ticker


        # 7. Update Portfolio Value based on *Next Day's Close*
        self.stock_value = self.stock_shares * next_close_prices
        self.portfolio_value = self.cash + self.stock_value.sum()

        # Ensure portfolio value is not negative (can happen with high leverage/costs, though unlikely here)
        self.portfolio_value = max(self.portfolio_value, 0)

        # Update current weights based on end-of-day valuation
        if self.portfolio_value > 0:
             self.current_weights = self.stock_value / self.portfolio_value
        else:
             self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)


        # 8. Calculate Reward
        reward = self._calculate_reward(previous_portfolio_value, self.portfolio_value)
        daily_return = (self.portfolio_value / previous_portfolio_value - 1) if previous_portfolio_value > 0 else 0
        self.portfolio_return_history.append(daily_return)


        # 9. Update State and Check Termination
        self.current_step += 1
        terminated = self.current_step >= self.end_step or self.portfolio_value <= 0 # End if bankrupt

        # Truncated ( Gymnasium specific - differentiate from termination )
        truncated = False # Set to True if an external condition cuts episode short (e.g. time limit not related to task goal)

        if self.portfolio_value <= 0:
            logging.warning(f"Portfolio value reached zero or below on {current_date}. Terminating episode.")
            # Optionally add a large negative reward for bankruptcy
            # reward -= 100

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def _calculate_reward(self, prev_val, current_val):
        """Calculates the reward for the current step."""
        if self.reward_strategy == 'log_return':
             # Log return of the portfolio value
             if prev_val <= 0 or current_val <= 0: return 0
             # Add small epsilon to prevent log(0) issues if values are very small
             epsilon = 1e-10
             reward = np.log(max(current_val, epsilon) / max(prev_val, epsilon))

        elif self.reward_strategy == 'sharpe':
             # Approximate incremental Sharpe ratio change (more complex)
             # Or simply use log return and let the agent learn risk aversion implicitly
             # A simple proxy could be return penalized by volatility
             returns = np.array(self.portfolio_return_history[-20:]) # Lookback for std dev
             if len(returns) < 5 or np.std(returns) == 0:
                  # Use simple log return if not enough history or zero std dev
                  epsilon = 1e-10
                  reward = np.log(max(current_val, epsilon) / max(prev_val, epsilon))
             else:
                  # (Daily Return - Daily Risk Free) / Daily Std Dev
                  daily_rf = config.TARGET_RISK_FREE_RATE / 252.0 # Approximate daily risk-free rate
                  daily_return = (current_val / prev_val - 1) if prev_val > 0 else 0
                  reward = (daily_return - daily_rf) / np.std(returns)

        else: # Default to simple return
             reward = (current_val / prev_val - 1) if prev_val > 0 else 0

        # Optional: Add penalty for excessive trading (turnover)
        # turnover = calculate_turnover(...)
        # reward -= turnover_penalty_factor * turnover

        # Optional: Penalize holding too much cash? (Depends on objective)
        # cash_penalty = (self.cash / self.portfolio_value) * cash_penalty_factor
        # reward -= cash_penalty

        return reward


    def render(self, mode='human'):
        """Renders the environment state (optional)."""
        if mode == 'human':
            info = self._get_info()
            print(f"--- Step: {info['step']}, Date: {info['date']} ---")
            print(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
            print(f"Cash: ${info['cash']:,.2f}")
            print(f"Stock Value: ${info['stock_value']:,.2f}")
            print("Holdings (Shares):")
            # Filter to show only stocks with shares > 0
            held_shares = {k: v for k, v in info['stock_shares'].items() if v > 0}
            if held_shares:
                 for ticker, shares in held_shares.items():
                      print(f"  {ticker}: {shares}")
            else:
                 print("  None")
            print("-" * 20)

    def close(self):
        """Clean up environment resources."""
        pass # Nothing specific to clean up here for now
