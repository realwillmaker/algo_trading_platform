import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for Reinforcement Learning using Gymnasium.
    Handles a portfolio of multiple stocks and aims to maximize portfolio value.
    """
    metadata = {'render_modes': ['human', 'ansi']} # Added 'ansi' for potential text rendering

    def __init__(self, features_dict, stock_tickers, initial_capital=config.INITIAL_CAPITAL,
                 lookback_window=config.LOOKBACK_WINDOW, commission=config.COMMISSION_PER_SHARE,
                 slippage=config.SLIPPAGE_PERCENT, reward_strategy=config.REWARD_STRATEGY):
        super().__init__()
        logging.info("Initializing StockTradingEnv...")

        self.stock_tickers = sorted(list(stock_tickers)) # Ensure consistent order
        self.num_stocks = len(self.stock_tickers)
        if self.num_stocks == 0:
             raise ValueError("Stock tickers list cannot be empty.")

        self.features_dict = features_dict # Should contain dataframes with processed features
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.commission = commission
        self.slippage = slippage
        self.reward_strategy = reward_strategy

        # --- Data Validation and Setup ---
        common_dates = None
        valid_tickers_with_data = []
        feature_cols_set = None
        min_len = float('inf')

        logging.debug(f"Initial tickers for env: {self.stock_tickers}")

        for ticker in list(self.stock_tickers): # Iterate over a copy
            if ticker not in self.features_dict or self.features_dict[ticker].empty:
                logging.warning(f"Ticker {ticker} missing or has empty features in features_dict. Excluding from env.")
                continue # Skip this ticker

            df_ticker = self.features_dict[ticker]

            # Check for required columns (Open, Close + features)
            required_cols_check = ['Open', 'Close']
            if not all(col in df_ticker.columns for col in required_cols_check):
                 logging.warning(f"Ticker {ticker} missing required columns (Open/Close). Excluding from env.")
                 continue

            # Identify feature columns (assuming they are all cols except Open, Close)
            current_cols = set(col for col in df_ticker.columns if col not in ['Open', 'Close'])
            if not current_cols:
                 logging.warning(f"Ticker {ticker} has no identifiable feature columns. Excluding from env.")
                 continue

            # Check for NaNs in crucial columns within the DataFrame
            if df_ticker[['Open', 'Close']].isnull().any().any():
                 logging.warning(f"Ticker {ticker} contains NaNs in Open/Close columns. Excluding from env.")
                 continue
            if df_ticker[list(current_cols)].isnull().any().any():
                 logging.warning(f"Ticker {ticker} contains NaNs in feature columns. Excluding from env.")
                 # You might implement imputation here instead of exclusion if preferred
                 continue


            # Update common dates and features
            valid_tickers_with_data.append(ticker)
            dates = df_ticker.index
            min_len = min(min_len, len(dates))

            if feature_cols_set is None:
                feature_cols_set = current_cols
            else:
                feature_cols_set.intersection_update(current_cols)

            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)


        # --- Update tickers list based on valid data ---
        self.stock_tickers = sorted(valid_tickers_with_data)
        self.num_stocks = len(self.stock_tickers)
        if self.num_stocks == 0:
             raise ValueError("No valid tickers remaining after data validation in environment.")
        logging.info(f"Environment using {self.num_stocks} validated stocks: {self.stock_tickers}")


        # --- Final common feature columns ---
        if feature_cols_set is None or not feature_cols_set:
             raise ValueError("No common feature columns found across valid tickers.")
        self.feature_columns = sorted(list(feature_cols_set))
        self.num_features_per_stock = len(self.feature_columns)
        logging.info(f"Environment using {self.num_features_per_stock} common feature columns: {self.feature_columns}")


        # --- Date Range Check ---
        if common_dates is None or len(common_dates) < self.lookback_window + 1:
             raise ValueError(f"Not enough common dates ({len(common_dates) if common_dates is not None else 0}) across {self.num_stocks} tickers for lookback {self.lookback_window}.")

        self.dates = sorted(list(common_dates)) # Ensure dates are sorted
        self.start_step = self.lookback_window
        self.end_step = len(self.dates) - 1 # Last valid index for step end valuation
        logging.info(f"Common date range: {self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')} ({len(self.dates)} days)")


        # --- Prepare Data Structures ---
        logging.debug("Preparing price and feature data structures...")
        self.prices = self._prepare_price_data() # Uses final self.stock_tickers
        self.features = self._prepare_feature_data() # Uses final self.stock_tickers and self.feature_columns
        logging.debug("Data structures prepared.")


        # --- Define Observation Space ---
        self.state_feature_dim = self.lookback_window * self.num_features_per_stock * self.num_stocks
        self.state_weight_dim = self.num_stocks # Current weight for each stock
        # self.state_cash_dim = 1 # Optional: Add cash percentage as state?
        self.obs_dim = self.state_feature_dim + self.state_weight_dim # + self.state_cash_dim
        logging.info(f"Calculated Observation Space Dim: {self.obs_dim} (Features: {self.state_feature_dim}, Weights: {self.state_weight_dim})")

        # Check for excessively large observation space
        if self.obs_dim > 1_000_000: # Example threshold
             logging.warning(f"Observation space dimension ({self.obs_dim}) is very large. Training might be extremely slow or memory intensive.")

        # Use float32 for observations, common practice in ML
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)


        # --- Define Action Space ---
        # Action: desired portfolio weights for the stocks [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # --- Initialize State Variables ---
        self.current_step = 0 # Will be set in reset
        self.cash = 0.0
        self.stock_shares = np.zeros(0)
        self.stock_value = np.zeros(0)
        self.portfolio_value = 0.0
        self.current_weights = np.zeros(0)
        self.portfolio_return_history = []
        self.trades = []

        # Call reset() to set initial state but capture result outside __init__
        # self.reset() # Reset sets up the actual state variables


    def _prepare_price_data(self):
        """Creates a DataFrame with 'Open' and 'Close' prices for selected stocks and common dates."""
        price_data = {}
        for ticker in self.stock_tickers:
            # Ensure columns exist before accessing
            if 'Open' in self.features_dict[ticker].columns:
                price_data[f"{ticker}_Open"] = self.features_dict[ticker]['Open']
            else: logging.error(f"'Open' column missing for {ticker} in _prepare_price_data")
            if 'Close' in self.features_dict[ticker].columns:
                price_data[f"{ticker}_Close"] = self.features_dict[ticker]['Close']
            else: logging.error(f"'Close' column missing for {ticker} in _prepare_price_data")
        # Align to common dates found in __init__
        prices_df = pd.DataFrame(price_data).loc[self.dates]
        # Check for NaNs introduced by reindexing or original data
        if prices_df.isnull().any().any():
             logging.warning("NaNs detected in price data after alignment. Forward filling.")
             prices_df.ffill(inplace=True)
             # Check again after fill
             if prices_df.isnull().any().any():
                  logging.error("NaNs remain in price data even after ffill! Backfilling.")
                  prices_df.bfill(inplace=True) # Use bfill as last resort for NaNs at start
                  if prices_df.isnull().any().any():
                       logging.error("FATAL: NaNs persist in price data after ffill and bfill. Cannot proceed.")
                       # Optionally, raise an error or handle appropriately
                       # For now, let it proceed but issues might occur in step()
        return prices_df

    def _prepare_feature_data(self):
        """Extracts and aligns defined feature columns for selected stocks and common dates."""
        aligned_features = {}
        for ticker in self.stock_tickers:
            # Select ONLY the common feature columns identified in __init__
            try:
                 aligned_features[ticker] = self.features_dict[ticker][self.feature_columns].loc[self.dates].copy()
                 # Check for NaNs after alignment
                 if aligned_features[ticker].isnull().any().any():
                      logging.warning(f"NaNs detected in features for {ticker} after alignment. Forward filling.")
                      aligned_features[ticker].ffill(inplace=True)
                      if aligned_features[ticker].isnull().any().any():
                           logging.warning(f"NaNs remain for {ticker} after ffill. Backfilling.")
                           aligned_features[ticker].bfill(inplace=True)
                           if aligned_features[ticker].isnull().any().any():
                                logging.error(f"FATAL: NaNs persist in features for {ticker} after ffill and bfill. Filling remaining with 0.")
                                aligned_features[ticker].fillna(0, inplace=True) # Last resort: fill with 0
            except KeyError as e:
                 logging.error(f"KeyError preparing features for {ticker}. Missing columns? {e}", exc_info=True)
                 # Handle error, maybe return empty dict or raise?
                 # For now, continue, but this ticker will likely cause issues later
            except Exception as e:
                 logging.error(f"Unexpected error preparing features for {ticker}: {e}", exc_info=True)

        return aligned_features


    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed) # Important for Gymnasium compatibility

        logging.debug("Resetting environment...")
        self.current_step = self.start_step # Start after lookback period
        self.cash = self.initial_capital
        self.stock_shares = np.zeros(self.num_stocks, dtype=int)
        self.stock_value = np.zeros(self.num_stocks, dtype=np.float32)
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.num_stocks, dtype=np.float32) # Stock weights only
        self.portfolio_return_history = [0.0] # Reset history
        self.trades = [] # Reset trades log

        # Get the initial observation
        observation = self._get_observation()
        info = self._get_info() # Get initial info dict

        logging.debug(f"Environment reset complete. Initial Portfolio Value: {self.portfolio_value:.2f}")
        # Check observation shape matches space
        if observation.shape != self.observation_space.shape:
             logging.error(f"Observation shape mismatch after reset! Expected {self.observation_space.shape}, Got {observation.shape}. Check _get_observation.")
             # Return a zero array of the correct shape to avoid immediate crash
             observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation, info


    def _get_observation(self):
        """Constructs the observation state vector."""
        # Check if current step allows lookback
        if self.current_step < self.lookback_window:
             logging.error(f"Called _get_observation at step {self.current_step} which is less than lookback {self.lookback_window}")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step # Python slicing is exclusive

        feature_frames = []
        # Ensure indices are valid for self.dates
        if start_idx < 0 or end_idx > len(self.dates):
             logging.error(f"Invalid indices for self.dates in _get_observation. start={start_idx}, end={end_idx}, len={len(self.dates)}")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Iterate through the lookback window dates
        for i in range(start_idx, end_idx):
            daily_features_list = []
            current_date = self.dates[i]

            # Iterate through each stock ticker
            for ticker in self.stock_tickers:
                try:
                    # Access the pre-prepared features for the current date
                    # self.features[ticker] has shape (n_dates, num_features_per_stock)
                    feature_vector = self.features[ticker].loc[current_date].values

                    # --- Shape Verification ---
                    if len(feature_vector) != self.num_features_per_stock:
                        logging.error(f"FATAL: Feature vector length mismatch for {ticker} on {current_date}. Expected {self.num_features_per_stock}, Got {len(feature_vector)}. Check _prepare_feature_data.")
                        return np.zeros(self.observation_space.shape, dtype=np.float32)

                    daily_features_list.append(feature_vector)

                except KeyError:
                    # Should not happen if self.dates is intersection and self.features is aligned
                    logging.error(f"KeyError accessing features for {ticker} on {current_date}. Imputing zeros.")
                    daily_features_list.append(np.zeros(self.num_features_per_stock, dtype=np.float32))
                except Exception as e:
                    logging.error(f"Error accessing features for {ticker} on {current_date}: {e}. Imputing zeros.", exc_info=True)
                    daily_features_list.append(np.zeros(self.num_features_per_stock, dtype=np.float32))

            # Concatenate features for all stocks for that day
            try:
                if not daily_features_list:
                      logging.error(f"No daily features collected for date {current_date}.")
                      return np.zeros(self.observation_space.shape, dtype=np.float32)

                daily_concatenated = np.concatenate(daily_features_list) # Should have length num_stocks * num_features_per_stock

                # --- Shape Verification ---
                expected_daily_len = self.num_stocks * self.num_features_per_stock
                if daily_concatenated.shape[0] != expected_daily_len:
                    logging.error(f"Daily concatenated feature length mismatch for {current_date}. Expected {expected_daily_len}, got {daily_concatenated.shape[0]}.")
                    return np.zeros(self.observation_space.shape, dtype=np.float32)

                feature_frames.append(daily_concatenated)

            except ValueError as concat_err:
                 logging.error(f"Failed to concatenate daily features for {current_date}: {concat_err}. Shapes: {[f.shape for f in daily_features_list]}", exc_info=True)
                 return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Stack frames for the lookback window
        if len(feature_frames) != self.lookback_window:
             logging.error(f"Number of feature frames ({len(feature_frames)}) does not match lookback window ({self.lookback_window}).")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        # --- Create final state ---
        try:
            # Convert list of 1D arrays into a 2D array first
            historical_features_array = np.array(feature_frames, dtype=np.float32) # Shape: (lookback, num_stocks * num_features_per_stock)
            # Then flatten to a 1D array
            historical_features_flat = historical_features_array.flatten() # Shape: (lookback * num_stocks * num_features_per_stock,)

            # --- Shape Verification ---
            if historical_features_flat.shape[0] != self.state_feature_dim:
                 logging.error(f"Flattened historical features shape mismatch! Expected {self.state_feature_dim}, Got {historical_features_flat.shape[0]}.")
                 return np.zeros(self.observation_space.shape, dtype=np.float32)

            # Ensure weights vector has correct shape
            if self.current_weights.shape[0] != self.num_stocks:
                 logging.warning(f"Correcting current_weights shape before observation. Expected {self.num_stocks}, got {self.current_weights.shape[0]}.")
                 correct_weights = np.zeros(self.num_stocks, dtype=np.float32)
                 len_to_copy = min(len(self.current_weights), self.num_stocks)
                 correct_weights[:len_to_copy] = self.current_weights[:len_to_copy]
                 self.current_weights = correct_weights

            # Concatenate features and weights
            observation = np.concatenate([historical_features_flat, self.current_weights])

            # --- Final Shape Check ---
            if observation.shape[0] != self.obs_dim:
                 logging.error(f"FINAL Observation shape mismatch! Expected {self.obs_dim}, got {observation.shape[0]}. Check calculation logic.")
                 return np.zeros(self.observation_space.shape, dtype=np.float32)

        except Exception as e:
             logging.error(f"Error constructing final observation vector: {e}", exc_info=True)
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation.astype(np.float32)


    def _get_info(self):
        """Returns supplementary information about the environment state."""
        # Ensure weights are a dictionary for info if possible
        weights_dict = dict(zip(self.stock_tickers, self.current_weights)) if len(self.stock_tickers) == len(self.current_weights) else {}
        shares_dict = dict(zip(self.stock_tickers, self.stock_shares)) if len(self.stock_tickers) == len(self.stock_shares) else {}

        return {
            "step": self.current_step,
            "date": self.dates[self.current_step] if self.current_step < len(self.dates) else None, # Handle potential off-by-one at end
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_value": self.stock_value.sum() if isinstance(self.stock_value, np.ndarray) else 0.0,
            "stock_shares": shares_dict,
            "current_weights": weights_dict,
            "trades": self.trades # Trades executed to reach this state (from *previous* step)
        }

    def step(self, action):
        """Executes one time step within the environment."""
        self.trades = [] # Clear trades from previous step
        previous_portfolio_value = self.portfolio_value
        current_date_index = self.current_step # Index for accessing data related to decision time T

        # Ensure action has the correct dimensions
        if not isinstance(action, np.ndarray) or action.shape != (self.num_stocks,):
             logging.error(f"Invalid action shape received in step(). Expected ({self.num_stocks},), Got {action.shape if hasattr(action, 'shape') else type(action)}. Using zero weights.")
             action = np.zeros(self.num_stocks, dtype=np.float32)

        # 1. Normalize Action (Target Weights)
        target_weights = np.clip(action, 0, 1)
        total_weight = np.sum(target_weights)
        if total_weight > 1.0001: # Allow small float tolerance
            target_weights = target_weights / total_weight # Normalize to sum to 1

        # 2. Determine Target Holdings based on Weights and *Current* Portfolio Value
        target_stock_values = target_weights * previous_portfolio_value

        # 3. Get Prices for Trading (Next Day's Open) & Valuation (Next Day's Close)
        next_step_idx = self.current_step + 1
        if next_step_idx >= len(self.dates): # Check if next step is out of bounds
             terminated = True
             reward = 0.0 # No further price changes to calculate reward
             observation = self._get_observation() # Get obs for the last valid state
             info = self._get_info()
             info['message'] = "End of data reached."
             logging.info(f"End of data reached at step {self.current_step}. Final Portfolio Value: {self.portfolio_value:.2f}")
             return observation, reward, terminated, False, info

        next_date = self.dates[next_step_idx]
        try:
            next_open_prices = self.prices.loc[next_date][[f"{ticker}_Open" for ticker in self.stock_tickers]].values
            next_close_prices = self.prices.loc[next_date][[f"{ticker}_Close" for ticker in self.stock_tickers]].values
        except KeyError:
            logging.error(f"Date {next_date} not found in self.prices index. Ending episode.")
            terminated = True
            reward = -10 # Penalize for data error
            observation = self._get_observation()
            info = self._get_info()
            info['error'] = f"Missing price data for date {next_date}."
            return observation, reward, terminated, False, info


        # Handle potential NaN prices (e.g., missing data for next day)
        if np.isnan(next_open_prices).any() or np.isnan(next_close_prices).any():
             logging.warning(f"NaN prices encountered on {next_date}. Holding positions and ending episode.")
             terminated = True
             reward = 0.0 # No change in value if we can't trade or value
             # Update state to reflect holding through the day
             self.current_step += 1
             observation = self._get_observation()
             info = self._get_info()
             info['error'] = f"NaN prices encountered for date {next_date}."
             return observation, reward, terminated, False, info

        # 4. Calculate Target Shares (Integer Shares)
        # Use np.floor for buys, maybe np.ceil for sells if trying to match value closely? Sticking with floor.
        # Avoid division by zero
        safe_open_prices = np.where(next_open_prices > 1e-6, next_open_prices, np.inf) # Use small threshold
        target_shares = np.floor(target_stock_values / safe_open_prices).astype(int)

        # 5. Determine Trades Needed
        shares_to_trade = target_shares - self.stock_shares

        # 6. Simulate Execution (Applying costs and integer constraints)
        cash_change_sell = 0
        executed_sells = []

        # --- Simulate Sells ---
        sell_indices = np.where(shares_to_trade < 0)[0]
        for idx in sell_indices:
            shares = abs(shares_to_trade[idx]) # Positive number
            ticker = self.stock_tickers[idx]
            exec_price = next_open_prices[idx] * (1 - self.slippage) # Assume sell hits bid side
            commission_cost = shares * self.commission
            proceeds = shares * exec_price - commission_cost
            if proceeds > 0: # Only execute if proceeds are positive
                cash_change_sell += proceeds
                self.stock_shares[idx] -= shares # Update holdings
                executed_sells.append({'ticker': ticker, 'shares': -shares, 'price': exec_price})
            else:
                 logging.debug(f"Skipping sell for {shares} {ticker} due to zero/negative proceeds after costs.")


        self.cash += cash_change_sell # Add proceeds from sells

        # --- Simulate Buys ---
        cash_change_buy = 0
        executed_buys = []
        buy_indices = np.where(shares_to_trade > 0)[0]
        buy_orders_cost = [] # Store tuples of (cost, index, shares)

        for idx in buy_indices:
            shares = shares_to_trade[idx] # Positive number
            ticker = self.stock_tickers[idx]
            exec_price = next_open_prices[idx] * (1 + self.slippage) # Assume buy hits ask side
            commission_cost = shares * self.commission
            cost = shares * exec_price + commission_cost
            if cost > 1e-6: # Check if cost is non-negligible
                 buy_orders_cost.append({'cost': cost, 'idx': idx, 'shares': shares, 'ticker': ticker, 'price': exec_price})

        # Optional: Sort buys (e.g., by target weight difference, or just cost - simplest is cost)
        # buy_orders_cost.sort(key=lambda x: x['cost']) # Example: Prioritize cheaper orders first

        # Execute buys based on available cash
        for order in buy_orders_cost:
            if self.cash >= order['cost']:
                self.cash -= order['cost']
                cash_change_buy -= order['cost'] # Track cash spent on buys
                self.stock_shares[order['idx']] += order['shares'] # Update holdings
                executed_buys.append({'ticker': order['ticker'], 'shares': order['shares'], 'price': order['price']})
            else:
                # Cannot afford this buy, skip (or implement partial fill logic)
                logging.debug(f"Insufficient cash for buy order: {order['shares']} {order['ticker']}. Need {order['cost']:.2f}, Have {self.cash:.2f}")
                # break # Optional: stop trying to buy if one fails (conservative)


        # Record executed trades for info dict
        self.trades = executed_sells + executed_buys

        # 7. Update Portfolio Value based on *Next Day's Close* prices
        self.stock_value = self.stock_shares * next_close_prices
        self.portfolio_value = self.cash + self.stock_value.sum()

        # Prevent negative portfolio value (e.g., due to extreme costs/slippage)
        self.portfolio_value = max(self.portfolio_value, 0)

        # 8. Calculate Reward
        reward = self._calculate_reward(previous_portfolio_value, self.portfolio_value)
        # Store daily return for reward calculations if needed (e.g., Sharpe)
        daily_return = (self.portfolio_value / previous_portfolio_value - 1) if previous_portfolio_value > 1e-6 else 0
        self.portfolio_return_history.append(daily_return)


        # 9. Update State Variables for Next Step
        self.current_step += 1 # Advance time step index
        # Update current weights based on end-of-day valuation
        if self.portfolio_value > 1e-6:
             self.current_weights = self.stock_value / self.portfolio_value
        else:
             self.current_weights = np.zeros(self.num_stocks, dtype=np.float32)
             # Optional: Handle bankruptcy explicitly
             # terminated = True
             # reward -= 100 # Add penalty


        # 10. Check Termination Conditions
        terminated = self.current_step > self.end_step # Check if we passed the last valid step index
        if self.portfolio_value <= 1e-6: # Check for bankruptcy
             logging.warning(f"Portfolio value near zero ({self.portfolio_value:.4f}) at step {self.current_step}. Terminating.")
             terminated = True
             # Optionally add large negative reward for bankruptcy
             # reward -= 100


        # Truncated flag (Gymnasium standard) - Use if episode ends for reasons other than task goal
        truncated = False # e.g., time limits not related to reaching end date or bankruptcy

        # Get observation for the *next* state (t+1)
        observation = self._get_observation()
        info = self._get_info()

        # Log step summary for debugging
        logging.debug(
            f"Step: {self.current_step-1}->{self.current_step}, Date: {next_date.strftime('%Y-%m-%d')}, "
            f"Action: {np.round(action, 2)}, PV: {self.portfolio_value:.2f}, "
            f"Reward: {reward:.5f}, Term: {terminated}, Trunc: {truncated}"
        )


        return observation, reward, terminated, truncated, info


    def _calculate_reward(self, prev_val, current_val):
        """Calculates the reward for the current step based on config."""
        reward = 0.0
        if prev_val < 1e-6: # Avoid division by zero if previous value was near zero
             return 0.0

        if self.reward_strategy == 'log_return':
             # Log return of the portfolio value
             # Add small epsilon only if current_val is also near zero
             epsilon = 1e-10 if current_val < 1e-6 else 0
             reward = np.log((current_val + epsilon) / prev_val)

        elif self.reward_strategy == 'sharpe':
             # Use daily returns history to approximate Sharpe
             returns = np.array(self.portfolio_return_history[-20:]) # Lookback window for std dev calc
             if len(returns) < 5: # Need min points for std dev
                  # Fallback to log return if not enough history
                  epsilon = 1e-10 if current_val < 1e-6 else 0
                  reward = np.log((current_val + epsilon) / prev_val)
             else:
                  daily_return = (current_val / prev_val) - 1
                  std_dev = np.std(returns)
                  if std_dev < 1e-6: # Avoid division by zero std dev
                      reward = 0.0 # Or use simple return
                  else:
                      # Simplified Sharpe: (mean return - risk_free) / std_dev
                      # Using latest daily return as proxy for mean for simplicity
                      daily_rf = config.TARGET_RISK_FREE_RATE / 252.0
                      reward = (daily_return - daily_rf) / std_dev

        elif self.reward_strategy == 'simple_return': # Default/simple option
             reward = (current_val / prev_val) - 1
        else:
            logging.warning(f"Unknown reward strategy: {self.reward_strategy}. Using simple return.")
            reward = (current_val / prev_val) - 1

        # --- Optional Penalties ---
        # Example: Penalty for high turnover (requires calculating trades value)
        # trade_value = sum(abs(t['shares']) * t['price'] for t in self.trades)
        # turnover = trade_value / prev_val if prev_val > 1e-6 else 0
        # turnover_penalty = 0.01 # Example penalty factor
        # reward -= turnover * turnover_penalty

        # Example: Penalty for holding too much cash
        # cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 1e-6 else 1.0
        # cash_penalty_factor = 0.001 # Small penalty
        # reward -= cash_ratio * cash_penalty_factor

        return reward


    def render(self, mode='human'):
        """Renders the environment state."""
        if mode == 'ansi': # Return text representation
            info = self._get_info()
            output = f"--- Step: {info['step']}, Date: {info.get('date', 'N/A')} ---\n"
            output += f"Portfolio Value: ${info['portfolio_value']:,.2f}\n"
            output += f"Cash: ${info['cash']:,.2f}, Stock Value: ${info['stock_value']:,.2f}\n"
            output += "Holdings (Shares):\n"
            held_shares = {k: v for k, v in info['stock_shares'].items() if v > 0}
            if held_shares:
                 for ticker, shares in sorted(held_shares.items()): output += f"  {ticker}: {shares}\n"
            else: output += "  None\n"
            output += "---\n"
            return output
        elif mode == 'human': # Print to console
            print(self.render(mode='ansi'))

    def close(self):
        """Clean up environment resources."""
        logging.debug("Closing StockTradingEnv.")
        pass # No specific resources (like open files or connections) to close here currently
