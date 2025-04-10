#update_state_manual.py
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta # Import datetime classes

import config # For initial capital, file paths, maybe commission/slippage?
import utils # For load/save state

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
log_file = "update_state_manual.log" # Use a specific log file name

# File Handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Stream Handler (Console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Set root logger level
if logger.hasHandlers(): logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# --------------------


def apply_manual_trades(current_state, planned_trades, prices):
    """
    Simulates the execution of planned trades on the current state.
    Uses provided prices (e.g., previous day's close used for planning).

    Args:
        current_state (dict): Dict with 'cash' and 'positions' ({ticker: shares}).
        planned_trades (list): List of trade dicts [{'ticker': str, 'shares': int}].
        prices (dict): Dict of {ticker: price} for simulation.

    Returns:
        dict: The new state dict after applying trades.
    """
    new_cash = current_state.get('cash', 0.0)
    # Ensure positions keys are strings and values are integers
    new_positions = {str(k): int(v) for k, v in current_state.get('positions', {}).items()}

    logging.info("Applying planned trades to simulate state update...")
    if not prices:
         logging.error("Price dictionary is empty in apply_manual_trades. Cannot simulate accurately.")
         return current_state # Return current state if prices are missing

    # --- Simulate Sells ---
    sells = [t for t in planned_trades if t.get('shares', 0) < 0]
    logging.debug(f"Processing {len(sells)} sell trades...")
    for trade in sells:
        ticker = str(trade['ticker']) # Ensure ticker is string
        shares_to_sell = abs(int(trade['shares'])) # Ensure shares is positive int
        price = prices.get(ticker)

        if price is None or price <= 0:
            logging.warning(f"Cannot apply sell for {ticker}, price missing/invalid: {price}. Skipping trade.")
            continue
        if ticker not in new_positions or new_positions[ticker] < shares_to_sell:
            logging.warning(f"Cannot apply sell for {ticker}, not enough shares held ({new_positions.get(ticker, 0)} held < {shares_to_sell} to sell). Skipping trade.")
            continue

        # Apply estimated slippage/commission
        exec_price = price * (1 - config.SLIPPAGE_PERCENT)
        commission = shares_to_sell * config.COMMISSION_PER_SHARE
        proceeds = shares_to_sell * exec_price - commission

        if proceeds > -1e-6: # Allow slightly negative for rounding
            new_cash += proceeds
            new_positions[ticker] -= shares_to_sell
            if new_positions[ticker] <= 0: # Remove if shares go to zero
                del new_positions[ticker]
            logging.debug(f"Simulated Sell: {shares_to_sell} {ticker} @ ~{exec_price:.2f}. Cash +{proceeds:.2f}")
        else:
            logging.warning(f"Skipping sell for {shares_to_sell} {ticker} due to zero/negative proceeds ({proceeds:.2f}) after costs.")


    # --- Simulate Buys ---
    buys = [t for t in planned_trades if t.get('shares', 0) > 0]
    logging.debug(f"Processing {len(buys)} buy trades...")
    buys.sort(key=lambda x: x['ticker']) # Sort alphabetically for consistent processing order

    for trade in buys:
        ticker = str(trade['ticker'])
        shares_to_buy = int(trade['shares'])
        price = prices.get(ticker)

        if price is None or price <= 0:
            logging.warning(f"Cannot apply buy for {ticker}, price missing/invalid: {price}. Skipping trade.")
            continue

        exec_price = price * (1 + config.SLIPPAGE_PERCENT)
        commission = shares_to_buy * config.COMMISSION_PER_SHARE
        cost = shares_to_buy * exec_price + commission

        if cost < 1e-6: # Skip if cost is negligible
             logging.debug(f"Skipping buy for {ticker}, negligible cost.")
             continue

        if new_cash >= cost:
            new_cash -= cost
            new_positions[ticker] = new_positions.get(ticker, 0) + shares_to_buy
            logging.debug(f"Simulated Buy: {shares_to_buy} {ticker} @ ~{exec_price:.2f}. Cash -{cost:.2f}")
        else:
            logging.warning(f"Simulating buy for {ticker}, but estimated cost {cost:.2f} > available cash {new_cash:.2f}. State file might diverge from actual brokerage account if trade was skipped/modified.")
            # Simulate it anyway for the file state
            new_cash -= cost
            new_positions[ticker] = new_positions.get(ticker, 0) + shares_to_buy

    # Ensure final positions only contain stocks with > 0 shares
    final_positions = {k: v for k, v in new_positions.items() if v > 0}

    logging.info(f"State update simulation complete. New Cash: ${new_cash:,.2f}, New Holdings: {len(final_positions)} stocks.")
    return {'cash': new_cash, 'positions': final_positions}


if __name__ == "__main__":
    logging.info("--- Starting Manual State Update Script ---")

    # 1. Load the state *before* the trades were applied
    last_state = utils.load_portfolio_state()
    if not last_state:
        # load_portfolio_state logs error, just exit
        exit(1)
    logging.debug(f"Loaded last state: Cash={last_state.get('cash'):.2f}, Positions={last_state.get('positions')}")


    # 2. Load the planned trades
    orders_file = "planned_orders.json"
    if not os.path.exists(orders_file):
        logging.info("No planned orders file found ('planned_orders.json'). Assuming no trades were needed or file is missing. State remains unchanged.")
        # Optional: Still save the loaded state again? Or just exit? Let's exit.
        exit(0)

    try:
        with open(orders_file, 'r') as f:
            planned_trades = json.load(f)
        # Basic validation of trade format
        if not isinstance(planned_trades, list) or (planned_trades and not all(isinstance(t, dict) and 'ticker' in t and 'shares' in t for t in planned_trades)):
             raise ValueError("Invalid format in planned_orders.json")
        logging.info(f"Loaded {len(planned_trades)} planned trades from {orders_file}")
    except Exception as e:
        logging.error(f"Failed to load or parse {orders_file}: {e}", exc_info=True)
        exit(1)

    # Exit if there are no trades to apply
    if not planned_trades:
        logging.info("Planned orders file contained no trades. State remains unchanged.")
        # Save state back for consistency even if no trades
        utils.save_portfolio_state(last_state)
        try: os.remove(orders_file) # Clean up empty file
        except OSError: pass
        exit(0)


    # 3. Get the prices for simulation (need prices for all tickers involved in trades)
    logging.info("Fetching latest close prices for state update simulation...")
    tickers_in_trades = list(set(t['ticker'] for t in planned_trades))
    latest_prices = {}
    yesterday_dt = datetime.now() - timedelta(days=1)
    days_to_check = 5 # Check up to 5 days back for last trading day
    found_prices_date = None

    for i in range(days_to_check):
         check_date = yesterday_dt - timedelta(days=i)
         # Ensure check_date is timezone naive if data index is naive
         if check_date.tzinfo is not None: check_date = check_date.replace(tzinfo=None)
         # Use the check_date directly (datetime object) for comparison/lookup

         logging.debug(f"Trying to fetch prices for date: {check_date.strftime('%Y-%m-%d')}")
         all_prices_found = True
         temp_prices = {}
         missing_ticker_on_date = None # Track which ticker failed

         for ticker in tickers_in_trades:
             price_data = utils.load_data_from_file(ticker)
             if price_data is not None and not price_data.empty:
                  # Ensure index is datetime and naive
                  if not isinstance(price_data.index, pd.DatetimeIndex):
                      try: price_data.index = pd.to_datetime(price_data.index)
                      except Exception as e: logging.warning(f"Could not convert index for {ticker} to datetime: {e}"); all_prices_found=False; missing_ticker_on_date=f"{ticker} (Index Error)"; break
                  if price_data.index.tz is not None: price_data.index = price_data.index.tz_localize(None)

                  # Use the datetime object check_date for index lookup
                  if check_date in price_data.index:
                       # Check if 'Close' column exists
                       if 'Close' not in price_data.columns:
                           logging.warning(f"'Close' column missing for {ticker}. Skipping price check.")
                           all_prices_found = False; missing_ticker_on_date = f"{ticker} (No Close Col)"; break
                       price = price_data.loc[check_date]['Close']
                       if pd.notna(price) and price > 0:
                           temp_prices[ticker] = price
                       else: # Price is NaN or zero
                           all_prices_found = False; missing_ticker_on_date = f"{ticker} (Invalid Price: {price})"; logging.debug(f"Invalid price found for {ticker} on {check_date.strftime('%Y-%m-%d')}"); break
                  else: # Date missing in index
                      all_prices_found = False; missing_ticker_on_date = f"{ticker} (Date Missing)"; logging.debug(f"Date {check_date.strftime('%Y-%m-%d')} missing for {ticker}"); break
             else: # File missing or empty
                 all_prices_found = False; missing_ticker_on_date = f"{ticker} (File Missing/Empty)"; logging.debug(f"Data file missing or empty for {ticker}"); break

         # Check if prices for *all* tickers were found for this date
         if all_prices_found:
              latest_prices = temp_prices
              found_prices_date = check_date # Store the datetime object used
              logging.info(f"Using prices from {found_prices_date.strftime('%Y-%m-%d')} for simulation.")
              break # Exit outer loop once prices are found
         else:
             # Log *why* this date failed if a reason was identified
             reason = f"Reason: {missing_ticker_on_date}" if missing_ticker_on_date else "Unknown reason."
             logging.warning(f"Price check failed for date {check_date.strftime('%Y-%m-%d')}. {reason}")

    # Check if prices were successfully found for a recent date
    if not latest_prices:
          logging.error(f"Could not find recent valid close prices for all {len(tickers_in_trades)} traded tickers after checking {days_to_check} days. Cannot accurately update state.")
          exit(1)


    # 4. Apply trades to the state using the found prices
    new_state = apply_manual_trades(last_state, planned_trades, latest_prices)

    # 5. Save the NEW state, overwriting the old one
    # Ensure keys/values are JSON serializable
    save_state = {
        'cash': float(new_state.get('cash', 0.0)),
        'positions': {str(k): int(v) for k, v in new_state.get('positions', {}).items() if v > 0} # Save only >0 positions
    }
    utils.save_portfolio_state(save_state)
    logging.info("Successfully updated portfolio state file for next run.")

    # Optional: Clean up the orders file now that state is updated
    try:
        os.remove(orders_file)
        logging.info(f"Removed processed orders file: {orders_file}")
    except OSError as e:
        logging.warning(f"Could not remove orders file {orders_file}: {e}")

    logging.info("--- Manual State Update Script Finished ---")
