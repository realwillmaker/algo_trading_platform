import pandas as pd
import numpy as np
import json
import os
import logging
import config # For initial capital, file paths, maybe commission/slippage?
import utils # For load/save state

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

def apply_manual_trades(current_state, planned_trades, prices):
    """
    Simulates the execution of planned trades on the current state.
    Uses provided prices (e.g., previous day's close used for planning).
    """
    new_cash = current_state.get('cash', 0.0)
    new_positions = current_state.get('positions', {}).copy() # Work on a copy

    logging.info("Applying planned trades to update state...")

    # --- Simulate Sells ---
    sells = [t for t in planned_trades if t['shares'] < 0]
    for trade in sells:
        ticker = trade['ticker']
        shares_to_sell = abs(trade['shares'])
        price = prices.get(ticker)

        if price is None or price <= 0:
            logging.warning(f"Cannot apply sell for {ticker}, price missing/invalid: {price}. Skipping.")
            continue
        if ticker not in new_positions or new_positions[ticker] < shares_to_sell:
            logging.warning(f"Cannot apply sell for {ticker}, not enough shares held ({new_positions.get(ticker, 0)} < {shares_to_sell}). Skipping.")
            continue

        # Apply estimated slippage/commission (match portfolio_manager logic)
        exec_price = price * (1 - config.SLIPPAGE_PERCENT)
        commission = shares_to_sell * config.COMMISSION_PER_SHARE
        proceeds = shares_to_sell * exec_price - commission

        new_cash += proceeds
        new_positions[ticker] -= shares_to_sell
        if new_positions[ticker] == 0:
            del new_positions[ticker] # Remove if shares go to zero
        logging.debug(f"Applied Sell: {shares_to_sell} {ticker} @ ~{exec_price:.2f}. Cash +{proceeds:.2f}")

    # --- Simulate Buys ---
    buys = [t for t in planned_trades if t['shares'] > 0]
    # Optional: Sort buys if needed
    for trade in buys:
        ticker = trade['ticker']
        shares_to_buy = trade['shares']
        price = prices.get(ticker)

        if price is None or price <= 0:
            logging.warning(f"Cannot apply buy for {ticker}, price missing/invalid: {price}. Skipping.")
            continue

        exec_price = price * (1 + config.SLIPPAGE_PERCENT)
        commission = shares_to_buy * config.COMMISSION_PER_SHARE
        cost = shares_to_buy * exec_price + commission

        if new_cash >= cost:
            new_cash -= cost
            new_positions[ticker] = new_positions.get(ticker, 0) + shares_to_buy
            logging.debug(f"Applied Buy: {shares_to_buy} {ticker} @ ~{exec_price:.2f}. Cash -{cost:.2f}")
        else:
            # This assumes you perfectly executed the plan. If you couldn't afford a buy manually,
            # this simulation won't reflect that unless you modify the planned_trades input.
            logging.warning(f"Simulating buy for {ticker}, but estimated cost {cost:.2f} > available cash {new_cash:.2f}. State might diverge from reality.")
            # For simulation, proceed assuming it happened, or stop? Let's proceed but warn.
            new_cash -= cost # Allow cash to go negative in simulation if needed, but log
            new_positions[ticker] = new_positions.get(ticker, 0) + shares_to_buy


    return {'cash': new_cash, 'positions': new_positions}


if __name__ == "__main__":
    logging.info("--- Starting Manual State Update Script ---")

    # 1. Load the state *before* the trades were applied (saved by run_daily_job)
    last_state = utils.load_portfolio_state()
    if not last_state:
        logging.error("Could not load the last portfolio state. Cannot update.")
        exit(1)

    # 2. Load the planned trades from the file saved by run_daily_job
    orders_file = "planned_orders.json"
    if not os.path.exists(orders_file):
        logging.info("No planned orders file found. Assuming no trades were needed or file is missing. State remains unchanged.")
        # Optional: Still save the loaded state again? Or just exit? Let's exit.
        exit(0)

    try:
        with open(orders_file, 'r') as f:
            planned_trades = json.load(f)
        logging.info(f"Loaded {len(planned_trades)} planned trades from {orders_file}")
    except Exception as e:
        logging.error(f"Failed to load or parse {orders_file}: {e}", exc_info=True)
        exit(1)

    # 3. Get the prices used for the trade calculation (e.g., previous close)
    # Need the date reference, maybe save it alongside planned orders?
    # Or re-fetch latest close prices. Re-fetching is simpler.
    logging.info("Fetching latest close prices for state update simulation...")
    tickers_in_trades = list(set(t['ticker'] for t in planned_trades))
    latest_prices = {}
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_dt = datetime.now() - timedelta(days=1)
    # Find the most recent trading day (handle weekends/holidays simply)
    days_to_check = 5
    for i in range(days_to_check):
         check_date = yesterday_dt - timedelta(days=i)
         check_date_str = check_date.strftime('%Y-%m-%d')
         all_prices_found = True
         temp_prices = {}
         logging.debug(f"Trying to fetch prices for date: {check_date_str}")
         for ticker in tickers_in_trades:
             price_data = utils.load_data_from_file(ticker) # Assumes data_fetcher ran recently
             if price_data is not None:
                  if check_date in price_data.index:
                       price = price_data.loc[check_date]['Close']
                       if pd.notna(price) and price > 0:
                            temp_prices[ticker] = price
                       else:
                            all_prices_found = False; break # Stop checking this date if price invalid
                  else:
                       all_prices_found = False; break # Stop checking this date if date missing
             else:
                  all_prices_found = False; break # Stop checking this date if file missing
         if all_prices_found:
              latest_prices = temp_prices
              logging.info(f"Using prices from {check_date_str} for simulation.")
              break
         if i == days_to_check - 1:
              logging.error("Could not find recent valid close prices for all traded tickers. Cannot accurately update state.")
              exit(1)


    # 4. Apply trades to the state
    new_state = apply_manual_trades(last_state, planned_trades, latest_prices)

    # 5. Save the NEW state, overwriting the old one
    utils.save_portfolio_state(new_state)
    logging.info("Successfully updated portfolio state file for next run.")

    # Optional: Clean up the orders file
    try:
        os.remove(orders_file)
        logging.info(f"Removed processed orders file: {orders_file}")
    except OSError as e:
        logging.warning(f"Could not remove orders file {orders_file}: {e}")

    logging.info("--- Manual State Update Script Finished ---")
