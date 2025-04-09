import numpy as np
import pandas as pd
import logging

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_target_orders(current_holdings, current_cash, total_portfolio_value,
                             target_weights, stock_tickers, current_prices):
    """
    Calculates the orders needed to move from current holdings to target weights.

    Args:
        current_holdings (dict): {ticker: shares}
        current_cash (float): Current cash balance.
        total_portfolio_value (float): Total value (cash + stocks).
        target_weights (np.array): Target weights for each stock in stock_tickers order.
        stock_tickers (list): List of tickers corresponding to target_weights.
        current_prices (dict): {ticker: price} (use closing price from day T for calculation).

    Returns:
        list: List of order dictionaries [{'ticker': str, 'shares': int (positive for buy, negative for sell)}]
    """
    logging.info("Calculating target orders...")
    logging.debug(f"Input Holdings: {current_holdings}")
    logging.debug(f"Input Cash: {current_cash:.2f}")
    logging.debug(f"Input Portfolio Value: {total_portfolio_value:.2f}")
    logging.debug(f"Input Target Weights: {target_weights}")
    logging.debug(f"Input Prices: {current_prices}")


    if total_portfolio_value <= 0:
        logging.warning("Total portfolio value is zero or negative. Cannot calculate orders.")
        return []
    if len(target_weights) != len(stock_tickers):
         raise ValueError("Length of target_weights must match length of stock_tickers.")

    orders = []
    num_stocks = len(stock_tickers)
    current_shares = np.array([current_holdings.get(ticker, 0) for ticker in stock_tickers], dtype=int)

    # 1. Calculate target dollar value for each stock
    target_dollar_values = target_weights * total_portfolio_value

    # Apply constraints (e.g., max position size) - Optional
    max_allowed_value = config.MAX_POSITION_WEIGHT * total_portfolio_value
    target_dollar_values = np.minimum(target_dollar_values, max_allowed_value)
    # Recalculate target weights after capping (and potentially re-normalize if needed)
    adjusted_target_weights = target_dollar_values / total_portfolio_value
    # Ensure we don't target negative weights implicitly by capping
    adjusted_target_weights = np.maximum(adjusted_target_weights, 0)
    if np.sum(adjusted_target_weights) > 1.0001: # Allow for small float inaccuracies
        logging.warning(f"Adjusted target weights sum to {np.sum(adjusted_target_weights):.4f} > 1 after capping. Re-normalizing.")
        adjusted_target_weights /= np.sum(adjusted_target_weights)
        target_dollar_values = adjusted_target_weights * total_portfolio_value # Recalculate dollar values


    # 2. Calculate target number of shares (integer)
    target_shares = np.zeros(num_stocks, dtype=int)
    for i, ticker in enumerate(stock_tickers):
        price = current_prices.get(ticker)
        if price is not None and price > 0:
            target_shares[i] = np.floor(target_dollar_values[i] / price).astype(int)
        elif target_dollar_values[i] > 1e-6 : # Only warn if we intended to hold it
             logging.warning(f"Price for {ticker} is missing or zero ({price}). Cannot calculate target shares.")


    # 3. Determine shares to trade
    shares_to_trade = target_shares - current_shares

    # 4. Separate Buys and Sells
    sell_indices = np.where(shares_to_trade < 0)[0]
    buy_indices = np.where(shares_to_trade > 0)[0]

    # 5. Estimate cash change from sells (using provided prices, assuming they are executable)
    estimated_proceeds = 0
    for idx in sell_indices:
        ticker = stock_tickers[idx]
        shares = -shares_to_trade[idx] # Positive number
        price = current_prices.get(ticker)
        if price is not None and price > 0:
            # Simplified: Use the provided price directly. Real execution needs next open/slippage.
            proceeds = shares * price * (1 - config.SLIPPAGE_PERCENT) # Sim slippage
            commission = shares * config.COMMISSION_PER_SHARE
            estimated_proceeds += (proceeds - commission)
            orders.append({'ticker': ticker, 'shares': -shares}) # Negative for sell order
        else:
             logging.warning(f"Cannot estimate proceeds for selling {ticker}, price missing/zero.")

    # 6. Estimate cost of buys and check affordability
    available_cash_for_buys = current_cash + estimated_proceeds
    buy_orders_to_place = []
    estimated_buy_cost = 0

    # Create list of potential buys with their estimated costs
    potential_buys = []
    for idx in buy_indices:
        ticker = stock_tickers[idx]
        shares = shares_to_trade[idx] # Positive number
        price = current_prices.get(ticker)
        if price is not None and price > 0:
             cost = shares * price * (1 + config.SLIPPAGE_PERCENT) # Sim slippage
             commission = shares * config.COMMISSION_PER_SHARE
             total_cost = cost + commission
             potential_buys.append({'ticker': ticker, 'shares': shares, 'cost': total_cost, 'idx': idx})
        else:
             logging.warning(f"Cannot estimate cost for buying {ticker}, price missing/zero.")

    # Prioritize buys (optional, e.g., by target weight difference or just iterate)
    # potential_buys.sort(key=lambda x: target_dollar_values[x['idx']], reverse=True)

    # Allocate cash to buys
    for buy in potential_buys:
        if estimated_buy_cost + buy['cost'] <= available_cash_for_buys:
            estimated_buy_cost += buy['cost']
            buy_orders_to_place.append({'ticker': buy['ticker'], 'shares': buy['shares']})
        else:
            # Cannot afford full amount. Option 1: Skip. Option 2: Partial fill.
            # Simple: Skip the order if cannot fully afford. More complex logic needed for partial fills.
            logging.warning(f"Cannot afford full buy order for {buy['shares']} {buy['ticker']}. Need {buy['cost']:.2f}, Available for buys: {(available_cash_for_buys - estimated_buy_cost):.2f}. Skipping this buy.")
            # To implement partial fill:
            # affordable_shares = np.floor((available_cash_for_buys - estimated_buy_cost) / (buy['cost'] / buy['shares']))
            # if affordable_shares > 0:
            #    cost = affordable_shares * price * (1 + config.SLIPPAGE_PERCENT)
            #    commission = affordable_shares * config.COMMISSION_PER_SHARE
            #    estimated_buy_cost += (cost + commission)
            #    buy_orders_to_place.append({'ticker': buy['ticker'], 'shares': int(affordable_shares)})
            pass # Skipping


    # Combine sell and buy orders
    orders.extend(buy_orders_to_place)

    # Filter out zero-share orders
    final_orders = [o for o in orders if o['shares'] != 0]

    logging.info(f"Calculated {len(final_orders)} orders.")
    logging.debug(f"Final Orders: {final_orders}")

    # Sanity check logs
    final_cash_estimate = current_cash + estimated_proceeds - estimated_buy_cost
    logging.debug(f"Initial Cash: {current_cash:.2f}, Estimated Proceeds: {estimated_proceeds:.2f}, Estimated Buy Cost: {estimated_buy_cost:.2f}, Estimated Final Cash: {final_cash_estimate:.2f}")


    return final_orders
