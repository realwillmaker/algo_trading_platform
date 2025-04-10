import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import os
import json
import time         # <--- Add if not present
import yfinance as yf # <--- Add if not present

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- S&P 500 Ticker Fetching ---
def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    logging.info("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('table', {'id': 'constituents'})
        tickers = []
        for row in table.findAll('tr')[1:]: # Skip header row
            ticker = row.findAll('td')[0].text.strip()
            # Replace '.' with '-' for tickers like BRK.B -> BRK-B (common in Yahoo Finance)
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)
        logging.info(f"Fetched {len(tickers)} S&P 500 tickers.")
        # Also include the benchmark for data download
        if config.BENCHMARK_TICKER not in tickers:
             tickers.append(config.BENCHMARK_TICKER)
        if 'VIX' not in config.MACRO_FEATURES.values(): # Add VIX ticker if not already macro
             vix_ticker = config.MACRO_FEATURES.get('VIX')
             if vix_ticker and vix_ticker not in tickers:
                  tickers.append(vix_ticker)

        return sorted(list(set(tickers))) # Return unique sorted list
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request error fetching S&P 500 list: {e}")
    except Exception as e:
        logging.error(f"Error parsing S&P 500 list: {e}")
    # Fallback or raise error if fetch fails
    logging.warning("Could not fetch S&P 500 list from Wikipedia. Using fallback or failing.")
    # return config.S_AND_P_500_MANUAL_LIST # Or raise an error
    raise RuntimeError("Failed to get S&P 500 ticker list.")


# --- Data Loading ---
def load_data_from_file(ticker, data_dir=config.DATA_DIR):
    """Loads historical data for a ticker from a parquet file."""
    filepath = os.path.join(data_dir, f"{ticker}.parquet")
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    else:
        logging.warning(f"Data file not found for {ticker} at {filepath}")
        return None

def save_data_to_file(df, ticker, data_dir=config.DATA_DIR):
    """Saves historical data for a ticker to a parquet file."""
    filepath = os.path.join(data_dir, f"{ticker}.parquet")
    try:
        df.to_parquet(filepath)
        logging.debug(f"Saved data for {ticker} to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data for {ticker} to {filepath}: {e}")

# --- Other Utilities can be added here ---
# E.g., Functions for specific data provider interactions, logging setup, etc.

def save_portfolio_state(state_dict, filepath=config.PORTFOLIO_STATE_FILE):
    """Saves the portfolio state (cash, positions) to a JSON file."""
    logging.info(f"Saving portfolio state to {filepath}")
    try:
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=4)
        logging.debug(f"State saved: {state_dict}")
    except Exception as e:
        logging.error(f"Failed to save portfolio state to {filepath}: {e}", exc_info=True)

def load_portfolio_state(filepath=config.PORTFOLIO_STATE_FILE):
    """
    Loads the portfolio state from a JSON file.
    If the file doesn't exist, initializes with default values.
    """
    logging.info(f"Loading portfolio state from {filepath}")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                # Basic validation
                if 'cash' in state and 'positions' in state:
                    logging.info("Portfolio state loaded successfully.")
                    logging.debug(f"Loaded state: {state}")
                    # Ensure positions keys are strings if loaded from JSON nums
                    state['positions'] = {str(k): int(v) for k, v in state.get('positions', {}).items()}
                    return state
                else:
                    logging.warning(f"Portfolio state file {filepath} is missing required keys ('cash', 'positions'). Initializing.")
                    return {'cash': config.INITIAL_CAPITAL, 'positions': {}}
        except Exception as e:
            logging.error(f"Failed to load or parse portfolio state from {filepath}: {e}. Initializing.", exc_info=True)
            return {'cash': config.INITIAL_CAPITAL, 'positions': {}}
    else:
        logging.warning(f"Portfolio state file {filepath} not found. Initializing with default capital.")
        return {'cash': config.INITIAL_CAPITAL, 'positions': {}} # Initial state

def get_market_caps(tickers, delay=config.MARKET_CAP_FETCH_DELAY):
    """
    Fetches current market capitalization for a list of tickers using yfinance.

    Args:
        tickers (list): List of stock ticker symbols.
        delay (float): Seconds to wait between API calls to avoid rate limits.

    Returns:
        dict: Dictionary mapping ticker symbols to their market cap (or 0 if unavailable/error).
    """
    # Check if config has the attribute, provide default if missing
    fetch_delay = getattr(config, 'MARKET_CAP_FETCH_DELAY', 0.5) # Default 0.5s

    logging.info(f"Fetching market caps for {len(tickers)} tickers...")
    market_caps = {}
    count = 0
    total = len(tickers)
    for ticker in tickers:
        count += 1
        # Add check for valid ticker format
        if not isinstance(ticker, str) or not ticker or ticker.startswith("^"):
            logging.debug(f"Skipping market cap fetch for invalid/index ticker: {ticker}")
            market_caps[ticker] = 0
            continue

        logging.debug(f"Fetching market cap for {ticker} ({count}/{total})")
        try:
            stock_info = yf.Ticker(ticker).info
            cap = stock_info.get('marketCap', 0)
            if cap is None: cap = 0
            market_caps[ticker] = int(cap)
            time.sleep(fetch_delay) # Use the fetched or default delay
        except Exception as e:
            logging.warning(f"Could not fetch info/market cap for {ticker}: {e}")
            market_caps[ticker] = 0
    logging.info(f"Finished fetching market caps. Found caps for {sum(1 for cap in market_caps.values() if cap > 0)} tickers.")
    return market_caps

if __name__ == '__main__':
    # Example usage:
    tickers = get_sp500_tickers()
    print("Fetched S&P 500 Tickers:")
    print(tickers[:10], "...") # Print first 10
    print(f"Total: {len(tickers)}")
