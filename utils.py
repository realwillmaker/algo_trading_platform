import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging

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

if __name__ == '__main__':
    # Example usage:
    tickers = get_sp500_tickers()
    print("Fetched S&P 500 Tickers:")
    print(tickers[:10], "...") # Print first 10
    print(f"Total: {len(tickers)}")
