import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

NASDAQ_100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'COST',
    'PEP', 'ADBE', 'CSCO', 'TMUS', 'NFLX', 'AMD', 'INTC', 'CMCSA', 'QCOM', 'INTU',
    'TXN', 'AMGN', 'HON', 'AMAT', 'ISRG', 'BKNG', 'SBUX', 'VRTX', 'GILD', 'ADP',
    'MDLZ', 'ADI', 'REGN', 'PYPL', 'LRCX', 'CSX', 'MAR', 'MU', 'MELI', 'PANW',
    'SNPS', 'KLAC', 'CDNS', 'ASML', 'MNST', 'ORLY', 'AEP', 'CTAS', 'CHTR', 'FTNT',
    'PCAR', 'DXCM', 'MRNA', 'ABNB', 'PAYX', 'EXC', 'KDP', 'AZN', 'LULU', 'BIIB',
    'WDAY', 'ROST', 'IDXX', 'ODFL', 'CPRT', 'KHC', 'FAST', 'CSGP', 'XEL', 'MCHP',
    'WBA', 'BKR', 'DLTR', 'MRVL', 'ON', 'CTSH', 'ADSK', 'EA', 'VRSK', 'CEG',
    'CRWD', 'TEAM', 'DDOG', 'ZS', 'SIRI', 'GEHC', 'ILMN', 'PDD', 'JD', 'NTES',
    'BIDU', 'BABA', 'ZM', 'CRWD', 'TEAM', 'DDOG', 'ZS', 'SIRI'
]
NASDAQ_100_TICKERS = sorted(list(set(NASDAQ_100_TICKERS)))

DATA_PATH = "data"
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")

DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2024-01-01"

desired_fundamentals = {
    'longName': 'Company Name',
    'sector': 'Sector',
    'industry': 'Industry',
    'country': 'Country',
    'marketCap': 'Market Cap',
    'trailingPE': 'Trailing P/E',
    'forwardPE': 'Forward P/E',
    'dividendYield': 'Dividend Yield',
    'beta': 'Beta',
    'enterpriseValue': 'Enterprise Value',
    'bookValue': 'Book Value'
}

def create_directories():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

def download_and_process_ohlcv(ticker):
    data = yf.download(
        ticker,
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by='column'
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def fetch_ohlcv_data():
    for ticker in tqdm(NASDAQ_100_TICKERS):
        try:
            ohlcv_data = download_and_process_ohlcv(ticker)

            if ohlcv_data.empty:
                continue

            file_path = os.path.join(RAW_DATA_PATH, f"{ticker}_ohlcv.csv")
            ohlcv_data.to_csv(file_path)

        except Exception:
            pass

def fetch_fundamental_data():
    fundamentals_data = []
    for ticker_str in tqdm(NASDAQ_100_TICKERS):
        try:
            ticker_obj = yf.Ticker(ticker_str)
            info = ticker_obj.info
            
            fundamental_dict = {'Ticker': ticker_str}
            for key, name in desired_fundamentals.items():
                fundamental_dict[name] = info.get(key, 'N/A')
            fundamentals_data.append(fundamental_dict)

        except Exception as e:
            fundamentals_data.append({'Ticker': ticker_str, 'Error': str(e)})

    if fundamentals_data:
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        column_order = ['Ticker'] + list(desired_fundamentals.values())
        existing_columns = [col for col in column_order if col in fundamentals_df.columns]
        fundamentals_df = fundamentals_df[existing_columns]

        fundamentals_file_path = os.path.join(DATA_PATH, "nasdaq100_fundamentals.csv")
        fundamentals_df.to_csv(fundamentals_file_path, index=False, encoding='utf-8-sig')

def main():
    create_directories()
    fetch_ohlcv_data()
    fetch_fundamental_data()

if __name__ == "__main__":
    main()