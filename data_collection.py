import os
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --- 1. 获取纳斯达克100股票代码 ---
def get_nasdaq100_tickers():
    """
    从维基百科页面抓取纳斯达克100指数的成分股列表。
    使用pandas.read_html以提高解析的稳定性。

    Returns:
        list: 包含纳斯达克100股票代码的列表。如果抓取失败则返回None。
    """
    print("正在从维基百科获取最新的纳斯达克100成分股列表...")
    try:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        if table is None:
            print("错误：无法在页面上找到ID为'constituents'的表格。")
            return None

        # 使用pandas直接从HTML表格读取数据，更稳定
        df = pd.read_html(str(table))[0]
        
        if 'Ticker' in df.columns:
            tickers = df['Ticker'].dropna().tolist()
            # yfinance对于某些代码（如BRK.B, BF.B）需要'-'而不是'.'
            tickers = [str(ticker).replace('.', '-') for ticker in tickers]
            print(f"成功获取 {len(tickers)} 个股票代码。")
            return tickers
        else:
            print("错误：在表格中找不到'Ticker'列。")
            return None

    except requests.exceptions.RequestException as e:
        print(f"获取维基百科页面时发生网络错误: {e}")
        return None
    except Exception as e:
        print(f"解析维基百科页面时发生未知错误: {e}")
        return None

# --- 2. 高效下载股票数据 ---
def download_stock_data(tickers, start_date, end_date, output_dir):
    """
    高效下载指定股票列表的历史数据，并拆分为单独的CSV文件。

    Args:
        tickers (list): 股票代码列表。
        start_date (str): 数据开始日期 (YYYY-MM-DD)。
        end_date (str): 数据结束日期 (YYYY-MM-DD)。
        output_dir (str): 保存CSV文件的目录。
    """
    if not tickers:
        print("股票代码列表为空，跳过下载。")
        return

    print(f"\n--- 开始批量下载 {len(tickers)} 只股票的数据 ---")
    print(f"时间范围: {start_date} 到 {end_date}")

    os.makedirs(output_dir, exist_ok=True)
    
    # 一次性下载所有股票数据，yfinance会自动进行多线程处理，速度极快
    # progress=True 会显示一个总的进度条
    all_data = yf.download(tickers, start=start_date, end=end_date, progress=True)

    if all_data.empty:
        print("下载数据失败，返回的数据为空。请检查网络连接、股票代码或日期范围。")
        return

    print("\n--- 批量下载完成，开始将数据保存为单独的CSV文件 ---")
    
    saved_count = 0
    failed_tickers = []
    
    # 遍历最初请求的ticker列表
    for ticker in tickers:
        try:
            # 从大的DataFrame中提取单个股票的数据
            # yfinance返回的DataFrame列是MultiIndex: ('Open', 'AAPL'), ('Close', 'AAPL'), ...
            # 我们需要检查该ticker的数据是否存在于下载结果中
            if ticker in all_data.columns.get_level_values(1):
                single_stock_df = all_data.xs(ticker, level=1, axis=1)
                
                # 删除所有列都为NaN的行 (这些通常是某些股票的非交易日)
                single_stock_df.dropna(how='all', inplace=True)

                if not single_stock_df.empty:
                    output_path = os.path.join(output_dir, f"{ticker}.csv")
                    single_stock_df.to_csv(output_path)
                    saved_count += 1
                else:
                    # 下载到了数据，但数据为空（例如，只有NaN行被清除了）
                    failed_tickers.append(ticker)
            else:
                # yfinance未能下载此ticker的数据
                failed_tickers.append(ticker)

        except Exception as e:
            print(f"处理或保存 {ticker} 数据时发生错误: {e}")
            failed_tickers.append(ticker)

    print(f"\n成功将 {saved_count} 只股票的数据保存到 '{output_dir}' 文件夹中。")
    if failed_tickers:
        print(f"以下 {len(failed_tickers)} 个股票代码未能成功下载或保存数据: {', '.join(failed_tickers)}")


# --- 主执行程序 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    RAW_DATA_DIR = os.path.join('data', 'raw')
    START_DATE = "2000-01-01"
    # 动态获取当前日期作为结束日期
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # 1. 获取股票列表
    nasdaq_tickers = get_nasdaq100_tickers()

    if nasdaq_tickers:
        # 2. 高效下载所有股票的数据
        download_stock_data(
            tickers=nasdaq_tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=RAW_DATA_DIR
        )
        
        print("\n--- 所有股票数据处理完成! ---")
        data_folder_path = os.path.abspath(RAW_DATA_DIR)
        print(f"数据已保存至 '{data_folder_path}' 文件夹中。")
    else:
        print("\n未能获取股票列表，程序已终止。请检查网络连接或维基百科页面结构是否变更。")