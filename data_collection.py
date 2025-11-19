import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os

# --- 函数定义部分 ---

def get_nasdaq100_tickers():
    """
    从维基百科页面抓取纳斯da克100指数的成分股列表。

    Returns:
        list: 包含纳斯达克100成分股代码的列表，如果抓取失败则返回None。
    """
    print("--- 步骤 1: 正在从维基百科获取NASDAQ-100成分股列表 ---")
    try:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        if table is None:
            print("❌ 错误: 未能在页面上找到ID为'constituents'的表格。")
            return None

        # 使用pandas直接从HTML表格读取数据，更稳定
        df = pd.read_html(str(table))[0]

        if 'Ticker' in df.columns:
            # 清理和格式化股票代码
            tickers = df['Ticker'].dropna().tolist()
            # 某些代码可能包含'.', 需要替换
            tickers = [str(ticker).replace('.', '-') for ticker in tickers]
            print(f"✅ 成功获取到 {len(tickers)} 个股票代码。")
            return tickers
        else:
            print("❌ 错误: 在表格中找不到名为 'Ticker' 的列。")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 错误: 网络请求失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 错误: 解析HTML时发生未知错误: {e}")
        return None

def download_stock_data(tickers, start_date, end_date, output_dir):
    """
    下载给定股票列表的历史市场数据并分别保存为CSV文件。

    Args:
        tickers (list): 股票代码列表。
        start_date (str): 数据开始日期 (YYYY-MM-DD)。
        end_date (str): 数据结束日期 (YYYY-MM-DD)。
        output_dir (str): 保存CSV文件的目录。
    """
    if not tickers:
        print("股票代码列表为空，跳过下载。")
        return

    print(f"\n--- 步骤 2: 开始下载 {len(tickers)} 只股票的历史数据 ---")
    print(f"时间范围: 从 {start_date} 到 {end_date}")

    os.makedirs(output_dir, exist_ok=True)

    # 一次性下载所有股票数据，yfinance会自动处理多线程，速度更快
    all_data = yf.download(tickers, start=start_date, end=end_date, progress=True)

    if all_data.empty:
        print("❌ 下载失败，未能获取任何数据。请检查网络连接、股票代码或日期范围。")
        return

    print(f"\n--- 下载完成，开始将数据保存为单独的CSV文件到 '{output_dir}' 文件夹 ---")

    saved_count = 0
    failed_tickers = []

    # 遍历股票代码列表
    for ticker in tickers:
        try:
            # yfinance下载的数据使用MultiIndex: ('Open', 'AAPL'), ('Close', 'AAPL'), ...
            # 我们需要检查该ticker的数据是否存在于下载的DataFrame中
            if ticker in all_data.columns.get_level_values(1):
                single_stock_df = all_data.loc[:, (slice(None), ticker)]
                single_stock_df.columns = single_stock_df.columns.droplevel(1) # 移除股票代码层级，简化列名
                
                # 删除所有值都为NaN的行（例如，股票在假期没有交易数据）
                single_stock_df.dropna(how='all', inplace=True)

                if not single_stock_df.empty:
                    output_path = os.path.join(output_dir, f"{ticker}.csv")
                    single_stock_df.to_csv(output_path)
                    saved_count += 1
                else:
                    # 下载到了数据，但全是NaN，也算失败
                    failed_tickers.append(ticker)
            else:
                # yfinance未能下载到此ticker的任何数据
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"❌ 处理或保存 {ticker} 的数据时发生错误: {e}")
            failed_tickers.append(ticker)

    print(f"\n✅ 成功将 {saved_count} 只股票的数据保存到 '{output_dir}' 文件夹中。")
    if failed_tickers:
        print(f"⚠️ 以下 {len(failed_tickers)} 只股票的数据下载或保存失败 (可能是退市或代码错误): {', '.join(failed_tickers)}")

# ==================================================================
# >>> 新增函数：获取基本面数据 <<<
# ==================================================================
def get_fundamental_data(tickers):
    """
    获取给定股票列表的基本面数据。

    Args:
        tickers (list): 股票代码列表。

    Returns:
        pd.DataFrame: 包含所有股票基本面数据的DataFrame，失败则返回None。
    """
    print("\n--- 步骤 3: 开始获取基本面数据 ---")
    all_fundamentals = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            # 打印进度
            print(f"正在获取 {ticker} ({i+1}/{len(tickers)}) 的基本面数据...")
            
            # 创建Ticker对象
            stock_info = yf.Ticker(ticker)
            
            # .info 属性返回一个包含大量基本面信息的字典
            info_dict = stock_info.info
            
            # 将股票代码添加到字典中，以便后续识别
            info_dict['ticker'] = ticker
            
            all_fundamentals.append(info_dict)
            
        except Exception as e:
            print(f"❌ 获取 {ticker} 数据失败: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\n⚠️ 以下股票的基本面数据获取失败: {', '.join(failed_tickers)}")
        
    if not all_fundamentals:
        print("❌ 未能获取任何基本面数据。")
        return None
        
    # 将字典列表转换为DataFrame
    # 由于不同股票返回的字段可能不同，直接转换可能会产生很多列和NaN值
    # 这是预期的行为，我们可以在后续数据处理步骤中选择需要的列
    return pd.DataFrame(all_fundamentals)

# --- 主执行部分 ---
if __name__ == "__main__":
    # --- 配置参数 ---
    RAW_DATA_DIR = os.path.join('data', 'raw')
    START_DATE = "2000-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # --- 流程开始 ---
    # 1. 获取股票列表
    nasdaq_tickers = get_nasdaq100_tickers()

    if nasdaq_tickers:
        # 2. 下载所有股票的历史数据
        download_stock_data(
            tickers=nasdaq_tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=RAW_DATA_DIR
        )

        # ==================================================================
        # >>> 新增调用：获取并保存基本面数据 <<<
        # ==================================================================
        # 3. 获取并保存所有股票的基本面数据
        fundamental_data = get_fundamental_data(nasdaq_tickers)
        if fundamental_data is not None:
            fundamentals_path = os.path.join(RAW_DATA_DIR, 'fundamentals.csv')
            try:
                fundamental_data.to_csv(fundamentals_path, index=False, encoding='utf-8-sig')
                print(f"\n✅ 基本面数据已成功保存至 '{fundamentals_path}'")
            except Exception as e:
                print(f"❌ 保存基本面数据到CSV时出错: {e}")
        else:
            print("\n⚠️ 未能生成基本面数据文件。")
        # ==================================================================
        
        print("\n--- 所有数据采集处理完成! ---")
    else:
        print("\n❌ 未能获取股票列表，程序已停止。请检查网络连接或维基百科页面结构是否变更。")