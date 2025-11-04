# data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 常量定义 (Constants) ---
PROCESSED_DATA_PATH = os.path.join("data", "processed")

def plot_stock_price(ticker):
    """
    读取指定股票代码的已处理数据并绘制收盘价走势图。
    (Reads the processed data for a given stock ticker and plots its closing price history.)

    Args:
        ticker (str): 股票代码 (The stock ticker symbol, e.g., 'AAPL').
    """
    print(f"\n--- 正在为 {ticker} 生成股价走势图 ---")
    print(f"--- Generating stock price chart for {ticker} ---")
    
    file_path = os.path.join(PROCESSED_DATA_PATH, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到 {ticker} 的数据文件 '{file_path}'。")
        print("请检查 'data/processed' 文件夹下是否存在对应的 .csv 文件。")
        return
        
    try:
        # 读取数据 (Read the data)
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        if 'Close' not in df.columns:
            print(f"错误: 文件 '{file_path}' 中找不到 'Close' 列。")
            return

        # --- 使用英文进行绘图 (Plotting in English) ---
            
        # 开始绘图 (Start plotting)
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # 使用英文设置图例 (Use English for the plot legend)
        ax.plot(df.index, df['Close'], label=f'{ticker} Close Price', color='deepskyblue', linewidth=2)
        
        # 使用英文设置图表标题和标签 (Set chart title and labels in English)
        ax.set_title(f'{ticker} Historical Stock Price', fontsize=18, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Close Price (USD)', fontsize=12)
        ax.legend(fontsize=12)
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        print(f"✅ {ticker} 股价图生成完毕，即将显示...")
        print(f"✅ {ticker} chart generated successfully. Displaying now...")
        plt.show()

    except Exception as e:
        print(f"为 {ticker} 绘图时发生错误 (An error occurred while plotting for {ticker}): {e}")

# --- 主程序入口 (Main entry point) ---
if __name__ == "__main__":
    ticker_to_analyze = 'AAPL' 
    plot_stock_price(ticker_to_analyze)