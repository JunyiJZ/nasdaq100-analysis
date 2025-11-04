<<<<<<< HEAD
# main.py
import os
from data_collection import fetch_stock_data, get_nasdaq100_tickers
from data_preprocessing import preprocess_data # <--- 修正 #1: 导入正确的函数名
from data_analysis import plot_stock_price

def main():
    """
    主函数，协调整个数据分析流程。
    """
    # --- 1. 数据采集 ---
    # 定义要分析的股票列表
    # 您可以从纳斯达克100列表中选择，或者手动指定
    # tickers = get_nasdaq100_tickers() # 如果想获取所有100支股票，取消此行注释
    tickers = ['AAPL', 'MSFT', 'GOOGL'] # 我们先从几支代表性的股票开始
    
    print("--- 开始数据采集 ---")
    # 确保数据目录存在
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    for ticker in tickers:
        fetch_stock_data(ticker)
    print("--- 数据采集完成 ---\n")

    # --- 2. 数据预处理 ---
    print("--- 开始数据预处理 ---")
    # 确保处理后数据的目录存在
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    for ticker in tickers:
        # <--- 修正 #2: 调用正确的函数，并传入ticker参数
        preprocess_data(ticker) 
    print("--- 数据预处理完成 ---\n")

    # --- 3. 数据分析与可视化 ---
    print("--- 开始数据分析与可视化 ---")
    # 我们以苹果公司（AAPL）为例进行绘图
    target_ticker = 'AAPL'
    print(f"准备为 {target_ticker} 生成股价图...")
    plot_stock_price(target_ticker)
    print(f"--- {target_ticker} 分析完成 ---")


if __name__ == "__main__":
=======
# main.py
import os
from data_collection import fetch_stock_data, get_nasdaq100_tickers
from data_preprocessing import preprocess_data # <--- 修正 #1: 导入正确的函数名
from data_analysis import plot_stock_price

def main():
    """
    主函数，协调整个数据分析流程。
    """
    # --- 1. 数据采集 ---
    # 定义要分析的股票列表
    # 您可以从纳斯达克100列表中选择，或者手动指定
    # tickers = get_nasdaq100_tickers() # 如果想获取所有100支股票，取消此行注释
    tickers = ['AAPL', 'MSFT', 'GOOGL'] # 我们先从几支代表性的股票开始
    
    print("--- 开始数据采集 ---")
    # 确保数据目录存在
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    for ticker in tickers:
        fetch_stock_data(ticker)
    print("--- 数据采集完成 ---\n")

    # --- 2. 数据预处理 ---
    print("--- 开始数据预处理 ---")
    # 确保处理后数据的目录存在
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    for ticker in tickers:
        # <--- 修正 #2: 调用正确的函数，并传入ticker参数
        preprocess_data(ticker) 
    print("--- 数据预处理完成 ---\n")

    # --- 3. 数据分析与可视化 ---
    print("--- 开始数据分析与可视化 ---")
    # 我们以苹果公司（AAPL）为例进行绘图
    target_ticker = 'AAPL'
    print(f"准备为 {target_ticker} 生成股价图...")
    plot_stock_price(target_ticker)
    print(f"--- {target_ticker} 分析完成 ---")


if __name__ == "__main__":
>>>>>>> 1a8baaf5dd38c97fc217a897f7260f6b9dadaf9f
    main()