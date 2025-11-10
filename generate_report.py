
import pandas as pd
from pathlib import Path
import os

# --- 配置区 ---
# 定义数据目录和报告输出路径
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')
REPORT_OUTPUT_PATH = Path('data_quality_report.csv')

def generate_data_quality_report():
    """
    遍历原始数据和处理后数据，生成一份数据质量报告，
    内容包括每个股票的数据范围、缺失值比例等。
    """
    print("🚀 开始生成数据质量报告...")

    # 确保目标目录存在
    if not RAW_DATA_DIR.exists() or not PROCESSED_DATA_DIR.exists():
        print(f"❌ 错误：无法找到数据目录 {RAW_DATA_DIR} 或 {PROCESSED_DATA_DIR}。")
        print("请先运行 data_download.py 和 data_preprocessing.py。")
        return

    # 从原始数据目录获取所有股票代码
    tickers = sorted([f.stem for f in RAW_DATA_DIR.glob('*.csv')])
    if not tickers:
        print("❌ 错误：在 data/raw 目录中没有找到任何CSV文件。")
        return

    report_data = []

    # 遍历每支股票
    for ticker in tickers:
        print(f"   - 正在分析 {ticker}...")
        stock_report = {'Ticker': ticker}

        # --- 1. 分析原始数据 (Before) ---
        raw_file_path = RAW_DATA_DIR / f"{ticker}.csv"
        try:
            df_raw = pd.read_csv(raw_file_path, index_col='Date', parse_dates=True)
            
            # 计算原始数据指标
            stock_report['Raw Start Date'] = df_raw.index.min().strftime('%Y-%m-%d')
            stock_report['Raw End Date'] = df_raw.index.max().strftime('%Y-%m-%d')
            
            # 计算缺失值百分比
            total_cells = df_raw.size # df.size = rows * columns
            missing_cells = df_raw.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            stock_report['Raw Missing %'] = missing_percentage

        except FileNotFoundError:
            stock_report['Raw Start Date'] = 'N/A'
            stock_report['Raw End Date'] = 'N/A'
            stock_report['Raw Missing %'] = 'N/A'
        except Exception as e:
            print(f"      读取原始文件 {raw_file_path} 时出错: {e}")
            continue

        # --- 2. 分析处理后数据 (After) ---
        processed_file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
        try:
            df_processed = pd.read_csv(processed_file_path, index_col='Date', parse_dates=True)
            
            # 计算处理后数据指标
            stock_report['Processed Start Date'] = df_processed.index.min().strftime('%Y-%m-%d')
            stock_report['Processed End Date'] = df_processed.index.max().strftime('%Y-%m-%d')

            # 计算缺失值百分比 (理论上应为0)
            total_cells = df_processed.size
            missing_cells = df_processed.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            stock_report['Processed Missing %'] = missing_percentage

        except FileNotFoundError:
            stock_report['Processed Start Date'] = 'N/A'
            stock_report['Processed End Date'] = 'N/A'
            stock_report['Processed Missing %'] = 'N/A'
        except Exception as e:
            print(f"      读取处理后文件 {processed_file_path} 时出错: {e}")
            continue
            
        report_data.append(stock_report)

    # --- 3. 创建并保存报告 ---
    if not report_data:
        print("🤷 没有收集到任何数据来生成报告。")
        return

    # 将列表转换为DataFrame
    df_report = pd.DataFrame(report_data)

    # 调整列顺序，使其更具可读性
    column_order = [
        'Ticker', 
        'Raw Start Date', 'Raw End Date', 'Raw Missing %',
        'Processed Start Date', 'Processed End Date', 'Processed Missing %'
    ]
    df_report = df_report[column_order]

    # --- 修正部分 ---
    # 格式化百分比列，使其在CSV中更易读
    # 使用 apply 和 lambda 函数来安全地格式化，只处理数字，忽略字符串（如 'N/A'）
    df_report['Raw Missing %'] = df_report['Raw Missing %'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
    )
    df_report['Processed Missing %'] = df_report['Processed Missing %'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
    )

    # 保存到CSV文件
    df_report.to_csv(REPORT_OUTPUT_PATH, index=False)

    print("\n✅ 数据质量报告生成成功！")
    print(f"报告已保存至: {REPORT_OUTPUT_PATH.resolve()}")
    print("\n报告预览:")
    print(df_report.to_string())


if __name__ == "__main__":
 import pandas as pd
from pathlib import Path
import os

# --- 配置区 ---
# 定义数据目录和报告输出路径
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')
REPORT_OUTPUT_PATH = Path('data_quality_report.csv')

def generate_data_quality_report():
    """
    遍历原始数据和处理后数据，生成一份数据质量报告，
    内容包括每个股票的数据范围、缺失值比例等。
    """
    print("🚀 开始生成数据质量报告...")

    # 确保目标目录存在
    if not RAW_DATA_DIR.exists() or not PROCESSED_DATA_DIR.exists():
        print(f"❌ 错误：无法找到数据目录 {RAW_DATA_DIR} 或 {PROCESSED_DATA_DIR}。")
        print("请先运行 data_download.py 和 data_preprocessing.py。")
        return

    # 从原始数据目录获取所有股票代码
    tickers = sorted([f.stem for f in RAW_DATA_DIR.glob('*.csv')])
    if not tickers:
        print("❌ 错误：在 data/raw 目录中没有找到任何CSV文件。")
        return

    report_data = []

    # 遍历每支股票
    for ticker in tickers:
        print(f"   - 正在分析 {ticker}...")
        stock_report = {'Ticker': ticker}

        # --- 1. 分析原始数据 (Before) ---
        raw_file_path = RAW_DATA_DIR / f"{ticker}.csv"
        try:
            df_raw = pd.read_csv(raw_file_path, index_col='Date', parse_dates=True)
            
            # 计算原始数据指标
            stock_report['Raw Start Date'] = df_raw.index.min().strftime('%Y-%m-%d')
            stock_report['Raw End Date'] = df_raw.index.max().strftime('%Y-%m-%d')
            
            # 计算缺失值百分比
            total_cells = df_raw.size # df.size = rows * columns
            missing_cells = df_raw.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            stock_report['Raw Missing %'] = missing_percentage

        except FileNotFoundError:
            stock_report['Raw Start Date'] = 'N/A'
            stock_report['Raw End Date'] = 'N/A'
            stock_report['Raw Missing %'] = 'N/A'
        except Exception as e:
            print(f"      读取原始文件 {raw_file_path} 时出错: {e}")
            continue

        # --- 2. 分析处理后数据 (After) ---
        processed_file_path = PROCESSED_DATA_DIR / f"{ticker}.csv"
        try:
            df_processed = pd.read_csv(processed_file_path, index_col='Date', parse_dates=True)
            
            # 计算处理后数据指标
            stock_report['Processed Start Date'] = df_processed.index.min().strftime('%Y-%m-%d')
            stock_report['Processed End Date'] = df_processed.index.max().strftime('%Y-%m-%d')

            # 计算缺失值百分比 (理论上应为0)
            total_cells = df_processed.size
            missing_cells = df_processed.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            stock_report['Processed Missing %'] = missing_percentage

        except FileNotFoundError:
            stock_report['Processed Start Date'] = 'N/A'
            stock_report['Processed End Date'] = 'N/A'
            stock_report['Processed Missing %'] = 'N/A'
        except Exception as e:
            print(f"      读取处理后文件 {processed_file_path} 时出错: {e}")
            continue
            
        report_data.append(stock_report)

    # --- 3. 创建并保存报告 ---
    if not report_data:
        print("🤷 没有收集到任何数据来生成报告。")
        return

    # 将列表转换为DataFrame
    df_report = pd.DataFrame(report_data)

    # 调整列顺序，使其更具可读性
    column_order = [
        'Ticker', 
        'Raw Start Date', 'Raw End Date', 'Raw Missing %',
        'Processed Start Date', 'Processed End Date', 'Processed Missing %'
    ]
    df_report = df_report[column_order]

    # --- 修正部分 ---
    # 格式化百分比列，使其在CSV中更易读
    # 使用 apply 和 lambda 函数来安全地格式化，只处理数字，忽略字符串（如 'N/A'）
    df_report['Raw Missing %'] = df_report['Raw Missing %'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
    )
    df_report['Processed Missing %'] = df_report['Processed Missing %'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
    )

    # 保存到CSV文件
    df_report.to_csv(REPORT_OUTPUT_PATH, index=False)

    print("\n✅ 数据质量报告生成成功！")
    print(f"报告已保存至: {REPORT_OUTPUT_PATH.resolve()}")
    print("\n报告预览:")
    print(df_report.to_string())


if __name__ == "__main__":

    generate_data_quality_report()