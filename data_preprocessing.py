
import os
import pandas as pd
import glob

# --- 1. 定义稳健的路径 ---
# 获取当前脚本文件所在的目录
# __file__ 是一个特殊变量，代表当前脚本的路径
# os.path.dirname() 获取该路径的目录部分
# os.path.abspath() 将其转换为绝对路径，以避免任何混淆
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 基于脚本目录构建项目根目录（假设脚本在项目根目录下）
# 如果你的脚本在子文件夹里，你可能需要用 os.path.join(SCRIPT_DIR, '..')
PROJECT_ROOT = SCRIPT_DIR

# 使用此根目录来定义所有其他数据路径
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
# 如果后续步骤需要，也可以在这里定义
# FINALIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'finalized')


def preprocess_data():
    """
    加载原始数据，进行预处理，并保存到processed文件夹。
    """
    # --- 2. 确保输出目录存在 ---
    print("Ensuring output directories exist...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"-> Processed data directory: {PROCESSED_DATA_DIR}")

    # --- 3. 查找所有原始CSV文件 ---
    # 使用glob模块可以方便地查找所有匹配特定模式的文件
    # os.path.join(RAW_DATA_DIR, '*.csv') 会生成像 '.../data/raw/*.csv' 这样的路径
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))

    # --- 4. 检查是否找到了文件 (这是你出错的地方) ---
    if not raw_files:
        # 这个错误信息现在可以准确地反映问题
        print(f"\n[Error] No raw data files found in '{RAW_DATA_DIR}'.")
        print("Please run data_collection.py first.")
        return # 提前退出函数

    print(f"\nFound {len(raw_files)} raw data files. Starting preprocessing...")
    
    # --- 5. 循环处理每个文件 ---
    for i, file_path in enumerate(raw_files):
        # 从完整路径中提取股票代码
        ticker = os.path.basename(file_path).replace('.csv', '')
        print(f"({i+1}/{len(raw_files)}) Processing: {ticker}")

        try:
            # 读取数据
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

            # --- 预处理步骤示例 ---
            # 1. 检查并报告缺失值
            if df.isnull().values.any():
                # print(f"  - Warning: Found missing values in {ticker}. Filling forward.")
                # 使用前一个有效值填充NaN
                df.fillna(method='ffill', inplace=True)
                # 如果填充后仍有NaN（例如第一行就是NaN），则用后一个值向后填充
                df.fillna(method='bfill', inplace=True)

            # 2. 确保数据类型正确 (yfinance通常处理得很好，但这是个好习惯)
            df = df.astype({'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64', 'Adj Close': 'float64', 'Volume': 'int64'})

            # 3. 移除交易量为0的行 (通常是节假日或错误数据)
            initial_rows = len(df)
            df = df[df['Volume'] > 0]
            if len(df) < initial_rows:
                print(f"  - Info: Removed {initial_rows - len(df)} rows with zero volume for {ticker}.")

            # 如果处理后数据为空，则跳过
            if df.empty:
                print(f"  - Warning: No data left for {ticker} after preprocessing. Skipping.")
                continue

            # --- 保存处理后的文件 ---
            output_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}.csv")
            df.to_csv(output_path)

        except Exception as e:
            print(f"  - Error processing {ticker}: {e}")

    print("\n--- Data preprocessing complete! ---")
    print(f"Processed files are saved in '{PROCESSED_DATA_DIR}'.")


if __name__ == "__main__":
    preprocess_data()

