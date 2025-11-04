<<<<<<< HEAD
import os
import pandas as pd
from tqdm import tqdm

# --- 1. 定义常量 ---
# (假设您的文件顶部有类似的定义，如果没有，请根据您的项目结构调整)
# 获取当前脚本所在的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 定义数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
FINAL_DATA_DIR = DATA_DIR # 保持与您之前的逻辑一致

def preprocess_and_combine_data():
    """
    加载所有原始CSV文件，进行预处理，然后合并成一个单一的DataFrame。
    """
    # 检查原始数据目录是否存在
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at '{RAW_DATA_DIR}'")
        print("Please ensure you have run the data_collection.py script first.")
        return None # 如果目录不存在，则提前退出函数

    # 获取所有原始数据文件名
    all_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]

    if not all_files:
        print(f"Error: No CSV files found in '{RAW_DATA_DIR}'.")
        print("Please ensure the data collection was successful.")
        return None # 如果没有文件，也提前退出

    print(f"Found {len(all_files)} ticker files in '{RAW_DATA_DIR}'. Starting preprocessing...")

    list_of_dfs = []
    # 使用tqdm来显示进度条
    for filename in tqdm(all_files, desc="Processing files"):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        try:
            df = pd.read_csv(file_path)

            # --- 核心预处理逻辑 ---
            # 1. 转换 'Date' 列为 datetime 对象
            df['Date'] = pd.to_datetime(df['Date'])

            # 2. 将 'Date' 设置为索引
            df.set_index('Date', inplace=True)

            # 3. 添加 'ticker' 列
            ticker = filename.split('.')[0] # 从文件名获取股票代码
            df['ticker'] = ticker

            # 4. 检查并确保所有必需的列都存在
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping {filename} due to missing columns.")
                continue

            # 5. 处理缺失值 (这里使用向前填充，您可以根据策略调整)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True) # 处理开头可能存在的NaN

            list_of_dfs.append(df)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if not list_of_dfs:
        print("No data was processed successfully. Exiting.")
        return None

    print("Combining all processed dataframes...")
    # 合并所有的DataFrame
    combined_df = pd.concat(list_of_dfs)

    # 重置索引，让 'Date' 成为普通列
    combined_df.reset_index(inplace=True)

    # 重新排列列的顺序，让 'ticker' 和 'Date' 在前面
    cols = ['ticker', 'Date'] + [col for col in combined_df.columns if col not in ['ticker', 'Date']]
    combined_df = combined_df[cols]

    print("Data combination complete.")

    # --- 7. 保存合并后的文件 ---
    # (这是从全局范围移动到这里来的代码)
    output_filename = 'all_tickers_preprocessed.csv'

    # 确保 'processed' 子文件夹存在
    processed_dir = os.path.join(FINAL_DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # 构建指向 'processed' 文件夹的完整输出路径
    output_path = os.path.join(processed_dir, output_filename)

    print(f"Saving combined and preprocessed data to: {output_path}")
    # 现在，当这行代码执行时，combined_df 肯定已经存在了
    combined_df.to_csv(output_path, index=False)
    
    # 返回创建好的 DataFrame，虽然在这个脚本里没用到，但是个好习惯
    return combined_df


if __name__ == "__main__":
    print("--- Starting data preprocessing and combination ---")
    
    # 调用主函数
    preprocess_and_combine_data()
    
    print("\n--- Data preprocessing and combination complete! ---")
    # 注意：这里的打印语句只是表示脚本执行完毕，真正的保存路径在函数内部打印
    print("Check the 'data/processed' directory for the output file.")
=======
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
>>>>>>> 1a8baaf5dd38c97fc217a897f7260f6b9dadaf9f
