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