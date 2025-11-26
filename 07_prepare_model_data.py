import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- 配置 ---
# 定义最终数据集的正确文件路径
FINAL_MASTER_DATASET_FILE = os.path.join('data', 'finalized', 'final_master_dataset.csv')

# 定义模型输入数据的输出文件夹
MODEL_INPUT_PATH = Path('data') / 'final'

def prepare_model_data():
    """
    主函数：加载主数据集，进行特征缩放，划分训练/验证/测试集，并保存。
    """
    print("--- 开始执行步骤 7: 准备模型数据 ---")

    # --- 1. 加载数据 ---
    try:
        df = pd.read_csv(FINAL_MASTER_DATASET_FILE, parse_dates=['Date'])
        print(f"成功加载 '{FINAL_MASTER_DATASET_FILE}'")
        print(f"初始数据形状: {df.shape}") # <-- 诊断信息1
    except FileNotFoundError:
        print(f"错误：找不到主数据文件 '{FINAL_MASTER_DATASET_FILE}'。")
        print("请确保已成功运行 'feature_engineering.py' 并生成了该文件。")
        return

    # --- 2. 特征选择与定义 ---
    TARGET = 'target_5d_return' 
    
    if TARGET not in df.columns:
        print(f"错误：在数据文件中找不到目标列 '{TARGET}'。")
        return

    cols_to_remove = [TARGET, 'Date', 'Ticker', 'Symbol']
    potential_feature_cols = [col for col in df.columns if col not in cols_to_remove]
    features = df[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 仅基于特征列和目标列来删除含有 NaN 的行
    print(f"在第一次dropna之前的数据形状: {df.shape}") # <-- 诊断信息2
    df.dropna(subset=features + [TARGET], inplace=True)
    print(f"在第一次dropna之后的数据形状: {df.shape}") # <-- 诊断信息3
    
    if df.empty:
        print("错误：在数据清洗后，DataFrame 变为空。请检查 'final_master_dataset.csv' 的数据质量。")
        return

    print(f"选定的数值特征数量: {len(features)}")

    # --- 3. 按股票分组进行特征缩放 (Min-Max Scaling) ---
    print("正在按股票分组对特征进行 Min-Max 缩放...")
    
    scaler = MinMaxScaler()
    
    if 'Ticker' not in df.columns:
        print("错误: 'Ticker' 列不存在，无法按股票分组进行缩放。")
        return
    
    df[features] = df.groupby('Ticker', group_keys=False)[features].apply(
        lambda x: pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)
    )
    
    # --- 关键修改：注释掉或移除第二次 dropna ---
    # 缩放后可能会产生新的NaN（如果一个组内所有值都相同），但这个操作可能过于激进。
    # 我们先注释掉它，因为第一次dropna已经保证了核心数据的完整性。
    # df.dropna(inplace=True) 
    print(f"特征缩放后，第二次dropna前的数据形状: {df.shape}") # <-- 诊断信息4
    
    print("特征缩放完成。")

    # --- 4. 划分数据集 (训练集、验证集、测试集) ---
    print("正在按时间划分数据集...")
    
    df.reset_index(drop=True, inplace=True)
    df.set_index('Date', inplace=True)
    
    times = sorted(df.index.unique())
    print(f"用于划分的唯一日期数量: {len(times)}") # <-- 诊断信息5
    
    if len(times) < 4: 
        print("错误：数据量太少，无法划分为训练/验证/测试集。")
        print("这很可能是因为在数据清洗(dropna)阶段移除了过多的行。请检查上方的'数据形状'诊断信息。")
        return

    last_15pct_date = times[-int(0.15 * len(times))]
    last_30pct_date = times[-int(0.30 * len(times))]

    train_df = df[(df.index < last_30pct_date)]
    val_df = df[(df.index >= last_30pct_date) & (df.index < last_15pct_date)]
    test_df = df[(df.index >= last_15pct_date)]

    # --- 5. 创建 X (特征) 和 y (目标) ---
    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # --- 6. 保存处理好的数据 ---
    os.makedirs(MODEL_INPUT_PATH, exist_ok=True)
    
    X_train.to_csv(MODEL_INPUT_PATH / 'X_train.csv', index=True)
    y_train.to_csv(MODEL_INPUT_PATH / 'y_train.csv', index=True, header=True)
    X_val.to_csv(MODEL_INPUT_PATH / 'X_val.csv', index=True)
    y_val.to_csv(MODEL_INPUT_PATH / 'y_val.csv', index=True, header=True)
    X_test.to_csv(MODEL_INPUT_PATH / 'X_test.csv', index=True)
    y_test.to_csv(MODEL_INPUT_PATH / 'y_test.csv', index=True, header=True)

    print(f"所有模型输入文件已保存到 '{MODEL_INPUT_PATH}'")
    print("--- 步骤 7 完成 ---")


if __name__ == '__main__':
    prepare_model_data()