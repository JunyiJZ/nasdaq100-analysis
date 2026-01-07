import pandas as pd
import os
from pathlib import Path

# --- 配置路径 ---
DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FINAL_DIR = DATA_DIR / 'finalized'

# 输出文件
OUTPUT_FILE = FINAL_DIR / 'final_master_dataset.csv'

def standardize_columns(df, source_name="Unknown"):
    """
    统一列名，处理大小写问题。
    """
    # 0. 去除列名两端的空格
    df.columns = [str(c).strip() for c in df.columns]

    # 1. 尝试识别各种变体的 Date
    if 'Date' not in df.columns:
        for col in df.columns:
            if col.lower() in ['date', 'time', 'timestamp', 'datetime']:
                print(f"  -> 在 {source_name} 中将 '{col}' 重命名为 'Date'")
                df.rename(columns={col: 'Date'}, inplace=True)
                break
    
    # 2. 解决 ticker -> Ticker (这是你之前遇到问题的关键)
    if 'Ticker' not in df.columns:
        for col in df.columns:
            if col.lower() in ['ticker', 'symbol', 'code']:
                print(f"  -> 在 {source_name} 中将 '{col}' 重命名为 'Ticker'")
                df.rename(columns={col: 'Ticker'}, inplace=True)
                break
    
    # 3. 确保 Date 是 datetime 类型
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    return df

def load_price_data():
    """加载价格/技术指标数据"""
    potential_files = [
        DATA_DIR / 'features_technical.csv', 
        DATA_DIR / 'processed' / 'technical_indicators.csv'
    ]
    
    target_file = None
    for f in potential_files:
        if f.exists():
            target_file = f
            break
            
    if target_file:
        print(f"✅ 找到价格/特征数据: {target_file}")
        df = pd.read_csv(target_file)
        df = standardize_columns(df, "Price Data")
        return df
    return None

def load_sentiment_data():
    sent_file = PROCESSED_DIR / 'sentiment_scores.csv'
    if not sent_file.exists():
        return None
    
    print(f"✅ 加载情感数据: {sent_file}")
    df = pd.read_csv(sent_file)
    df = standardize_columns(df, "Sentiment Data")
    return df

def load_fundamental_data():
    fund_file = RAW_DIR / 'fundamentals.csv' 
    if not fund_file.exists():
        print(f"⚠️ 警告: 找不到基本面数据 {fund_file}")
        return None

    print(f"✅ 加载基本面数据: {fund_file}")
    try:
        df = pd.read_csv(fund_file)
        df = standardize_columns(df, "Fundamental Data")
        
        # 筛选一些有用的列，防止数据表过大包含无用信息（如地址、电话等）
        # 如果你想保留所有列，可以注释掉下面这段
        useful_keywords = [
            'Ticker', 'Date', 'sector', 'industry', 'marketCap', 'trailingPE', 
            'forwardPE', 'bookValue', 'priceToBook', 'trailingEps', 'forwardEps',
            'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow', 'averageVolume',
            'profitMargins', 'revenueGrowth', 'operatingMargins'
        ]
        # 找出df中存在的且包含在useful_keywords里的列，或者是Ticker/Date
        cols_to_keep = [c for c in df.columns if c in useful_keywords or c in ['Ticker', 'Date']]
        
        # 如果筛选后列太少（说明列名可能不匹配），就保留所有数值型列
        if len(cols_to_keep) < 3:
            print("  -> 未能自动筛选核心基本面列，将保留所有列。")
        else:
            print(f"  -> 筛选出 {len(cols_to_keep)} 个核心基本面特征。")
            df = df[cols_to_keep]

        return df
    except Exception as e:
        print(f"❌ 读取基本面数据失败: {e}")
        return None

def merge_datasets():
    print("--- 开始数据合并 (Data Integration) ---")
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 1. 加载主数据
    df_main = load_price_data()
    if df_main is None: 
        print("❌ 无法找到主价格数据，程序终止。")
        return
    
    # 2. 合并情感数据
    df_sent = load_sentiment_data()
    if df_sent is not None:
        merge_cols = ['Date']
        if 'Ticker' in df_sent.columns and 'Ticker' in df_main.columns:
            merge_cols = ['Date', 'Ticker']
        
        print(f"正在合并情感数据 (Keys: {merge_cols})...")
        df_main = pd.merge(df_main, df_sent, on=merge_cols, how='left')
        if 'Sentiment_Score' in df_main.columns:
            df_main['Sentiment_Score'] = df_main['Sentiment_Score'].fillna(0)

    # 3. 合并基本面数据 (修复版逻辑)
    df_fund = load_fundamental_data()
    
    if df_fund is not None:
        if 'Ticker' in df_fund.columns:
            # 判断合并策略
            if 'Date' in df_fund.columns:
                print("ℹ️ 检测到历史基本面数据 (含 Date)，执行 [Date, Ticker] 精确合并...")
                df_main['Date'] = pd.to_datetime(df_main['Date'])
                df_fund['Date'] = pd.to_datetime(df_fund['Date'])
                df_main = pd.merge(df_main, df_fund, on=['Date', 'Ticker'], how='left')
                # 历史数据通常需要向下填充 (ffill)
                df_main.groupby('Ticker').ffill(inplace=True)
            else:
                print("ℹ️ 检测到静态基本面数据 (无 Date)，执行 [Ticker] 广播合并...")
                # 这种合并会将基本面信息复制到该 Ticker 的每一行
                df_main = pd.merge(df_main, df_fund, on=['Ticker'], how='left')
        else:
            print("⚠️ 基本面数据缺少 'Ticker' 列，无法合并。")
    else:
        print("ℹ️ 本次运行未包含基本面数据。")

    # 4. 生成目标变量
    if 'target_5d_return' not in df_main.columns:
        print("ℹ️ 计算目标变量: target_5d_return")
        df_main.sort_values(['Ticker', 'Date'], inplace=True)
        # 确保 Close 是数值型
        df_main['Close'] = pd.to_numeric(df_main['Close'], errors='coerce')
        df_main['target_5d_return'] = df_main.groupby('Ticker')['Close'].transform(lambda x: x.shift(-5) / x - 1)

    # 5. 清洗
    initial_len = len(df_main)
    df_main.dropna(subset=['target_5d_return'], inplace=True) 
    
    # 填充缺失值（数值型填0，非数值型填Unknown）
    num_cols = df_main.select_dtypes(include=['number']).columns
    df_main[num_cols] = df_main[num_cols].fillna(0)
    
    print(f"合并完成。原始行数: {initial_len}, 清洗后行数: {len(df_main)}")
    
    # 6. 保存
    df_main.to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 最终主数据集已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_datasets()