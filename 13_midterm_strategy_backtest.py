import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 配置与数据加载 (Configuration & Data)
# ==========================================
DATA_DIR = 'data'
OUTPUT_DIR = 'tuned_models'
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001

PREDICTIONS_FILE = 'deep_learning_comparison_results.csv' 
SENTIMENT_FILE = 'sentiment_analysis_results.csv' 
PRICE_FILE = os.path.join(DATA_DIR, 'features_technical.csv')

def load_and_merge_data():
    """
    加载数据 (修复版：去除了模拟数据中的未来函数)
    """
    print("正在加载数据...")
    
    # 1. 加载真实价格
    try:
        df_price = pd.read_csv(PRICE_FILE)
        df_price.rename(columns={'date': 'Date', 'ticker': 'Ticker', 'close': 'Close'}, inplace=True)
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price = df_price[['Date', 'Ticker', 'Close']]
        print(f"成功加载价格数据，共 {len(df_price)} 行。")
    except Exception as e:
        print(f"严重错误：无法加载价格文件 {PRICE_FILE}。错误: {e}")
        return None

    # 2. 尝试加载预测数据
    try:
        df_pred = pd.read_csv(PREDICTIONS_FILE)
        
        # 检查是否是那个错误的“精度对比文件”
        if 'Accuracy' in df_pred.columns and 'Date' not in df_pred.columns:
            raise ValueError("文件格式不匹配：这是精度报告")
            
        df_pred.rename(columns={
            'date': 'Date', 'ticker': 'Ticker', 
            'prediction': 'Predicted_Return', 'predicted_return': 'Predicted_Return'
        }, inplace=True)
        
        if 'Date' not in df_pred.columns:
            raise KeyError("缺少 Date 列")

        df_pred['Date'] = pd.to_datetime(df_pred['Date'])
        print("成功加载真实预测文件。")
        
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"【提示】无法加载有效预测数据 ({e})。正在生成模拟预测数据...")
        
        # === 修复点：生成模拟预测数据 (去除未来函数) ===
        df_pred = df_price.copy()
        
        # 方案 A: 纯随机预测 (用于测试代码逻辑是否跑通，预期收益应接近 0 或负数)
        # np.random.seed(42)
        # df_pred['Predicted_Return'] = np.random.normal(0, 0.02, size=len(df_pred))
        
        # 方案 B: 基于历史动量的模拟 (更真实的模拟)
        # 假设：预测值 = 过去5天的收益率 + 噪音 (只用过去数据！)
        df_pred['Past_Return'] = df_pred.groupby('Ticker')['Close'].pct_change(5) # 过去5天
        df_pred['Predicted_Return'] = df_pred['Past_Return'] + np.random.normal(0, 0.01, size=len(df_pred))
        
        # 填补空值
        df_pred['Predicted_Return'] = df_pred['Predicted_Return'].fillna(0)
        
        df_pred = df_pred[['Date', 'Ticker', 'Predicted_Return']]
        print("模拟预测数据生成完毕 (基于历史数据，无未来函数)。")

    # 3. 加载情绪数据
    try:
        df_sent = pd.read_csv(SENTIMENT_FILE)
        df_sent.rename(columns={'date': 'Date', 'ticker': 'Ticker', 'sentiment_score': 'Sentiment_Score'}, inplace=True)
        df_sent['Date'] = pd.to_datetime(df_sent['Date'])
    except Exception:
        print(f"提示: 找不到 {SENTIMENT_FILE}，使用模拟情绪数据。")
        df_sent = df_price[['Date', 'Ticker']].copy()
        df_sent['Sentiment_Score'] = np.random.uniform(-1, 1, size=len(df_sent))

    # --- 合并数据 ---
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_sent['Date'] = pd.to_datetime(df_sent['Date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])

    df_merged = pd.merge(df_pred, df_sent, on=['Date', 'Ticker'], how='inner')
    df_merged = pd.merge(df_merged, df_price, on=['Date', 'Ticker'], how='inner')
    
    return df_merged

# ==========================================
# 2. 策略逻辑 (Strategy Logic) - 保持不变
# ==========================================
def calculate_weekly_returns(df):
    # ... (代码与您之前的一致) ...
    # 设置索引方便重采样
    df = df.set_index('Date')
    weekly_data = df.groupby('Ticker').resample('W-FRI').agg({
        'Predicted_Return': 'last',
        'Sentiment_Score': 'mean',
        'Close': 'last'
    }).dropna()
    
    # 这里的 shift(-1) 是正确的，因为我们在计算作为标签的“真实结果”
    weekly_data['Next_Week_Close'] = weekly_data.groupby('Ticker')['Close'].shift(-1)
    weekly_data['Actual_Next_Week_Return'] = (weekly_data['Next_Week_Close'] - weekly_data['Close']) / weekly_data['Close']
    
    weekly_data = weekly_data.dropna()
    return weekly_data.reset_index()

def strategy_weekly_rebalance(df_weekly, top_k=5, weight_pred=0.7, weight_sent=0.3):
    # ... (代码与您之前的一致) ...
    df_weekly['Final_Score'] = (df_weekly['Predicted_Return'] * weight_pred) + \
                               (df_weekly['Sentiment_Score'] * 0.01 * weight_sent) 

    dates = sorted(df_weekly['Date'].unique())
    portfolio_values = []
    current_capital = INITIAL_CAPITAL
    
    for date in dates:
        current_week_data = df_weekly[df_weekly['Date'] == date]
        if len(current_week_data) < top_k:
            continue
            
        picks = current_week_data.nlargest(top_k, 'Final_Score')
        avg_return = picks['Actual_Next_Week_Return'].mean()
        net_return = avg_return - (TRANSACTION_COST * 2) 
        
        profit = current_capital * net_return
        current_capital += profit
        
        portfolio_values.append({
            'Date': date,
            'Portfolio_Value': current_capital,
            'Weekly_Return': net_return
        })
        
    return pd.DataFrame(portfolio_values)

# ==========================================
# 3. 指标计算 (Metrics) - 保持不变
# ==========================================
def calculate_metrics(strategy_df):
    returns = strategy_df['Weekly_Return']
    total_return = (strategy_df['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL) - 1
    n_weeks = len(strategy_df)
    n_years = n_weeks / 52.0
    # 避免 n_years 为 0
    if n_years == 0: n_years = 1/52 
    
    cagr = (1 + total_return) ** (1 / n_years) - 1
    volatility = returns.std() * np.sqrt(52)
    
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(52)
    annualized_return = returns.mean() * 52
    
    sortino = annualized_return / downside_std if downside_std != 0 else 0
    
    return {
        'CAGR': cagr,
        'Volatility': volatility,
        'Sortino_Ratio': sortino,
        'Total_Return': total_return
    }

# ==========================================
# 4. 可视化 (Visualization) - 保持不变
# ==========================================
def plot_performance(strategy_df):
    dates = strategy_df['Date']
    n = len(dates)
    # 简单的基准模拟
    benchmark_returns = np.random.normal(0.0015, 0.02, n) 
    benchmark_equity = INITIAL_CAPITAL * np.cumprod(1 + benchmark_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df['Date'], strategy_df['Portfolio_Value'], label='Weekly Strategy', linewidth=2)
    plt.plot(strategy_df['Date'], benchmark_equity, label='Benchmark', linestyle='--', alpha=0.7)
    
    plt.title('Corrected Strategy Backtest', fontsize=16)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    df_merged = load_and_merge_data()
    
    if df_merged is not None and not df_merged.empty:
        df_weekly = calculate_weekly_returns(df_merged)
        
        if not df_weekly.empty:
            strategy_results = strategy_weekly_rebalance(df_weekly)
            
            if not strategy_results.empty:
                metrics = calculate_metrics(strategy_results)
                
                print("\n" + "="*30)
                print("Week 13 Strategy Performance (Fixed)")
                print("="*30)
                print(f"CAGR (年化收益): {metrics['CAGR']:.2%}")
                print(f"Volatility (波动率): {metrics['Volatility']:.2%}")
                print(f"Sortino Ratio: {metrics['Sortino_Ratio']:.2f}")
                print(f"Total Return: {metrics['Total_Return']:.2%}")
                print("="*30)
                
                plot_performance(strategy_results)
            else:
                print("策略未生成任何交易记录。")
        else:
            print("周频数据为空，请检查日期范围。")