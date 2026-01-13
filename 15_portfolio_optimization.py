import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import os

# ================= 配置区域 =================
FILE_PATH = 'data/finalized/all_ohlcv_data.csv'
PORTFOLIO_VALUE = 100000 
RISK_FREE_RATE = 0.02  # 统一设置无风险利率 (2%)
SAVE_DIR = 'data/backtest_results' # 结果保存目录
# ===========================================

plt.style.use('seaborn-v0_8-darkgrid')

def load_and_process_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到文件 {csv_path}")
        return None

    print(f"📂 正在读取数据: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        else:
            price_col = 'Close'

        prices = df.pivot(index='Date', columns='Ticker', values=price_col)
        prices = prices.dropna(axis=0, how='all')
        prices = prices.dropna(axis=1, how='any')
        
        return prices
    except Exception as e:
        print(f"❌ 数据处理发生错误: {e}")
        return None

def optimize_portfolio(prices):
    print("\n🚀 开始计算最优投资组合 (Max Sharpe)...")
    
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S)
    
    try:
        # 使用统一的无风险利率
        raw_weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        cleaned_weights = ef.clean_weights()
        
        print("\n📊 预期投资组合表现:")
        # 修复警告：在这里也传入 risk_free_rate
        ef.portfolio_performance(verbose=True, risk_free_rate=RISK_FREE_RATE)
        
        return cleaned_weights
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        return None

def save_results(allocation, leftover, weights):
    """
    保存具体的持仓建议到 CSV
    """
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # 1. 保存持仓数量
    df_alloc = pd.DataFrame(list(allocation.items()), columns=['Ticker', 'Shares'])
    df_alloc['Type'] = 'Stock'
    
    # 添加剩余现金行
    new_row = pd.DataFrame([{'Ticker': 'CASH', 'Shares': leftover, 'Type': 'Cash'}])
    df_alloc = pd.concat([df_alloc, new_row], ignore_index=True)
    
    save_path = os.path.join(SAVE_DIR, 'optimized_portfolio_allocation.csv')
    df_alloc.to_csv(save_path, index=False)
    print(f"\n💾 持仓结果已保存至: {save_path}")

    # 2. 保存权重配置 (用于后续分析)
    df_weights = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
    df_weights = df_weights[df_weights['Weight'] > 0].sort_values(by='Weight', ascending=False)
    weight_path = os.path.join(SAVE_DIR, 'optimized_weights.csv')
    df_weights.to_csv(weight_path, index=False)
    print(f"💾 权重配置已保存至: {weight_path}")

def calculate_allocation(cleaned_weights, prices, total_value):
    print(f"\n💰 计算具体持仓 (总资金: ${total_value})...")
    
    latest_prices = get_latest_prices(prices)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value)
    allocation, leftover = da.greedy_portfolio()
    
    print("\n📋 建议购买清单 (Top 5):")
    sorted_alloc = sorted(allocation.items(), key=lambda x: x[1] * latest_prices[x[0]], reverse=True)
    for ticker, num in sorted_alloc[:5]:
        print(f"  - {ticker}: {num} 股")
    print(f"  ... (完整列表已保存)")
    print(f"💵 剩余现金: ${leftover:.2f}")
    
    return allocation, leftover

def plot_weights(weights):
    filtered_weights = {k: v for k, v in weights.items() if v > 0}
    if not filtered_weights: return

    sorted_weights = dict(sorted(filtered_weights.items(), key=lambda item: item[1], reverse=True))
    labels = list(sorted_weights.keys())
    sizes = list(sorted_weights.values())
    
    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(f'Optimized Portfolio Allocation (Max Sharpe)\nRisk Free Rate: {RISK_FREE_RATE*100}%', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prices_df = load_and_process_data(FILE_PATH)
    
    if prices_df is not None:
        weights = optimize_portfolio(prices_df)
        
        if weights:
            allocation, leftover = calculate_allocation(weights, prices_df, PORTFOLIO_VALUE)
            
            # 保存结果
            save_results(allocation, leftover, weights)
            
            # 绘图
            plot_weights(weights)