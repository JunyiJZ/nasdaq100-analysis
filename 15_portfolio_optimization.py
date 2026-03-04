import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import os

# ==========================================
# 1. 配置与路径设置
# ==========================================
IMG_DIR = 'images'

# 根据你的文件结构
FILE_SHORT = os.path.join('backtest_results', 'backtest_shortterm.csv') 
FILE_MID = '13_midterm_strategy_backtest.csv'
FILE_LONG = 'long_term_backtest.csv'

OUTPUT_WEIGHTS = 'optimized_portfolio_allocation.csv'
OUTPUT_CHART_FRONTIER = os.path.join(IMG_DIR, 'efficient_frontier.png')
OUTPUT_CHART_CORR = os.path.join(IMG_DIR, 'strategy_correlation.png')

os.makedirs(IMG_DIR, exist_ok=True)

# ==========================================
# 2. 数据加载函数
# ==========================================
def load_strategy_curve(filepath, strategy_name):
    print(f"Checking {strategy_name} file: {filepath} ...", end=" ")
    
    if not os.path.exists(filepath):
        print(f"⚠️ Not Found (Skipping)")
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # 1. 检查是否是汇总表
        if len(df) < 10:
            print(f"❌ File looks like a summary report (Rows={len(df)}). Skipping.")
            return None

        # 2. 处理日期索引
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        # 3. 强制移除时区 (关键)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 4. 寻找净值列
        target_col = None
        candidates = ['Portfolio_Value', 'Strategy_Value', 'Total_Value', 'Close', 'Equity', 'algorithm_period_return']
        
        for col in candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # 模糊匹配
            for col in df.columns:
                if 'value' in col.lower() or 'equity' in col.lower():
                    target_col = col
                    break

        if target_col is None:
            print(f"❌ No suitable value column found. Columns: {list(df.columns)}")
            return None
            
        print(f"✅ Loaded. (Rows: {len(df)}, Freq: {'Daily' if len(df)>1000 else 'Monthly/Weekly?'})")
        
        series = df[target_col].copy()
        series.name = strategy_name
        
        # 归一化：从1.0开始
        series = series / series.iloc[0]
        
        # 删除重复索引
        series = series[~series.index.duplicated(keep='first')]
        
        return series

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

# ==========================================
# 3. MPT 优化逻辑 (已修改为收益优先)
# ==========================================
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# 新增：以收益为主，风险为辅的目标函数
def objective_max_return_penalty(weights, mean_returns, cov_matrix, risk_aversion):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    # 目标：最大化 (收益 - 风险惩罚)
    # 因为 scipy.minimize 求的是最小值，所以我们返回负值
    return -(p_ret - risk_aversion * p_var)

# 新增：调用新的目标函数进行优化
def optimize_for_return(mean_returns, cov_matrix, risk_aversion=0.5):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_aversion)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    init_guess = num_assets * [1./num_assets,]
    
    result = sco.minimize(objective_max_return_penalty, init_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# ==========================================
# 4. 主程序
# ==========================================
def run_portfolio_optimization():
    print("--- Starting Portfolio Optimization (v3 Auto-Resample) ---")

    strategies = {}
    
    # 加载策略
    s1 = load_strategy_curve(FILE_SHORT, 'Short-Term')
    if s1 is not None: strategies['Short-Term'] = s1
    
    s2 = load_strategy_curve(FILE_MID, 'Mid-Term')
    if s2 is not None: strategies['Mid-Term'] = s2
    
    s3 = load_strategy_curve(FILE_LONG, 'Long-Term')
    if s3 is not None: strategies['Long-Term'] = s3

    if len(strategies) < 2:
        print("\n❌ Not enough valid strategies loaded (Need at least 2).")
        return

    print(f"\n✅ Strategies ready: {list(strategies.keys())}")

    # --- 核心修复：智能对齐日期 ---
    print("🔄 Aligning dates and resampling to Daily frequency...")
    
    # 1. 找到所有策略的共同时间段
    start_date = max([s.index.min() for s in strategies.values()])
    end_date = min([s.index.max() for s in strategies.values()])
    
    print(f"   Common Time Range: {start_date.date()} to {end_date.date()}")
    
    if start_date >= end_date:
        print("❌ Error: Start date is after End date. No overlap.")
        return

    # 2. 创建一个标准的交易日索引 (Business Days)
    common_index = pd.date_range(start=start_date, end=end_date, freq='B')
    
    aligned_data = {}
    for name, s in strategies.items():
        # 3. 重新索引并前向填充 (ffill)
        aligned_s = s.reindex(common_index, method='ffill')
        aligned_data[name] = aligned_s

    df_combined = pd.DataFrame(aligned_data).dropna()
    print(f"✅ Aligned Data Points: {len(df_combined)} days")

    if len(df_combined) < 10:
        print("❌ Error: Still not enough overlapping data after resampling.")
        return
    
    # 计算日收益率
    returns = df_combined.pct_change().dropna()
    
    # 简单的异常值清洗
    returns = returns[np.abs(returns) < 0.5] 

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # 相关性热力图
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Strategy Correlation Matrix')
        plt.tight_layout()
        plt.savefig(OUTPUT_CHART_CORR)
        print(f"📊 Correlation chart saved: {OUTPUT_CHART_CORR}")
    except Exception as e:
        print(f"⚠️ Could not save correlation chart: {e}")

    # 优化：使用新的优化函数
    print("\n--- Optimizing (Target: Maximize Return with Risk Penalty) ---")
    
    # 这里的 risk_aversion 参数决定了风险的惩罚力度。
    # 设为 0.2 表示非常看重收益，不太在乎风险。
    # 设为 1.0 表示收益和风险同等重要。
    # 你可以根据需要调整这个值 (例如 0.1 到 0.5 之间)
    RISK_AVERSION_COEF = 1.0 
    
    result = optimize_for_return(mean_returns, cov_matrix, risk_aversion=RISK_AVERSION_COEF)
    optimal_weights = result.x
    
    allocation = pd.DataFrame({
        'Strategy': returns.columns,
        'Weight': np.round(optimal_weights, 4),
        'Percentage': np.round(optimal_weights * 100, 2)
    })
    
    print(f"\n🏆 Optimal Portfolio Allocation (Risk Aversion = {RISK_AVERSION_COEF}):")
    print(allocation)
    allocation.to_csv(OUTPUT_WEIGHTS, index=False)

    # 有效前沿图
    print("\n--- Generating Efficient Frontier ---")
    num_assets = len(strategies)
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = p_std
        results[1,i] = p_ret
        # 颜色映射依然保留夏普比率作为参考
        results[2,i] = (p_ret - 0.02) / p_std if p_std != 0 else 0

    opt_std, opt_ret = portfolio_annualised_performance(optimal_weights, mean_returns, cov_matrix)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', s=10, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio (Reference)')
    
    # 标记新的最优点
    plt.scatter(opt_std, opt_ret, marker='*', color='red', s=250, label='Optimal Portfolio (Return Focused)')
    
    plt.title('Efficient Frontier (Max Return Objective)')
    plt.xlabel('Risk (Volatility)')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_CHART_FRONTIER)
    print(f"📈 Frontier chart saved: {OUTPUT_CHART_FRONTIER}")
    
    print("\n✅ Optimization Complete!")

if __name__ == "__main__":
    run_portfolio_optimization()