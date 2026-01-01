import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置与参数
# ==========================================
PREDICTIONS_PATH = 'data/backtest_results/model_predictions.csv'
MASTER_DATA_PATH = 'data/finalized/final_master_dataset.csv'
RESULTS_DIR = 'data/backtest_results'

# 确保输出目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 策略参数
CONFIDENCE_THRESHOLD = 0.55  # 买入信号阈值
SELL_THRESHOLD = 0.45        # 卖出信号阈值
MAX_POSITION_PCT = 0.20      # 单只股票最大仓位
INITIAL_CAPITAL = 10000      # 初始资金

# ==========================================
# 2. 数据加载与清洗 (增强版)
# ==========================================
def load_and_prepare_data():
    print("🔍 [Step 1] 正在加载并对齐数据...")
    
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"❌ 找不到预测文件: {PREDICTIONS_PATH}")
    
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds['Date'] = pd.to_datetime(preds['Date'])
    
    # --- 修复逻辑: 确保有 Close 价格 ---
    if 'Close' not in preds.columns:
        print("⚠️ 预测数据缺少 'Close' 列，尝试从主数据集合并...")
        if not os.path.exists(MASTER_DATA_PATH):
            raise FileNotFoundError(f"❌ 缺少 Close 列且找不到主数据集: {MASTER_DATA_PATH}")
            
        master = pd.read_csv(MASTER_DATA_PATH)
        master['Date'] = pd.to_datetime(master['Date'])
        
        # 模糊匹配列名 (处理 close, Close, adj_close 等情况)
        price_col = next((c for c in master.columns if c.lower() == 'close'), None)
        if not price_col:
            raise ValueError("❌ 无法在主数据集中找到收盘价列。")
            
        # 合并价格
        price_df = master[['Date', 'Ticker', price_col]].rename(columns={price_col: 'Close'})
        preds = pd.merge(preds, price_df, on=['Date', 'Ticker'], how='left')
        print(f"✅ 已成功合并收盘价数据。")

    # --- 集成逻辑: 计算多模型平均概率 ---
    # 自动检测所有以 'Prob_' 开头的列 (例如 Prob_LSTM, Prob_GRU)
    model_cols = [c for c in preds.columns if c.startswith('Prob_')]
    
    if model_cols:
        print(f"🧠 检测到集成模型列: {model_cols}")
        preds['Probability'] = preds[model_cols].mean(axis=1)
    elif 'Probability' not in preds.columns:
        # 备用方案：寻找常见的概率列名
        candidates = ['Predicted_Probability', 'Prob', 'Confidence', 'Prediction']
        found = False
        for name in candidates:
            if name in preds.columns:
                preds = preds.rename(columns={name: 'Probability'})
                found = True
                break
        if not found:
            raise KeyError("❌ 数据中找不到预测概率列 (Probability)。")

    # 清洗无效数据
    initial_len = len(preds)
    preds = preds.dropna(subset=['Close', 'Probability'])
    if len(preds) < initial_len:
        print(f"⚠️ 移除了 {initial_len - len(preds)} 行缺失价格或概率的数据。")
        
    # 按时间排序，这对回测至关重要
    preds = preds.sort_values(['Date', 'Ticker'])
    
    print(f"✅ 数据准备完成，共 {len(preds)} 条交易信号。")
    return preds

# ==========================================
# 3. 核心回测引擎
# ==========================================
def run_backtest(df):
    print("\n🚀 [Step 2] 开始执行策略回测...")
    
    cash = INITIAL_CAPITAL
    holdings = {} # {Ticker: shares}
    portfolio_history = []
    
    # 获取所有交易日
    dates = sorted(df['Date'].unique())
    
    for current_date in dates:
        daily_data = df[df['Date'] == current_date]
        
        # 1. 卖出逻辑 (Sell Logic)
        # 检查持仓，如果预测概率下降则卖出
        tickers_to_sell = []
        for ticker, shares in holdings.items():
            row = daily_data[daily_data['Ticker'] == ticker]
            if not row.empty:
                current_price = row.iloc[0]['Close']
                current_prob = row.iloc[0]['Probability']
                
                if current_prob < SELL_THRESHOLD:
                    cash += shares * current_price
                    tickers_to_sell.append(ticker)
            else:
                # 如果当天该股票停牌或无数据，保持持仓不动
                pass
        
        for t in tickers_to_sell:
            del holdings[t]
            
        # 2. 买入逻辑 (Buy Logic)
        # 筛选出高置信度的股票
        buy_candidates = daily_data[daily_data['Probability'] > CONFIDENCE_THRESHOLD]
        # 按概率从高到低排序，优先买入确定性最高的
        buy_candidates = buy_candidates.sort_values('Probability', ascending=False)
        
        for _, row in buy_candidates.iterrows():
            ticker = row['Ticker']
            price = row['Close']
            
            # 资金管理：每只股票最多占总资金的 20%
            if ticker not in holdings and cash > 0:
                target_position_size = (cash + sum(h * daily_data.loc[daily_data['Ticker']==t, 'Close'].values[0] for t, h in holdings.items() if not daily_data[daily_data['Ticker']==t].empty)) * MAX_POSITION_PCT
                # 简化：直接用当前现金的 20% 尝试买入
                invest_amount = min(cash, INITIAL_CAPITAL * MAX_POSITION_PCT)
                
                if invest_amount > price:
                    shares_to_buy = invest_amount / price
                    holdings[ticker] = shares_to_buy
                    cash -= (shares_to_buy * price)

        # 3. 每日资产结算
        current_equity = 0
        for ticker, shares in holdings.items():
            row = daily_data[daily_data['Ticker'] == ticker]
            if not row.empty:
                current_equity += shares * row.iloc[0]['Close']
            else:
                # 如果无当日数据，暂时用上一次已知价值估算（简化处理）
                # 实际生产中应获取最近一个交易日价格
                pass 
        
        total_value = cash + current_equity
        portfolio_history.append({
            'Date': current_date,
            'Portfolio_Value': total_value
        })
    
    return pd.DataFrame(portfolio_history)

# ==========================================
# 4. 金融指标计算 (Week 13 核心任务)
# ==========================================
def calculate_financial_metrics(portfolio_df):
    print("\n📊 [Step 3] 计算金融指标 (KPIs)...")
    
    df = portfolio_df.copy()
    df['Daily_Return'] = df['Portfolio_Value'].pct_change().fillna(0)
    
    # 1. 总收益率
    total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
    
    # 2. CAGR (年化复合增长率)
    days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    years = days / 365.25
    if years > 0:
        cagr = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (1/years) - 1
    else:
        cagr = 0
        
    # 3. Volatility (年化波动率)
    # 假设一年 252 个交易日
    volatility = df['Daily_Return'].std() * np.sqrt(252)
    
    # 4. Sortino Ratio (索提诺比率)
    # 只考虑下行风险 (Downside Deviation)
    risk_free_rate = 0.04 # 假设无风险利率 4%
    daily_rf = risk_free_rate / 252
    
    downside_returns = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
    downside_std = downside_returns.std() * np.sqrt(252)
    
    # 年化收益 - 无风险利率
    excess_return = cagr - risk_free_rate
    
    if downside_std == 0:
        sortino = np.nan
    else:
        sortino = excess_return / downside_std

    # 打印报告
    print("-" * 40)
    print(f"💰 初始资金: ${INITIAL_CAPITAL}")
    print(f"💰 最终资金: ${df['Portfolio_Value'].iloc[-1]:.2f}")
    print(f"📈 总收益率: {total_return*100:.2f}%")
    print("-" * 40)
    print(f"🚀 CAGR (年化增长): {cagr*100:.2f}%")
    print(f"🌊 Volatility (波动率): {volatility*100:.2f}%")
    print(f"🛡️ Sortino Ratio: {sortino:.2f}")
    print("-" * 40)
    
    return df

# ==========================================
# 5. 可视化
# ==========================================
def plot_results(perf_df, benchmark_data):
    plt.figure(figsize=(12, 8))
    
    # 准备基准 (Market Average)
    bench = benchmark_data.groupby('Date')['Close'].mean().reset_index()
    bench = bench[bench['Date'].isin(perf_df['Date'])]
    
    # 归一化对比
    strategy_norm = perf_df['Portfolio_Value'] / perf_df['Portfolio_Value'].iloc[0]
    bench_norm = bench['Close'] / bench['Close'].iloc[0]
    
    plt.subplot(2, 1, 1)
    plt.plot(perf_df['Date'], strategy_norm, label='AI Strategy', color='#00ff00', linewidth=1.5)
    plt.plot(bench['Date'], bench_norm, label='Market Benchmark', color='gray', linestyle='--', alpha=0.7)
    plt.title('Strategy vs Market (Normalized)')
    plt.ylabel('Growth ($1 = Start)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 绘制回撤 (Drawdown)
    plt.subplot(2, 1, 2)
    rolling_max = perf_df['Portfolio_Value'].cummax()
    drawdown = (perf_df['Portfolio_Value'] - rolling_max) / rolling_max
    plt.fill_between(perf_df['Date'], drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    plt.title('Strategy Drawdown (Risk)')
    plt.ylabel('Drawdown %')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'week13_strategy_metrics.png')
    plt.savefig(save_path)
    print(f"🖼️ 图表已保存至: {save_path}")
    plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 准备数据
        data = load_and_prepare_data()
        
        # 2. 运行回测
        portfolio = run_backtest(data)
        
        if not portfolio.empty:
            # 3. 计算指标 (Week 13 重点)
            portfolio_with_metrics = calculate_financial_metrics(portfolio)
            
            # 4. 绘图
            plot_results(portfolio_with_metrics, data[['Date', 'Close']])
        else:
            print("❌ 回测生成了空的结果，请检查数据日期范围。")
            
    except Exception as e:
        print(f"\n❌ 程序崩溃: {e}")
        import traceback
        traceback.print_exc()