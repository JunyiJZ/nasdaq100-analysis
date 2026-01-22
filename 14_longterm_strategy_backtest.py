import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yfinance as yf
import warnings

# 忽略 pandas 的一些警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
# 只读取基础清洗后的数据，不再读取预测结果，避免泄露
FILE_PATH = r"data/finalized/final_master_dataset.csv" 

# 输出文件配置
OUTPUT_CSV = "backtest_results/long_term_strategy_backtest.csv" 
OUTPUT_CHART = "backtest_results/long_term_strategy_backtest.png"

# 确保输出目录存在
os.makedirs("backtest_results", exist_ok=True)

# 回测参数
INITIAL_CAPITAL = 100000
TOP_K = 10              # 持仓数量
RISK_FREE_RATE = 0.04   # 无风险利率
TRANSACTION_COST = 0.002 # 交易成本 (双边千分之二)

# 风控配置
USE_MARKET_FILTER = True        # 启用大盘风控 (QQQ MA200)
MIN_MARKET_CAP_PERCENTILE = 0.5 # 仅交易市值排名前 50% 的股票

# ==========================================
# 2. 辅助函数
# ==========================================
def find_column_fuzzy(df, keywords):
    """模糊查找列名"""
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

def get_qqq_history(start_date, end_date):
    """下载 QQQ 历史数据用于计算均线 (大盘风控)"""
    print(f"📥 Downloading QQQ data for Market Regime Filter...")
    # 多下载一年的数据以计算 MA200
    start_buffer = start_date - pd.DateOffset(years=1)
    try:
        qqq = yf.download("QQQ", start=start_buffer, end=end_date + pd.DateOffset(days=10), progress=False)
        
        # 兼容 yfinance 新旧版本的数据结构
        if isinstance(qqq.columns, pd.MultiIndex):
            if 'Adj Close' in qqq.columns.get_level_values(0):
                qqq = qqq['Adj Close']
            elif 'Close' in qqq.columns.get_level_values(0):
                qqq = qqq['Close']
            if isinstance(qqq, pd.DataFrame) and not qqq.empty:
                qqq = qqq.iloc[:, 0]
        else:
            qqq = qqq['Adj Close'] if 'Adj Close' in qqq.columns else qqq['Close']
            
        qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
        qqq_df = pd.DataFrame({'Close': qqq})
        # 计算大盘的 200 日均线
        qqq_df['MA200'] = qqq_df['Close'].rolling(window=200).mean()
        return qqq_df
    except Exception as e:
        print(f"⚠️ Warning: QQQ download failed ({e}). Market filter disabled.")
        return pd.DataFrame()

# ==========================================
# 3. 数据加载与指标计算 (核心修改)
# ==========================================
def load_and_process_data():
    print("-" * 50)
    print("🚀 Starting LONG-TERM Backtest Pipeline (Trend + Low Volatility)")
    
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"❌ File not found: {FILE_PATH}")
    
    df = pd.read_csv(FILE_PATH)
    
    # 标准化列名
    col_map = {c: c.lower() for c in df.columns}
    if 'date' in col_map.values():
        original_date_col = list(col_map.keys())[list(col_map.values()).index('date')]
        df.rename(columns={original_date_col: 'Date'}, inplace=True)
    if 'ticker' in col_map.values():
        original_ticker_col = list(col_map.keys())[list(col_map.values()).index('ticker')]
        df.rename(columns={original_ticker_col: 'Ticker'}, inplace=True)
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 确保有 Close 价格
    close_col = find_column_fuzzy(df, ['close', 'adj']) or 'Close'
    if close_col not in df.columns:
        if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        else: raise ValueError("Could not find Close price column.")
    else:
        df['Close'] = df[close_col]

    # 处理市值 (用于过滤小盘股)
    mcap_col = find_column_fuzzy(df, ['market', 'cap'])
    df['MCap_Raw'] = df[mcap_col] if mcap_col else 1e9

    # 排序以便计算滚动指标
    df.sort_values(['Ticker', 'Date'], inplace=True)

    # =========================================================
    # 🔥 核心修改：计算长线技术指标 (完全基于历史价格，无泄露)
    # =========================================================
    print("🔄 Calculating Technical Indicators (SMA200 & Volatility)...")
    
    # 1. 个股 200 日均线 (判断长期趋势)
    df['SMA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=200).mean())
    
    # 2. 个股 60 日波动率 (判断稳定性，越低越好)
    # 计算日收益率的标准差
    df['Volatility_60'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(window=60).std())

    # 移除计算产生的 NaN (前200天无法交易)
    df.dropna(subset=['SMA_200', 'Volatility_60'], inplace=True)

    return df

# ==========================================
# 4. 策略逻辑 (按月调仓)
# ==========================================
def run_strategy(df):
    # 将日期转换为月份周期
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # 每个月取最后一天的数据作为决策点
    monthly_groups = df.sort_values('Date').groupby(['Ticker', 'YearMonth']).last().reset_index()
    unique_months = sorted(monthly_groups['YearMonth'].unique())
    
    if len(unique_months) < 2:
        print("❌ Not enough data for monthly backtest.")
        return pd.DataFrame()

    start_dt = unique_months[0].start_time
    end_dt = unique_months[-1].end_time
    
    # 获取大盘数据用于风控
    qqq_df = get_qqq_history(start_dt, end_dt)
    
    strategy_capital = INITIAL_CAPITAL
    history = [{'Date': start_dt, 'Strategy_Value': INITIAL_CAPITAL, 'In_Market': 1}]
    
    print(f"⏳ Executing Strategy over {len(unique_months)} months...")
    
    for i in range(len(unique_months) - 1):
        curr_month = unique_months[i]
        next_month = unique_months[i+1]
        decision_date = curr_month.end_time 
        
        # --- 1. 市场风控检查 (Market Regime Filter) ---
        in_market = True
        if USE_MARKET_FILTER and not qqq_df.empty:
            # 找到离决策日期最近的 QQQ 数据
            idx_loc = qqq_df.index.get_indexer([decision_date], method='nearest')[0]
            current_qqq = qqq_df.iloc[idx_loc]
            
            # 如果 QQQ 价格低于 200日均线，视为熊市，空仓
            if pd.notna(current_qqq['MA200']) and current_qqq['Close'] < current_qqq['MA200']:
                in_market = False
        
        if not in_market:
            history.append({'Date': next_month.end_time, 'Strategy_Value': strategy_capital, 'In_Market': 0})
            continue 

        # --- 2. 选股逻辑 (Trend + Low Volatility) ---
        curr_df = monthly_groups[monthly_groups['YearMonth'] == curr_month].copy()
        
        # A. 市值过滤 (只做大票)
        if 'MCap_Raw' in curr_df.columns:
            mcap_threshold = curr_df['MCap_Raw'].quantile(MIN_MARKET_CAP_PERCENTILE)
            curr_df = curr_df[curr_df['MCap_Raw'] >= mcap_threshold]
        
        # B. 趋势过滤: 股价必须在 200 日均线之上
        trend_candidates = curr_df[curr_df['Close'] > curr_df['SMA_200']].copy()
        
        # C. 优中选优: 在趋势向上的股票中，选波动率最低的 Top K
        # (低波动率通常意味着走势稳健，适合长线)
        if not trend_candidates.empty:
            top_picks = trend_candidates.sort_values('Volatility_60', ascending=True).head(TOP_K)
            selected_tickers = top_picks['Ticker'].tolist()
        else:
            selected_tickers = []
        
        # --- 3. 计算下个月收益 ---
        next_df = monthly_groups[monthly_groups['YearMonth'] == next_month]
        monthly_returns = []
        
        if not selected_tickers:
            # 如果没有选出股票，持有现金
            history.append({'Date': next_month.end_time, 'Strategy_Value': strategy_capital, 'In_Market': 0})
            continue

        for ticker in selected_tickers:
            price_curr = curr_df.loc[curr_df['Ticker'] == ticker, 'Close'].values
            price_next = next_df.loc[next_df['Ticker'] == ticker, 'Close'].values
            
            if len(price_curr) > 0 and len(price_next) > 0:
                entry_price = price_curr[0]
                exit_price = price_next[0]
                # 计算收益率并扣除交易成本
                ret = ((exit_price - entry_price) / entry_price) - TRANSACTION_COST
                monthly_returns.append(ret)
            else:
                monthly_returns.append(0.0) 
            
        if monthly_returns:
            # 等权重分配资金
            avg_return = np.mean(monthly_returns)
            strategy_capital *= (1 + avg_return)
        
        history.append({'Date': next_month.end_time, 'Strategy_Value': strategy_capital, 'In_Market': 1})

    res_df = pd.DataFrame(history)
    res_df.set_index('Date', inplace=True)
    return res_df

# ==========================================
# 5. 绩效评估与绘图
# ==========================================
def evaluate_performance(df):
    if df.empty: 
        print("❌ No results to evaluate.")
        return

    # 1. 准备基准数据 (QQQ)
    print("📊 Calculating Benchmark Performance...")
    try:
        qqq_bench = yf.download("QQQ", start=df.index[0], end=df.index[-1] + pd.DateOffset(days=5), progress=False)
        
        if isinstance(qqq_bench.columns, pd.MultiIndex):
            qqq_bench = qqq_bench['Adj Close'] if 'Adj Close' in qqq_bench.columns.get_level_values(0) else qqq_bench['Close']
            if isinstance(qqq_bench, pd.DataFrame): qqq_bench = qqq_bench.iloc[:, 0]
        else:
            qqq_bench = qqq_bench['Adj Close'] if 'Adj Close' in qqq_bench.columns else qqq_bench['Close']
            
        qqq_bench.index = pd.to_datetime(qqq_bench.index).tz_localize(None)
        
        # 对齐数据
        df['Benchmark_QQQ_Price'] = qqq_bench.reindex(df.index, method='nearest')
        initial_bench_price = df['Benchmark_QQQ_Price'].iloc[0]
        df['Benchmark_QQQ'] = (df['Benchmark_QQQ_Price'] / initial_bench_price) * INITIAL_CAPITAL
        
    except Exception as e:
        print(f"⚠️ Benchmark download failed: {e}. Plotting strategy only.")
        df['Benchmark_QQQ'] = np.nan

    # 2. 计算收益率
    df['Strat_Ret'] = df['Strategy_Value'].pct_change().fillna(0)
    df['Bench_Ret'] = df['Benchmark_QQQ'].pct_change().fillna(0)
    
    # 3. 计算夏普比率
    def calc_sharpe(returns):
        std = returns.std()
        if std == 0: return 0
        return (returns.mean() * 12 - RISK_FREE_RATE) / (std * np.sqrt(12))

    sharpe_strat = calc_sharpe(df['Strat_Ret'])
    sharpe_bench = calc_sharpe(df['Bench_Ret'])

    # 4. 其他指标
    total_ret_strat = (df['Strategy_Value'].iloc[-1] / INITIAL_CAPITAL) - 1
    total_ret_bench = (df['Benchmark_QQQ'].iloc[-1] / INITIAL_CAPITAL) - 1 if 'Benchmark_QQQ' in df else 0
    
    roll_max = df['Strategy_Value'].cummax()
    max_dd = ((df['Strategy_Value'] - roll_max) / roll_max).min()

    # 5. 打印对比报告
    print("\n" + "="*50)
    print("📊 PERFORMANCE COMPARISON (RISK-ADJUSTED)")
    print("="*50)
    print(f"{'Metric':<20} | {'Strategy':<12} | {'Benchmark (QQQ)':<15}")
    print("-" * 53)
    print(f"{'Total Return':<20} | {total_ret_strat*100:11.2f}% | {total_ret_bench*100:14.2f}%")
    print(f"{'Sharpe Ratio':<20} | {sharpe_strat:11.2f}  | {sharpe_bench:14.2f}")
    print(f"{'Max Drawdown':<20} | {max_dd*100:11.2f}% | {'--':>14}")
    print("-" * 53)
    
    # 6. 保存结果
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Milestones Met:")
    print(f"  1. Results saved to: {OUTPUT_CSV}")
    print(f"  2. Risk-adjusted comparison displayed above.")

    # 7. 绘图
    plt.figure(figsize=(12, 6))
    if 'Benchmark_QQQ' in df.columns and not df['Benchmark_QQQ'].isna().all():
        plt.plot(df.index, df['Benchmark_QQQ'], label=f'Benchmark (QQQ) Sharpe: {sharpe_bench:.2f}', color='gray', linestyle='--')
    
    plt.plot(df.index, df['Strategy_Value'], label=f'Strategy (Trend+LowVol) Sharpe: {sharpe_strat:.2f}', color='#1f77b4', linewidth=2)
    
    # 标记空仓区域
    if 'In_Market' in df.columns:
        y_min, y_max = plt.ylim()
        plt.fill_between(df.index, y_min, y_max, where=(df['In_Market'] == 0), color='red', alpha=0.1, label='Cash (Market Filter)')

    plt.yscale('log')
    plt.title('Long-Term Strategy (Trend + Low Volatility) vs QQQ')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    print(f"  3. Chart saved to: {OUTPUT_CHART}")
    plt.show()

if __name__ == "__main__":
    try:
        data = load_and_process_data()
        if data is not None and not data.empty:
            results = run_strategy(data)
            evaluate_performance(results)
        else:
            print("❌ Data loading failed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error: {e}")