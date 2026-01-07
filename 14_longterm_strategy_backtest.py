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
FILE_PATH = r"data/finalized/final_master_dataset.csv" 
PREDICTIONS_PATH = r"data/backtest_results/model_predictions.csv"

# [修正 1] 文件名严格匹配图片要求
OUTPUT_CSV = "long_term_backtest.csv" 
OUTPUT_CHART = "strategy_performance.png"

# 回测参数
INITIAL_CAPITAL = 100000
TOP_K = 10
RISK_FREE_RATE = 0.04
TRANSACTION_COST = 0.002 

# 风控配置
STRICT_LAG_MODE = True          
USE_MARKET_FILTER = True        
MIN_MARKET_CAP_PERCENTILE = 0.5 

# ==========================================
# 2. 辅助函数 (保持不变)
# ==========================================
def find_column_fuzzy(df, keywords):
    for col in df.columns:
        if all(k.lower() in col.lower() for k in keywords):
            return col
    return None

def get_qqq_history(start_date, end_date):
    print(f"📥 Downloading QQQ data for Market Regime Filter...")
    start_buffer = start_date - pd.DateOffset(years=1)
    try:
        qqq = yf.download("QQQ", start=start_buffer, end=end_date + pd.DateOffset(days=10), progress=False)
        if isinstance(qqq.columns, pd.MultiIndex):
            if 'Adj Close' in qqq.columns.get_level_values(0): qqq = qqq['Adj Close']
            elif 'Close' in qqq.columns.get_level_values(0): qqq = qqq['Close']
            if isinstance(qqq, pd.DataFrame): qqq = qqq.iloc[:, 0]
        else:
            qqq = qqq['Adj Close'] if 'Adj Close' in qqq.columns else qqq['Close']
        qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
        qqq_df = pd.DataFrame({'Close': qqq})
        qqq_df['MA200'] = qqq_df['Close'].rolling(window=200).mean()
        return qqq_df
    except Exception as e:
        print(f"⚠️ Warning: QQQ download failed ({e}). Market filter disabled.")
        return pd.DataFrame()

# ==========================================
# 3. 数据加载与处理 (保持不变)
# ==========================================
def load_and_process_data():
    print("-" * 50)
    print("🚀 Starting REALISTIC Backtest Pipeline")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")
    
    df = pd.read_csv(FILE_PATH)
    col_map = {c: c.lower() for c in df.columns}
    if 'date' in col_map.values():
        original_date_col = list(col_map.keys())[list(col_map.values()).index('date')]
        df.rename(columns={original_date_col: 'Date'}, inplace=True)
    if 'ticker' in col_map.values():
        original_ticker_col = list(col_map.keys())[list(col_map.values()).index('ticker')]
        df.rename(columns={original_ticker_col: 'Ticker'}, inplace=True)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['Ticker', 'Date'], inplace=True)

    if os.path.exists(PREDICTIONS_PATH):
        pred_df = pd.read_csv(PREDICTIONS_PATH)
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        prob_col = find_column_fuzzy(pred_df, ['prob']) or find_column_fuzzy(pred_df, ['pred']) or find_column_fuzzy(pred_df, ['score'])
        
        if prob_col:
            df = pd.merge(df, pred_df[['Date', 'Ticker', prob_col]], on=['Date', 'Ticker'], how='left')
            df['AI_Score'] = df[prob_col].fillna(0.5)
            if STRICT_LAG_MODE:
                df['AI_Score'] = df.groupby('Ticker')['AI_Score'].shift(1)
                df.dropna(subset=['AI_Score'], inplace=True)
        else:
            df['AI_Score'] = 0.5
    else:
        df['AI_Score'] = 0.5

    mcap_col = find_column_fuzzy(df, ['market', 'cap'])
    df['MCap_Raw'] = df[mcap_col] if mcap_col else 1e9
    sent_col = find_column_fuzzy(df, ['sentiment'])
    df['Sentiment_Raw'] = df[sent_col].fillna(0) if sent_col else 0
    
    close_col = find_column_fuzzy(df, ['close', 'adj']) or 'Close'
    if close_col not in df.columns:
        if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        else: raise ValueError("Could not find Close price column.")
    else:
        df['Close'] = df[close_col]

    return df

# ==========================================
# 4. 策略逻辑 (保持不变)
# ==========================================
def run_strategy(df):
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_groups = df.sort_values('Date').groupby(['Ticker', 'YearMonth']).last().reset_index()
    unique_months = sorted(monthly_groups['YearMonth'].unique())
    
    start_dt = unique_months[0].start_time
    end_dt = unique_months[-1].end_time
    qqq_df = get_qqq_history(start_dt, end_dt)
    
    strategy_capital = INITIAL_CAPITAL
    history = [{'Date': start_dt, 'Strategy_Value': INITIAL_CAPITAL, 'In_Market': 1}]
    
    print("⏳ Executing Strategy...")
    
    for i in range(len(unique_months) - 1):
        curr_month = unique_months[i]
        next_month = unique_months[i+1]
        decision_date = curr_month.end_time
        
        in_market = True
        if USE_MARKET_FILTER and not qqq_df.empty:
            idx_loc = qqq_df.index.get_indexer([decision_date], method='nearest')[0]
            current_qqq = qqq_df.iloc[idx_loc]
            if pd.notna(current_qqq['MA200']) and current_qqq['Close'] < current_qqq['MA200']:
                in_market = False
        
        if not in_market:
            history.append({'Date': next_month.end_time, 'Strategy_Value': strategy_capital, 'In_Market': 0})
            continue 

        curr_df = monthly_groups[monthly_groups['YearMonth'] == curr_month].copy()
        if 'MCap_Raw' in curr_df.columns:
            mcap_threshold = curr_df['MCap_Raw'].quantile(MIN_MARKET_CAP_PERCENTILE)
            curr_df = curr_df[curr_df['MCap_Raw'] >= mcap_threshold]
        
        cols_to_norm = ['AI_Score', 'Sentiment_Raw']
        for col in cols_to_norm:
            std = curr_df[col].std()
            curr_df[f'Z_{col}'] = 0 if (std == 0 or pd.isna(std)) else (curr_df[col] - curr_df[col].mean()) / std
            
        curr_df['Final_Score'] = 0.6 * curr_df['Z_AI_Score'] + 0.4 * curr_df['Z_Sentiment_Raw']
        
        top_picks = curr_df.sort_values('Final_Score', ascending=False).head(TOP_K)
        selected_tickers = top_picks['Ticker'].tolist()
        
        next_df = monthly_groups[monthly_groups['YearMonth'] == next_month]
        monthly_returns = []
        
        for ticker in selected_tickers:
            price_curr = curr_df.loc[curr_df['Ticker'] == ticker, 'Close'].values
            price_next = next_df.loc[next_df['Ticker'] == ticker, 'Close'].values
            if len(price_curr) > 0 and len(price_next) > 0:
                entry_price = price_curr[0]
                exit_price = price_next[0]
                ret = ((exit_price - entry_price) / entry_price) - TRANSACTION_COST
                monthly_returns.append(ret)
            else:
                monthly_returns.append(-0.10) 
            
        if monthly_returns:
            strategy_capital *= (1 + np.mean(monthly_returns))
        
        history.append({'Date': next_month.end_time, 'Strategy_Value': strategy_capital, 'In_Market': 1})

    res_df = pd.DataFrame(history)
    res_df.set_index('Date', inplace=True)
    return res_df

# ==========================================
# 5. 绩效评估与绘图 (已修正)
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
    df['Bench_Ret'] = df['Benchmark_QQQ'].pct_change().fillna(0) # [修正 2] 计算基准收益率
    
    # 3. 计算夏普比率 (Risk-Adjusted Performance)
    def calc_sharpe(returns):
        std = returns.std()
        if std == 0: return 0
        return (returns.mean() * 12 - RISK_FREE_RATE) / (std * np.sqrt(12))

    sharpe_strat = calc_sharpe(df['Strat_Ret'])
    sharpe_bench = calc_sharpe(df['Bench_Ret']) # [修正 2] 计算基准夏普

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
    print(f"{'Sharpe Ratio':<20} | {sharpe_strat:11.2f}  | {sharpe_bench:14.2f}") # [重点] 展示对比
    print(f"{'Max Drawdown':<20} | {max_dd*100:11.2f}% | {'--':>14}")
    print("-" * 53)
    
    # 6. 保存结果
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Milestones Met:")
    print(f"  1. Results saved to: {OUTPUT_CSV}")
    print(f"  2. Risk-adjusted comparison displayed above.")

    # 7. 绘图
    plt.figure(figsize=(12, 6))
    if 'Benchmark_QQQ' in df.columns:
        plt.plot(df.index, df['Benchmark_QQQ'], label=f'Benchmark (QQQ) Sharpe: {sharpe_bench:.2f}', color='gray', linestyle='--')
    plt.plot(df.index, df['Strategy_Value'], label=f'Strategy Sharpe: {sharpe_strat:.2f}', color='#1f77b4', linewidth=2)
    
    if 'In_Market' in df.columns:
        y_min, y_max = plt.ylim()
        plt.fill_between(df.index, y_min, y_max, where=(df['In_Market'] == 0), color='red', alpha=0.1, label='Cash (Market Filter)')

    plt.yscale('log')
    plt.title('Long-Term Strategy vs QQQ Benchmark')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
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
        print(f"\n❌ Error: {e}")