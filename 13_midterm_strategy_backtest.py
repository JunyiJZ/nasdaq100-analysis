import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置与路径
# ==========================================
# 【关键修改】这里指向我们刚刚用修复脚本生成的“正确格式”文件
PREDICTIONS_PATH = "backtest_results/daily_signals_midterm.csv" 

# 备用主数据路径 (如果预测文件缺价格，会从这里补)
MASTER_DATA_PATH = "data/features_technical.csv" 

OUTPUT_CSV = "13_midterm_strategy_backtest.csv"
OUTPUT_IMG = "13_midterm_strategy_backtest.png"

# 策略参数
INITIAL_CAPITAL = 100000  # 初始资金
TOP_K = 5                 # 每天持仓评分最高的 K 只股票
HOLDING_PERIOD = 5        # 持仓周期（天）
TRANSACTION_COST = 0.001  # 交易费率 (0.1%)

# ==========================================
# 2. 数据加载与清洗 (增强健壮性版)
# ==========================================
def load_and_prepare_data():
    print("🔍 [Step 1] 正在加载并对齐数据...")
    
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"❌ 找不到文件: {PREDICTIONS_PATH}")
        print("请先运行上一步提供的 '修复数据脚本' 生成此文件。")
        raise FileNotFoundError(f"找不到预测文件: {PREDICTIONS_PATH}")
    
    # 读取数据
    preds = pd.read_csv(PREDICTIONS_PATH)
    
    # --- 修复 1: 清理列名空格 ---
    preds.columns = [c.strip() for c in preds.columns]

    # --- 修复 2: 智能查找日期列 ---
    date_col_candidates = ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time', 'Unnamed: 0']
    date_col = None
    for col in date_col_candidates:
        if col in preds.columns:
            date_col = col
            break
    
    if date_col is None:
        raise KeyError(f"无法在预测文件中找到日期列。当前列: {preds.columns.tolist()}")
    
    print(f"✅ 识别到日期列名为: '{date_col}'，正在标准化...")
    preds.rename(columns={date_col: 'Date'}, inplace=True)
    preds['Date'] = pd.to_datetime(preds['Date'])
    
    # --- 修复 3: 智能查找 Ticker 列 ---
    ticker_col_candidates = ['Ticker', 'ticker', 'Symbol', 'symbol']
    ticker_col = None
    for col in ticker_col_candidates:
        if col in preds.columns:
            ticker_col = col
            break     
    if ticker_col:
        preds.rename(columns={ticker_col: 'Ticker'}, inplace=True)
    else:
        raise KeyError(f"无法找到股票代码列 (Ticker/Symbol)。当前列: {preds.columns.tolist()}")

    # --- 修复 4: 确保有 Close 价格 ---
    preds.columns = [c.capitalize() if c.lower() == 'close' else c for c in preds.columns]

    if 'Close' not in preds.columns:
        print("⚠️ 预测数据缺少 'Close' 列，尝试从主数据集合并...")
        if not os.path.exists(MASTER_DATA_PATH):
            raise FileNotFoundError(f"❌ 缺少 Close 列且找不到主数据集: {MASTER_DATA_PATH}")
            
        master = pd.read_csv(MASTER_DATA_PATH)
        master.columns = [c.strip() for c in master.columns]
        
        # 寻找主数据的日期列
        master_date_col = next((c for c in master.columns if c.lower() in ['date', 'datetime']), None)
        if not master_date_col: raise KeyError("主数据集中找不到日期列")
        master.rename(columns={master_date_col: 'Date'}, inplace=True)
        master['Date'] = pd.to_datetime(master['Date'])
        
        # 寻找主数据的 Ticker 列
        master_ticker_col = next((c for c in master.columns if c.lower() in ['ticker', 'symbol']), 'Ticker')
        
        # 寻找主数据的 Close 列
        master_price_col = next((c for c in master.columns if c.lower() == 'close'), None)
        
        if not master_price_col:
            raise ValueError("❌ 无法在主数据集中找到收盘价列。")
            
        # 准备合并的价格数据
        price_df = master[['Date', master_ticker_col, master_price_col]].rename(
            columns={master_price_col: 'Close', master_ticker_col: 'Ticker'}
        )
        # 去重
        price_df = price_df.drop_duplicates(subset=['Date', 'Ticker'])
        # 合并
        preds = pd.merge(preds, price_df, on=['Date', 'Ticker'], how='left')
        print(f"✅ 已成功合并收盘价数据。")

    # --- 集成逻辑: 计算多模型平均概率 ---
    # 自动寻找所有以 Prob_ 开头的列 (例如 Prob_LSTM, Prob_GRU)
    model_cols = [c for c in preds.columns if c.startswith('Prob_')]
    
    if model_cols:
        print(f"🧠 检测到集成模型列: {model_cols}，正在计算平均分...")
        preds['AI_Score'] = preds[model_cols].mean(axis=1)
    elif 'Probability' in preds.columns:
        preds['AI_Score'] = preds['Probability']
    else:
        # 最后的保底
        print("⚠️ 警告: 未找到概率列，将使用随机分数进行测试（请检查数据源列名）")
        preds['AI_Score'] = np.random.uniform(0, 1, len(preds))

    # --- 关键修复: 强制去重 ---
    duplicates = preds.duplicated(subset=['Date', 'Ticker']).sum()
    if duplicates > 0:
        print(f"⚠️ 检测到 {duplicates} 条重复数据 (Date + Ticker)，正在通过取平均值合并...")
        preds = preds.groupby(['Date', 'Ticker'], as_index=False).mean(numeric_only=True)
    
    # 清洗无效数据
    preds = preds.dropna(subset=['Close', 'AI_Score'])
    
    # 按时间排序
    preds = preds.sort_values(['Date', 'Ticker'])
    
    print(f"✅ 数据准备完成，唯一记录数: {len(preds)}")
    return preds

# ==========================================
# 3. 核心回测逻辑 (向量化加速版)
# ==========================================
def run_longterm_backtest(df):
    print("🚀 [Step 2] 开始执行中期策略回测...")
    
    # Pivot 数据
    try:
        close_prices = df.pivot(index='Date', columns='Ticker', values='Close').ffill()
        ai_scores = df.pivot(index='Date', columns='Ticker', values='AI_Score').fillna(0)
    except ValueError as e:
        print(f"❌ 数据透视失败: {e}")
        raise e

    # 初始化
    dates = close_prices.index
    portfolio_value = [INITIAL_CAPITAL]
    current_holdings = {} # {ticker: shares}
    cash = INITIAL_CAPITAL
    
    history = []

    # 模拟交易循环
    for i, date in enumerate(dates):
        if i == 0: continue 
        
        # --- 每日更新市值 ---
        current_value = cash
        todays_prices = close_prices.loc[date]
        
        # 计算持仓市值
        for ticker, shares in current_holdings.items():
            if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                current_value += shares * todays_prices[ticker]
        
        portfolio_value.append(current_value)
        
        # --- 调仓逻辑 (每隔 N 天) ---
        if i % HOLDING_PERIOD == 0:
            # 1. 全部卖出 (简化逻辑：先全卖再全买，方便计算)
            for ticker, shares in list(current_holdings.items()):
                price = todays_prices.get(ticker, 0)
                if price > 0:
                    cash += shares * price * (1 - TRANSACTION_COST)
            current_holdings = {}
            
            # 2. 选股买入 Top K
            todays_scores = ai_scores.loc[date]
            # 只看今天有价格的股票
            valid_tickers = todays_prices[todays_prices > 0].index
            todays_scores = todays_scores[todays_scores.index.isin(valid_tickers)]
            
            if not todays_scores.empty:
                # 选分最高的 K 个
                top_k_tickers = todays_scores.nlargest(TOP_K).index.tolist()
                
                if len(top_k_tickers) > 0:
                    target_per_stock = cash / len(top_k_tickers)
                    for ticker in top_k_tickers:
                        price = todays_prices[ticker]
                        if price > 0:
                            shares_to_buy = (target_per_stock * (1 - TRANSACTION_COST)) / price
                            current_holdings[ticker] = shares_to_buy
                            cash -= shares_to_buy * price
        
        history.append({
            'Date': date,
            'Portfolio_Value': current_value,
            'Cash': cash,
            'Num_Holdings': len(current_holdings)
        })

    # 结果整合
    results = pd.DataFrame(history)
    
    if not close_prices.empty and not results.empty:
        # 计算基准收益 (所有股票平均)
        market_returns = close_prices.mean(axis=1).pct_change().fillna(0)
        results = results.set_index('Date')
        
        # 对齐索引
        common_idx = results.index.intersection(market_returns.index)
        market_subset = market_returns.loc[common_idx]
        
        # 计算基准净值曲线
        results.loc[common_idx, 'Benchmark_Value'] = INITIAL_CAPITAL * (1 + market_subset).cumprod()

    return results

# ==========================================
# 4. 计算指标
# ==========================================
def calculate_financial_metrics(df):
    df['Daily_Return'] = df['Portfolio_Value'].pct_change()
    
    total_return = (df['Portfolio_Value'].iloc[-1] / INITIAL_CAPITAL) - 1
    sharpe_ratio = df['Daily_Return'].mean() / df['Daily_Return'].std() * np.sqrt(252)
    max_drawdown = (df['Portfolio_Value'] / df['Portfolio_Value'].cummax() - 1).min()
    
    print("\n📊 [策略表现报告]")
    print(f"💰 最终资金: ${df['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"📈 总收益率: {total_return*100:.2f}%")
    print(f"⚡ 夏普比率: {sharpe_ratio:.2f}")
    print(f"📉 最大回撤: {max_drawdown*100:.2f}%")
    
    return df

# ==========================================
# 5. 绘图与保存
# ==========================================
def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Portfolio_Value'], label='AI Strategy (Mid-term)', linewidth=2)
    
    if 'Benchmark_Value' in df.columns:
        plt.plot(df.index, df['Benchmark_Value'], label='Market Average (Benchmark)', linestyle='--', alpha=0.7)
        
    plt.title(f'Backtest Results: Top {TOP_K} Stocks (Rebalance every {HOLDING_PERIOD} days)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"🖼️ 图表已保存至: {OUTPUT_IMG}")

    df.to_csv(OUTPUT_CSV)
    print(f"💾 详细数据已保存至: {OUTPUT_CSV}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    try:
        data = load_and_prepare_data()
        portfolio = run_longterm_backtest(data)
        
        if not portfolio.empty:
            portfolio_with_metrics = calculate_financial_metrics(portfolio)
            plot_results(portfolio_with_metrics)
            print("\n✅ Week 13 任务成功完成！")
        else:
            print("❌ 回测结果为空。")
            
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()