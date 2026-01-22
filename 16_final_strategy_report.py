import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# ==========================================
# 1. 配置部分
# ==========================================

STRATEGY_FILES = {
    'Short-Term': 'backtest_shortterm.csv',
    'Mid-Term':   '13_midterm_strategy_backtest.csv',
    'Long-Term':  'long_term_backtest.csv'
}

WEIGHTS_FILE = 'portfolio_allocation_weights.csv'
OUTPUT_REPORT_CSV = 'final_system_performance.csv'
OUTPUT_CHART = 'final_system_chart.png'

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 辅助函数
# ==========================================

def find_file(filename):
    search_paths = [filename, os.path.join('backtest_results', filename), 
                    os.path.basename(filename), os.path.join('..', filename)]
    for path in search_paths:
        if os.path.exists(path): return path
    return None

def load_weights():
    weights = {}
    if os.path.exists(WEIGHTS_FILE):
        try:
            df = pd.read_csv(WEIGHTS_FILE)
            if 'Strategy' in df.columns and 'Weight' in df.columns:
                weights = dict(zip(df['Strategy'], df['Weight']))
        except: pass
    if not weights:
        weights = {k: 1.0/len(STRATEGY_FILES) for k in STRATEGY_FILES.keys()}
    return weights

def clean_series_data(series, name):
    series = series.dropna()
    if len(series) == 0: return None
    
    # 标准化索引：去掉时分秒，确保能对齐
    series.index = pd.to_datetime(series.index).normalize()
    
    # 检查是否是价格数据 (均值 > 0.5 视为价格/净值)
    if series.abs().mean() > 0.5:
        print(f"⚠️ [{name}] 检测为价格/净值数据，正在转换为收益率...")
        series = series.pct_change().fillna(0)
    
    # 异常值截断
    if series.max() > 5.0: # 单日/单月涨幅超过500%视为异常
        series = series.clip(upper=1.0)
        
    return series

def load_and_process_data(strategy_name, filename):
    filepath = find_file(filename)
    if not filepath:
        print(f"❌ 找不到文件: {filename}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # 智能查找日期列
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'datetime', 'index', 'timestamp']:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            df.index = pd.to_datetime(df.index)

        # 智能查找收益列
        target_col = None
        possible_cols = ['Daily_Return', 'Strat_Ret', 'Total_Return', 'Strategy_Value', 'Close']
        for col in possible_cols:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col: return None
        
        series = pd.to_numeric(df[target_col], errors='coerce')
        series = clean_series_data(series, strategy_name)
        series.name = strategy_name
        
        return series
        
    except Exception as e:
        print(f"❌ 读取错误 {strategy_name}: {e}")
        return None

def calculate_metrics(series):
    if len(series) < 2: return 0, 0, 0, 0, 0
    # 累计收益
    total_return = (1 + series).prod() - 1
    
    # 年化计算 (根据数据频率粗略估计，假设总天数跨度)
    days = (series.index.max() - series.index.min()).days
    if days < 30: days = 30 # 避免除零
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # 波动率 (简化版，假设日线)
    volatility = series.std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility != 0 else 0
    
    # 最大回撤
    cum_ret = (1 + series).cumprod()
    max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
    
    return total_return, cagr, sharpe, max_dd, volatility

# ==========================================
# 3. 主逻辑 (核心修复)
# ==========================================

def generate_report():
    print("\n🚀 启动第16步: 最终组合回测系统 (强制对齐版)")
    print("=" * 80)
    
    weights = load_weights()
    data_dict = {}
    
    # 1. 读取数据
    start_dates = []
    end_dates = []
    
    for name, filename in STRATEGY_FILES.items():
        s = load_and_process_data(name, filename)
        if s is not None:
            data_dict[name] = s
            start_dates.append(s.index.min())
            end_dates.append(s.index.max())
            print(f"📄 {name:<10} | 范围: {s.index.min().date()} -> {s.index.max().date()} | 行数: {len(s)}")

    if not data_dict: return

    # 2. 计算公共时间窗口 (Common Time Window)
    # 核心逻辑：起跑线取最晚的那个开始时间，终点线取最早的那个结束时间
    global_start = max(start_dates)
    global_end = max(end_dates) # 结束时间可以取最晚的，前面缺失的补0即可，但开始时间必须统一
    
    print("-" * 80)
    print(f"✂️ 强制裁切时间窗口: 从 {global_start.date()} 开始")
    
    # 3. 截断数据
    aligned_data = []
    for name, s in data_dict.items():
        # 只保留公共开始时间之后的数据
        s_trimmed = s[s.index >= global_start]
        aligned_data.append(s_trimmed)
    
    # 4. 合并数据 (使用 Outer Join + Fillna 0)
    # 允许日线和月线混合。如果某天Short-Term有数据但Long-Term没数据，Long-Term填0
    full_df = pd.concat(aligned_data, axis=1).fillna(0)
    
    # 再次按时间排序，确保安全
    full_df = full_df.sort_index()
    
    print(f"✅ 合并完成! 最终交易天数: {len(full_df)} 天")

    # 5. 计算组合收益
    full_df['Portfolio_Daily_Return'] = 0
    for name in STRATEGY_FILES.keys():
        if name in full_df.columns:
            w = weights.get(name, 0)
            full_df['Portfolio_Daily_Return'] += full_df[name] * w
            
    # 6. 绩效计算与输出
    full_df['Portfolio_Cumulative'] = (1 + full_df['Portfolio_Daily_Return']).cumprod()
    
    print("\n🏆 最终绩效报告 (基于 2020-2025 统一时段)")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Total Ret':<12} {'CAGR':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 80)
    
    metrics = calculate_metrics(full_df['Portfolio_Daily_Return'])
    print(f"{'AI Portfolio':<20} {metrics[0]*100:>9.2f}% {metrics[1]*100:>8.2f}% {metrics[2]:>8.2f} {metrics[3]*100:>8.2f}%")
    
    for name in STRATEGY_FILES.keys():
        if name in full_df.columns:
            # 计算单策略的累计净值用于绘图
            full_df[f'{name}_Cumulative'] = (1 + full_df[name]).cumprod()
            m = calculate_metrics(full_df[name])
            print(f"{name:<20} {m[0]*100:>9.2f}% {m[1]*100:>8.2f}% {m[2]:>8.2f} {m[3]*100:>8.2f}%")
            
    # 7. 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(full_df.index, full_df['Portfolio_Cumulative'], label='AI Portfolio', linewidth=3, color='black')
    
    colors = sns.color_palette("husl", len(STRATEGY_FILES))
    for i, name in enumerate(STRATEGY_FILES.keys()):
        if f'{name}_Cumulative' in full_df.columns:
            plt.plot(full_df.index, full_df[f'{name}_Cumulative'], label=name, alpha=0.6, linestyle='--', color=colors[i])
            
    plt.title(f'AI Strategy Performance ({global_start.date()} - {global_end.date()})')
    plt.ylabel('Normalized Value (Start=1.0)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART)
    full_df.to_csv(OUTPUT_REPORT_CSV)
    print(f"\n✅ 完成。图表: {OUTPUT_CHART}")

if __name__ == "__main__":
    generate_report()