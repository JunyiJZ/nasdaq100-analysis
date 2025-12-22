import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 配置项
# ==========================================
RESULTS_FILE = 'backtest_results/backtest_shortterm.csv'
OUTPUT_DIR = 'backtest_results'

def visualize_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Error: Results file not found at {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 平均总回报对比 (Bar Plot)
    sns.barplot(x='Model', y='Total_Return', data=df, ax=axes[0, 0], palette='viridis', errorbar=None)
    axes[0, 0].set_title('Average Total Return by Model', fontsize=14)
    axes[0, 0].set_ylabel('Total Return')
    
    # 2. 平均夏普比率对比 (Bar Plot)
    sns.barplot(x='Model', y='Sharpe_Ratio', data=df, ax=axes[0, 1], palette='magma', errorbar=None)
    axes[0, 1].set_title('Average Sharpe Ratio by Model', fontsize=14)
    axes[0, 1].set_ylabel('Sharpe Ratio')
    
    # 3. 收益率分布 (Box Plot) - 展示稳定性
    sns.boxplot(x='Model', y='Total_Return', data=df, ax=axes[1, 0], palette='coolwarm')
    axes[1, 0].set_title('Distribution of Returns (Stability)', fontsize=14)
    axes[1, 0].set_ylabel('Total Return')
    
    # 4. 胜率对比 (Bar Plot)
    sns.barplot(x='Model', y='Hit_Ratio', data=df, ax=axes[1, 1], palette='Blues_d', errorbar=None)
    axes[1, 1].set_title('Average Win Rate (Hit Ratio)', fontsize=14)
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].set_ylim(0.4, 0.7) # 设置y轴范围以便更清楚看到差异

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'model_comparison_summary.png')
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

    # 打印文字版摘要
    print("\n--- Final Summary ---")
    summary = df.groupby('Model')[['Total_Return', 'Sharpe_Ratio', 'Hit_Ratio', 'Max_Drawdown']].mean()
    print(summary)

if __name__ == "__main__":
    visualize_results()