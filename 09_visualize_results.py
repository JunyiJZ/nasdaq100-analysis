import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 读取结果文件
RESULTS_PATH = 'baseline_results.csv'

if not os.path.exists(RESULTS_PATH):
    print("❌ 找不到结果文件，请先运行模型训练代码。")
else:
    df = pd.read_csv(RESULTS_PATH)
    
    # 2. 简单的数据清洗（确保没有重复项干扰绘图）
    # 保留每个 Model 和 Ticker 组合的最新一次结果
    df = df.drop_duplicates(subset=['Model', 'Ticker'], keep='last')

    print("📊 加载的数据预览：")
    print(df.head())

    # 3. 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # ==========================================
    # 图表 1: RMSE 对比 (越低越好)
    # ==========================================
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Ticker', y='RMSE', hue='Model', palette='viridis')
    plt.title('Model Performance Comparison: RMSE (Lower is Better)')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    plt.legend(title='Model')

    # ==========================================
    # 图表 2: R2 Score 对比 (越高越好)
    # ==========================================
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Ticker', y='R2_Score', hue='Model', palette='magma')
    plt.title('Model Performance Comparison: R2 Score (Higher is Better)')
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # 添加0刻度线
    plt.xticks(rotation=45)
    plt.ylabel('R2 Score')
    plt.legend(title='Model')

    plt.tight_layout()
    
    # 保存图片
    save_path = 'model_comparison_chart.png'
    plt.savefig(save_path)
    print(f"\n✅ 图表已保存至: {save_path}")
    plt.show()