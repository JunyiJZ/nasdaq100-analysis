import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 设置风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_comparison():
    base_dir = os.getcwd()
    # 读取刚才生成的对比结果 CSV
    report_path = os.path.join(base_dir, 'data', 'tuned_models', 'performance_comparison.csv')
    
    if not os.path.exists(report_path):
        print(f"❌ 找不到文件: {report_path}，请先运行 12_evaluate_performance_gains.py")
        return

    df = pd.read_csv(report_path)
    
    if df.empty:
        print("⚠️ 结果文件是空的。")
        return

    # 创建一个画布
    plt.figure(figsize=(12, 6))

    # 数据转换格式以便绘图 (Melt)
    df_melted = df.melt(id_vars=['Ticker', 'Horizon'], 
                        value_vars=['Untuned_Acc', 'Tuned_Acc'], 
                        var_name='Model_Type', 
                        value_name='Accuracy')

    # 绘制柱状图
    chart = sns.barplot(x='Ticker', y='Accuracy', hue='Model_Type', data=df_melted, palette='viridis')

    # 添加标题和标签
    plt.title('模型调优前后准确率对比 (Untuned vs Tuned)', fontsize=16)
    plt.ylabel('Accuracy (准确率)', fontsize=12)
    plt.xlabel('股票代码', fontsize=12)
    plt.ylim(0, 1.0) # 准确率在 0 到 1 之间

    # 在柱子上标数值
    for container in chart.containers:
        chart.bar_label(container, fmt='%.4f', padding=3)

    # 保存图片
    output_img = os.path.join(base_dir, 'model_comparison_chart.png')
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"✅ 对比图表已保存至: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()