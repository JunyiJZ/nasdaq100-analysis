# sentiment_analysis.py

import os
import json
import pandas as pd
from tqdm import tqdm  # 引入tqdm，用于显示进度条
import nltk
# 检查并下载 NLTK 依赖后，再从 NLTK 导入其他模块是更安全的做法
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from transformers import pipeline

# --- 准备工作 ---
# 确保 VADER 词典已下载
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:  # <--- 修改为正确的异常类型
    print("正在下载 VADER 词典...")
    nltk.download('vader_lexicon')
    print("下载完成。")

# 现在可以安全地导入依赖于已下载数据的模块
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# --- 路径和常量定义 ---
RAW_NEWS_DIR = os.path.join('data', 'news_raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- 模型初始化 ---
# 1. 初始化 VADER
print("初始化 VADER...")
vader_analyzer = SentimentIntensityAnalyzer()

# 2. 初始化 FinBERT
# 使用 pipeline 会自动处理 tokenization 和模型加载
# 如果你有GPU，它会自动使用；如果没有，会使用CPU（速度会慢很多）
print("初始化 FinBERT 模型 (可能需要一些时间下载)...")
finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("模型初始化完成。")

def load_all_news_to_dataframe():
    """
    从 data/news_raw/ 加载所有JSON新闻文件到一个Pandas DataFrame中。
    """
    all_news_list = []
    print(f"从 '{RAW_NEWS_DIR}' 加载新闻文件...")
    
    # 检查目录是否存在
    if not os.path.isdir(RAW_NEWS_DIR):
        print(f"错误：目录 '{RAW_NEWS_DIR}' 不存在。")
        print("请先运行 news_collection.py 脚本来下载新闻数据。")
        return pd.DataFrame()

    file_names = [f for f in os.listdir(RAW_NEWS_DIR) if f.endswith('.json')]
    if not file_names:
        print(f"错误：在 '{RAW_NEWS_DIR}' 目录下没有找到任何 .json 文件。")
        print("请先运行 news_collection.py 脚本。")
        return pd.DataFrame()

    for file_name in tqdm(file_names, desc="加载JSON文件"):
        file_path = os.path.join(RAW_NEWS_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                articles = json.load(f)
                all_news_list.extend(articles)
            except json.JSONDecodeError:
                print(f"警告：文件 {file_name} 不是有效的JSON格式，已跳过。")
            
    if not all_news_list:
        print("错误：已加载的JSON文件均为空或格式错误。")
        return pd.DataFrame()

    df = pd.DataFrame(all_news_list)
    print(f"总共加载了 {len(df)} 篇新闻。")
    return df

def analyze_sentiment(df: pd.DataFrame):
    """
    对DataFrame中的新闻标题进行情感分析。
    """
    if df.empty:
        print("DataFrame 为空，跳过情感分析。")
        return df

    # 确保 'title' 列存在
    if 'title' not in df.columns:
        print("错误：DataFrame 中缺少 'title' 列。")
        return df

    # --- 1. VADER 分析 ---
    print("开始使用 VADER 进行情感分析...")
    # 我们只分析标题，因为它更简洁。也可以选择 'description' 或两者结合。
    # VADER 返回一个字典，我们主要关心 'compound' 分数，范围从 -1 (最负面) 到 1 (最正面)
    tqdm.pandas(desc="VADER 分析")
    df['vader_sentiment'] = df['title'].progress_apply(
        lambda title: vader_analyzer.polarity_scores(title)['compound'] if isinstance(title, str) else 0
    )

    # --- 2. FinBERT 分析 ---
    print("开始使用 FinBERT 进行情感分析 (这可能需要很长时间)...")
    # FinBERT 返回一个标签 ('positive', 'negative', 'neutral') 和一个分数
    # 为了便于比较，我们将其转换为一个从 -1 到 1 的数值
    # (positive * score) + (negative * -score)
    
    # 直接 apply 会很慢，最好是批量处理
    titles = df['title'].dropna().tolist()
    if not titles:
        print("没有有效的标题可供 FinBERT 分析。")
        df['finbert_sentiment'] = 0.0
    else:
        finbert_results = []
        # 使用 pipeline 进行批量处理会快很多
        for result in tqdm(finbert_pipeline(titles, batch_size=32), total=len(titles), desc="FinBERT 分析"):
            label = result['label']
            score = result['score']
            if label == 'positive':
                finbert_results.append(score)
            elif label == 'negative':
                finbert_results.append(-score)
            else: # neutral
                finbert_results.append(0.0)

        df.loc[df['title'].notna(), 'finbert_sentiment'] = finbert_results
    
    df['finbert_sentiment'].fillna(0, inplace=True)

    print("情感分析完成。")
    return df

def aggregate_and_save(df: pd.DataFrame):
    """
    按股票代码和日期聚合情感分数，并保存结果。
    """
    if df.empty or 'vader_sentiment' not in df.columns:
        print("数据不足，无法进行聚合和保存。")
        return

    # 确保关键列存在
    required_cols = ['publishedAt', 'ticker', 'vader_sentiment', 'finbert_sentiment', 'title']
    if not all(col in df.columns for col in required_cols):
        print(f"错误：DataFrame 缺少必要的列。需要 {required_cols}")
        return

    print("开始聚合情感分数...")
    # 转换日期列
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df.dropna(subset=['publishedAt'], inplace=True) # 删除无法转换的日期行
    df['date'] = df['publishedAt'].dt.date

    # 按 ticker 和 date 分组，计算平均情感分数
    daily_sentiment = df.groupby(['ticker', 'date']).agg(
        vader_sentiment_mean=('vader_sentiment', 'mean'),
        finbert_sentiment_mean=('finbert_sentiment', 'mean'),
        article_count=('title', 'count')
    ).reset_index()

    # 保存聚合后的结果，这是里程碑要求的文件
    output_path = os.path.join(PROCESSED_DATA_DIR, 'sentiment_scores.csv')
    daily_sentiment.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"每日聚合情感分数已保存至: {output_path}")

    # 也可以保存带有每篇文章情感分数的完整数据集，用于更详细的分析
    full_output_path = os.path.join(PROCESSED_DATA_DIR, 'news_with_sentiment.csv')
    df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
    print(f"带有详细情感分数的完整数据集已保存至: {full_output_path}")


def main():
    """
    主函数：加载数据，进行分析，保存结果。
    """
    # 1. 加载数据
    news_df = load_all_news_to_dataframe()
    
    # 2. 进行情感分析
    news_with_sentiment_df = analyze_sentiment(news_df)
    
    # 3. 聚合和保存
    aggregate_and_save(news_with_sentiment_df)
    
    print("情感分析流程全部完成！")

if __name__ == "__main__":
    main()