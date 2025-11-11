# news_collection.py

import os
import json
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# 从 .env 文件加载环境变量 (API_KEY)
load_dotenv()

# --- 配置常量 ---
# 从环境变量中获取API密钥
API_KEY = os.getenv("NEWS_API_KEY")
# 从您的 data_collection.py 中获取股票列表函数
from data_collection import get_nasdaq100_tickers

# 定义数据保存目录
# 确保这个目录存在，如果不存在则创建
RAW_NEWS_DIR = os.path.join('data', 'news_raw')
os.makedirs(RAW_NEWS_DIR, exist_ok=True)

# 定义要获取新闻的时间范围（例如，过去365天）
DAYS_AGO = 365
DATE_TO = datetime.now()
DATE_FROM = DATE_TO - timedelta(days=DAYS_AGO)

def fetch_news_for_ticker(ticker: str):
    """
    为单个股票代码获取新闻并保存为JSON文件。
    """
    print(f"--- 开始为 {ticker} 获取新闻 ---")
    
    # NewsAPI 的免费计划限制只能获取过去一个月的新闻，并且有请求频率限制
    # 如果您需要更早的数据，需要付费计划或寻找其他数据源
    # 这里我们以免费计划的限制为准
    
    # 免费版只能获取过去一个月的数据，我们调整一下时间
    query_date_from = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"from={query_date_from}&"
        "sortBy=publishedAt&"
        "language=en&"
        f"apiKey={API_KEY}"
    )

    all_articles = []
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败 (如 4xx, 5xx 错误)，则抛出异常
        
        data = response.json()
        
        if data['status'] == 'ok':
            articles = data.get('articles', [])
            print(f"找到 {len(articles)} 篇关于 {ticker} 的新闻。")
            
            # 为每篇文章添加 ticker 字段，方便后续处理
            for article in articles:
                article['ticker'] = ticker
            
            all_articles.extend(articles)
        else:
            print(f"API 返回错误: {data.get('message')}")
            return # 如果API返回错误，则跳过此ticker

    except requests.exceptions.RequestException as e:
        print(f"请求 {ticker} 新闻时发生网络错误: {e}")
        return # 如果网络错误，则跳过

    # 保存数据到文件
    if all_articles:
        file_path = os.path.join(RAW_NEWS_DIR, f"{ticker}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=4)
        print(f"新闻已保存至: {file_path}")
    else:
        print(f"没有为 {ticker} 找到任何新闻。")

def main():
    """
    主函数，获取所有纳斯达克100股票的新闻。
    """
    if not API_KEY:
        print("错误：未找到 NEWS_API_KEY。请检查你的 .env 文件。")
        return

    print("开始新闻数据收集流程...")
    
    tickers = get_nasdaq100_tickers()
    print(f"将为 {len(tickers)} 个股票代码获取新闻。")

    for i, ticker in enumerate(tickers):
        fetch_news_for_ticker(ticker)
        
        # NewsAPI 对免费用户有速率限制，在每个请求后稍作停顿，以示友好
        # 即使没有严格限制，这也是一个好习惯
        print(f"完成 ({i+1}/{len(tickers)})。等待1秒...")
        time.sleep(1)
    
    print("所有新闻数据收集完成！")

if __name__ == "__main__":
    main()