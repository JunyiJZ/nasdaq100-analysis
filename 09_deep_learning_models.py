import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import math

# ==========================================
# 配置项
# ==========================================
DATA_PATH = 'data/finalized/data_with_targets.csv'
MODEL_SAVE_DIR = 'models/dl_checkpoints'
RESULTS_PATH = 'deep_learning_comparison_results.csv'
# 新增：预测结果保存路径
PREDICTIONS_SAVE_PATH = 'data/backtest_results/model_predictions.csv' 

SEQ_LENGTH = 60      # 时间窗口
EPOCHS = 30          # 训练轮数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SAMPLE_TICKERS_COUNT = 10
TARGET_COL_SOURCE = 'target_1d_return'

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# 确保保存预测结果的目录存在
os.makedirs(os.path.dirname(PREDICTIONS_SAVE_PATH), exist_ok=True)

# ==========================================
# 1. 数据预处理工具
# ==========================================
def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    length = len(input_data)
    for i in range(length - seq_length):
        x = input_data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 2. 定义模型库 (LSTM, GRU, Transformer)
# ==========================================

# --- 模型 A: LSTM ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

# --- 模型 B: GRU ---
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# --- 模型 C: Transformer ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        src = self.input_embedding(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] 
        output = self.fc(output)
        return self.sigmoid(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==========================================
# 3. 训练与评估通用函数 (已修改：返回预测概率)
# ==========================================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, ticker):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    num_samples = X_train.size(0)
    num_batches = int(np.ceil(num_samples / BATCH_SIZE))

    for epoch in range(EPOCHS):
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, num_samples)
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # 评估
    model.eval()
    with torch.no_grad():
        pred_probs = model(X_test).cpu().numpy() # 获取概率值
        actual_labels = y_test.cpu().numpy()
    
    pred_labels = (pred_probs > 0.5).astype(float)
    acc = accuracy_score(actual_labels, pred_labels)
    
    # 保存模型
    save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{ticker}.pth")
    torch.save(model.state_dict(), save_path)
    
    # 返回 Acc 以及 预测概率数组 (flatten用于转为1维数组)
    return acc, pred_probs.flatten()

# ==========================================
# 4. 主流程
# ==========================================
def run_week9_workflow():
    print("🚀 Starting Week 9: Advanced DL Models (LSTM, GRU, Transformer)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 日期处理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values(['Ticker', 'Date'])
    
    top_tickers = df['Ticker'].value_counts().head(SAMPLE_TICKERS_COUNT).index.tolist()
    print(f"Sample Tickers: {top_tickers}")

    results_list = []
    
    # 用于收集所有股票的预测结果的大列表
    all_predictions = []

    for ticker in top_tickers:
        print(f"\nProcessing {ticker}...")
        
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.ffill().bfill()

        # 特征选择
        feature_cols = [c for c in ticker_df.columns if c not in ['Date', 'Ticker'] and not c.startswith('target_')]
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c in numeric_cols and 'Unnamed' not in c]
        
        X_values = ticker_df[feature_cols].values
        
        # 目标：分类 (涨=1, 跌=0)
        raw_targets = ticker_df[TARGET_COL_SOURCE].values
        y_labels = (raw_targets > 0).astype(float)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_values)
        
        # 创建序列
        X, y = create_sequences(X_scaled, y_labels, SEQ_LENGTH)
        
        if len(X) < 100:
            print(f"Skipping {ticker}: Not enough data.")
            continue

        # 划分数据
        train_size = int(len(X) * 0.8)
        X_train_t = torch.FloatTensor(X[:train_size]).to(device)
        y_train_t = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X[train_size:]).to(device)
        y_test_t = torch.FloatTensor(y[train_size:]).unsqueeze(1).to(device)

        input_dim = X.shape[2]

        # --- 1. 训练 LSTM ---
        print(f"  > Training LSTM...")
        lstm_model = LSTMClassifier(input_dim).to(device)
        lstm_acc, lstm_probs = train_and_evaluate(lstm_model, X_train_t, y_train_t, X_test_t, y_test_t, "LSTM", ticker)
        
        # --- 2. 训练 GRU ---
        print(f"  > Training GRU...")
        gru_model = GRUClassifier(input_dim).to(device)
        gru_acc, gru_probs = train_and_evaluate(gru_model, X_train_t, y_train_t, X_test_t, y_test_t, "GRU", ticker)

        # --- 3. 训练 Transformer ---
        print(f"  > Training Transformer...")
        tf_model = TransformerClassifier(input_dim, d_model=64).to(device)
        tf_acc, tf_probs = train_and_evaluate(tf_model, X_train_t, y_train_t, X_test_t, y_test_t, "Transformer", ticker)

        print(f"  ✅ Results for {ticker}: LSTM={lstm_acc:.4f}, GRU={gru_acc:.4f}, Transformer={tf_acc:.4f}")

        results_list.append({'Ticker': ticker, 'Model': 'LSTM', 'Accuracy': lstm_acc})
        results_list.append({'Ticker': ticker, 'Model': 'GRU', 'Accuracy': gru_acc})
        results_list.append({'Ticker': ticker, 'Model': 'Transformer', 'Accuracy': tf_acc})
        
        # ==========================================
        # 关键修改：收集预测数据用于回测
        # ==========================================
        # 计算测试集对应的日期
        # create_sequences 中，y[i] 对应的是 input_data[i + seq_length] 的时间点
        # 测试集从 train_size 开始，所以测试集第一个点对应原始数据的索引是 train_size + seq_length
        test_start_index = train_size + SEQ_LENGTH
        test_dates = ticker_df['Date'].iloc[test_start_index : test_start_index + len(y_test_t)].values
        
        # 确保长度一致 (防止切片越界导致的微小差异)
        min_len = min(len(test_dates), len(lstm_probs))
        
        # 创建该股票的预测 DataFrame
        ticker_pred_df = pd.DataFrame({
            'Date': test_dates[:min_len],
            'Ticker': [ticker] * min_len,
            'Actual_Label': y_labels[train_size : train_size + min_len], # 真实涨跌
            'Prob_LSTM': lstm_probs[:min_len],
            'Prob_GRU': gru_probs[:min_len],
            'Prob_Transformer': tf_probs[:min_len]
        })
        
        all_predictions.append(ticker_pred_df)

    # 保存模型对比结果
    if results_list:
        res_df = pd.DataFrame(results_list)
        res_df.to_csv(RESULTS_PATH, index=False)
        print(f"\n🎉 All models trained. Comparison results saved to {RESULTS_PATH}")
        
        avg_perf = res_df.groupby('Model')['Accuracy'].mean()
        print("\nAverage Performance:")
        print(avg_perf)
        
    # 保存详细预测结果 (供回测使用)
    if all_predictions:
        final_pred_df = pd.concat(all_predictions, ignore_index=True)
        final_pred_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
        print(f"\n💾 Predictions saved for backtesting: {PREDICTIONS_SAVE_PATH}")
        print(f"   Columns: {final_pred_df.columns.tolist()}")
        print(f"   Total rows: {len(final_pred_df)}")

if __name__ == "__main__":
    run_week9_workflow()