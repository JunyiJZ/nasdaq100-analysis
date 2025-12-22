import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

# ==========================================
# 配置项
# ==========================================
DATA_PATH = 'data/finalized/data_with_targets.csv'
MODEL_DIR = 'models/dl_checkpoints'
RESULTS_DIR = 'backtest_results'
SEQ_LENGTH = 60      # 必须与训练时保持一致
BATCH_SIZE = 64
TARGET_COL_SOURCE = 'target_1d_return' # 预测目标列

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. 模型定义 (保持不变)
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
        device = x.device
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
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# --- 模型 C: Transformer ---
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

# ==========================================
# 2. 数据处理辅助函数
# ==========================================
def create_sequences(input_data, seq_length):
    xs = []
    length = len(input_data)
    for i in range(length - seq_length):
        x = input_data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

def load_model_for_ticker(ticker, model_type, input_dim):
    """
    根据 Ticker 和模型类型加载对应的 .pth 文件
    """
    filename = f"{model_type}_{ticker}.pth"
    model_path = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(model_path):
        return None

    # 实例化对应的模型类
    if model_type == 'LSTM':
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif model_type == 'GRU':
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif model_type == 'Transformer':
        model = TransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    else:
        return None
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"  ❌ Error loading {filename}: {e}")
        return None

# ==========================================
# 3. 回测主逻辑
# ==========================================
def run_backtest():
    print("🚀 Starting Week 12: DL Models Backtest...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])

    # 找出所有有模型文件的 Ticker
    model_files = os.listdir(MODEL_DIR)
    available_tickers = set()
    for f in model_files:
        if f.endswith('.pth'):
            parts = f.replace('.pth', '').split('_')
            if len(parts) >= 2:
                ticker = parts[1]
                available_tickers.add(ticker)
    
    print(f"Found models for tickers: {list(available_tickers)}")
    
    summary_results = []

    for ticker in available_tickers:
        print(f"\nBacktesting {ticker}...")
        
        # 1. 筛选数据
        ticker_df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
        
        # 清理 NaN 和 Inf 值
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        feature_cols = [c for c in ticker_df.columns if c not in ['Date', 'Ticker'] and not c.startswith('target_')]
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c in numeric_cols and 'Unnamed' not in c]
        
        cols_to_check = feature_cols + [TARGET_COL_SOURCE]
        
        initial_len = len(ticker_df)
        ticker_df.dropna(subset=cols_to_check, inplace=True)
        ticker_df.reset_index(drop=True, inplace=True)
        
        if len(ticker_df) < SEQ_LENGTH + 10:
            print(f"  Skipping {ticker}: Not enough data after cleaning.")
            continue

        # 2. 准备特征数据
        X_values = ticker_df[feature_cols].values
        
        if np.isnan(X_values).any():
            print(f"  Skipping {ticker}: Data still contains NaNs after cleaning.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_values)
        
        X_seq = create_sequences(X_scaled, SEQ_LENGTH)
        
        if len(X_seq) == 0:
            print(f"  Skipping {ticker}: No sequences created.")
            continue

        valid_indices = range(SEQ_LENGTH, len(ticker_df))
        test_dates = ticker_df.iloc[valid_indices]['Date'].values
        actual_returns = ticker_df.iloc[valid_indices][TARGET_COL_SOURCE].values
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        input_dim = X_seq.shape[2]

        for model_type in ['LSTM', 'GRU', 'Transformer']:
            model = load_model_for_ticker(ticker, model_type, input_dim)
            if model is None:
                continue
            
            # 预测
            try:
                with torch.no_grad():
                    preds_prob = model(X_tensor).cpu().numpy().flatten()
            except Exception as e:
                print(f"  Error predicting {model_type}: {e}")
                continue
            
            # 策略信号
            signals = (preds_prob > 0.5).astype(int)
            
            # 收益计算
            strategy_returns = signals * actual_returns
            cum_strategy = np.cumsum(strategy_returns)
            cum_market = np.cumsum(actual_returns)
            
            final_return = cum_strategy[-1] if len(cum_strategy) > 0 else 0
            
            if np.isnan(final_return):
                print(f"  ⚠️ {model_type}: Return is NaN")
                continue

            # ======================================================
            # 新增：计算高级指标 (Sharpe, Drawdown, Precision/Recall)
            # ======================================================
            
            # 1. 基础分类指标
            y_true_binary = (actual_returns > 0).astype(int)
            hit_ratio = accuracy_score(y_true_binary, signals)
            precision = precision_score(y_true_binary, signals, zero_division=0)
            recall = recall_score(y_true_binary, signals, zero_division=0)

            # 2. 金融指标
            # 夏普比率 (假设无风险利率为0，年化252天)
            daily_std = np.std(strategy_returns)
            if daily_std > 1e-9:
                sharpe_ratio = (np.mean(strategy_returns) / daily_std) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # 最大回撤 (Max Drawdown)
            # 假设 cum_strategy 是累积收益率 (additive returns)
            running_max = np.maximum.accumulate(cum_strategy)
            drawdown_curve = cum_strategy - running_max
            max_drawdown = np.min(drawdown_curve)

            summary_results.append({
                'Ticker': ticker,
                'Model': model_type,
                'Total_Return': final_return,
                'Hit_Ratio': hit_ratio,
                'Precision': precision,
                'Recall': recall,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown
            })

            # ======================================================
            # 绘图：Equity Curve + Drawdowns
            # ======================================================
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # 上图：资金曲线
            ax1.plot(test_dates, cum_market, label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
            ax1.plot(test_dates, cum_strategy, label=f'{model_type} Strategy', color='blue', linewidth=1.5)
            ax1.set_title(f'{ticker} - {model_type} Performance')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # 下图：回撤曲线
            ax2.plot(test_dates, drawdown_curve, label='Drawdown', color='red', linewidth=1)
            ax2.fill_between(test_dates, drawdown_curve, 0, color='red', alpha=0.2)
            ax2.set_ylabel('Drawdown')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'{ticker}_{model_type}_performance.png'))
            plt.close()
            
            print(f"  > {model_type}: Ret={final_return:.2f}, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2f}")

    # 保存汇总结果
    if summary_results:
        res_df = pd.DataFrame(summary_results)
        # 按照任务要求保存为 backtest_shortterm.csv
        res_path = os.path.join(RESULTS_DIR, 'backtest_shortterm.csv')
        res_df.to_csv(res_path, index=False)
        print(f"\n✅ Backtest complete. Results saved to {res_path}")
        
        # 打印平均指标
        print("\n--- Average Metrics by Model ---")
        print(res_df.groupby('Model')[['Total_Return', 'Sharpe_Ratio', 'Hit_Ratio', 'Max_Drawdown']].mean())
    else:
        print("\n⚠️ No results generated.")

if __name__ == "__main__":
    run_backtest()