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
SEQ_LENGTH = 60      
BATCH_SIZE = 64
TARGET_COL_SOURCE = 'target_1d_return' 

# 🛑 新增：交易成本设置 (关键修复)
# 0.001 代表单次交易成本 0.1% (包含佣金+滑点)
# 如果是双边交易（买+卖），这会显著降低不切实际的高收益
COST_RATE = 0.001 

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================

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
    filename = f"{model_type}_{ticker}.pth"
    model_path = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(model_path):
        return None

    if model_type == 'LSTM':
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif model_type == 'GRU':
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif model_type == 'Transformer':
        model = TransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    else:
        return None
    
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
    print("🚀 Starting Week 12: DL Models Backtest (With Transaction Costs)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])

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
    daily_returns_collection = {} 

    for ticker in available_tickers:
        print(f"\nBacktesting {ticker}...")
        
        # 1. 筛选数据
        ticker_df = df[df['Ticker'] == ticker].copy().reset_index(drop=True)
        ticker_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        feature_cols = [c for c in ticker_df.columns if c not in ['Date', 'Ticker'] and not c.startswith('target_')]
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c in numeric_cols and 'Unnamed' not in c]
        
        cols_to_check = feature_cols + [TARGET_COL_SOURCE]
        ticker_df.dropna(subset=cols_to_check, inplace=True)
        ticker_df.reset_index(drop=True, inplace=True)
        
        if len(ticker_df) < SEQ_LENGTH + 10:
            continue

        X_values = ticker_df[feature_cols].values

        # 🛑 修复数据泄露：Scaler 只能 fit 在训练集上！
        # 假设训练集是前 80% (与训练代码保持一致)
        train_size = int(len(X_values) * 0.8)
        scaler = StandardScaler()
        
        # Fit on Train ONLY
        scaler.fit(X_values[:train_size])
        # Transform ALL
        X_scaled = scaler.transform(X_values)

        X_seq = create_sequences(X_scaled, SEQ_LENGTH)
        
        if len(X_seq) == 0:
            continue

        # 这里的 valid_indices 对应的是 X_seq 的预测目标
        valid_indices = range(SEQ_LENGTH, len(ticker_df))
        
        # 我们只关心 Test 集部分的回测 (后 20%)
        # 因为在 Train 集上回测没有意义 (模型已经见过答案了)
        test_start_idx = train_size - SEQ_LENGTH 
        if test_start_idx < 0: test_start_idx = 0

        # 切片出测试集的数据
        X_seq_test = X_seq[test_start_idx:] 
        
        # 对应的日期和真实回报
        # 注意：valid_indices 是相对于原始 ticker_df 的索引
        test_indices = valid_indices[test_start_idx:]
        test_dates = ticker_df.iloc[test_indices]['Date'].values
        actual_returns = ticker_df.iloc[test_indices][TARGET_COL_SOURCE].values
        
        X_tensor = torch.FloatTensor(X_seq_test).to(device)
        input_dim = X_seq.shape[2]

        for model_type in ['LSTM', 'GRU', 'Transformer']:
            model = load_model_for_ticker(ticker, model_type, input_dim)
            if model is None:
                continue
            
            try:
                with torch.no_grad():
                    preds_prob = model(X_tensor).cpu().numpy().flatten()
            except Exception as e:
                print(f"  Error predicting {model_type}: {e}")
                continue
            
            # 生成信号 (1: Buy, 0: Hold/Sell)
            signals = (preds_prob > 0.5).astype(int)
            
            # 🛑 修复：计算交易成本
            # 计算换手率：如果昨天是0，今天是1，说明买入；昨天1，今天0，说明卖出。
            # np.roll 将数组向后移一位
            prev_signals = np.roll(signals, 1)
            prev_signals[0] = 0 # 第一天假设之前是空仓
            
            # 换手动作 (0->1 或 1->0 都是 1)
            turnover = np.abs(signals - prev_signals)
            
            # 扣除成本：每次换手扣除 COST_RATE
            costs = turnover * COST_RATE
            
            # 净收益 = (持仓 * 涨跌幅) - 交易成本
            strategy_returns = (signals * actual_returns) - costs
            
            # 收集每日收益数据
            series_name = f"{ticker}_{model_type}"
            daily_returns_collection[series_name] = pd.Series(strategy_returns, index=test_dates)

            # --- 计算指标 ---
            # 使用复利计算总回报 (Compound Return)
            cum_strategy = (1 + strategy_returns).cumprod()
            final_return = cum_strategy[-1] - 1 if len(cum_strategy) > 0 else 0
            
            daily_std = np.std(strategy_returns)
            sharpe_ratio = (np.mean(strategy_returns) / daily_std) * np.sqrt(252) if daily_std > 1e-9 else 0.0
            
            # 最大回撤计算
            running_max = np.maximum.accumulate(cum_strategy)
            drawdown_curve = (cum_strategy - running_max) / running_max
            max_drawdown = np.min(drawdown_curve)

            y_true_binary = (actual_returns > 0).astype(int)
            hit_ratio = accuracy_score(y_true_binary, signals)

            summary_results.append({
                'Ticker': ticker,
                'Model': model_type,
                'Total_Return': final_return,
                'Hit_Ratio': hit_ratio,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown
            })
            
            print(f"  > {model_type}: Ret={final_return:.2%}, Sharpe={sharpe_ratio:.2f}")

    # ======================================================
    # 4. 保存结果
    # ======================================================
    
    if summary_results:
        res_df = pd.DataFrame(summary_results)
        metrics_path = os.path.join(RESULTS_DIR, 'backtest_shortterm_metrics.csv')
        res_df.to_csv(metrics_path, index=False)
        print(f"\n✅ Summary metrics saved to {metrics_path}")
    
    if daily_returns_collection:
        print("\n🔄 Aggregating daily returns for Portfolio Optimizer...")
        
        df_all_returns = pd.DataFrame(daily_returns_collection)
        df_all_returns.sort_index(inplace=True)
        df_all_returns.fillna(0, inplace=True)
        
        df_all_returns['Composite_Daily_Return'] = df_all_returns.mean(axis=1)
        
        cumulative_returns = (1 + df_all_returns['Composite_Daily_Return']).cumprod()
        
        initial_capital = 10000.0
        
        final_ts_df = pd.DataFrame({
            'Date': df_all_returns.index,
            'Daily_Return': df_all_returns['Composite_Daily_Return'],
            'Total_Return': cumulative_returns,
            'Portfolio Value': initial_capital * cumulative_returns 
        })
        
        ts_path = os.path.join(RESULTS_DIR, 'backtest_shortterm.csv')
        final_ts_df.to_csv(ts_path, index=False)
        
        print(f"✅ Time-series data saved to {ts_path}")
        
    else:
        print("\n⚠️ No returns data collected. Check if models exist.")

if __name__ == "__main__":
    run_backtest()