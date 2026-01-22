import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 配置与参数
# ==========================================
DATA_PATH = 'data/finalized/data_with_targets.csv'
PARAMS_PATH = 'models/tuned_models/best_hyperparameters.json'
RESULTS_DIR = 'backtest_results'  # 确保结果目录正确
SEQ_LENGTH = 60

# 交易策略配置
INITIAL_CAPITAL = 10000
CONFIDENCE_THRESHOLD = 0.05 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, src):
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output[:, -1, :])
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
# 3. 辅助函数
# ==========================================
def create_sequences(data_x, data_y, prices, dates, seq_length):
    xs, ys, ps, ds = [], [], [], []
    for i in range(len(data_x) - seq_length):
        xs.append(data_x[i:(i + seq_length)])
        ys.append(data_y[i + seq_length])
        ps.append(prices[i + seq_length])
        ds.append(dates[i + seq_length]) # 同时保存日期
    return np.array(xs), np.array(ys), np.array(ps), np.array(ds)

def prepare_data_split(df, ticker, horizon_days):
    t_df = df[df['Ticker'] == ticker].copy()
    if 'Date' in t_df.columns:
        t_df['Date'] = pd.to_datetime(t_df['Date'])
        t_df = t_df.sort_values('Date')
    
    price_col = 'Close' if 'Close' in t_df.columns else t_df.select_dtypes(include=[np.number]).columns[0]
    
    # 生成 Target
    t_df['Target'] = (t_df[price_col].shift(-horizon_days) > t_df[price_col]).astype(float)
    t_df = t_df.dropna(subset=['Target'])
    
    feature_cols = [c for c in t_df.columns if c not in ['Date', 'Ticker', 'Target'] and not c.startswith('target_')]
    numeric_cols = t_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    raw_x = t_df[feature_cols].values
    raw_y = t_df['Target'].values
    raw_prices = t_df[price_col].values
    raw_dates = t_df['Date'].values # 获取原始日期
    
    # 切分 Train/Test
    split_idx = int(len(raw_x) * 0.8)
    
    train_x_raw = raw_x[:split_idx]
    test_x_raw = raw_x[split_idx:]
    
    train_y = raw_y[:split_idx]
    test_y = raw_y[split_idx:]
    
    train_prices = raw_prices[:split_idx]
    test_prices = raw_prices[split_idx:]
    
    train_dates = raw_dates[:split_idx]
    test_dates = raw_dates[split_idx:]
    
    # Scaling
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x_raw)
    test_x_scaled = scaler.transform(test_x_raw)
    
    # 生成序列 (注意这里增加了 dates 返回)
    X_train, y_train, _, _ = create_sequences(train_x_scaled, train_y, train_prices, train_dates, SEQ_LENGTH)
    X_test, y_test, prices_test, dates_test = create_sequences(test_x_scaled, test_y, test_prices, test_dates, SEQ_LENGTH)
    
    return X_train, y_train, X_test, y_test, prices_test, dates_test, len(feature_cols)

def train_and_predict(model_cls, params, input_dim, X_train, y_train, X_test):
    model_type = params['model_type']
    dropout = params['dropout']
    
    if model_type == 'LSTM':
        model = LSTMClassifier(input_dim, params['lstm_hidden'], params['lstm_layers'], dropout).to(device)
    elif model_type == 'GRU':
        model = GRUClassifier(input_dim, params['gru_hidden'], params['gru_layers'], dropout).to(device)
    elif model_type == 'Transformer':
        d_model = params['nhead'] * params['d_model_mult']
        model = TransformerClassifier(input_dim, d_model, params['nhead'], params['tf_layers'], dropout).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    model.train()
    epochs = 15 # 演示用，实际可增加
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        xv = torch.FloatTensor(X_test).to(device)
        preds = model(xv).cpu().numpy().flatten()
        
    return preds

# ==========================================
# 4. 主回测引擎
# ==========================================
def run_backtest_engine():
    print("🚀 Starting Week 11: Backtesting Engine...")
    print(f"📂 Results will be saved to: {RESULTS_DIR}")
    
    if not os.path.exists(PARAMS_PATH):
        print("❌ Best hyperparameters not found.")
        return

    with open(PARAMS_PATH, 'r') as f:
        best_params_registry = json.load(f)
        
    df = pd.read_csv(DATA_PATH)
    results = []
    
    # === 新增：专门用于收集 Mid-term 数据的列表 ===
    midterm_predictions = [] 
    
    for ticker, horizons in best_params_registry.items():
        for horizon_name, params in horizons.items():
            print(f"\n🔄 Backtesting: {ticker} [{horizon_name}]...")
            
            horizon_days = {'Short': 1, 'Mid': 5, 'Long': 10}.get(horizon_name, 1)
            
            try:
                X_train, y_train, X_test, y_test, prices_test, dates_test, input_dim = prepare_data_split(df, ticker, horizon_days)
            except ValueError:
                print("   ⚠️ Not enough data to split.")
                continue

            if len(X_train) < 50 or len(X_test) < 10:
                print("   ⚠️ Not enough data.")
                continue
                
            try:
                probs = train_and_predict(None, params, input_dim, X_train, y_train, X_test)
            except Exception as e:
                print(f"   ❌ Model Error: {e}")
                continue
            
            # === 关键修改：如果是 Mid 策略，收集详细数据供 Week 13 使用 ===
            if horizon_name == 'Mid':
                mid_df = pd.DataFrame({
                    'Date': dates_test,
                    'Ticker': ticker,
                    'Signal': probs,       # 预测概率
                    'Close': prices_test,  # 收盘价
                    'Target': y_test       # 真实标签
                })
                midterm_predictions.append(mid_df)
            
            # --- 执行交易策略 ---
            cash = INITIAL_CAPITAL
            position = 0 
            trades = 0
            
            buy_thresh = 0.50 + CONFIDENCE_THRESHOLD
            sell_thresh = 0.50 - CONFIDENCE_THRESHOLD
            
            for i in range(len(probs) - 1):
                current_price = prices_test[i]
                prob = probs[i]
                
                if prob > buy_thresh and position == 0:
                    position = cash / current_price
                    cash = 0
                    trades += 1
                elif prob < sell_thresh and position > 0:
                    cash = position * current_price
                    position = 0
                    trades += 1
                
            final_price = prices_test[-1]
            final_value = cash + (position * final_price)
            roi = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            initial_price = prices_test[0]
            buy_hold_roi = (final_price - initial_price) / initial_price * 100
            
            pred_dirs = (probs > 0.5).astype(float)
            accuracy = (pred_dirs == y_test).mean() * 100
            
            print(f"   💰 Final: ${final_value:.2f} | ROI: {roi:.2f}% | Trades: {trades}")
            print(f"   🎯 Win Rate: {accuracy:.2f}%")
            
            results.append({
                'Ticker': ticker,
                'Horizon': horizon_name,
                'Model': params['model_type'],
                'ROI': roi,
                'Buy_Hold_ROI': buy_hold_roi,
                'Trades': trades,
                'Win_Rate': f"{accuracy:.1f}%"
            })

    # 1. 保存回测摘要
    res_df = pd.DataFrame(results)
    summary_path = os.path.join(RESULTS_DIR, 'backtest_summary.csv')
    res_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # 2. 保存 Week 13 所需的 Mid-term 详细预测文件
    if midterm_predictions:
        all_mid_df = pd.concat(midterm_predictions)
        # 确保日期格式正确
        all_mid_df['Date'] = pd.to_datetime(all_mid_df['Date'])
        
        mid_file_path = os.path.join(RESULTS_DIR, 'daily_signals_midterm.csv')
        all_mid_df.to_csv(mid_file_path, index=False)
        print(f"✅ [重要] Week 13 所需文件已生成: {mid_file_path}")
    else:
        print("⚠️ 警告: 未生成 Mid-term 预测数据，Week 13 可能会报错。请检查 best_hyperparameters.json 是否包含 'Mid' 策略。")

    if not res_df.empty:
        print("\n" + "="*60)
        print(res_df[['Ticker', 'Horizon', 'Model', 'ROI', 'Buy_Hold_ROI', 'Trades', 'Win_Rate']].to_string(index=False))

if __name__ == "__main__":
    run_backtest_engine()



























































    