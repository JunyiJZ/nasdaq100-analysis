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
RESULTS_DIR = 'data/backtest_results'
SEQ_LENGTH = 60

# 交易策略配置 (激进模式)
BUY_THRESHOLD = 0.50  # 预测概率 > 0.50 就买
SELL_THRESHOLD = 0.50 # 预测概率 < 0.50 就卖
INITIAL_CAPITAL = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. 模型定义 (必须与 Week 10 保持一致)
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
def prepare_data(df, ticker, horizon_days):
    """与 Week 10 逻辑一致的数据准备"""
    t_df = df[df['Ticker'] == ticker].copy()
    if 'Date' in t_df.columns:
        t_df['Date'] = pd.to_datetime(t_df['Date'])
        t_df = t_df.sort_values('Date')
    
    # 确定价格列
    price_col = 'Close' if 'Close' in t_df.columns else t_df.select_dtypes(include=[np.number]).columns[0]
    
    # 生成 Target
    t_df['Target'] = (t_df[price_col].shift(-horizon_days) > t_df[price_col]).astype(float)
    t_df = t_df.dropna(subset=['Target'])
    
    # 特征
    feature_cols = [c for c in t_df.columns if c not in ['Date', 'Ticker', 'Target'] and not c.startswith('target_')]
    numeric_cols = t_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    # 原始价格用于回测计算收益
    raw_prices = t_df[price_col].values
    
    data_x = t_df[feature_cols].values
    data_y = t_df['Target'].values
    
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    
    xs, ys, prices = [], [], []
    for i in range(len(data_x) - SEQ_LENGTH):
        xs.append(data_x[i:(i + SEQ_LENGTH)])
        ys.append(data_y[i + SEQ_LENGTH])
        prices.append(raw_prices[i + SEQ_LENGTH]) # 记录对应当天的价格
        
    return np.array(xs), np.array(ys), np.array(prices), len(feature_cols)

def train_and_predict(model_cls, params, input_dim, X_train, y_train, X_test):
    """根据参数实例化模型，训练，并预测"""
    
    # 1. 实例化模型
    model_type = params['model_type']
    dropout = params['dropout']
    
    if model_type == 'LSTM':
        model = LSTMClassifier(input_dim, params['lstm_hidden'], params['lstm_layers'], dropout).to(device)
    elif model_type == 'GRU':
        model = GRUClassifier(input_dim, params['gru_hidden'], params['gru_layers'], dropout).to(device)
    elif model_type == 'Transformer':
        # 重新计算 d_model
        d_model = params['nhead'] * params['d_model_mult']
        model = TransformerClassifier(input_dim, d_model, params['nhead'], params['tf_layers'], dropout).to(device)
    
    # 2. 训练
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    model.train()
    # 稍微多训练几轮以确保收敛
    epochs = 15 
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()
        
    # 3. 预测
    model.eval()
    with torch.no_grad():
        xv = torch.FloatTensor(X_test).to(device)
        preds = model(xv).cpu().numpy().flatten()
        
    return preds

# ==========================================
# 4. 主回测引擎
# ==========================================
def run_backtest_engine():
    print("🚀 Starting Week 11: Backtesting Engine (PyTorch)...")
    
    if not os.path.exists(PARAMS_PATH):
        print("❌ Best hyperparameters not found. Please run Week 10 first.")
        return

    with open(PARAMS_PATH, 'r') as f:
        best_params_registry = json.load(f)
        
    df = pd.read_csv(DATA_PATH)
    
    results = []
    
    # 遍历所有已调优的股票和周期
    for ticker, horizons in best_params_registry.items():
        for horizon_name, params in horizons.items():
            print(f"\n🔄 Backtesting: {ticker} [{horizon_name}] using {params['model_type']}...")
            
            horizon_days = {'Short': 1, 'Mid': 5, 'Long': 10}.get(horizon_name, 1)
            
            # 1. 准备数据
            X, y, prices, input_dim = prepare_data(df, ticker, horizon_days)
            
            if len(X) < 100:
                print("   ⚠️ Not enough data.")
                continue
                
            # 2. 划分训练/测试集 (80% 训练, 20% 回测)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            prices_test = prices[split:]
            
            # 3. 重新训练并预测
            try:
                probs = train_and_predict(None, params, input_dim, X_train, y_train, X_test)
            except Exception as e:
                print(f"   ❌ Model Error: {e}")
                continue
            
            # 4. 执行交易策略 (Simulation)
            cash = INITIAL_CAPITAL
            position = 0 # 0 = 空仓, >0 = 持仓股数
            trades = 0
            
            # 记录资产曲线
            portfolio_values = []
            
            for i in range(len(probs) - 1):
                current_price = prices_test[i]
                prob = probs[i]
                
                # 激进策略逻辑
                if prob > BUY_THRESHOLD and position == 0:
                    # 买入 (全仓)
                    position = cash / current_price
                    cash = 0
                    trades += 1
                elif prob < SELL_THRESHOLD and position > 0:
                    # 卖出 (清仓)
                    cash = position * current_price
                    position = 0
                    trades += 1
                
                # 计算当前总资产
                curr_val = cash + (position * current_price)
                portfolio_values.append(curr_val)
            
            # 5. 结算
            final_price = prices_test[-1]
            final_value = cash + (position * final_price)
            roi = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            
            # 简单的基准对比 (Buy & Hold)
            initial_price = prices_test[0]
            buy_hold_roi = (final_price - initial_price) / initial_price * 100
            
            print(f"   💰 Final Value: ${final_value:.2f} | ROI: {roi:.2f}% | Trades: {trades}")
            print(f"   📊 Buy & Hold ROI: {buy_hold_roi:.2f}%")
            
            results.append({
                'Ticker': ticker,
                'Horizon': horizon_name,
                'Model': params['model_type'],
                'ROI': roi,
                'Buy_Hold_ROI': buy_hold_roi,
                'Trades': trades,
                'Win_Rate': 'N/A' # 暂略
            })

    # 保存结果
    res_df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'backtest_summary.csv')
    res_df.to_csv(save_path, index=False)
    print("\n" + "="*40)
    print(f"✅ Backtest Complete. Results saved to: {save_path}")
    if not res_df.empty:
        print(res_df[['Ticker', 'Horizon', 'Model', 'ROI', 'Trades']])

if __name__ == "__main__":
    run_backtest_engine()