import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
import json
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 配置项
# ==========================================
DATA_PATH = 'data/finalized/data_with_targets.csv'
TUNED_MODELS_DIR = 'models/tuned_models'
RESULTS_LOG_PATH = 'tuning_performance_results.csv'
SEQ_LENGTH = 60      # 回溯窗口
N_TRIALS = 15        # 每个任务尝试多少次优化 (正式跑建议设为 50+)
EPOCHS_PER_TRIAL = 10 # 每次尝试训练几轮 (为了速度设低一点，正式跑可设 20)
SAMPLE_TICKERS = 3   # 限制调优的股票数量，节省时间

# 定义三个预测周期 (Horizon)
HORIZONS = {
    'Short': 1,   # 预测 1 天后
    'Mid': 5,     # 预测 5 天后
    'Long': 10    # 预测 10 天后
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(TUNED_MODELS_DIR, exist_ok=True)

# ==========================================
# 1. 模型定义 (动态参数版)
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
# 2. 数据处理工具
# ==========================================
def prepare_data_for_horizon(df, ticker, horizon_days):
    """
    为特定的 Horizon 动态生成 Target
    """
    t_df = df[df['Ticker'] == ticker].copy()
    
    # 必须按时间排序
    if 'Date' in t_df.columns:
        t_df['Date'] = pd.to_datetime(t_df['Date'])
        t_df = t_df.sort_values('Date')
    
    # 动态生成 Target: 未来 N 天的收盘价 > 当前收盘价
    # 假设 'Close' 或 'Adj Close' 存在，如果没有，尝试用 numeric columns 的第一列作为价格代理
    price_col = 'Close' if 'Close' in t_df.columns else t_df.select_dtypes(include=[np.number]).columns[0]
    
    t_df['Target_Horizon'] = (t_df[price_col].shift(-horizon_days) > t_df[price_col]).astype(float)
    
    # 移除最后 horizon_days 行 (因为没有 Target)
    t_df = t_df.dropna(subset=['Target_Horizon'])
    
    # 特征选择 (排除非数值和 Target)
    feature_cols = [c for c in t_df.columns if c not in ['Date', 'Ticker', 'Target_Horizon'] and not c.startswith('target_')]
    numeric_cols = t_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    data_x = t_df[feature_cols].values
    data_y = t_df['Target_Horizon'].values
    
    # 标准化
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    
    # 创建序列
    xs, ys = [], []
    for i in range(len(data_x) - SEQ_LENGTH):
        xs.append(data_x[i:(i + SEQ_LENGTH)])
        ys.append(data_y[i + SEQ_LENGTH])
        
    return np.array(xs), np.array(ys), len(feature_cols)

# ==========================================
# 3. Optuna Objective Function
# ==========================================
def objective(trial, X_train, y_train, X_val, y_val, input_dim):
    # 1. 采样超参数
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'Transformer'])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    model = None
    
    if model_type == 'LSTM':
        hidden_dim = trial.suggest_int('lstm_hidden', 32, 128)
        num_layers = trial.suggest_int('lstm_layers', 1, 3)
        model = LSTMClassifier(input_dim, hidden_dim, num_layers, dropout).to(device)
        
    elif model_type == 'GRU':
        hidden_dim = trial.suggest_int('gru_hidden', 32, 128)
        num_layers = trial.suggest_int('gru_layers', 1, 3)
        model = GRUClassifier(input_dim, hidden_dim, num_layers, dropout).to(device)
        
    elif model_type == 'Transformer':
        # Transformer 要求 d_model 能被 nhead 整除
        nhead = trial.suggest_categorical('nhead', [2, 4])
        # 这里的 d_model 设为 nhead 的倍数
        d_model_multiplier = trial.suggest_int('d_model_mult', 8, 32) 
        d_model = nhead * d_model_multiplier
        num_layers = trial.suggest_int('tf_layers', 1, 3)
        model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dropout).to(device)

    # 2. 训练过程
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 转换为 Tensor
    # 注意：为了速度，这里不使用 DataLoader，直接全量或简单切片。
    # 在正式大规模训练中应使用 DataLoader。
    xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    model.train()
    for epoch in range(EPOCHS_PER_TRIAL):
        optimizer.zero_grad()
        output = model(xt)
        loss = criterion(output, yt)
        loss.backward()
        optimizer.step()
        
        # 剪枝 (Pruning): 如果效果太差提前终止
        # trial.report(loss.item(), epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # 3. 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(xv)
        val_labels = (val_pred > 0.5).float()
        acc = accuracy_score(yv.cpu(), val_labels.cpu())
        
    return acc

# ==========================================
# 4. 主流程
# ==========================================
def run_optimization():
    print("🚀 Starting Week 10: Hyperparameter Optimization (Optuna)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    tickers = df['Ticker'].value_counts().head(SAMPLE_TICKERS).index.tolist()
    
    results = []
    best_params_registry = {}

    for ticker in tickers:
        print(f"\n🔹 Optimizing for Ticker: {ticker}")
        best_params_registry[ticker] = {}
        
        for horizon_name, horizon_days in HORIZONS.items():
            print(f"   > Horizon: {horizon_name} ({horizon_days} days)")
            
            # 准备数据
            X, y, input_dim = prepare_data_for_horizon(df, ticker, horizon_days)
            
            if len(X) < 100:
                print("     Not enough data, skipping.")
                continue
                
            # 划分训练/验证 (80/20)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            # 定义 Optuna Study
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            
            # 包装 objective 函数以传递额外参数
            func = lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim)
            
            # 开始优化
            study.optimize(func, n_trials=N_TRIALS)
            
            best_acc = study.best_value
            best_params = study.best_params
            
            print(f"     ✅ Best Acc: {best_acc:.4f} | Model: {best_params['model_type']}")
            
            # 记录结果
            results.append({
                'Ticker': ticker,
                'Horizon': horizon_name,
                'Best_Model': best_params['model_type'],
                'Best_Accuracy': best_acc,
                'Trials': N_TRIALS
            })
            
            best_params_registry[ticker][horizon_name] = best_params

    # 保存最佳参数
    params_path = os.path.join(TUNED_MODELS_DIR, 'best_hyperparameters.json')
    with open(params_path, 'w') as f:
        json.dump(best_params_registry, f, indent=4)
        
    # 保存性能对比
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_LOG_PATH, index=False)
    
    print("\n" + "="*40)
    print(f"🎉 Optimization Complete!")
    print(f"📄 Best parameters saved to: {params_path}")
    print(f"📊 Performance log saved to: {RESULTS_LOG_PATH}")
    print("="*40)

if __name__ == "__main__":
    run_optimization()