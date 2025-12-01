import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# ==========================================
# 配置项
# ==========================================
DATA_PATH = 'data/finalized/data_with_targets.csv'
MODEL_SAVE_DIR = 'models/dl_checkpoints'
RESULTS_PATH = 'baseline_results.csv' 
SEQ_LENGTH = 60   # 时间窗口：用过去60天预测第61天
EPOCHS = 20       # 训练轮数
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ⚠️ 修正点：根据您的截图，这里改成了 'target_1d_return'
# 您也可以根据需要改为 'target_5d_return' 等
TARGET_COL = 'target_1d_return' 

SAMPLE_TICKERS_COUNT = 10    

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 确保保存目录存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==========================================
# 1. 数据预处理工具：创建时间序列
# ==========================================
def create_sequences(input_data, target_data, seq_length):
    """
    input_data: 特征数据 (N, features)
    target_data: 目标数据 (N,)
    返回: 
    xs: (samples, seq_length, features)
    ys: (samples,)
    """
    xs, ys = [], []
    # 确保长度一致
    length = len(input_data)
    for i in range(length - seq_length):
        x = input_data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 2. 定义 LSTM 模型
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出用于预测
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 3. 主执行逻辑
# ==========================================
def run_week9_workflow():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: 文件不存在 -> {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # --- 检查目标列是否存在 ---
    if TARGET_COL not in df.columns:
        print(f"\n❌ 错误: 在CSV中找不到列名 '{TARGET_COL}'")
        print(f"ℹ️ CSV文件中的可用列名有: {df.columns.tolist()}")
        print("👉 请检查代码第 23 行的 TARGET_COL 变量。\n")
        return
    # ----------------------------------

    # 处理日期
    if 'Date' in df.columns:
        # errors='coerce' 会将无法解析的日期设为 NaT，防止报错
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # 删除日期解析失败的行（如果有）
        df = df.dropna(subset=['Date'])
        df = df.sort_values(['Ticker', 'Date'])
    
    # 筛选样本股票
    top_tickers = df['Ticker'].value_counts().head(SAMPLE_TICKERS_COUNT).index.tolist()
    print(f"Selected Sample Tickers: {top_tickers}")

    results_list = []

    for ticker in top_tickers:
        print(f"\nProcessing {ticker}...")
        
        # 1. 获取该股票数据
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # 填充缺失值
        ticker_df = ticker_df.ffill().bfill()

        # --- 动态选择数值特征列 ---
        # 排除 Date, Ticker 和 所有的 target_ 开头的列（防止数据泄露）
        feature_cols = [c for c in ticker_df.columns if c not in ['Date', 'Ticker'] and not c.startswith('target_')]
        
        # 确保只选数值类型
        numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c in numeric_cols]

        # 移除可能的干扰列 (如 Unnamed)
        feature_cols = [c for c in feature_cols if 'Unnamed' not in c]
        
        # 提取数据
        X_values = ticker_df[feature_cols].values
        y_values = ticker_df[TARGET_COL].values
        # ----------------------------------

        # 2. 归一化
        # 特征归一化
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler_X.fit_transform(X_values)
        
        # 目标归一化
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler_y.fit_transform(y_values.reshape(-1, 1)).flatten()

        # 3. 创建序列数据
        X, y = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)
        
        if len(X) < 100:
            print(f"Not enough data for {ticker}, skipping.")
            continue

        # 4. 划分训练集/测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 转换为 Tensor
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

        # 5. 初始化模型
        input_dim = X_train.shape[2]
        model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 6. 训练循环
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

        # 7. 保存模型
        model_path = os.path.join(MODEL_SAVE_DIR, f"lstm_{ticker}.pth")
        torch.save(model.state_dict(), model_path)

        # 8. 评估
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy()
            actual_scaled = y_test_t.cpu().numpy()

        # 反归一化以计算真实的 RMSE
        pred_actual = scaler_y.inverse_transform(pred_scaled)
        y_actual = scaler_y.inverse_transform(actual_scaled)

        mse = mean_squared_error(y_actual, pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, pred_actual)
        r2 = r2_score(y_actual, pred_actual)

        print(f"  {ticker} | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

        # 记录结果
        results_list.append({
            'Model': 'LSTM (DL)',
            'Ticker': ticker,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        })

    # ==========================================
    # 4. 保存对比结果
    # ==========================================
    if results_list:
        new_results_df = pd.DataFrame(results_list)
        
        if os.path.exists(RESULTS_PATH):
            old_results = pd.read_csv(RESULTS_PATH)
            final_df = pd.concat([old_results, new_results_df], ignore_index=True)
        else:
            final_df = new_results_df
        
        final_df.to_csv(RESULTS_PATH, index=False)
        print(f"\n✅ Updated results saved to {RESULTS_PATH}")
        print(final_df.tail())
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_week9_workflow()