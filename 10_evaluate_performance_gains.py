import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 配置与辅助函数
# ==========================================
# 为了节省时间，演示时我们只选前 3 个股票进行对比报告
# 如果你想跑全量，把这个列表设为 None
TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL'] 

def prepare_data(df, ticker, horizon_days, look_back=60):
    # (这里直接复用之前的代码，保持一致性)
    data = df[df['Ticker'] == ticker].copy()
    if len(data) == 0: return None, None, None, None, None
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']
    data['Target'] = (data['Close'].shift(-horizon_days) > data['Close']).astype(int)
    data = data[feature_cols + ['Target']].dropna()
    if len(data) == 0: return None, None, None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols]) 

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(data['Target'].iloc[i])
    X, y = np.array(X), np.array(y)
    
    if len(X) < 100: return None, None, None, None, None # 数据太少跳过

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler

# ==========================================
# 2. 构建模型函数
# ==========================================

# A. 默认模型 (Untuned / Baseline)
# 这是一个“凭感觉”设置的普通模型，用来做对比基准
def build_untuned_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape)) # 默认50个单元
    model.add(Dropout(0.2)) # 默认0.2
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# B. 调优后的模型 (Tuned)
# 从 JSON 读取最佳参数
def build_tuned_model(params, input_shape):
    model = Sequential()
    model.add(LSTM(units=params['units_1'], return_sequences=params['return_sequences'], input_shape=input_shape))
    model.add(Dropout(params['dropout_1']))
    if params['return_sequences']:
        model.add(LSTM(units=params['units_2'], return_sequences=False))
        model.add(Dropout(params['dropout_2']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 3. 主对比流程
# ==========================================
def evaluate_gains():
    # 路径设置
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'data', 'finalized', 'final_master_dataset.csv')
    params_path = os.path.join(base_dir, 'data', 'tuned_models', 'best_hyperparameters.json')
    
    if not os.path.exists(params_path):
        print("❌ 找不到最佳参数文件，请先运行 Step 10！")
        return

    print("正在读取数据...")
    df = pd.read_csv(data_path)
    with open(params_path, 'r') as f:
        best_params_record = json.load(f)

    results = []

    # 遍历 JSON 里记录的每一个 (Ticker + Horizon)
    for key, record in best_params_record.items():
        parts = key.split('_')
        horizon_name = parts[-1]
        ticker = "_".join(parts[:-1])
        
        # 如果设置了测试列表，且该股票不在列表里，跳过（为了快速生成报告）
        if TEST_TICKERS and ticker not in TEST_TICKERS:
            continue

        horizon_map = {'Short': 1, 'Medium': 5, 'Long': 20}
        horizon_days = horizon_map.get(horizon_name, 1)

        print(f"\n📊 正在对比: {ticker} - {horizon_name} ...")

        # 1. 准备数据
        X_train, y_train, X_test, y_test, _ = prepare_data(df, ticker, horizon_days)
        if X_train is None: continue

        # --- 跑默认模型 (Untuned) ---
        start_time = time.time()
        model_untuned = build_untuned_model((X_train.shape[1], X_train.shape[2]))
        # 简单训练 10 个 epoch 看效果
        hist_untuned = model_untuned.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=0)
        time_untuned = time.time() - start_time
        acc_untuned = max(hist_untuned.history['val_accuracy']) # 取验证集最佳精度

        # --- 跑调优模型 (Tuned) ---
        start_time = time.time()
        model_tuned = build_tuned_model(record['best_params'], (X_train.shape[1], X_train.shape[2]))
        hist_tuned = model_tuned.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=0)
        time_tuned = time.time() - start_time
        acc_tuned = max(hist_tuned.history['val_accuracy'])

        # --- 计算提升 ---
        gain = acc_tuned - acc_untuned
        print(f"   🔹 默认精度: {acc_untuned:.4f} (耗时 {time_untuned:.1f}s)")
        print(f"   🔸 调优精度: {acc_tuned:.4f} (耗时 {time_tuned:.1f}s)")
        print(f"   🚀 提升: {gain*100:.2f}%")

        results.append({
            'Ticker': ticker,
            'Horizon': horizon_name,
            'Untuned_Acc': acc_untuned,
            'Tuned_Acc': acc_tuned,
            'Gain': gain,
            'Untuned_Time': time_untuned,
            'Tuned_Time': time_tuned
        })

    # ==========================================
    # 4. 生成报告
    # ==========================================
    if results:
        res_df = pd.DataFrame(results)
        report_path = os.path.join(base_dir, 'data', 'tuned_models', 'performance_comparison.csv')
        res_df.to_csv(report_path, index=False)
        print(f"\n✅ 对比报告已保存: {report_path}")
        
        # 简单打印平均提升
        avg_gain = res_df['Gain'].mean()
        print(f"\n🏆 平均准确率提升: {avg_gain*100:.2f}%")
        print("💡 提示：如果提升为负，说明默认参数运气好，或者调优次数(n_trials)还不够多。")

if __name__ == "__main__":
    evaluate_gains()