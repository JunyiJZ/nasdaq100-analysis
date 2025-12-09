import pandas as pd
import numpy as np
import os
import json
import optuna
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 数据准备函数 (修复版：先筛选列，再dropna)
# ==========================================
def prepare_data(df, ticker, horizon_days, look_back=60):
    # 1. 筛选特定 Ticker 的数据
    data = df[df['Ticker'] == ticker].copy()
    
    if len(data) == 0:
        # Ticker 不存在
        return None, None, None, None, None

    # 2. 确保按时间排序
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')

    # 3. 定义特征列 (请确保这些列名在下面的【列名侦探】输出中能找到)
    # 根据你的截图，我猜测列名如下，如果报错，请看控制台输出的实际列名
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9' 
    ]
    
    # 【检查列是否存在】
    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        print(f"   ⚠️ [跳过] {ticker} 缺少列: {missing_cols}")
        return None, None, None, None, None

    # 4. 创建目标变量
    data['Target'] = (data['Close'].shift(-horizon_days) > data['Close']).astype(int)
    
    # =========================================================
    # 【核心修复】 只保留我们需要的列，防止被无关列的空值误杀
    # =========================================================
    needed_cols = feature_cols + ['Target']
    data = data[needed_cols] 

    # 5. 去除空值
    rows_before = len(data)
    data = data.dropna()
    rows_after = len(data)

    if rows_after == 0:
        print(f"   ⚠️ [跳过] {ticker} dropna() 后为空 (原: {rows_before} -> 0)。请检查特征列是否全是NaN。")
        return None, None, None, None, None

    # 6. 数据归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols]) 

    # 7. 构建 LSTM 序列
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(data['Target'].iloc[i])

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        print(f"   ⚠️ [跳过] {ticker} 构建序列后数据不足 (行数 < look_back)")
        return None, None, None, None, None

    # 8. 划分数据集
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test, scaler

# ==========================================
# 2. 模型构建函数
# ==========================================
def create_lstm_model(trial, input_shape):
    model = Sequential()
    units_1 = trial.suggest_int('units_1', 32, 128)
    return_sequences = trial.suggest_categorical('return_sequences', [True, False])
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5)
    
    model.add(LSTM(units=units_1, return_sequences=return_sequences, input_shape=input_shape))
    model.add(Dropout(dropout_1))
    
    if return_sequences:
        units_2 = trial.suggest_int('units_2', 16, 64)
        dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.5)
        model.add(LSTM(units=units_2, return_sequences=False))
        model.add(Dropout(dropout_2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 3. 主优化流程
# ==========================================
def run_optimization():
    current_dir = os.getcwd()
    DATA_PATH = os.path.join(current_dir, 'data', 'finalized', 'final_master_dataset.csv')
    RESULTS_DIR = os.path.join(current_dir, 'data', 'tuned_models') 
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到数据文件: {DATA_PATH}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"正在读取数据: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    
    # 清洗 Ticker
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
    
    # ==========================================
    # 🕵️ 列名侦探：打印出所有列名，方便核对
    # ==========================================
    print("\n" + "="*40)
    print("🕵️  列名侦探报告 (请核对MACD列名是否一致):")
    print(df.columns.tolist())
    print("="*40 + "\n")

    tickers = df['Ticker'].unique()
    print(f"检测到 {len(tickers)} 个股票。")
    
    # ⚠️ 调试模式：只跑前 2 个股票。如果成功了，把 [:2] 去掉改成 tickers
    target_tickers = tickers[:2]  
    
    horizons = {'Short': 1, 'Medium': 5, 'Long': 20}
    best_params_record = {}

    for ticker in target_tickers:
        for horizon_name, horizon_days in horizons.items():
            print(f"\n>>> 正在处理: {ticker} - {horizon_name} ...")

            X_train, y_train, X_test, y_test, scaler = prepare_data(df, ticker, horizon_days)

            if X_train is None:
                continue

            # 定义 Optuna 目标函数
            def objective(trial):
                model = create_lstm_model(trial, (X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=5,  # 调试用 5，正式跑改 10-20
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                return max(history.history['val_accuracy'])

            # 调试用 2 次 trial，正式跑改 10-20
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=2) 

            print(f"   ✅ 成功! 最佳准确率: {study.best_value:.4f}")

            key = f"{ticker}_{horizon_name}"
            best_params_record[key] = {
                'best_params': study.best_params,
                'best_accuracy': study.best_value
            }

    # --- 保存结果 ---
    print("\n" + "="*30)
    if not best_params_record:
        print("❌ 依然没有生成结果。请检查上方报错信息。")
    else:
        params_file = os.path.join(RESULTS_DIR, 'best_hyperparameters.json')
        with open(params_file, 'w') as f:
            json.dump(best_params_record, f, indent=4)
        print(f"✅ 完美！结果已保存至: {params_file}")

if __name__ == "__main__":
    run_optimization()