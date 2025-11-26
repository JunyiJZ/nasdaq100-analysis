import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ==========================================
# 配置路径
# ==========================================
DATA_DIR = Path('data/final')
MODELS_DIR = Path('models')
RESULTS_FILE = Path('baseline_results.csv')

MODELS_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. 加载数据
# ==========================================
print("\n--- Loading Data ---")
try:
    # --- 读取特征 X ---
    X_train_raw = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test_raw = pd.read_csv(DATA_DIR / 'X_test.csv')
    
    # 过滤 X：只保留数值类型的列
    X_train = X_train_raw.select_dtypes(include=[np.number])
    X_test = X_test_raw.select_dtypes(include=[np.number])

    # --- 读取标签 y ---
    y_train_raw = pd.read_csv(DATA_DIR / 'y_train.csv')
    y_test_raw = pd.read_csv(DATA_DIR / 'y_test.csv')

    # 过滤 y：只保留数值类型的列，并展平
    y_train = y_train_raw.select_dtypes(include=[np.number]).values.ravel()
    y_test = y_test_raw.select_dtypes(include=[np.number]).values.ravel()

    print(f"Data loaded successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # 检查 NaN
    if np.isnan(y_train).any():
        y_train = np.nan_to_num(y_train)
    if np.isnan(y_test).any():
        y_test = np.nan_to_num(y_test)

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# 准备结果列表
results = []

# ==========================================
# 2. 定义并训练模型
# ==========================================

# --- Model A: Linear Regression ---
print("\n--- Training Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 修复：手动计算 RMSE，兼容所有 sklearn 版本
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - RMSE: {rmse_lr:.4f}, R2: {r2_lr:.4f}")
results.append({
    'Model': 'Linear Regression',
    'RMSE': rmse_lr,
    'R2': r2_lr,
    'Description': 'Simple baseline'
})
joblib.dump(lr_model, MODELS_DIR / 'linear_regression.joblib')


# --- Model B: Random Forest ---
print("\n--- Training Random Forest (This may take a while) ---")
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 修复：手动计算 RMSE
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")
results.append({
    'Model': 'Random Forest',
    'RMSE': rmse_rf,
    'R2': r2_rf,
    'Description': 'Ensemble baseline'
})
joblib.dump(rf_model, MODELS_DIR / 'random_forest.joblib')


# --- Model C: XGBoost ---
print("\n--- Training XGBoost ---")
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 修复：手动计算 RMSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost - RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")
results.append({
    'Model': 'XGBoost',
    'RMSE': rmse_xgb,
    'R2': r2_xgb,
    'Description': 'Gradient Boosting'
})
joblib.dump(xgb_model, MODELS_DIR / 'xgboost_model_v2.joblib')

# ==========================================
# 3. 生成 Milestone 1: baseline_results.csv
# ==========================================
print("\n--- Saving Results to CSV ---")
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)
print(f"Successfully saved results to {RESULTS_FILE}")

# ==========================================
# 4. 展示 Milestone 2 & 3: 对比报告
# ==========================================
print("\n" + "="*60)
print("FINAL MODEL COMPARISON REPORT (Week 8 Milestones)")
print("="*60)
print(results_df[['Model', 'RMSE', 'R2', 'Description']].to_string(index=False))
print("="*60)

# 验证 Insight
if r2_xgb > r2_rf:
    print("\n✅ Insight Verified: XGBoost outperformed Random Forest.")
elif r2_xgb > r2_lr:
    print("\n✅ XGBoost outperformed Linear Regression, but RF was strong.")
else:
    print("\n⚠️ Insight Check: XGBoost did not outperform baselines in this specific run.")