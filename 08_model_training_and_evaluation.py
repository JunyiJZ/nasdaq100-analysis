import pandas as pd
import xgboost as xgb
# Updated import: Added root_mean_squared_error for sklearn > 1.4
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import os
import joblib
from pathlib import Path
from xgboost.callback import EarlyStopping

# ==============================================================================
# Task 1: Load Data
# ==============================================================================
print("--- Task 1: Loading Model Input Data ---")

# Use pathlib for cross-platform compatibility
input_dir = Path('data') / 'final'

try:
    print("Loading X_train...")
    X_train = pd.read_csv(input_dir / 'X_train.csv', index_col='Date', parse_dates=True)
    
    print("Loading y_train...")
    y_train = pd.read_csv(input_dir / 'y_train.csv', index_col='Date', parse_dates=True).squeeze('columns')
    
    print("Loading X_val...")
    X_val = pd.read_csv(input_dir / 'X_val.csv', index_col='Date', parse_dates=True)
    
    print("Loading y_val...")
    y_val = pd.read_csv(input_dir / 'y_val.csv', index_col='Date', parse_dates=True).squeeze('columns')
    
    print("Loading X_test...")
    X_test = pd.read_csv(input_dir / 'X_test.csv', index_col='Date', parse_dates=True)
    
    print("Loading y_test...")
    y_test = pd.read_csv(input_dir / 'y_test.csv', index_col='Date', parse_dates=True).squeeze('columns')
    
    print("All datasets loaded successfully.")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}")
    print("Please ensure '07_prepare_model_data.py' has run successfully and files exist in 'data/final/'.")
    exit()

# ==============================================================================
# Task 2: Define and Train XGBoost Model (Modern Syntax)
# ==============================================================================
print("\n--- Task 2: Defining and Training XGBoost Model ---")

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

model = xgb.XGBRegressor(**params)

eval_set = [(X_val, y_val)]

print("Starting model training...")

# Create EarlyStopping callback
# This monitors the validation set and stops if metric doesn't improve for 50 rounds
early_stop_callback = EarlyStopping(
    rounds=50,      
    save_best=True  
)

model.fit(
    X_train, y_train,
    eval_set=eval_set,
    callbacks=[early_stop_callback], 
    verbose=100 
)

print("Model training completed.")

# ==============================================================================
# Task 3: Evaluate Model
# ==============================================================================
print("\n--- Task 3: Evaluating Model on Test Set ---")

y_pred = model.predict(X_test)

# --- FIX IS HERE ---
# Since you are using scikit-learn 1.7.2, 'squared=False' is removed.
# We use the new root_mean_squared_error function instead.
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - R2:   {r2:.4f}")

# ==============================================================================
# Task 4: Save Model
# ==============================================================================
print("\n--- Task 4: Saving Trained Model ---")

model_dir = Path('models')
model_dir.mkdir(exist_ok=True)
model_path = model_dir / 'xgboost_model_v1.joblib'

joblib.dump(model, model_path)

print(f"🎉 Task Complete: Model successfully saved to '{model_path}'.")