import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

# ==========================================
# 1. Risk Manager Logic (保持不变)
# ==========================================
class RiskManager:
    def __init__(self, initial_capital, atr_period, atr_multiplier, trailing_stop_pct, risk_per_trade):
        self.initial_capital = initial_capital
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trailing_stop_pct = trailing_stop_pct
        self.risk_per_trade = risk_per_trade

    def calculate_atr(self, df):
        df = df.copy()
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(self.atr_period).mean()
        return df

    def calculate_position_size(self, current_capital, entry_price, stop_loss_price):
        if isinstance(entry_price, pd.Series): entry_price = float(entry_price.iloc[0])
        if isinstance(stop_loss_price, pd.Series): stop_loss_price = float(stop_loss_price.iloc[0])
        
        if entry_price <= stop_loss_price: return 0
        
        risk_amount = current_capital * self.risk_per_trade
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share == 0: return 0
        return int(risk_amount / risk_per_share)

def run_simulation(ticker, start_date, end_date, params):
    warmup_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    
    try:
        df = yf.download(ticker, start=warmup_start, end=end_date, progress=False, auto_adjust=False)
    except Exception:
        return None, None, None

    if df.empty: return None, None, None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    rm = RiskManager(params['capital'], 14, params['atr_mult'], params['trail_stop'], params['risk_pct'])
    df = rm.calculate_atr(df)
    
    df['SMA_Fast'] = df['Close'].rolling(10).mean()
    df['SMA_Slow'] = df['Close'].rolling(30).mean()
    
    df.dropna(inplace=True)
    df = df[df.index >= pd.to_datetime(start_date)]
    
    if df.empty: return None, None, None

    capital = rm.initial_capital
    shares = 0
    equity_curve = []
    trade_log = []
    stop_loss_price = 0.0
    high_water_mark = 0.0
    entry_price = 0.0

    for i in range(len(df)):
        date = df.index[i]
        price = float(df['Close'].iloc[i])
        atr = float(df['ATR'].iloc[i])
        sma_fast = float(df['SMA_Fast'].iloc[i])
        sma_slow = float(df['SMA_Slow'].iloc[i])
        
        # Sell Logic
        if shares > 0:
            if price > high_water_mark: high_water_mark = price
            trailing_stop = high_water_mark * (1 - rm.trailing_stop_pct)
            
            is_stop_loss = price < stop_loss_price
            is_trailing_stop = price < trailing_stop
            is_signal_sell = sma_fast < sma_slow
            
            if is_stop_loss or is_trailing_stop or is_signal_sell:
                reason = "Stop Loss" if is_stop_loss else ("Trailing Stop" if is_trailing_stop else "Trend Reversal")
                pnl_amt = (price - entry_price) * shares
                pnl_pct = ((price - entry_price) / entry_price) * 100 
                capital += shares * price
                
                trade_log.append({
                    'Date': date, 'Type': 'SELL', 'Price': price, 
                    'Shares': shares, 'Reason': reason, 'PnL (%)': pnl_pct, 'PnL ($)': pnl_amt
                })
                shares = 0
                stop_loss_price = 0.0
                high_water_mark = 0.0
        
        # Buy Logic
        elif shares == 0 and sma_fast > sma_slow:
            initial_stop = price - (atr * rm.atr_multiplier)
            shares_to_buy = rm.calculate_position_size(capital, price, initial_stop)
            cost = shares_to_buy * price
            
            if cost < capital and shares_to_buy > 0:
                shares = shares_to_buy
                capital -= cost
                entry_price = price
                stop_loss_price = initial_stop
                high_water_mark = price
                
                trade_log.append({
                    'Date': date, 'Type': 'BUY', 'Price': price, 
                    'Shares': shares, 'Reason': 'Golden Cross', 'PnL (%)': 0.0, 'PnL ($)': 0.0
                })

        equity_curve.append(capital + (shares * price))

    df['Equity'] = equity_curve
    df['Peak_Equity'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak_Equity']) / df['Peak_Equity']
    max_drawdown = df['Drawdown'].min()
    
    return df, pd.DataFrame(trade_log), max_drawdown

# ==========================================
# 2. AI Engine (Week 20 & 21: LSTM + Explainability)
# ==========================================

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

class AIRecommender:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = 60 
        self.data_scaled = None
        self.df_raw = None
        
    def fetch_and_prepare_data(self):
        try:
            df = yf.download(self.ticker, period="2y", progress=False, auto_adjust=False)
            if len(df) < 100: return None
            
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            self.df_raw = df 
            data = df['Close'].values.reshape(-1, 1)
            self.data_scaled = self.scaler.fit_transform(data)
            return df
        except Exception:
            return None

    def calculate_risk_metrics(self):
        if self.df_raw is None: return 0, 0
        returns = self.df_raw['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        mean_return = returns.mean() * 252
        risk_free_rate = 0.04
        if volatility == 0: return 0, 0
        sharpe_ratio = (mean_return - risk_free_rate) / volatility
        return volatility, sharpe_ratio

    def train_model(self):
        if self.data_scaled is None: return None
        
        x_train, y_train = [], []
        for i in range(len(self.data_scaled) - self.look_back):
            x_train.append(self.data_scaled[i:i+self.look_back])
            y_train.append(self.data_scaled[i+self.look_back])
        
        x_train = torch.from_numpy(np.array(x_train)).type(torch.Tensor)
        y_train = torch.from_numpy(np.array(y_train)).type(torch.Tensor)
        
        model = LSTMNet(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 30
        for epoch in range(epochs):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        self.model = model
        return model

    def predict_future_recursive(self, horizon=30):
        if self.model is None: return []
        current_seq = self.data_scaled[-self.look_back:] 
        current_seq = torch.from_numpy(np.array([current_seq])).type(torch.Tensor) 
        future_preds_scaled = []
        
        with torch.no_grad():
            for _ in range(horizon):
                pred = self.model(current_seq)
                future_preds_scaled.append(pred.item())
                next_val = pred.unsqueeze(1) 
                current_seq = torch.cat((current_seq[:, 1:, :], next_val), dim=1)
        
        future_preds = self.scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
        return future_preds.flatten().tolist()

    # ==========================================
    # Week 21: Feature Importance & Explainability
    # ==========================================
    def get_feature_importance(self):
        """
        使用随机森林作为代理模型 (Surrogate Model)，
        提取技术指标的 Feature Importance，并生成自然语言解释。
        """
        if self.df_raw is None or len(self.df_raw) < 50:
            return "HOLD", 0.5, {"No Data": 0}, "Insufficient data to generate explanation."

        df = self.df_raw.copy()
        
        # 1. 构建解释性特征 (Feature Engineering for Explainability)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Volatility'] = df['Close'].rolling(14).std()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # 距离均线的乖离率
        df['SMA_20_Dist'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()

        # 目标变量：未来5天的收益率
        df['Target'] = df['Close'].shift(-5) / df['Close'] - 1
        df.dropna(inplace=True)

        if len(df) < 20:
            return "HOLD", 0.5, {"Data too short": 0}, "Not enough valid feature data."

        features = ['Return_5d', 'Volatility', 'RSI_14', 'MACD', 'SMA_20_Dist']
        X = df[features]
        y = df['Target']

        # 2. 训练代理模型 (Surrogate Random Forest)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_

        # 3. 结合当前最新状态，赋予重要性方向 (正向/负向影响)
        latest_data = df.iloc[-1]
        features_dict = {}
        
        for idx, feat in enumerate(features):
            imp = importances[idx]
            val = latest_data[feat]
            
            # 简单的业务逻辑判断方向：
            # 例如：RSI < 30 是看涨(正向)，RSI > 70 是看跌(负向)
            direction = 1 
            if feat == 'RSI_14':
                if val > 65: direction = -1
                elif val < 35: direction = 1
                else: direction = 0.2 # 中性偏正
            elif feat == 'SMA_20_Dist':
                direction = -1 if val > 0.05 else 1 # 均线回归逻辑
            elif feat == 'Return_5d':
                direction = 1 if val > 0 else -1 # 动量逻辑
            elif feat == 'MACD':
                direction = 1 if val > 0 else -1
            elif feat == 'Volatility':
                direction = -1 # 波动率大通常视为负面风险
                
            # 最终的定向影响力 (Impact)
            features_dict[feat] = float(imp * direction)

        # 4. 生成信号与置信度
        total_impact = sum(features_dict.values())
        confidence = min(abs(total_impact) * 2 + 0.5, 0.95) # 模拟置信度 50%~95%
        
        if total_impact > 0.1: signal = "STRONG BUY"
        elif total_impact > 0.02: signal = "BUY"
        elif total_impact < -0.1: signal = "STRONG SELL"
        elif total_impact < -0.02: signal = "SELL"
        else: signal = "HOLD"

        # 5. 生成自然语言解释 (Natural Language Generation)
        sorted_feats = sorted(features_dict.items(), key=lambda item: item[1])
        top_positive = sorted_feats[-1] if sorted_feats[-1][1] > 0 else None
        top_negative = sorted_feats[0] if sorted_feats[0][1] < 0 else None

        action = "Buy" if "BUY" in signal else "Sell" if "SELL" in signal else "Hold"
        
        explanation_text = f"**Recommendation:** {action} {self.ticker}. \n\n"
        explanation_text += f"**Confidence Level:** {confidence:.1%} \n\n"
        explanation_text += f"**AI Reasoning:** "
        
        if top_positive:
            explanation_text += f"The primary driving factor for this decision is **{top_positive[0]}**, which currently has a highly positive/bullish setup. "
        if top_negative:
            explanation_text += f"However, investors should note that **{top_negative[0]}** is showing a negative trend, acting as a slight headwind."
            
        if not top_positive and not top_negative:
            explanation_text += "The model sees mixed signals across all technical indicators, suggesting a neutral stance."

        return signal, confidence, features_dict, explanation_text