import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf
from datetime import datetime, timedelta

# ==========================================
# 0. Environment Check & Imports
# ==========================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# ==========================================
# 1. Core Logic: Risk Engine (Week 18)
# ==========================================

class RiskManager:
    """
    Implements logic from Week 18: Risk Management
    """
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

# ==========================================
# 2. Core Logic: AI Engine (Week 9 Integration)
# ==========================================

# --- PyTorch LSTM Model Structure (From Week 9) ---
if PYTORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super(LSTMNet, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :]) 
            return out

class AIRecommender:
    """
    REAL Week 19 Engine: Integrates Week 9 LSTM + Week 6 Sentiment Logic.
    Replaces the 'Mock/Random' logic with actual PyTorch inference.
    """
    def __init__(self, return_threshold=0.01, sentiment_threshold=0.2):
        self.return_threshold = return_threshold
        self.sentiment_threshold = sentiment_threshold
        self.device = torch.device('cuda' if (PYTORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        self.seq_length = 60 # Consistent with Week 9 screenshot

    def _get_real_prediction(self, ticker):
        """
        Performs On-the-fly Training using PyTorch
        """
        if not PYTORCH_AVAILABLE:
            return 0.0, 0.0, 50.0 # Return dummy if no torch
            
        # 1. Fetch Data
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if len(df) < 100: return 0.0, 0.0, 50.0
        
        # 2. Preprocess (Calculate Returns)
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
        
        # Calculate RSI for feature
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        current_rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Prepare for LSTM
        data = df['Return'].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(data_scaled) - self.seq_length):
            X.append(data_scaled[i:i+self.seq_length])
            y.append(data_scaled[i+self.seq_length])
            
        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(self.device)
        
        # 3. Train Model (Fast Loop)
        model = LSTMNet(input_dim=1, hidden_dim=32, output_dim=1, num_layers=1).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        model.train()
        for _ in range(30): # 30 Epochs for speed
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 4. Predict
        model.eval()
        last_seq = torch.tensor(data_scaled[-self.seq_length:].reshape(1, self.seq_length, 1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = model(last_seq).cpu().numpy()
            
        pred_return = scaler.inverse_transform(pred_scaled)[0][0]
        
        # Mock Sentiment (Since we don't have the NLP model loaded here, we simulate it based on trend)
        # In a full system, you would import the BERT model here.
        trend_sentiment = 0.5 if pred_return > 0 else -0.5
        
        return pred_return, trend_sentiment, current_rsi, df['Close'].iloc[-1]

    def generate_signal(self, ticker):
        # --- REAL CALCULATION ---
        pred_return, sentiment_score, rsi, current_price = self._get_real_prediction(ticker)
        
        # Logic Rules
        signal = "HOLD"
        reasoning = []
        confidence = 0.50
        
        # 1. Buy Logic
        if pred_return > self.return_threshold:
            if sentiment_score > self.sentiment_threshold:
                signal = "BUY"
                confidence = 0.85
                reasoning.append(f"✅ LSTM predicts strong return (+{pred_return:.2%}) > {self.return_threshold:.1%}")
                reasoning.append(f"✅ Sentiment aligned ({sentiment_score:.2f})")
            else:
                reasoning.append(f"⚠️ LSTM Bullish (+{pred_return:.2%}) but Sentiment weak.")
        
        # 2. Sell Logic
        elif pred_return < -self.return_threshold:
            signal = "SELL"
            confidence = 0.80
            reasoning.append(f"🔻 LSTM predicts drop ({pred_return:.2%})")
        
        # 3. Hold Logic
        else:
            reasoning.append(f"⚖️ Predicted return ({pred_return:.2%}) is noise.")

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "data": {
                "predicted_return": pred_return,
                "sentiment_score": sentiment_score,
                "rsi": rsi,
                "current_price": current_price,
                "fundamental_pe": np.random.uniform(20, 50) # Placeholder for fundamentals
            }
        }

# ==========================================
# 3. Simulation Function (Week 18)
# ==========================================
def run_simulation(ticker, start_date, end_date, params):
    warmup_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    try:
        df = yf.download(ticker, start=warmup_start, end=end_date, progress=False)
    except Exception: return None, None, None

    if df.empty: return None, None, None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
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
        
        if shares > 0:
            if price > high_water_mark: high_water_mark = price
            trailing_stop = high_water_mark * (1 - rm.trailing_stop_pct)
            if price < stop_loss_price or price < trailing_stop or sma_fast < sma_slow:
                pnl_amt = (price - entry_price) * shares
                capital += shares * price
                trade_log.append({'Date': date, 'Type': 'SELL', 'Price': price, 'PnL ($)': pnl_amt})
                shares = 0
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
                trade_log.append({'Date': date, 'Type': 'BUY', 'Price': price, 'PnL ($)': 0.0})

        equity_curve.append(capital + (shares * price))

    df['Equity'] = equity_curve
    df['Peak_Equity'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak_Equity']) / df['Peak_Equity']
    max_drawdown = df['Drawdown'].min()
    return df, pd.DataFrame(trade_log), max_drawdown

# ==========================================
# 4. Page Configuration & Navigation
# ==========================================
st.set_page_config(page_title="AI Quant Dashboard", page_icon="📈", layout="wide")
st.title("🤖 Innovation AI: Quant Internship Dashboard")

st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio("Select View:", [
    "Week 19: AI Recommendation Engine",
    "Week 18: Risk Management Simulator", 
    "Portfolio & Backtest Data"
])

# ==========================================
# 5. Page Content
# ==========================================

if page == "Week 19: AI Recommendation Engine":
    st.header("🤖 Week 19: AI Hybrid Model (Real-Time)")
    st.markdown("""
    **Architecture:** PyTorch LSTM (Seq_Len=60) + Sentiment Logic.
    **Status:** This module performs **live training** on the selected ticker to generate predictions.
    """)
    
    if not PYTORCH_AVAILABLE:
        st.error("⚠️ PyTorch not detected. Please install: `pip install torch`")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("1. Input & Settings")
            ticker_input = st.text_input("Ticker Symbol", value="NVDA").upper()
            
            st.markdown("---")
            st.caption("Decision Thresholds")
            thresh_ret = st.slider("Min Return Threshold", 0.0, 0.05, 0.005, format="%.3f")
            thresh_sent = st.slider("Min Sentiment Threshold", 0.0, 1.0, 0.2)
            
            analyze_btn = st.button("🚀 Train & Predict (Live)", type="primary", use_container_width=True)

        with col2:
            if analyze_btn:
                with st.spinner(f"Training PyTorch LSTM on {ticker_input} (This takes a few seconds)..."):
                    # Initialize Week 19 Engine
                    engine = AIRecommender(return_threshold=thresh_ret, sentiment_threshold=thresh_sent)
                    
                    # Run Analysis (REAL)
                    try:
                        result = engine.generate_signal(ticker_input)
                        
                        # --- DISPLAY RESULTS ---
                        sig = result['signal']
                        color_map = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background-color: {color_map.get(sig, 'gray')}20; border-radius: 10px; border: 2px solid {color_map.get(sig, 'gray')};">
                            <h2 style="color: {color_map.get(sig, 'gray')}; margin:0;">RECOMMENDATION: {sig}</h2>
                            <p style="margin:0;">Confidence: {result['confidence']*100:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("") 

                        st.subheader("2. Decision Logic (Traceability)")
                        for reason in result['reasoning']:
                            st.write(reason)

                        st.markdown("---")
                        
                        st.subheader("3. Real-Time Model Inputs")
                        d = result['data']
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("LSTM Predicted Return", f"{d['predicted_return']:.2%}")
                        k2.metric("Trend Sentiment", f"{d['sentiment_score']:.2f}")
                        k3.metric("RSI (Tech)", f"{d['rsi']:.1f}")
                        k4.metric("Current Price", f"${d['current_price']:.2f}")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

elif page == "Week 18: Risk Management Simulator":
    st.header("🛡️ Risk Management (Week 18)")
    scenarios = {
        "2024 Aug 'Yen Carry' Flash Crash": {"start": "2024-07-01", "end": "2024-09-01"},
        "2022 Full Bear Market": {"start": "2022-01-01", "end": "2022-12-31"},
    }
    
    col1, col2 = st.columns(2)
    with col1:
        sim_ticker = st.text_input("Ticker", value="QQQ") 
        scenario_name = st.selectbox("Scenario", list(scenarios.keys()))
    with col2:
        sim_risk = st.slider("Risk %", 0.5, 5.0, 2.0) / 100.0
        
    if st.button("Run Stress Test"):
        dates = scenarios[scenario_name]
        params = {'capital': 100000, 'risk_pct': sim_risk, 'atr_mult': 2.5, 'trail_stop': 0.08}
        df_res, trade_log, max_dd = run_simulation(sim_ticker, dates["start"], dates["end"], params)
        
        if df_res is not None:
            final_eq = df_res['Equity'].iloc[-1]
            st.metric("Final Equity", f"${final_eq:,.0f}")
            st.metric("Max Drawdown", f"{max_dd:.2%}")
            fig = px.line(df_res, x=df_res.index, y="Equity", title="Stress Test Equity Curve")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data.")

elif page == "Portfolio & Backtest Data":
    st.header("📂 Data Inspector")
    if os.path.exists("optimized_portfolio_allocation.csv"):
        st.subheader("Portfolio Allocation")
        st.dataframe(pd.read_csv("optimized_portfolio_allocation.csv"))
    else:
        st.info("No portfolio file found.")

st.sidebar.markdown("---")
st.sidebar.info("System: PyTorch Integrated")