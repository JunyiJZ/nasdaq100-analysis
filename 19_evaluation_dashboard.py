import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf
from datetime import timedelta, datetime
import warnings

# ==========================================
# 0. Environment Setup & Warning Suppression
# ==========================================
# Suppress warnings for cleaner terminal output (Addressing your screenshot issues)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. Core Logic: Risk Engine (Week 18)
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
        # Fix: Explicitly set auto_adjust to suppress yfinance warnings
        df = yf.download(ticker, start=warmup_start, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        return None, None, None

    if df.empty: return None, None, None
    
    # Handle MultiIndex columns (common in new yfinance versions)
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
# 2. Core Logic: AI Engine (Week 19 NEW)
# ==========================================

if AI_AVAILABLE:
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
            
        def fetch_and_prepare_data(self):
            # Fetch 2 years of data for sufficient training samples
            try:
                df = yf.download(self.ticker, period="2y", progress=False, auto_adjust=False)
                if len(df) < 100: return None
                
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                    
                data = df['Close'].values.reshape(-1, 1)
                self.data_scaled = self.scaler.fit_transform(data)
                return df
            except Exception:
                return None

        def train_model(self):
            if self.data_scaled is None: return None
            
            x_train, y_train = [], []
            for i in range(len(self.data_scaled) - self.look_back):
                x_train.append(self.data_scaled[i:i+self.look_back])
                y_train.append(self.data_scaled[i+self.look_back])
            
            x_train = torch.from_numpy(np.array(x_train)).type(torch.Tensor)
            y_train = torch.from_numpy(np.array(y_train)).type(torch.Tensor)
            
            # Model Architecture
            model = LSTMNet(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Training Loop (Increased epochs slightly for better convergence in demo)
            epochs = 30
            for epoch in range(epochs):
                y_train_pred = model(x_train)
                loss = criterion(y_train_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            self.model = model
            return model

        def predict_next_day(self):
            if self.model is None: return None
            
            # Take the last sequence from the actual data
            last_sequence = self.data_scaled[-self.look_back:]
            last_sequence = torch.from_numpy(np.array([last_sequence])).type(torch.Tensor)
            
            with torch.no_grad():
                prediction = self.model(last_sequence)
            
            predicted_price = self.scaler.inverse_transform(prediction.numpy())[0][0]
            return predicted_price

# ==========================================
# 3. Streamlit Page Configuration
# ==========================================
st.set_page_config(page_title="AI Quant Dashboard", page_icon="📈", layout="wide")
st.title("🤖 Innovation AI: Quant Internship Dashboard")

# ==========================================
# 4. Data Loading Helper
# ==========================================
@st.cache_data
def load_data(filename):
    if os.path.exists(filename): return pd.read_csv(filename)
    else: return None

# ==========================================
# 5. Navigation (Updated with 6 Options)
# ==========================================
st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio("Select View:", [
    "Overview & Portfolio", 
    "Backtest Performance", 
    "Model Metrics", 
    "Risk Management Simulator", 
    "Week 19: AI Prediction Engine",  # <--- The New Module
    "Raw Data Inspector"
])

# ==========================================
# 6. Page Content Logic
# ==========================================

# --- Page 1: Overview ---
if page == "Overview & Portfolio":
    st.header("📊 Portfolio Allocation")
    df_alloc = load_data("optimized_portfolio_allocation.csv")
    if df_alloc is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            numeric_cols = df_alloc.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) > 0:
                fig = px.pie(df_alloc, values=numeric_cols[0], names=df_alloc.columns[0], title="Asset Allocation", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(df_alloc.head(10), use_container_width=True)
    else: st.error("Data file not found. Please run portfolio optimization first.")

# --- Page 2: Backtest ---
elif page == "Backtest Performance":
    st.header("📈 Strategy Performance")
    strategy_file = st.selectbox("Select Strategy:", ["long_term_backtest.csv", "long_term_backtest_realistic.csv"])
    df_backtest = load_data(strategy_file)
    if df_backtest is not None:
        if 'Date' in df_backtest.columns: df_backtest['Date'] = pd.to_datetime(df_backtest['Date'])
        numeric_cols = df_backtest.select_dtypes(include=['float', 'int']).columns.tolist()
        selected_cols = st.multiselect("Select Metrics", numeric_cols, default=numeric_cols[:2])
        if selected_cols:
            fig = px.line(df_backtest, x='Date' if 'Date' in df_backtest.columns else df_backtest.index, y=selected_cols)
            st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Metrics ---
elif page == "Model Metrics":
    st.header("🤖 AI Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        df_base = load_data("baseline_results.csv")
        if df_base is not None: st.dataframe(df_base, use_container_width=True)
    with col2:
        st.subheader("Tuning")
        df_tune = load_data("tuning_performance_results.csv")
        if df_tune is not None: st.dataframe(df_tune, use_container_width=True)

# --- Page 4: Risk Sim (Week 18) ---
elif page == "Risk Management Simulator":
    st.header("🛡️ Week 18: Modern Era Stress Testing")
    
    scenarios = {
        "2024 Aug 'Yen Carry' Flash Crash": {"start": "2024-07-01", "end": "2024-09-01"},
        "2024 April Correction": {"start": "2024-03-01", "end": "2024-05-31"},
        "2023 Banking Crisis (SVB)": {"start": "2023-02-01", "end": "2023-05-01"},
        "2022 Full Bear Market (Rates)": {"start": "2022-01-01", "end": "2022-12-31"},
        "2022 H1 Tech Crash (Nasdaq)": {"start": "2022-01-01", "end": "2022-06-30"},
    }

    with st.expander("⚙️ Simulation Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sim_ticker = st.text_input("Ticker", value="QQQ") 
            sim_capital = st.number_input("Capital ($)", value=100000)
            selected_scenario_name = st.selectbox("Recent Event", list(scenarios.keys()))
        with col2:
            sim_risk = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0) / 100.0
        with col3:
            sim_atr = st.slider("ATR Multiplier (Stop dist)", 1.0, 4.0, 2.5)
        with col4:
            sim_trail = st.slider("Trailing Stop (%)", 1.0, 20.0, 8.0) / 100.0
            
        run_sim = st.button("🚀 Run Modern Era Test", type="primary")

    if run_sim:
        scenario_dates = scenarios[selected_scenario_name]
        start_date = scenario_dates["start"]
        end_date = scenario_dates["end"]
        
        st.info(f"Testing {sim_ticker} during: {selected_scenario_name} ({start_date} to {end_date})")

        with st.spinner(f"Downloading recent data (with warm-up buffer)..."):
            params = {'capital': sim_capital, 'risk_pct': sim_risk, 'atr_mult': sim_atr, 'trail_stop': sim_trail}
            df_res, trade_log, max_dd = run_simulation(sim_ticker, start_date, end_date, params)
            
            if df_res is not None and not df_res.empty:
                final_eq = df_res['Equity'].iloc[-1]
                ret = (final_eq - sim_capital) / sim_capital
                
                if not trade_log.empty:
                    wins = trade_log[trade_log['PnL ($)'] > 0]
                    sells = trade_log[trade_log['Type'] == 'SELL']
                    win_rate = (len(wins) / len(sells) * 100) if len(sells) > 0 else 0
                    total_trades_count = len(sells)
                else:
                    win_rate = 0
                    total_trades_count = 0

                st.markdown("### 📊 Test Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Final Equity", f"${final_eq:,.0f}", f"{ret:.2%}")
                m2.metric("Completed Trades", total_trades_count)
                m3.metric("Max Drawdown", f"{max_dd:.2%}", delta_color="inverse")
                m4.metric("Win Rate", f"{win_rate:.1f}%")
                m5.metric("Scenario", selected_scenario_name)

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                                    subplot_titles=(f"{sim_ticker} Price Action", "Account Equity", "Drawdown"))

                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], name='Price', line=dict(color='gray')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['SMA_Fast'], name='SMA Fast', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['SMA_Slow'], name='SMA Slow', line=dict(color='blue', width=1)), row=1, col=1)

                if not trade_log.empty:
                    buys = trade_log[trade_log['Type'] == 'BUY']
                    sells = trade_log[trade_log['Type'] == 'SELL']
                    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Equity'], name='Equity', line=dict(color='#2563eb', width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Drawdown'], name='Drawdown', line=dict(color='#ef4444'), fill='tozeroy'), row=3, col=1)
                
                fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📝 Trade Log Details")
                if not trade_log.empty:
                    trade_log['Date'] = pd.to_datetime(trade_log['Date']).dt.date
                    st.dataframe(trade_log, use_container_width=True, hide_index=True)
            else:
                st.error(f"No data found for {sim_ticker} in this period.")

# ==========================================
# Page 5: Week 19 AI Engine (NEW ADDITION)
# ==========================================
elif page == "Week 19: AI Prediction Engine":
    st.header("🧠 Week 19: AI Deep Learning Engine")
    st.markdown("""
    **Real-time LSTM Neural Network Training.**
    This module fetches live market data, trains a PyTorch LSTM model on the fly, and predicts the next day's price.
    """)
    
    if not AI_AVAILABLE:
        st.error("⚠️ PyTorch is not installed. Please run `pip install torch scikit-learn` to use this feature.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("AI Settings")
            ai_ticker = st.text_input("Target Ticker", value="NVDA")
            st.info("Note: Training happens in real-time. Please wait a few seconds after clicking.")
            start_ai = st.button("🚀 Train & Predict", type="primary")
            
        with col2:
            if start_ai:
                ai_engine = AIRecommender(ai_ticker)
                
                with st.spinner(f"1. Fetching real-time data for {ai_ticker}..."):
                    df = ai_engine.fetch_and_prepare_data()
                
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    
                    with st.spinner("2. Training LSTM Neural Network (PyTorch)..."):
                        ai_engine.train_model()
                        
                    with st.spinner("3. Generating Inference..."):
                        predicted_price = ai_engine.predict_next_day()
                    
                    change_pct = (predicted_price - current_price) / current_price * 100
                    
                    st.success("Analysis Complete!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"${current_price:.2f}")
                    m2.metric("AI Predicted Price (Next Day)", f"${predicted_price:.2f}", delta=f"{change_pct:.2f}%")
                    
                    signal = "HOLD"
                    if change_pct > 1.0: signal = "STRONG BUY"
                    elif change_pct > 0.2: signal = "BUY"
                    elif change_pct < -1.0: signal = "STRONG SELL"
                    elif change_pct < -0.2: signal = "SELL"
                    
                    m3.metric("AI Signal", signal, delta_color="off")
                    
                    st.subheader("Price Projection")
                    # Charting
                    chart_data = df['Close'].tail(90)
                    fig_ai = go.Figure()
                    
                    # Historical Line
                    fig_ai.add_trace(go.Scatter(
                        x=chart_data.index, 
                        y=chart_data.values, 
                        mode='lines', 
                        name='Historical Price', 
                        line=dict(color='#00CC96', width=2)
                    ))
                    
                    # Prediction Point
                    next_date = chart_data.index[-1] + timedelta(days=1)
                    
                    # Connect last point to prediction
                    fig_ai.add_trace(go.Scatter(
                        x=[chart_data.index[-1], next_date], 
                        y=[chart_data.values[-1], predicted_price], 
                        mode='lines+markers', 
                        name='AI Projection', 
                        line=dict(color='#EF553B', width=2, dash='dot'), 
                        marker=dict(size=8)
                    ))
                    
                    fig_ai.update_layout(
                        title=f"LSTM Model Projection for {ai_ticker}", 
                        xaxis_title="Date", 
                        yaxis_title="Price ($)", 
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_ai, use_container_width=True)
                else:
                    st.error(f"Could not fetch data for {ai_ticker}. Please check the symbol.")

# --- Page 6: Raw Data (Moved to end) ---
elif page == "Raw Data Inspector":
    st.header("🔍 File Inspector")
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if all_files:
        selected_file = st.selectbox("Select CSV:", all_files)
        if selected_file:
            df = pd.read_csv(selected_file)
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("No CSV files found in the current directory.")

st.sidebar.markdown("---")
st.sidebar.info("Week 19: AI Engine Integrated")