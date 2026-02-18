import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import timedelta

# --- Import from your Engine ---
# 确保 20_ai_engine.py 在同一目录下
try:
    from importlib import import_module
    ai_engine_module = import_module("20_ai_engine")
    RiskManager = ai_engine_module.RiskManager
    AIRecommender = ai_engine_module.AIRecommender
    run_simulation = ai_engine_module.run_simulation
    AI_AVAILABLE = True
except ImportError:
    st.error("Could not import '20_ai_engine.py'. Make sure both files are in the same folder.")
    AI_AVAILABLE = False

# ==========================================
# Streamlit Page Configuration
# ==========================================
st.set_page_config(page_title="AI Quant Dashboard", page_icon="📈", layout="wide")
st.title("🤖 Innovation AI: Quant Internship Dashboard (Week 20)")

# ==========================================
# Data Loading Helper
# ==========================================
@st.cache_data
def load_data(filename):
    if os.path.exists(filename): return pd.read_csv(filename)
    else: return None

# ==========================================
# Navigation
# ==========================================
st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio("Select View:", [
    "Overview & Portfolio", 
    "Backtest Performance", 
    "Model Metrics", 
    "Risk Management Simulator", 
    "Week 20: AI Prediction & Risk",
    "Raw Data Inspector"
])

# ==========================================
# Page Content Logic
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

# --- Page 4: Risk Sim ---
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

    if run_sim and AI_AVAILABLE:
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
# Page 5: Week 20 AI Engine (UPDATED)
# ==========================================
elif page == "Week 20: AI Prediction & Risk":
    st.header("🧠 Week 20: AI Prediction & Risk Analysis")
    st.markdown("""
    **Recursive Forecasting & Risk Metrics.**
    This module trains a real-time LSTM model to predict future price trends (up to 30 days) 
    and analyzes the asset's historical volatility and Sharpe Ratio.
    """)
    
    if not AI_AVAILABLE:
        st.error("⚠️ AI Engine not loaded. Check if '20_ai_engine.py' exists.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Configuration")
            ai_ticker = st.text_input("Target Ticker", value="NVDA")
            horizon = st.slider("Forecast Horizon (Days)", 1, 30, 30)
            st.info("Training happens in real-time. Please wait a few seconds.")
            start_ai = st.button("🚀 Run Analysis", type="primary")
            
        with col2:
            if start_ai:
                ai_engine = AIRecommender(ai_ticker)
                
                with st.spinner(f"1. Fetching data & Calculating Risk for {ai_ticker}..."):
                    df = ai_engine.fetch_and_prepare_data()
                    vol, sharpe = ai_engine.calculate_risk_metrics()
                
                if df is not None:
                    current_price = df['Close'].iloc[-1]
                    
                    with st.spinner("2. Training LSTM Neural Network..."):
                        ai_engine.train_model()
                        
                    with st.spinner(f"3. Generating {horizon}-Day Forecast..."):
                        future_prices = ai_engine.predict_future_recursive(horizon=horizon)
                    
                    final_pred = future_prices[-1]
                    change_pct = (final_pred - current_price) / current_price * 100
                    
                    # --- Metrics Display ---
                    st.success("Analysis Complete!")
                    
                    # Row 1: Price & Prediction
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"${current_price:.2f}")
                    m2.metric(f"Predicted ({horizon} Days)", f"${final_pred:.2f}", delta=f"{change_pct:.2f}%")
                    
                    signal = "HOLD"
                    if change_pct > 2.0 and sharpe > 1.0: signal = "STRONG BUY"
                    elif change_pct > 0.5: signal = "BUY"
                    elif change_pct < -2.0: signal = "STRONG SELL"
                    elif change_pct < -0.5: signal = "SELL"
                    
                    m3.metric("AI Signal", signal, delta_color="off")
                    
                    # Row 2: Risk Metrics (Week 20 New)
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Annualized Volatility", f"{vol*100:.1f}%", help="Higher means more risk")
                    r2.metric("Sharpe Ratio", f"{sharpe:.2f}", help=">1.0 is considered good")
                    
                    risk_label = "Low"
                    if vol > 0.4: risk_label = "High"
                    elif vol > 0.2: risk_label = "Medium"
                    r3.metric("Risk Level", risk_label)
                    
                    st.divider()
                    
                    # --- Charting ---
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
                    
                    # Future Dates
                    last_date = chart_data.index[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
                    
                    # Prediction Line
                    fig_ai.add_trace(go.Scatter(
                        x=future_dates, 
                        y=future_prices, 
                        mode='lines+markers', 
                        name=f'AI Forecast ({horizon}d)', 
                        line=dict(color='#EF553B', width=2, dash='dash'), 
                        marker=dict(size=6)
                    ))
                    
                    # Connect the lines
                    fig_ai.add_trace(go.Scatter(
                        x=[last_date, future_dates[0]],
                        y=[chart_data.values[-1], future_prices[0]],
                        mode='lines',
                        showlegend=False,
                        line=dict(color='#EF553B', width=2, dash='dash')
                    ))
                    
                    fig_ai.update_layout(
                        title=f"LSTM Forecast: {ai_ticker} (Horizon: {horizon} Days)", 
                        xaxis_title="Date", 
                        yaxis_title="Price ($)", 
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_ai, use_container_width=True)
                    
                    with st.expander("See Detailed Forecast Data"):
                        st.dataframe(pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices}))
                        
                else:
                    st.error(f"Could not fetch data for {ai_ticker}. Please check the symbol.")

# --- Page 6: Raw Data ---
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
st.sidebar.info("Week 20: Recursive Forecast & Risk Metrics")