import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import yfinance as yf
from datetime import timedelta

# ==========================================
# 0. Core Logic (Risk Engine)
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
    # --- FIX 1: Data Warm-up ---
    # Download extra data (60 days) prior to start_date to ensure indicators (SMA30) are ready on day 1
    warmup_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
    
    try:
        df = yf.download(ticker, start=warmup_start, end=end_date, progress=False)
    except Exception as e:
        return None, None, None

    if df.empty: return None, None, None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Calculate Indicators on the FULL dataset first
    rm = RiskManager(params['capital'], 14, params['atr_mult'], params['trail_stop'], params['risk_pct'])
    df = rm.calculate_atr(df)
    
    df['SMA_Fast'] = df['Close'].rolling(10).mean()
    df['SMA_Slow'] = df['Close'].rolling(30).mean()
    
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    
    # --- FIX 2: Slice to User Requested Range ---
    # Now that indicators are calculated, we cut the data to the user's requested start date
    df = df[df.index >= pd.to_datetime(start_date)]
    
    if df.empty: return None, None, None

    # 3. Simulation Variables
    capital = rm.initial_capital
    shares = 0
    equity_curve = []
    trade_log = []
    stop_loss_price = 0.0
    high_water_mark = 0.0
    entry_price = 0.0

    # 4. Loop
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
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="AI Quant Dashboard", page_icon="📈", layout="wide")
st.title("🤖 Innovation AI: Quant Internship Dashboard")

# ==========================================
# 2. Data Loading
# ==========================================
@st.cache_data
def load_data(filename):
    if os.path.exists(filename): return pd.read_csv(filename)
    else: return None

# ==========================================
# 3. Navigation
# ==========================================
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio("Select View:", ["Overview & Portfolio", "Backtest Performance", "Model Metrics", "Risk Management Simulator", "Raw Data Inspector"])

# ==========================================
# 4. Page Content
# ==========================================

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

elif page == "Model Metrics":
    st.header("🤖 AI Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        df_base = load_data("baseline_results.csv")
        if df_base is not None: st.dataframe(df_base)
    with col2:
        st.subheader("Tuning")
        df_tune = load_data("tuning_performance_results.csv")
        if df_tune is not None: st.dataframe(df_tune)

elif page == "Risk Management Simulator":
    st.header("🛡️ Week 18: Modern Era Stress Testing")
    st.markdown("""
    Test your risk parameters against **Recent Market Events (2022-2024)**.
    This avoids the structural differences of older eras (like 2008) and focuses on the current market regime.
    """)

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
                
                # Calculate Win Rate
                if not trade_log.empty:
                    wins = trade_log[trade_log['PnL ($)'] > 0]
                    sells = trade_log[trade_log['Type'] == 'SELL']
                    win_rate = (len(wins) / len(sells) * 100) if len(sells) > 0 else 0
                    total_trades_count = len(sells) # Count completed trades
                else:
                    win_rate = 0
                    total_trades_count = 0

                # Display Metrics
                st.markdown("### 📊 Test Results")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Final Equity", f"${final_eq:,.0f}", f"{ret:.2%}")
                m2.metric("Completed Trades", total_trades_count)
                m3.metric("Max Drawdown", f"{max_dd:.2%}", delta_color="inverse")
                m4.metric("Win Rate", f"{win_rate:.1f}%")
                m5.metric("Scenario", selected_scenario_name)

                # Plotting
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                                    subplot_titles=(f"{sim_ticker} Price Action", "Account Equity", "Drawdown"))

                # 1. Price Chart
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Close'], name='Price', line=dict(color='gray')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['SMA_Fast'], name='SMA Fast', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['SMA_Slow'], name='SMA Slow', line=dict(color='blue', width=1)), row=1, col=1)

                if not trade_log.empty:
                    buys = trade_log[trade_log['Type'] == 'BUY']
                    sells = trade_log[trade_log['Type'] == 'SELL']
                    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'), row=1, col=1)

                # 2. Equity Curve - FIX: Removed fill='tozeroy' to allow auto-scaling
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Equity'], name='Equity', line=dict(color='#2563eb', width=2)), row=2, col=1)
                
                # 3. Drawdown Chart
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Drawdown'], name='Drawdown', line=dict(color='#ef4444'), fill='tozeroy'), row=3, col=1)
                
                fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # --- Trade Log Table ---
                st.subheader("📝 Trade Log Details")
                if not trade_log.empty:
                    trade_log['Date'] = pd.to_datetime(trade_log['Date']).dt.date
                    st.dataframe(
                        trade_log,
                        use_container_width=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date"),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "PnL (%)": st.column_config.NumberColumn("PnL %", format="%.2f %%"),
                            "PnL ($)": st.column_config.NumberColumn("PnL $", format="$%.2f"),
                        },
                        hide_index=True
                    )
            else:
                st.error(f"No data found for {sim_ticker} in this period.")

elif page == "Raw Data Inspector":
    st.header("🔍 File Inspector")
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    selected_file = st.selectbox("Select CSV:", all_files)
    if selected_file:
        df = pd.read_csv(selected_file)
        st.dataframe(df.head(20))

st.sidebar.markdown("---")
st.sidebar.info("Week 18: Modern Era Stress Testing Loaded")