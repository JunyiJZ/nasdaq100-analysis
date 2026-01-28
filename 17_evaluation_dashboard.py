import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. 页面配置 (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="AI Quant Project Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 Innovation AI: Quant Internship Dashboard")
st.markdown("### Week 17: Strategy Evaluation & Risk Analysis")

# ==========================================
# 2. 数据加载函数 (Data Loading)
# ==========================================
@st.cache_data
def load_data(filename):
    """
    尝试加载 CSV 文件，如果文件不存在返回 None
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return None

# ==========================================
# 3. 侧边栏导航 (Sidebar Navigation)
# ==========================================
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio(
    "Select View:",
    ["Overview & Portfolio", "Backtest Performance", "Model Metrics", "Raw Data Inspector"]
)

# ==========================================
# 4. 页面内容逻辑
# ==========================================

# --- 页面 1: 投资组合概览 ---
if page == "Overview & Portfolio":
    st.header("📊 Portfolio Allocation (Week 15 Result)")
    
    # 读取你的优化结果文件
    df_alloc = load_data("optimized_portfolio_allocation.csv")
    
    if df_alloc is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 绘制饼图
            # 假设你的CSV里有 'Ticker' 和 'Weight' 列，如果列名不同，请在此处修改
            # 如果没有列名，Streamlit会显示数据让你检查
            numeric_cols = df_alloc.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) > 0:
                fig = px.pie(
                    df_alloc, 
                    values=numeric_cols[0], # 取第一列数值作为权重
                    names=df_alloc.columns[0], # 取第一列作为股票代码
                    title="Optimized Asset Allocation",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not identify numeric columns for Pie Chart.")
        
        with col2:
            st.subheader("Top Holdings")
            st.dataframe(df_alloc.head(10), use_container_width=True)
    else:
        st.error("File 'optimized_portfolio_allocation.csv' not found. Please run Week 15 script first.")

# --- 页面 2: 回测表现 ---
elif page == "Backtest Performance":
    st.header("📈 Strategy vs Benchmark (Week 12-14 Results)")
    
    # 选择要查看的策略文件
    strategy_file = st.selectbox(
        "Select Strategy Result:",
        ["long_term_backtest.csv", "long_term_backtest_realistic.csv", "midterm_strategy_backtest.csv"]
    )
    
    df_backtest = load_data(strategy_file)
    
    if df_backtest is not None:
        # 尝试转换日期列
        if 'Date' in df_backtest.columns:
            df_backtest['Date'] = pd.to_datetime(df_backtest['Date'])
        
        # 绘制资金曲线
        st.subheader("Equity Curve")
        
        # 自动识别数值列进行绘图
        numeric_cols = df_backtest.select_dtypes(include=['float', 'int']).columns.tolist()
        selected_cols = st.multiselect("Select Metrics to Plot", numeric_cols, default=numeric_cols[:2])
        
        if selected_cols:
            fig = px.line(
                df_backtest, 
                x='Date' if 'Date' in df_backtest.columns else df_backtest.index, 
                y=selected_cols,
                title=f"Performance: {strategy_file}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 计算简单的回撤 (Drawdown)
        if len(selected_cols) > 0:
            st.subheader("Drawdown Analysis")
            primary_col = selected_cols[0]
            # 计算回撤逻辑
            rolling_max = df_backtest[primary_col].cummax()
            drawdown = (df_backtest[primary_col] - rolling_max) / rolling_max
            
            fig_dd = px.area(
                x=df_backtest['Date'] if 'Date' in df_backtest.columns else df_backtest.index, 
                y=drawdown,
                title=f"Drawdown: {primary_col}"
            )
            st.plotly_chart(fig_dd, use_container_width=True)
            
    else:
        st.warning(f"File '{strategy_file}' not found.")

# --- 页面 3: 模型指标 ---
elif page == "Model Metrics":
    st.header("🤖 AI Model Performance (Week 8-10 Results)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Models")
        df_base = load_data("baseline_results.csv")
        if df_base is not None:
            st.dataframe(df_base)
            # 假设有一列叫 'RMSE' 或 'Accuracy'
            if 'RMSE' in df_base.columns:
                fig = px.bar(df_base, x=df_base.columns[0], y='RMSE', title="Model RMSE Comparison")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("baseline_results.csv not found.")

    with col2:
        st.subheader("Hyperparameter Tuning")
        df_tune = load_data("tuning_performance_results.csv")
        if df_tune is not None:
            st.dataframe(df_tune)
        else:
            st.info("tuning_performance_results.csv not found.")

# --- 页面 4: 原始数据检查器 ---
elif page == "Raw Data Inspector":
    st.header("🔍 File Inspector")
    st.markdown("Use this tab to check if your CSV files are formatted correctly.")
    
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    selected_file = st.selectbox("Select a CSV file to inspect:", all_files)
    
    if selected_file:
        df = pd.read_csv(selected_file)
        st.write(f"**Shape:** {df.shape}")
        st.write("**Columns:**", df.columns.tolist())
        st.dataframe(df.head(20))

# ==========================================
# Footer
# ==========================================
st.sidebar.markdown("---")
st.sidebar.info("Run this app via terminal: \n`streamlit run 17_evaluation_dashboard.py`")