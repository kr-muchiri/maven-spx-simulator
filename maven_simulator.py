import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import random

# Configure page
st.set_page_config(
    page_title="Maven SPX Market Making Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Maven branding
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #f0f2f6;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        color: #1e3c72;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        background: #f8f9fa;
        border-radius: 5px;
        font-style: italic;
        color: #6c757d;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Professional Header for Maven
st.markdown("""
<div class="main-header">
    <h1>📊 MAVEN TRADING</h1>
    <p>SPX Options Market Making Simulator | Built by Muchiri Kahwai</p>
    <p style="font-size: 1rem; margin-top: 1rem;">
        <strong>Demonstrating:</strong> SPX Options Pricing • CBOE Pit Trading • Operational Tools Development
    </p>
</div>
""", unsafe_allow_html=True)

# Description and Disclaimer Section
st.markdown("### 📋 About This Application")

st.info("""
**SPX Options Market Making Simulator** - Demonstrating quantitative finance and operational tools development 
relevant to Maven Trading's **Junior Trader** position focused on SPX options in the CBOE pit.
""")

st.markdown("**Key Features:**")
st.markdown("""
• **SPX Options Pricing:** Index options with dividend yield adjustments  
• **CBOE Pit Simulation:** Physical trading environment with rapid execution  
• **Operational Tools:** Automated position management and risk monitoring  
• **Strategy Analysis:** Position sizing and exposure optimization for SPX products  
• **Real-time P&L:** Continuous tracking across multiple SPX expirations  
""")

st.warning("""
**⚠️ Disclaimer:** This application is **not affiliated with or property of Maven Trading**. 
It was independently developed by Muchiri Kahwai to showcase quantitative finance and programming skills 
for consideration in the **Junior Trader** role. All market data is simulated for demonstration purposes.
""")

st.caption("*Built with Python, Streamlit, NumPy, Pandas, and Plotly • Operational tools focus for Maven Trading*")

st.markdown("---")

# Black-Scholes Functions adapted for SPX (with dividend yield)
def black_scholes_call_index(S, K, T, r, sigma, q=0.02):
    """Calculate Black-Scholes call option price for index with dividend yield"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put_index(S, K, T, r, sigma, q=0.02):
    """Calculate Black-Scholes put option price for index with dividend yield"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def calculate_greeks_index(S, K, T, r, sigma, option_type='call', q=0.02):
    """Calculate option Greeks for index options"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta (adjusted for dividend yield)
    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    # Gamma
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (adjusted for dividend yield)
    if option_type == 'call':
        theta = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                + q * S * np.exp(-q * T) * norm.cdf(d1) \
                - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) \
                - q * S * np.exp(-q * T) * norm.cdf(-d1) \
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = theta / 365  # Convert to daily
    
    # Vega
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# Initialize session state
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'current_spx' not in st.session_state:
    st.session_state.current_spx = 4500.0
if 'pnl_history' not in st.session_state:
    st.session_state.pnl_history = []

# Enhanced Sidebar for SPX Trading
st.sidebar.markdown('<p class="sidebar-header">📊 SPX Market Parameters</p>', unsafe_allow_html=True)

# SPX Market Data
st.sidebar.markdown("**📈 SPX Index Data**")
spx_price = st.sidebar.number_input("SPX Level", value=4500.0, min_value=3000.0, max_value=6000.0, step=1.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=5.25, min_value=0.0, max_value=10.0, step=0.05) / 100
dividend_yield = st.sidebar.number_input("SPX Dividend Yield (%)", value=2.0, min_value=0.0, max_value=5.0, step=0.1) / 100
implied_vol = st.sidebar.number_input("Implied Volatility (%)", value=18.0, min_value=5.0, max_value=50.0, step=0.5) / 100

st.sidebar.markdown("---")

# CBOE Pit Trading Parameters
st.sidebar.markdown("**🏛️ CBOE Pit Setup**")
bid_spread = st.sidebar.slider("Bid Spread (cents)", 5, 50, 15, help="Spread below theoretical price") / 100
ask_spread = st.sidebar.slider("Ask Spread (cents)", 5, 50, 15, help="Spread above theoretical price") / 100
max_position = st.sidebar.number_input("Max Position Size", value=500, min_value=50, max_value=2000, step=50)
pit_speed = st.sidebar.slider("Pit Execution Speed (ms)", 50, 500, 150, help="Simulated pit trading latency")

st.sidebar.markdown("---")

# Market Status
current_time = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown("**📈 CBOE SPX Pit Status**")
st.sidebar.success(f"🟢 **OPEN** - {current_time}")
st.sidebar.metric("SPX Level", f"{spx_price:.2f}", delta="Regular Trading")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏛️ SPX Pit Trading", "📊 Risk Dashboard", "💰 P&L Analysis", "📋 Position Manager", "⚙️ Operational Tools"])

with tab1:
    st.markdown("### 🏛️ SPX Options Pit Trading Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # SPX Options Chain
        st.markdown("#### SPX Options Chain")
        
        # SPX typically trades in 25-point increments around ATM
        current_strike_base = int(spx_price / 25) * 25
        strikes = np.arange(current_strike_base - 100, current_strike_base + 125, 25)
        expirations = [3, 7, 14, 30, 60, 90]  # SPX has frequent expirations
        
        col_exp, col_refresh = st.columns([3, 1])
        with col_exp:
            selected_expiry = st.selectbox("📅 Select Expiration (Days)", expirations, index=3)
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_spx_chain"):
                st.rerun()
        
        time_to_expiry = selected_expiry / 365
        
        # Build SPX options chain
        chain_data = []
        
        for strike in strikes:
            # Calculate theoretical prices with dividend yield
            call_price = black_scholes_call_index(spx_price, strike, time_to_expiry, risk_free_rate, implied_vol, dividend_yield)
            put_price = black_scholes_put_index(spx_price, strike, time_to_expiry, risk_free_rate, implied_vol, dividend_yield)
            
            # Add pit trading spreads (wider than electronic)
            call_bid = max(0.05, call_price - bid_spread)
            call_ask = call_price + ask_spread
            put_bid = max(0.05, put_price - bid_spread)
            put_ask = put_price + ask_spread
            
            # Calculate Greeks for SPX
            call_greeks = calculate_greeks_index(spx_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'call', dividend_yield)
            put_greeks = calculate_greeks_index(spx_price, strike, time_to_expiry, risk_free_rate, implied_vol, 'put', dividend_yield)
            
            # Moneyness indicators
            if abs(strike - spx_price) <= 12.5:
                moneyness = "🔵 ATM"
            elif strike < spx_price:
                moneyness = "🟢 ITM" if call_price > put_price else "🔴 OTM"
            else:
                moneyness = "🔴 OTM" if call_price > put_price else "🟢 ITM"
            
            chain_data.append({
                'Strike': f"{strike:.0f}",
                'Type': moneyness,
                'Call Bid': f"{call_bid:.2f}",
                'Call Ask': f"{call_ask:.2f}",
                'Call Δ': f"{call_greeks['delta']:.3f}",
                'Put Bid': f"{put_bid:.2f}",
                'Put Ask': f"{put_ask:.2f}",
                'Put Δ': f"{put_greeks['delta']:.3f}"
            })
        
        chain_df = pd.DataFrame(chain_data)
        st.dataframe(chain_df, use_container_width=True, hide_index=True)
    
    with col2:
        # CBOE Pit Trading Panel
        st.markdown("#### ⚡ Pit Execution")
        
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            
            trade_strike = st.selectbox("📍 Strike", strikes, index=len(strikes)//2)
            trade_type = st.selectbox("📊 Option Type", ['Call', 'Put'])
            trade_side = st.selectbox("📈 Side", ['Buy', 'Sell'])
            trade_quantity = st.number_input("📦 Quantity", value=50, min_value=1, max_value=500, step=10)
            
            # Calculate estimated price for SPX
            option_price = black_scholes_call_index(spx_price, trade_strike, time_to_expiry, risk_free_rate, implied_vol, dividend_yield) if trade_type == 'Call' else black_scholes_put_index(spx_price, trade_strike, time_to_expiry, risk_free_rate, implied_vol, dividend_yield)
            
            if trade_side == 'Buy':
                estimated_price = option_price + ask_spread
                price_color = "#dc3545"
            else:
                estimated_price = max(0.05, option_price - bid_spread)
                price_color = "#28a745"
            
            st.markdown(f"**Est. Price:** <span style='color: {price_color}; font-weight: bold;'>${estimated_price:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Notional:** ${estimated_price * trade_quantity * 100:.0f}")  # SPX multiplier is 100
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Execute in Pit", type="primary", use_container_width=True):
                # Simulate pit execution delay
                with st.spinner(f'Executing in CBOE pit... ({pit_speed}ms)'):
                    time.sleep(pit_speed / 1000)  # Convert ms to seconds
                
                # Calculate trade price
                if trade_side == 'Buy':
                    trade_price = option_price + ask_spread
                else:
                    trade_price = max(0.05, option_price - bid_spread)
                
                # Record trade
                trade = {
                    'timestamp': datetime.now(),
                    'strike': trade_strike,
                    'type': trade_type,
                    'side': trade_side,
                    'quantity': trade_quantity if trade_side == 'Buy' else -trade_quantity,
                    'price': trade_price,
                    'expiry_days': selected_expiry,
                    'notional': trade_price * abs(trade_quantity) * 100
                }
                
                st.session_state.trades.append(trade)
                
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>SPX Trade Executed!</strong><br>
                    {trade_side} {trade_quantity} SPX {trade_type} {trade_strike} @ ${trade_price:.2f}<br>
                    Notional: ${trade['notional']:.0f}
                </div>
                """, unsafe_allow_html=True)
        
        # SPX Market Insights
        st.markdown("#### 📊 SPX Market Insights")
        
        with st.container():
            # SPX-specific metrics
            vix_level = np.random.uniform(15, 25)  # Simulated VIX
            skew = np.random.uniform(-2, 2)  # Simulated skew
            
            st.metric("VIX Level", f"{vix_level:.1f}", delta="Vol Environment")
            st.metric("Put/Call Skew", f"{skew:.2f}%", delta="Skew Direction")
            st.metric("SPX Div Yield", f"{dividend_yield*100:.1f}%")
            
            # Portfolio delta exposure check
            total_delta = sum([
                calculate_greeks_index(spx_price, trade['strike'], trade['expiry_days']/365, risk_free_rate, implied_vol, trade['type'].lower(), dividend_yield)['delta'] * trade['quantity']
                for trade in st.session_state.trades
            ])
            
            if abs(total_delta) > 100:
                st.warning(f"⚠️ High SPX Delta: {total_delta:.0f}")
            else:
                st.success(f"✅ Delta Managed: {total_delta:.0f}")

with tab2:
    st.markdown("### 🛡️ SPX Portfolio Risk Dashboard")
    
    if st.session_state.trades:
        # Calculate SPX portfolio Greeks
        portfolio_delta = 0
        portfolio_gamma = 0
        portfolio_theta = 0
        portfolio_vega = 0
        total_notional = 0
        
        for trade in st.session_state.trades:
            time_to_exp = trade['expiry_days'] / 365
            greeks = calculate_greeks_index(spx_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol, trade['type'].lower(), dividend_yield)
            
            portfolio_delta += greeks['delta'] * trade['quantity']
            portfolio_gamma += greeks['gamma'] * trade['quantity']
            portfolio_theta += greeks['theta'] * trade['quantity']
            portfolio_vega += greeks['vega'] * trade['quantity']
            total_notional += trade['notional']
        
        # SPX Portfolio Greeks with enhanced presentation
        st.markdown("#### 📊 SPX Portfolio Greeks")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if abs(portfolio_delta) < 100 else "inverse"
            st.metric(
                "Portfolio Delta", 
                f"{portfolio_delta:.0f}", 
                delta=f"{'⚠️ High' if abs(portfolio_delta) > 200 else '✅ Normal'} Exposure",
                delta_color=delta_color
            )
        
        with col2:
            gamma_color = "normal" if abs(portfolio_gamma) < 10 else "inverse" 
            st.metric(
                "Portfolio Gamma", 
                f"{portfolio_gamma:.2f}", 
                delta=f"{'⚠️ High' if abs(portfolio_gamma) > 20 else '✅ Normal'} Convexity",
                delta_color=gamma_color
            )
        
        with col3:
            theta_color = "inverse" if portfolio_theta < -100 else "normal"
            st.metric(
                "Portfolio Theta", 
                f"${portfolio_theta:.0f}", 
                delta=f"{'📉 High Decay' if portfolio_theta < -200 else '📈 Low Decay'}",
                delta_color=theta_color
            )
        
        with col4:
            vega_color = "normal" if abs(portfolio_vega) < 200 else "inverse"
            st.metric(
                "Portfolio Vega", 
                f"{portfolio_vega:.0f}", 
                delta=f"{'⚠️ High' if abs(portfolio_vega) > 500 else '✅ Normal'} Vol Risk",
                delta_color=vega_color
            )
        
        st.markdown("---")
        
        # SPX Risk Limits Monitoring
        st.markdown("#### 🎯 SPX Risk Limits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Delta limit for SPX (higher than individual stocks)
            delta_limit = 200
            delta_utilization = min(abs(portfolio_delta) / delta_limit, 1.0)
            
            if delta_utilization < 0.6:
                bar_color = "#28a745"
            elif delta_utilization < 0.8:
                bar_color = "#ffc107"
            else:
                bar_color = "#dc3545"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = abs(portfolio_delta),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "🔺 SPX Delta Risk", 'font': {'size': 16}},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, delta_limit], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.8},
                    'steps': [
                        {'range': [0, delta_limit*0.6], 'color': "#e8f5e8"},
                        {'range': [delta_limit*0.6, delta_limit*0.8], 'color': "#fff3cd"},
                        {'range': [delta_limit*0.8, delta_limit], 'color': "#f8d7da"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': delta_limit}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vega limit for SPX
            vega_limit = 500
            vega_utilization = min(abs(portfolio_vega) / vega_limit, 1.0)
            
            if vega_utilization < 0.6:
                bar_color = "#28a745"
            elif vega_utilization < 0.8:
                bar_color = "#ffc107" 
            else:
                bar_color = "#dc3545"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = abs(portfolio_vega),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "📊 SPX Vega Risk", 'font': {'size': 16}},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, vega_limit], 'tickwidth': 1},
                    'bar': {'color': bar_color, 'thickness': 0.8},
                    'steps': [
                        {'range': [0, vega_limit*0.6], 'color': "#e8f5e8"},
                        {'range': [vega_limit*0.6, vega_limit*0.8], 'color': "#fff3cd"},
                        {'range': [vega_limit*0.8, vega_limit], 'color': "#f8d7da"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': vega_limit}}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # SPX Portfolio summary
            st.markdown("**📈 SPX Portfolio Summary**")
            
            total_positions = len([t for t in st.session_state.trades if t['quantity'] != 0])
            net_notional = sum([t['notional'] if t['quantity'] > 0 else -t['notional'] for t in st.session_state.trades])
            
            st.metric("Active SPX Positions", total_positions)
            st.metric("Net Notional", f"${net_notional:.0f}")
            st.metric("Total Notional", f"${total_notional:.0f}")
            
            # Risk status
            risk_level = "🟢 LOW" if max(delta_utilization, vega_utilization) < 0.6 else "🟡 MEDIUM" if max(delta_utilization, vega_utilization) < 0.8 else "🔴 HIGH"
            st.markdown(f"**Overall Risk:** {risk_level}")
        
        # SPX Scenario Analysis
        st.markdown("---")
        st.markdown("#### 📈 SPX Scenario Analysis")
        
        spx_scenarios = np.linspace(spx_price * 0.85, spx_price * 1.15, 50)
        pnl_scenarios = []
        
        for scenario_spx in spx_scenarios:
            scenario_pnl = 0
            for trade in st.session_state.trades:
                time_to_exp = trade['expiry_days'] / 365
                if trade['type'] == 'Call':
                    current_price = black_scholes_call_index(scenario_spx, trade['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
                else:
                    current_price = black_scholes_put_index(scenario_spx, trade['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
                
                pnl = (current_price - trade['price']) * trade['quantity'] * 100  # SPX multiplier
                scenario_pnl += pnl
            
            pnl_scenarios.append(scenario_pnl)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spx_scenarios, 
            y=pnl_scenarios, 
            mode='lines+markers',
            line=dict(color='#1e3c72', width=3),
            fill='tonexty',
            fillcolor='rgba(30, 60, 114, 0.1)',
            name='SPX Portfolio P&L',
            hovertemplate='SPX: %{x:.0f}<br>P&L: $%{y:.0f}<extra></extra>'
        ))
        
        fig.add_vline(
            x=spx_price, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text=f"Current SPX: {spx_price:.0f}",
            annotation_position="top"
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig.update_layout(
            title="📊 SPX Portfolio P&L vs Index Level",
            xaxis_title="SPX Index Level",
            yaxis_title="Portfolio P&L ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No SPX Positions</strong><br>
            Go to the SPX Pit Trading tab to start building positions and see risk metrics here.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 💰 SPX P&L Analysis")
    
    if st.session_state.trades:
        # Calculate current SPX P&L
        total_pnl = 0
        trade_pnls = []
        
        for i, trade in enumerate(st.session_state.trades):
            time_to_exp = trade['expiry_days'] / 365
            if trade['type'] == 'Call':
                current_price = black_scholes_call_index(spx_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
            else:
                current_price = black_scholes_put_index(spx_price, trade['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
            
            trade_pnl = (current_price - trade['price']) * trade['quantity'] * 100  # SPX multiplier
            total_pnl += trade_pnl
            
            trade_pnls.append({
                'Trade_ID': i+1,
                'Strike': trade['strike'],
                'Type': trade['type'],
                'Quantity': trade['quantity'],
                'Entry_Price': f"${trade['price']:.2f}",
                'Current_Price': f"${current_price:.2f}",
                'Notional': f"${trade['notional']:.0f}",
                'P&L': f"${trade_pnl:.0f}"
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total SPX P&L", f"${total_pnl:.0f}", delta=f"{total_pnl:.0f}")
        with col2:
            total_notional = sum([trade['notional'] for trade in st.session_state.trades])
            pnl_percentage = (total_pnl / total_notional * 100) if total_notional > 0 else 0
            st.metric("P&L %", f"{pnl_percentage:.2f}%", delta=f"{pnl_percentage:.2f}%")
        with col3:
            st.metric("Total Notional", f"${total_notional:.0f}")
        
        # P&L by Trade
        st.subheader("SPX P&L by Trade")
        pnl_df = pd.DataFrame(trade_pnls)
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No SPX trades executed yet. P&L analysis will appear after trading.")

with tab4:
    st.markdown("### 📋 SPX Position Manager")
    
    if st.session_state.trades:
        # Aggregate SPX positions
        positions = {}
        
        for trade in st.session_state.trades:
            key = f"{trade['type']}_{trade['strike']}_{trade['expiry_days']}"
            if key not in positions:
                positions[key] = {
                    'strike': trade['strike'],
                    'type': trade['type'],
                    'expiry_days': trade['expiry_days'],
                    'quantity': 0,
                    'avg_price': 0,
                    'total_notional': 0
                }
            
            positions[key]['quantity'] += trade['quantity']
            positions[key]['total_notional'] += trade['notional'] if trade['quantity'] > 0 else -trade['notional']
        
        # Calculate average prices
        for pos in positions.values():
            if pos['quantity'] != 0:
                pos['avg_price'] = abs(pos['total_notional']) / (abs(pos['quantity']) * 100)
        
        # Filter active positions
        active_positions = {k: v for k, v in positions.items() if v['quantity'] != 0}
        
        if active_positions:
            position_data = []
            
            for pos_id, pos in active_positions.items():
                time_to_exp = pos['expiry_days'] / 365
                if pos['type'] == 'Call':
                    current_price = black_scholes_call_index(spx_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
                else:
                    current_price = black_scholes_put_index(spx_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol, dividend_yield)
                
                greeks = calculate_greeks_index(spx_price, pos['strike'], time_to_exp, risk_free_rate, implied_vol, pos['type'].lower(), dividend_yield)
                position_pnl = (current_price - pos['avg_price']) * pos['quantity'] * 100
                
                position_data.append({
                    'Position_ID': pos_id,
                    'Strike': pos['strike'],
                    'Type': pos['type'],
                    'Quantity': pos['quantity'],
                    'Avg_Price': f"${pos['avg_price']:.2f}",
                    'Current_Price': f"${current_price:.2f}",
                    'P&L': f"${position_pnl:.0f}",
                    'Delta': f"{greeks['delta'] * pos['quantity']:.1f}",
                    'Gamma': f"{greeks['gamma'] * pos['quantity']:.2f}",
                    'Theta': f"${greeks['theta'] * pos['quantity']:.0f}",
                    'Vega': f"{greeks['vega'] * pos['quantity']:.0f}"
                })
            
            positions_df = pd.DataFrame(position_data)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            
            # Position management controls
            st.subheader("Position Management")
            
            if st.button("Close All SPX Positions", type="secondary"):
                st.session_state.trades = []
                st.session_state.pnl_history = []
                st.success("All SPX positions closed!")
                st.rerun()
        
        else:
            st.info("No active SPX positions.")
    
    else:
        st.info("No SPX positions to manage.")

with tab5:
    st.markdown("### ⚙️ Operational Tools")
    
    st.markdown("#### 📊 Trading Analytics")
    
    if st.session_state.trades:
        # Trading statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trade Volume Analysis**")
            
            total_trades = len(st.session_state.trades)
            total_volume = sum([abs(trade['quantity']) for trade in st.session_state.trades])
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
            
            st.metric("Total Trades", total_trades)
            st.metric("Total Volume", f"{total_volume} contracts")
            st.metric("Avg Trade Size", f"{avg_trade_size:.0f} contracts")
        
        with col2:
            st.markdown("**Strike Distribution**")
            
            strike_counts = {}
            for trade in st.session_state.trades:
                strike = trade['strike']
                if strike not in strike_counts:
                    strike_counts[strike] = 0
                strike_counts[strike] += abs(trade['quantity'])
            
            if strike_counts:
                strikes = list(strike_counts.keys())
                volumes = list(strike_counts.values())
                
                fig = go.Figure(data=[go.Bar(x=strikes, y=volumes)])
                fig.update_layout(
                    title="Volume by Strike",
                    xaxis_title="Strike Price",
                    yaxis_title="Volume",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🛠️ Risk Management Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Position Flattening Tool
        st.markdown("**Position Flattening Tool**")
        
        if st.session_state.trades:
            total_delta = sum([
                calculate_greeks_index(spx_price, trade['strike'], trade['expiry_days']/365, risk_free_rate, implied_vol, trade['type'].lower(), dividend_yield)['delta'] * trade['quantity']
                for trade in st.session_state.trades
            ])
            
            st.metric("Current Portfolio Delta", f"{total_delta:.0f}")
            
            if abs(total_delta) > 50:
                if st.button("🎯 Auto-Hedge Delta", type="primary"):
                    st.success(f"Would hedge {total_delta:.0f} delta using SPX futures or ATM options")
            else:
                st.success("✅ Portfolio is delta neutral")
        else:
            st.info("No positions to hedge")
    
    with col2:
        # Performance Metrics
        st.markdown("**Performance Metrics**")
        
        if st.session_state.trades:
            # Calculate some basic performance metrics
            total_pnl = sum([
                (black_scholes_call_index(spx_price, trade['strike'], trade['expiry_days']/365, risk_free_rate, implied_vol, dividend_yield) if trade['type'] == 'Call' 
                 else black_scholes_put_index(spx_price, trade['strike'], trade['expiry_days']/365, risk_free_rate, implied_vol, dividend_yield)
                 - trade['price']) * trade['quantity'] * 100
                for trade in st.session_state.trades
            ])
            
            total_notional = sum([trade['notional'] for trade in st.session_state.trades])
            
            win_rate = len([t for t in st.session_state.trades if ((black_scholes_call_index(spx_price, t['strike'], t['expiry_days']/365, risk_free_rate, implied_vol, dividend_yield) if t['type'] == 'Call' else black_scholes_put_index(spx_price, t['strike'], t['expiry_days']/365, risk_free_rate, implied_vol, dividend_yield)) - t['price']) * t['quantity'] > 0]) / len(st.session_state.trades) * 100
            
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("ROI", f"{(total_pnl/total_notional*100):.2f}%" if total_notional > 0 else "0%")
            st.metric("Avg P&L per Trade", f"${total_pnl/len(st.session_state.trades):.0f}" if st.session_state.trades else "$0")
        else:
            st.info("No trading data available")

# Real-time updates with enhanced styling
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh SPX Data", use_container_width=True):
    # Simulate SPX movement
    st.session_state.current_spx = spx_price + np.random.normal(0, 5.0)
    st.success("SPX market data refreshed!")
    st.rerun()

# Enhanced Footer
st.markdown("""
<div class="footer">
    <h4>📊 MAVEN TRADING SPX MARKET MAKING SIMULATOR</h4>
    <p><strong>Built by Muchiri Kahwai</strong> | Demonstrating SPX Options Trading & Operational Tools</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        <em>Featuring: SPX Index Options • CBOE Pit Trading • Risk Management • Operational Analytics</em>
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem; color: #6c757d;">
        📧 mk@Muchiri.tech | 📱 +1(859)319-6196 | 
        <a href="https://linkedin.com/in/muchiri-kahwai" style="color: #1e3c72;">LinkedIn</a> | 
        <a href="https://github.com/muchirikahwai" style="color: #1e3c72;">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)