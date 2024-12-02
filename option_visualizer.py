import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import requests
import datetime
import plotly.express as px

# API Key for MarketAux
MARKETAUX_API_KEY = 'qbxbDtY9kpNK6YrAKGrg8vn892LIdKcJPQSIDzFv'

# Setup
st.set_page_config(page_title="Advanced Options Visualizer", layout="wide")

def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    return data['Close'].iloc[-1] if not data.empty else None

def safe_divide(numerator, denominator):
    """Safely divides two numbers, returning NaN if the denominator is zero."""
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)

def black_scholes(S, K, T, r, v, div, option_type):
    if S is None or S <= 0 or K <= 0 or T <= 0 or v <= 0:
        return None, {}
    d1 = safe_divide((np.log(S / K) + (r - div + 0.5 * v**2) * T), (v * np.sqrt(T)))
    d2 = d1 - v * np.sqrt(T)
    call_price = S * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-div * T) * norm.cdf(-d1)
    price = call_price if option_type == 'call' else put_price
    greeks = {
        'Delta': norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1),
        'Gamma': safe_divide(norm.pdf(d1), (S * v * np.sqrt(T))),
        'Vega': S * norm.pdf(d1) * np.sqrt(T) * np.exp(-div * T),
        'Theta': (-S * norm.pdf(d1) * v * np.exp(-div * T) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' \
                 else (-S * norm.pdf(d1) * v * np.exp(-div * T) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)),
        'Rho': T * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -T * K * np.exp(-r * T) * norm.cdf(-d2)
    }
    return price, greeks

def calculate_higher_order_greeks(S, K, T, r, v, div):
    if S <= 0 or K <= 0 or T <= 0 or v <= 0:
        return {}
    d1 = safe_divide((np.log(S / K) + (r - div + 0.5 * v**2) * T), (v * np.sqrt(T)))
    d2 = d1 - v * np.sqrt(T)
    gamma = safe_divide(norm.pdf(d1), (S * v * np.sqrt(T)))
    vega = S * norm.pdf(d1) * np.sqrt(T) * np.exp(-div * T)
    higher_order_greeks = {
        'Charm': safe_divide(-norm.pdf(d1) * (2 * (r - div) * T - d2 * v * np.sqrt(T)), (2 * T * v * np.sqrt(T))),
        'Color': safe_divide(-norm.pdf(d1) * ((2 * div - r) / (v * T * np.sqrt(T))) - (2 * gamma / T), 1),
        'Speed': safe_divide(-gamma / S * (d1 / (v * np.sqrt(T)) + 1), 1),
        'Vanna': safe_divide(vega * (1 - d1 * d2), v),
        'Vomma': safe_divide(vega * d1 * d2, v),
        'Zomma': safe_divide(gamma * (d1 * d2 - 1), v)
    }
    return higher_order_greeks

def fetch_market_news():
    try:
        url = f'https://api.marketaux.com/v1/news/all?api_token={MARKETAUX_API_KEY}&language=en'
        response = requests.get(url)
        news_items = response.json().get('data', [])
        return news_items[:5]  # Return the latest 5 news items
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Input Parameters
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
S = fetch_data(ticker)
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
expiration = st.sidebar.date_input("Expiration Date")
T = max((expiration - datetime.date.today()).days / 365.25, 0.0001)
r = st.sidebar.number_input("Risk-Free Rate (%)", value=1.0) / 100
v = st.sidebar.number_input("Volatility (%)", value=20.0) / 100
div = st.sidebar.number_input("Dividend Yield (%)", value=0.0) / 100
option_type = st.sidebar.radio("Option Type", ['call', 'put'])

price, greeks = black_scholes(S, K, T, r, v, div, option_type) if S else (0, {})
higher_order_greeks = calculate_higher_order_greeks(S, K, T, r, v, div) if S else {}

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["About", "Option Details", "Market News", "Volatility & Probability", "Higher Order Greeks"])

# About  Tab
with tab1:
    st.header("By: Miguel Angel Crespo III")
    st.write("""
    Being passionate about financial modeling and software development, I developed this tool to enhance the understanding of options pricing and its Greeks.Thanks for checking it out!
    Connect with me on [LinkedIn](https://www.linkedin.com/in/miguel-%C3%A1ngel-crespo-iii-26b344223/).
    """)

# Option Details Tab
with tab2:
    st.header("Option Pricing and Greeks")
    if price:
        st.write(f"Calculated {option_type.capitalize()} Price: ${price:.2f}")
        st.write("Black-Scholes Greeks:")
        for greek, value in greeks.items():
            st.write(f"{greek}: {value:.4f}")

        # Chart: Option Price Relative to Stock Price
        stock_prices = np.linspace(S * 0.8, S * 1.2, 50)
        option_prices = [black_scholes(price, K, T, r, v, div, option_type)[0] for price in stock_prices]
        fig1 = px.line(x=stock_prices, y=option_prices, title="Option Price vs. Stock Price")
        fig1.update_layout(xaxis_title="Stock Price", yaxis_title="Option Price", title_x=0.5)
        st.plotly_chart(fig1, key="option_vs_stock")

        # Chart: Option Price Relative to Expiration Date
        expiration_days = np.linspace(1, 365, 50) / 365  # Range of time to expiration (1 day to 1 year)
        option_prices_expiration = [black_scholes(S, K, t, r, v, div, option_type)[0] for t in expiration_days]
        expiration_dates = [datetime.date.today() + datetime.timedelta(days=int(t * 365)) for t in expiration_days]
        fig2 = px.line(x=expiration_dates, y=option_prices_expiration, title="Option Price vs. Expiration Date")
        fig2.update_layout(xaxis_title="Expiration Date", yaxis_title="Option Price", title_x=0.5)
        st.plotly_chart(fig2, key="option_vs_expiration")

# Market News Tab
with tab3:
    st.header("Latest Financial News")
    news = fetch_market_news()
    for item in news:
        st.write(f"[{item.get('title', 'No Title Available')}]({item.get('url', '#')}) - {item.get('summary', 'No summary available')}")

# Heatmaps in Tab 4
with tab4:
    st.header("Heatmaps")
    strikes = np.arange(K - 10, K + 11, 1)
    volatilities = np.linspace(0.1, 1, 10)
    probabilities = np.array([
        [norm.cdf((np.log(S / strike) + (r - div + 0.5 * vol**2) * T) / (vol * np.sqrt(T))) if S > 0 else 0
         for strike in strikes] for vol in volatilities])

    fig_prob = px.imshow(probabilities, x=strikes,                      y=volatilities,
                      color_continuous_scale="RdYlGn",
                      title="Probability Heatmap",
                      labels={"x": "Strike Price", "y": "Volatility (%)", "color": "Probability"},
                      aspect="auto")

    fig_prob.update_layout(
        title=dict(font=dict(size=20, color="white"), x=0.5),
        xaxis=dict(title="Strike Price", tickangle=45, showgrid=True, gridcolor="gray"),
        yaxis=dict(title="Volatility (%)", showgrid=True, gridcolor="gray"),
        coloraxis_colorbar=dict(title="Probability"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )
    fig_prob.update_traces(hovertemplate="Strike Price: %{x}<br>Volatility: %{y}<br>Probability: %{z:.2f}")
    st.plotly_chart(fig_prob, key="probability_heatmap")

    # Volatility Heatmap
    volatility_data = np.array([
        [strike * vol for strike in strikes] for vol in volatilities
    ])
    fig_volatility = px.imshow(volatility_data, x=strikes, y=volatilities,
                               color_continuous_scale="Cividis",
                               title="Volatility Heatmap",
                               labels={"x": "Strike Price", "y": "Volatility (%)", "color": "Volatility"},
                               aspect="auto")

    fig_volatility.update_layout(
        title=dict(font=dict(size=20, color="white"), x=0.5),
        xaxis=dict(title="Strike Price", tickangle=45, showgrid=True, gridcolor="gray"),
        yaxis=dict(title="Volatility (%)", showgrid=True, gridcolor="gray"),
        coloraxis_colorbar=dict(title="Volatility"),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )
    fig_volatility.update_traces(hovertemplate="Strike Price: %{x}<br>Volatility: %{y}<br>Value: %{z:.2f}")
    st.plotly_chart(fig_volatility, key="volatility_heatmap")

# Higher Order Greeks Tab
with tab5:
    st.header("Higher Order Greeks")
    st.write("Below are the calculated higher-order Greeks with their definitions:")

    # Create data for the higher order Greeks
    data = []
    for greek, value in higher_order_greeks.items():
        explanation = {
            'Charm': "Rate of change of delta with respect to the passage of time.",
            'Color': "Rate of change of gamma with respect to the passage of time.",
            'Speed': "Rate of change of gamma with respect to changes in the underlying price.",
            'Vanna': "Sensitivity of delta to changes in volatility.",
            'Vomma': "Sensitivity of vega to changes in volatility.",
            'Zomma': "Sensitivity of gamma to changes in volatility."
        }.get(greek, "No explanation available.")
        data.append({'Greek': greek, 'Value': value, 'Explanation': explanation})

    # Create DataFrame for higher-order Greeks
    df_higher_order = pd.DataFrame(data)

    # Display Data Table
    st.dataframe(df_higher_order.style.format({'Value': "{:.4f}"}))

    # Visualization for Higher Order Greeks
    fig_greeks = px.bar(
        df_higher_order,
        x='Greek',
        y='Value',
        text='Value',
        title="Higher Order Greeks Visualization",
        color='Greek',
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    fig_greeks.update_layout(
        xaxis_title="Greek",
        yaxis_title="Value",
        title_x=0.5,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )
    st.plotly_chart(fig_greeks, key="higher_order_greeks_visualization")
