"""
Streamlit dashboard for the AI Stock Price Predictor.

Run with: streamlit run app.py

Lets you type any stock ticker, pulls fresh data, trains a model live,
and shows actual vs predicted price in the browser.
"""

import json
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

SAVED_STOCKS_FILE = "saved_stocks.json"

# ---- Page setup ----
st.set_page_config(page_title="AI Stock Price Predictor", page_icon="📈", layout="wide")

if "hero_ticker" not in st.session_state:
    st.session_state.hero_ticker = None

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    .wordmark {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.4rem;
        text-align: center;
        letter-spacing: -0.01em;
    }
    .wordmark span { color: #E3A857; }
    .tagline {
        text-align: center;
        color: #838EA3;
        font-size: 1.02rem;
        margin-top: 0.3rem;
        margin-bottom: 2.2rem;
    }

    div[data-testid="stTextInput"] input {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Ticker pill buttons */
    div[data-testid="column"] .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        border-radius: 999px;
        border: 1px solid #262C39;
        background-color: #12161F;
        color: #C7CEDB;
        padding: 6px 16px;
    }
    div[data-testid="column"] .stButton > button:hover {
        border-color: #E3A857;
        color: #E3A857;
    }

    [data-testid="stMetric"] {
        background-color: #12161F;
        border: 1px solid #262C39;
        border-top: 2px solid #E3A857;
        border-radius: 8px;
        padding: 16px 18px 10px 18px;
    }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wordmark">stock<span>Predictor</span></div>
    <div class="tagline">Predicts tomorrow's closing price from historical patterns — trained live, in your browser.</div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar controls ----
st.sidebar.header("Settings")

POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google / Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
}


def load_saved_stocks():
    """Load community-added stocks from disk. Returns {} if file doesn't exist yet."""
    if os.path.exists(SAVED_STOCKS_FILE):
        try:
            with open(SAVED_STOCKS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def add_saved_stock(label, symbol):
    """Add a new stock to the shared list and save it to disk."""
    saved = load_saved_stocks()
    if label not in saved and label not in POPULAR_STOCKS:
        saved[label] = symbol
        with open(SAVED_STOCKS_FILE, "w") as f:
            json.dump(saved, f, indent=2)


saved_stocks = load_saved_stocks()
all_quick_picks = {**POPULAR_STOCKS, **saved_stocks}

st.sidebar.caption("Quick picks, or search any company below:")

quick_pick = st.sidebar.selectbox("Quick picks", ["-- none --"] + sorted(all_quick_picks.keys()))

search_query = st.sidebar.text_input(
    "Search any company",
    placeholder="e.g. Britannia, Netflix, Infosys...",
)

ticker = None
if search_query:
    try:
        results = yf.Search(search_query, max_results=8).quotes
    except Exception:
        results = []

    if results:
        options = {
            f"{r.get('shortname', r.get('longname', '?'))} ({r['symbol']})": r["symbol"]
            for r in results
            if "symbol" in r
        }
        picked = st.sidebar.selectbox("Matches — pick one", list(options.keys()))
        ticker = options[picked]
        # Add it to the shared quick-picks list for next time / other users
        add_saved_stock(picked, ticker)
    else:
        st.sidebar.warning("No matches found. Try a different name, or it may not be publicly traded.")
elif quick_pick != "-- none --":
    ticker = all_quick_picks[quick_pick]

if not ticker:
    ticker = st.session_state.hero_ticker

period = st.sidebar.selectbox("History to use", ["1y", "2y", "5y", "max"], index=2)
run_button = st.sidebar.button("Run Prediction")

st.sidebar.markdown("---")
st.sidebar.caption("Note: only publicly traded companies have stock tickers — private companies won't show up in search.")


def run_pipeline(ticker, period):
    raw = yf.download(ticker, period=period, interval="1d")
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    data["Prev_Close"] = data["Close"].shift(1)
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["Daily_Range"] = data["High"] - data["Low"]
    data["Target"] = data["Close"].shift(-1)

    features = ["Prev_Close", "MA5", "MA10", "Volume", "Daily_Range"]
    clean = data.dropna()

    X = clean[features]
    y = clean["Target"]

    split_index = int(len(clean) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # ---- Directional accuracy ----
    # Did the model correctly predict whether the price would go UP or DOWN
    # compared to today's close? This often matters more in practice than
    # the exact dollar error, since a trader mainly cares about direction.
    today_close_for_test = clean["Close"][split_index:]
    actual_direction = (y_test.values > today_close_for_test.values)
    predicted_direction = (predictions > today_close_for_test.values)
    directional_accuracy = (actual_direction == predicted_direction).mean() * 100

    # Predict the actual next trading day (using the most recent row)
    latest_features = data[features].iloc[[-1]].dropna()
    next_day_pred = None
    if not latest_features.empty:
        next_day_pred = model.predict(latest_features)[0]

    return {
        "raw": data,
        "y_test": y_test,
        "predictions": predictions,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "next_day_pred": next_day_pred,
        "last_close": data["Close"].iloc[-1],
    }


# ---- Currency symbol based on exchange suffix ----
def currency_symbol(ticker):
    ticker = ticker.upper()
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹"
    if ticker.endswith(".L"):
        return "£"
    if ticker.endswith(".T"):
        return "¥"
    if ticker.endswith(".DE") or ticker.endswith(".PA"):
        return "€"
    return "$"  # default: US exchanges


if not ticker:
    st.markdown(
        "<div style='text-align:center; color:#838EA3; margin-bottom:10px; font-size:0.92rem;'>Or jump straight in —</div>",
        unsafe_allow_html=True,
    )

    demo_tickers = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "NFLX"]
    cols = st.columns(len(demo_tickers))
    for col, sym in zip(cols, demo_tickers):
        with col:
            if st.button(sym, key=f"hero_{sym}", use_container_width=True):
                st.session_state.hero_ticker = sym
                st.rerun()

    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            "**Search anything**<br><span style='color:#838EA3; font-size:0.9rem;'>"
            "Live lookup across public companies worldwide — no fixed list to maintain.</span>",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            "**Trained live**<br><span style='color:#838EA3; font-size:0.9rem;'>"
            "A real Linear Regression model fits fresh on every run, not a cached demo.</span>",
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            "**Honest evaluation**<br><span style='color:#838EA3; font-size:0.9rem;'>"
            "Error (MAE) and directional accuracy, not just a chart that looks convincing.</span>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='margin-top:32px; padding-top:16px; border-top:1px solid #262C39; "
        "color:#838EA3; font-size:0.85rem;'>"
        "Built as a learning project. Predicts short-term price movement from historical "
        "patterns only — not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    with st.spinner(f"Downloading data and training model for {ticker}..."):
        result = run_pipeline(ticker, period)

    if result is None:
        st.error(f"Couldn't find data for '{ticker}'. Check the ticker symbol and try again.")
    else:
        sym = currency_symbol(ticker)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Close Price", f"{sym}{result['last_close']:.2f}")
        if result["next_day_pred"] is not None:
            delta = result["next_day_pred"] - result["last_close"]
            col2.metric("Predicted Next Close", f"{sym}{result['next_day_pred']:.2f}", f"{delta:+.2f}")
        col3.metric("Model Error (MAE)", f"{sym}{result['mae']:.2f}")
        col4.metric("Directional Accuracy", f"{result['directional_accuracy']:.1f}%")
        st.caption(
            "Directional accuracy = % of test days where the model correctly predicted "
            "whether the price would go up or down (not just how close the number was). "
            "50% is what random guessing would get."
        )

        st.subheader(f"{ticker}: Actual vs Predicted (Test Period)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["y_test"].index, y=result["y_test"].values,
            name="Actual Price", line=dict(color="#4ADE9C", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=result["y_test"].index, y=result["predictions"],
            name="Predicted Price", line=dict(color="#E3A857", width=2.5, dash="dot"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title=f"Price ({sym})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See raw data"):
            st.dataframe(result["raw"].tail(20))

        st.caption(
            "⚠️ This is a simplified model for learning/portfolio purposes. "
            "It is not financial advice and should not be used for real trading decisions. "
            "Stock prices are influenced by countless factors this model doesn't capture."
        )