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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

SAVED_STOCKS_FILE = "saved_stocks.json"

# ---- Page setup ----
st.set_page_config(page_title="AI Stock Price Predictor", layout="wide")
st.title("📈 AI Stock Price Predictor")
st.write("Predicts tomorrow's closing price using yesterday's price and recent trends.")

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
    st.info("👈 Pick a quick pick or search for a company in the sidebar to get started.")
else:
    with st.spinner(f"Downloading data and training model for {ticker}..."):
        result = run_pipeline(ticker, period)

    if result is None:
        st.error(f"Couldn't find data for '{ticker}'. Check the ticker symbol and try again.")
    else:
        sym = currency_symbol(ticker)

        col1, col2, col3 = st.columns(3)
        col1.metric("Last Close Price", f"{sym}{result['last_close']:.2f}")
        if result["next_day_pred"] is not None:
            delta = result["next_day_pred"] - result["last_close"]
            col2.metric("Predicted Next Close", f"{sym}{result['next_day_pred']:.2f}", f"{delta:+.2f}")
        col3.metric("Model Error (MAE)", f"{sym}{result['mae']:.2f}")

        st.subheader(f"{ticker}: Actual vs Predicted (Test Period)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(result["y_test"].index, result["y_test"].values, label="Actual Price", linewidth=2)
        ax.plot(result["y_test"].index, result["predictions"], label="Predicted Price", linewidth=2, alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Price ({sym})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        with st.expander("See raw data"):
            st.dataframe(result["raw"].tail(20))

        st.caption(
            "⚠️ This is a simplified model for learning/portfolio purposes. "
            "It is not financial advice and should not be used for real trading decisions. "
            "Stock prices are influenced by countless factors this model doesn't capture."
        )