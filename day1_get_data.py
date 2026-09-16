"""
Day 1: Pull historical stock data and plot it.

This is step 1 of the AI Stock Price Predictor project.
Goal today: get data, look at it, make sure it makes sense.
"""

import yfinance as yf
import matplotlib.pyplot as plt

# ---- 1. Pick a stock ----
# Try a company you know / use. Examples: "AAPL" (Apple), "TSLA" (Tesla),
# "MSFT" (Microsoft), "RELIANCE.NS" (Reliance, India), "TCS.NS" (TCS, India)
TICKER = "AAPL"

# ---- 2. Pull historical data ----
# period options: "1y", "2y", "5y", "max"
# interval options: "1d" (daily), "1wk" (weekly)
data = yf.download(TICKER, period="5y", interval="1d")

# ---- 3. Look at it ----
print(f"Downloaded {len(data)} rows for {TICKER}")
print(data.head())
print(data.tail())

# ---- 4. Save it so we don't have to re-download every time ----
data.to_csv(f"{TICKER}_history.csv")
print(f"\nSaved to {TICKER}_history.csv")

# ---- 5. Plot the closing price over time ----
plt.figure(figsize=(12, 5))
plt.plot(data.index, data["Close"], label=f"{TICKER} Close Price")
plt.title(f"{TICKER} Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TICKER}_price_history.png")
plt.show()

print("\nDone. You should see a chart pop up and a PNG saved.")