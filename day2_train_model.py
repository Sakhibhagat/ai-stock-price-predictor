"""
Day 2: Build a simple model to predict tomorrow's closing price.

Uses the CSV saved by day1_get_data.py.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

TICKER = "AAPL"

# ---- 1. Load the data we saved in Day 1 ----
data = pd.read_csv(f"{TICKER}_history.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
data.columns = ["Close", "High", "Low", "Open", "Volume"]  # rename to be safe
data = data[["Open", "High", "Low", "Close", "Volume"]]

# ---- 2. Build simple features ----
# "Yesterday's close" -> today's row
data["Prev_Close"] = data["Close"].shift(1)
# 5-day moving average of closing price
data["MA5"] = data["Close"].rolling(window=5).mean()
# What we want to predict: TOMORROW's close
data["Target"] = data["Close"].shift(-1)

# Drop rows with missing values (first few rows won't have MA5/Prev_Close,
# last row won't have a Target since there's no "tomorrow" yet)
data = data.dropna()

# ---- 3. Split into features (X) and target (y) ----
features = ["Prev_Close", "MA5", "Volume"]
X = data[features]
y = data["Target"]

# ---- 4. Train/test split — CHRONOLOGICAL, not random ----
# We train on the older 80% of days, test on the newest 20%.
# This mimics reality: predicting the future from the past.
split_index = int(len(data) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ---- 5. Train the model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- 6. Predict on the test set ----
predictions = model.predict(X_test)

# ---- 7. See how well it did ----
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: ${mae:.2f}")
print("(On average, predictions are off by this many dollars)")

# ---- 8. Plot actual vs predicted ----
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price", linewidth=2)
plt.plot(y_test.index, predictions, label="Predicted Price", linewidth=2, alpha=0.7)
plt.title(f"{TICKER}: Actual vs Predicted Closing Price (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TICKER}_predictions.png")
plt.show()

print("\nDone. Chart saved as", f"{TICKER}_predictions.png")