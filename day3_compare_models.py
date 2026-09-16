"""
Day 3: Compare Linear Regression vs Random Forest for predicting
tomorrow's closing price.

Uses the CSV saved by day1_get_data.py.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

TICKER = "AAPL"

# ---- 1. Load the data ----
data = pd.read_csv(f"{TICKER}_history.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
data.columns = ["Close", "High", "Low", "Open", "Volume"]
data = data[["Open", "High", "Low", "Close", "Volume"]]

# ---- 2. Features (same as Day 2, plus a couple more) ----
data["Prev_Close"] = data["Close"].shift(1)
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA10"] = data["Close"].rolling(window=10).mean()
data["Daily_Range"] = data["High"] - data["Low"]  # how much it moved that day
data["Target"] = data["Close"].shift(-1)

data = data.dropna()

features = ["Prev_Close", "MA5", "MA10", "Volume", "Daily_Range"]
X = data[features]
y = data["Target"]

# ---- 3. Chronological train/test split ----
split_index = int(len(data) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ---- 4. Train BOTH models ----
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ---- 5. Compare errors ----
lr_mae = mean_absolute_error(y_test, lr_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f"Linear Regression MAE: ${lr_mae:.2f}")
print(f"Random Forest MAE:     ${rf_mae:.2f}")
print()
if rf_mae < lr_mae:
    print("Random Forest did better on this stock.")
else:
    print("Linear Regression did better on this stock (this happens - simpler isn't always worse).")

# ---- 6. Which features mattered most to Random Forest? ----
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance (Random Forest):")
print(importances)

# ---- 7. Plot all three: actual, LR prediction, RF prediction ----
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price", linewidth=2)
plt.plot(y_test.index, lr_predictions, label="Linear Regression", linewidth=1.5, alpha=0.7)
plt.plot(y_test.index, rf_predictions, label="Random Forest", linewidth=1.5, alpha=0.7)
plt.title(f"{TICKER}: Model Comparison (Test Set)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{TICKER}_model_comparison.png")
plt.show()

print("\nDone. Chart saved as", f"{TICKER}_model_comparison.png")