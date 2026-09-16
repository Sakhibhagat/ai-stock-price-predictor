# AI Stock Price Predictor

An interactive web app that predicts a stock's next-day closing price using
historical price data and simple machine learning models — with a live
dashboard where you can search and try any publicly traded company.

**Live demo:** _add your Streamlit Cloud link here after deploying_

## What it does

- Pulls historical daily price data for any stock (via `yfinance`)
- Engineers simple features from raw prices: previous close, 5-day and
  10-day moving averages, daily price range, trading volume
- Trains a model on the older portion of the data and evaluates it on the
  most recent, unseen portion (a proper chronological train/test split —
  never shuffled, since that would leak future information into training)
- Predicts the next trading day's closing price
- Visualizes actual vs. predicted prices, and reports Mean Absolute Error (MAE)
- Lets users search for any public company by name and adds new lookups to a
  shared quick-picks list for future users

## What I found

I compared two models: **Linear Regression** and **Random Forest**.

Linear Regression performed noticeably better (MAE ≈ $4.79) than Random
Forest (MAE ≈ $32.27) on trending stock data. This isn't because Random
Forest is a worse model in general — it's because tree-based models can't
extrapolate beyond the price range they saw during training. Since the test
period contained prices higher than anything in the training data, the
Random Forest's predictions got "stuck" near the highest price it had seen,
while Linear Regression could follow the trend upward.

Feature importance from the Random Forest also showed that **yesterday's
closing price alone accounts for ~75% of the predictive power**, with the
5-day moving average adding most of the rest. This lines up with the
"random walk" idea in finance — that a stock's next price is best explained
by its current price, since if strong predictable patterns existed reliably
beyond that, they'd typically get traded away.

## Honest limitations

- This predicts short-term price movement using only historical price/volume
  patterns. It does not account for news, earnings, macroeconomic events,
  or company fundamentals — all of which move real stock prices.
- Predicting exact future prices is a genuinely hard, largely unsolved
  problem (this is why it's a good learning project, not a trading tool).
- This project is for educational/portfolio purposes only and is **not**
  financial advice.

## Tech stack

Python, yfinance, pandas, scikit-learn, matplotlib, Streamlit

## Project structure

```
stock-predictor/
├── app.py                  # Interactive Streamlit dashboard (main entry point)
├── day1_get_data.py        # Step 1: pull and visualize historical data
├── day2_train_model.py     # Step 2: baseline Linear Regression model
├── day3_compare_models.py  # Step 3: Linear Regression vs Random Forest comparison
├── requirements.txt
└── README.md
```

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What I'd add next

- A deep learning model (LSTM) to compare against the two above
- Directional accuracy (did it correctly predict up/down, not just the
  price number) as an additional evaluation metric
- Testing across multiple stocks to see if these findings generalize