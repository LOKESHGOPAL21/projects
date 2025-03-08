# Time Series Forecasting Using Yahoo Finance Data

## 📌 Project Overview

This project performs **time series forecasting** using real-time **Yahoo Finance** stock market data. The models used include:

- **Naïve Approach**
- **Moving Average**
- **Simple Exponential Smoothing (SES)**
- **Holt’s Linear Trend Model**
- **ARIMA (AutoRegressive Integrated Moving Average)**

The goal is to analyze stock price trends, evaluate stationarity using **Dickey-Fuller tests**, and compare the forecasting accuracy of different models using **Root Mean Square Error (RMSE).**

---

## 📊 Dataset Information

- **Source:** Yahoo Finance API
- **Ticker Used:** `AAPL` (Apple Inc.) (Can be changed to any stock)
- **Time Period:** `2020-01-01` to `2024-01-01`
- **Feature Used:** `Close` price of the stock

---

## 🛠 Models Implemented

1. **Naïve Approach:** Assumes the next day's price is the same as the last observed price.
2. **Moving Average:** Smoothens data by averaging the last `n` days (default: 10 days).
3. **Simple Exponential Smoothing (SES):** Assigns exponentially decreasing weights to past observations.
4. **Holt’s Linear Trend Model:** Accounts for level and trend in time series data.
5. **ARIMA Model:** Uses autoregressive, differencing, and moving average components to make accurate forecasts.

Each model's predictions are visualized in separate plots.

---

## 📈 Results: Model Performance Comparison

The models were evaluated using **Root Mean Square Error (RMSE)**, where lower values indicate better forecasting accuracy.

| Model                        | RMSE     |
| ---------------------------- | -------- |
| Naïve Forecasting            | 2.601887 |
| Moving Average               | 4.195622 |
| Simple Exponential Smoothing | 4.143332 |
| Holt’s Linear Trend          | 2.737787 |
| ARIMA Model                  | 2.599180 |

---

## 📝 Conclusion

From the results:

- The **ARIMA model** achieved the lowest RMSE (**2.599**), making it the most effective model for forecasting stock prices.
- The **Naïve Approach** performed surprisingly well, with an RMSE of **2.601**, showing that past price trends strongly influence future values.
- **Holt’s Linear Trend Model** was also a competitive alternative with an RMSE of **2.737**.
- **Moving Average and Simple Exponential Smoothing** had the highest errors, suggesting they are less suitable for stock price forecasting.

### **Key Takeaways**

✅ ARIMA is the most accurate model for this dataset.
✅ The Naïve approach is a strong baseline.
✅ Moving averages are not always reliable for volatile stock price data.
✅ Further improvements can be made by using **LSTM or Transformer-based** deep learning models.

---

## 🚀 How to Run This Project

### **1. Install Required Libraries**

Ensure you have the necessary dependencies:

```bash
pip install yfinance numpy pandas matplotlib seaborn statsmodels scikit-learn
```

### **2. Run the Python Script**

Execute the script to fetch stock data and apply forecasting models:

```bash
python time_series_forecasting.py
```

### **3. Change Stock Symbol (Optional)**

Modify the `ticker` variable in the script to analyze different stocks:

```python
ticker = "TSLA"  # Change to Tesla stock data
```

🔗 **Contributions and feedback are welcome!** 🚀
