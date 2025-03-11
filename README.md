# ARIMA Time Series Forecasting

This repository contains an example of **ARIMA (AutoRegressive Integrated Moving Average)** model for time series forecasting using Python. The example uses the **AirPassengers dataset** to demonstrate trend detection, stationarity testing, and forecasting.

## 📌 Features
- **Load & visualize time series data**
- **Check stationarity using ADF Test**
- **Use differencing to make data stationary**
- **Identify ARIMA parameters (p, d, q) using ACF & PACF plots**
- **Auto-select best (p, d, q) using `auto_arima()`**
- **Train an ARIMA model and forecast future values**

## 📜 Dataset
The dataset used is the **AirPassengers dataset**, which contains monthly airline passenger data from 1949 to 1960.

## 📊 Steps in ARIMA Forecasting
### **1. Load & Visualize the Data**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                 parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']
df.plot(figsize=(12,6))
plt.title("Airline Passengers Data")
plt.show()
```

### **2. Check for Stationarity (ADF Test)**
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Passengers'])
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
```
If **p-value > 0.05**, the data is non-stationary and requires differencing.

### **3. Differencing for Stationarity**
```python
df['Passengers_diff'] = df['Passengers'].diff().dropna()
```

### **4. Identify ARIMA Parameters using ACF & PACF**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['Passengers_diff'].dropna())
plot_pacf(df['Passengers_diff'].dropna())
plt.show()
```

### **5. Auto-Select Best ARIMA Model**
```python
from pmdarima import auto_arima
auto_model = auto_arima(df['Passengers'], seasonal=False, trace=True)
print(auto_model.summary())
```

### **6. Train ARIMA Model & Forecast**
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['Passengers'], order=(2,1,2))  # Replace with optimal (p,d,q)
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
```

## 📌 Output Example
- **Original time series plot**
- **ACF & PACF plots for p and q selection**
- **ARIMA model summary**
- **Forecast plot for next 12 months**

## 📖 References
- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [pmdarima auto_arima Documentation](https://alkaline-ml.com/pmdarima/)

