import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Title
st.title("📈 Stock Price Prediction Using LSTM")

# User input
stock = st.text_input("Enter Stock Symbol", "AAPL")

# Download data
data = yf.download(stock, start="2018-01-01", end="2024-01-01")

# Check if data is empty
if data.empty:
    st.error("❌ Failed to fetch data. Try AAPL, TSLA, MSFT")
    st.stop()

# Show data
st.subheader("Stock Data (Last 5 rows)")
st.write(data.tail())

# Ploting closing price
st.subheader("Closing Price Chart")
fig = plt.figure()
plt.plot(data['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig)

#### Preprocessing

# Using only Close price
data_close = data[['Close']]

# Normalizing data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_close)

# Creating sequences
X, y = [], []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Reshape
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Model Buidling

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(64))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=15, batch_size=32, verbose=0)

# -------------------------
# PREDICTION
# -------------------------

# Predict on training data (for evaluation)
y_pred = model.predict(X)

# Convert back to original scale
y_pred = scaler.inverse_transform(y_pred)
y_true = scaler.inverse_transform(y.reshape(-1, 1))

# -------------------------
# METRICS
# -------------------------

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

st.subheader("Model Evaluation")
st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")

# -------------------------
# NEXT DAY PREDICTION
# -------------------------

last_60_days = scaled_data[-60:]
X_test = np.array([last_60_days])
X_test = np.reshape(X_test, (1, 60, 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

st.subheader("📊 Predicted Next Day Price")
st.success(f"{predicted_price[0][0]:.2f}")

mean_price = np.mean(y_true)
accuracy = (1 - (mae / mean_price)) * 100

st.write(f"Accuracy: {accuracy:.2f}%") 

# -------------------------
# PLOT ACTUAL VS PREDICTED
# -------------------------

st.subheader("Actual vs Predicted")

fig2 = plt.figure()
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
st.pyplot(fig2)