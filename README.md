# 📈 Stock Price Prediction Using LSTM

An end-to-end Deep Learning project that predicts next-day stock closing prices using an LSTM (Long Short-Term Memory) neural network with an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🔗 Live Application: 

> Note: Initial loading may take a few seconds due to Streamlit Community Cloud cold-start behavior.

---

## 📌 Project Overview

Stock price prediction is a challenging problem because financial markets are highly volatile and influenced by multiple factors such as market trends, company performance, and investor sentiment.

This project uses an LSTM-based deep learning model to learn temporal dependencies from historical stock data and forecast future stock prices.

The application allows users to:

* Enter stock symbols dynamically
* Visualize stock closing prices
* Predict next-day stock prices
* Compare actual vs predicted prices
* Evaluate model performance using metrics

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* LSTM Neural Networks
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* yFinance API

---

## 📂 Project Structure

```bash
├── app.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Features

- ✅ Real-time stock data collection using Yahoo Finance API
- ✅ Data normalization using MinMaxScaler
- ✅ Sequence generation for time-series forecasting
- ✅ LSTM-based Deep Learning model
- ✅ Next-day stock price prediction
- ✅ Model evaluation using MSE and MAE
- ✅ Interactive Streamlit dashboard
- ✅ Actual vs Predicted visualization

---

## 📊 Workflow

1. User enters stock symbol
2. Historical stock data is fetched using yFinance
3. Data preprocessing and normalization
4. Sequence generation using previous 60 days
5. LSTM model training
6. Stock price prediction
7. Visualization and evaluation

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(64))
model.add(Dense(1))
```

### Model Details

* Two stacked LSTM layers
* Adam optimizer
* Mean Squared Error loss function
* 60-day sliding window approach

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Approximate Accuracy Calculation

---

## ▶️ Installation & Setup

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
streamlit
yfinance
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

---

## 🔮 Future Enhancements

* Add technical indicators (RSI, MACD, EMA)
* Multi-step forecasting
* Transformer/GRU architectures
* Sentiment analysis integration
* Candlestick chart visualization
* Deploy using Docker and cloud platforms

---

## 👨‍💻 Author

**Shaik Mahaboob Vali**

* LinkedIn: https://linkedin.com/in/mahaboobvalishaik
* GitHub: https://github.com/ValiShaik03

---
