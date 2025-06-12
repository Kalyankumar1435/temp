# 📊 Machine Learning for Finance: Tick Data Feature Engineering and Price Direction Prediction

This project focuses on building a **feature matrix** from tick-by-tick ETH/USDT data using order book imbalance, trade flow imbalance, volatility clusters, and microstructure features. The goal is to predict **short-term (1-minute horizon) price movement direction** using machine learning and evaluate with F1-score and AUC-ROC.

---

## 🗂 Project Structure

kkalyan/
├── ethusdt_ticks.csv # Tick-level raw data with price, volume, order book info
├── ml_predict.py # Python script implementing feature engineering & prediction
|---generate_eth_ticks.py
├── README.md # This documentation file

---

## 📈 Features Used

- **Order Book Imbalance (OBI):** Measures bid vs ask size imbalance  
- **Trade Flow Imbalance (TFI):** Captures buy vs sell volume imbalance over rolling windows  
- **Volatility Clusters:** Rolling standard deviation of log returns to capture changing volatility  
- **Microstructure Features:** Bid-ask spread, mid-price, and other order book metrics  

---

## 🎯 Task

- Build feature matrix from tick data aggregated to 1-minute intervals  
- Create target label as 1-minute ahead price movement direction (up or down)  
- Train a classification model (e.g., Random Forest)  
- Evaluate model with **F1-score** and **AUC-ROC**

---

## 🔧 Requirements

Install the necessary Python packages:

```bash
pip install pandas numpy scikit-learn

How to Run
Place your tick data CSV ethusdt_ticks.csv inside the kkalyan folder.

Run the Python script:
python ml_predict.py

The script will output:

F1 Score

AUC-ROC score