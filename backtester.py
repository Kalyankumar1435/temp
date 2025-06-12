import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Load Data
# --------------------
data = pd.read_csv("ohlcv_minute_data.csv", parse_dates=['datetime'], index_col='datetime')
data = data.sort_index().dropna()

# --------------------
# Signal Generation
# --------------------
window = 30
data['mean'] = data['close'].rolling(window).mean()
data['std'] = data['close'].rolling(window).std()
data['zscore'] = (data['close'] - data['mean']) / data['std']

# --------------------
# Volatility-Scaled Position
# --------------------
vol_window = 60
data['vol'] = data['close'].pct_change().rolling(vol_window).std()
vol_target = 0.01
data['raw_signal'] = np.where(data['zscore'] < -1, 1.0, 0.0)
data['position'] = (vol_target / data['vol']) * data['raw_signal']
data['position'] = data['position'].clip(upper=1.0)

# --------------------
# Simulate Execution: Latency, Slippage, Costs
# --------------------
slippage_bps = 2
transaction_cost = 0.0005
latency = 1

data['executed_position'] = data['position'].shift(latency)
data['returns'] = data['close'].pct_change()
data['strategy_returns'] = data['executed_position'].shift(1) * data['returns']
trades = data['executed_position'].diff().abs()
data['strategy_returns'] -= (slippage_bps / 10000 + transaction_cost) * trades

# --------------------
# Metrics Function
# --------------------
def calculate_metrics(df):
    cumulative = (1 + df['strategy_returns']).cumprod()
    cagr = cumulative.iloc[-1] ** (252*24*60 / len(df)) - 1
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252*24*60)
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    turnover = trades.sum() / len(df)
    hit_rate = (df['strategy_returns'] > 0).mean()
    return {
        'CAGR': round(cagr, 4),
        'Sharpe': round(sharpe, 2),
        'Max Drawdown': round(max_dd, 4),
        'Turnover': round(turnover, 4),
        'Hit Rate': round(hit_rate, 2)
    }

# --------------------
# Display Performance
# --------------------
metrics = calculate_metrics(data)
print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# --------------------
# Plot Equity Curve
# --------------------
data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(data.index, data['cumulative_returns'], label='Strategy')
plt.title("Mean-Reversion Strategy Performance")
plt.xlabel("Time")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid()
plt.show()
