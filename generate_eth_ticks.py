import pandas as pd
import numpy as np

# Generate 60 seconds of fake tick data
np.random.seed(42)
n_ticks = 10000


timestamps = pd.date_range(start='2024-06-01 10:00:00', periods=n_ticks, freq='S')
prices = 3760 + np.cumsum(np.random.randn(n_ticks) * 0.1)  # price with small noise
volumes = np.random.uniform(0.1, 1.0, size=n_ticks).round(3)
sides = np.random.choice(['buy', 'sell'], size=n_ticks)

bid_prices = prices - np.random.uniform(0.1, 0.3, size=n_ticks)
ask_prices = prices + np.random.uniform(0.1, 0.3, size=n_ticks)
bid_sizes = np.random.uniform(5, 15, size=n_ticks).round(2)
ask_sizes = np.random.uniform(5, 15, size=n_ticks).round(2)

df = pd.DataFrame({
    'timestamp': timestamps,
    'price': prices.round(2),
    'volume': volumes,
    'side': sides,
    'bid_price': bid_prices.round(2),
    'bid_size': bid_sizes,
    'ask_price': ask_prices.round(2),
    'ask_size': ask_sizes
})

df.to_csv('ethusdt_ticks.csv', index=False)
print("ethusdt_ticks.csv created!")
