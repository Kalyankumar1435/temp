import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

# Step 1: Load tick data
df = pd.read_csv("ethusdt_ticks.csv", parse_dates=['timestamp'])
df = df.sort_values('timestamp')

# Step 2: Preprocessing
df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
df['side'] = df['side'].map({'buy': 1, 'sell': -1})

# Step 3: Resample for labels
df.set_index('timestamp', inplace=True)
df_minute = df['mid_price'].resample('1Min').last()
future_return = df_minute.pct_change().shift(-1)
label = (future_return > 0).astype(int)

# Step 4: Feature Engineering
df['obi'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
df['signed_volume'] = df['volume'] * df['side']
df['tfi'] = df['signed_volume'].rolling(10).sum()
df['returns'] = df['mid_price'].pct_change()
df['volatility'] = df['returns'].rolling(10).std()
df['spread'] = df['ask_price'] - df['bid_price']
df['log_price'] = np.log(df['mid_price'])

# Step 5: Resample features
features = df[['obi', 'tfi', 'volatility', 'spread', 'log_price']].resample('1Min').mean()
features['label'] = label
features = features.dropna()

# Step 6: Split and Train
X = features.drop('label', axis=1)
y = features['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("✅ Evaluation Results:")
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("AUC-ROC:", round(roc_auc_score(y_test, y_proba), 4))
