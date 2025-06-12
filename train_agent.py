# train_agent.py
from stable_baselines3 import PPO
from orderbook_env import OrderBookEnv  # Make sure you have this file defined

env = OrderBookEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_orderbook")

# --- Evaluation ---
obs = env.reset()
total_reward = 0

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        obs = env.reset()

print("Total Reward (Proxy for PnL):", total_reward)
