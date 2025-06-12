import gym
from gym import spaces
import numpy as np

class OrderBookEnv(gym.Env):
    def __init__(self):
        super(OrderBookEnv, self).__init__()
        
        # Example order book state: [bid_price, bid_size, ask_price, ask_size]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        # Actions: 0 = do nothing, 1 = join spread, 2 = cancel order, 3 = cross spread
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self):
        self.bid_price = 100
        self.ask_price = 101
        self.position = 0
        self.order_status = 'none'
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.bid_price, 10, self.ask_price, 10], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if action == 1:  # Join spread
            self.order_status = 'limit'
        elif action == 2:  # Cancel
            self.order_status = 'none'
        elif action == 3:  # Cross spread
            self.position += 1
            reward = -self.ask_price

        # Simulate price movement
        self.bid_price += np.random.choice([-1, 0, 1])
        self.ask_price = self.bid_price + 1

        obs = self._get_obs()
        return obs, reward, done, info
