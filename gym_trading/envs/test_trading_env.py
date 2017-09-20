# import gym
import pandas as pd
import trading_env

pd.set_option('display.width', 500)

env = trading_env.TradingEnv()
# env = gym.make('trading-v0')
env.time_cost_bps = 0

episodes = 100

obs = []

for _ in range(episodes):
    observation = env.reset()
    done = False
    count = 0
    while not done:
        action = env.action_space.sample()  # random
        observation, reward, done, info = env.step(action)
        obs = obs + [observation]
        # print observation,reward,done,info
        count += 1
        if done:
            print reward
            print count

df = env.sim.to_df()

df.head()
df.tail()

buyhold = lambda x, y: 2
df = env.run_strat(buyhold)

df10 = env.run_strats(buyhold, episodes)
