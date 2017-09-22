# import gym
import pandas as pd
from gym.spaces import prng
import multi_stock_trading_env

pd.set_option('display.width', 500)

env = multi_stock_trading_env.MultiStockTradingEnv()
# env = gym.make('trading-v0')
env.time_cost_bps = 0

episodes = 1

obs = []

for _ in range(episodes):
    observation = env.reset()
    done = False
    count = 0
    while not done:
        action_ops = env.action_options
        # print 'action_options: ', action_ops
        action = action_ops[prng.np_random.choice(range(len(action_ops)))]  # random
        # print 'action: ', action
        observation, reward, done, info = env.step(action)
        # print 'obs: ', observation
        # print env.sim.to_df().head()
        obs = obs + [observation]
        # print observation,reward,done,info
        count += 1
        if done:
            print reward, count

print 'check sim to_df'
df = env.sim.to_df()
print df

print 'run buy and hold strategy'
buyhold = lambda x, y: [0, 2]
df = env.run_strat(buyhold)
print df

print 'run buy and hold strategy'
df = env.run_strats(buyhold, 10)
print df


