# import gym
import pandas as pd
import fast_trading_env

pd.set_option('display.width', 500)

env = fast_trading_env.FastTradingEnv(name='000333.SZ', days=200)
# env = gym.make('trading-v0')

episodes = 100

obs = []

for _ in range(episodes):
    observation = env.reset()
    done = False
    count = 0
    acc_reward = 0.0
    while not done:
        action = env.action_space.sample()  # random
        observation, reward, done, info = env.step(action)
        print '--------', count
        print 'obs', observation
        print 'action:{a} reward:{r} info:{i}'.format(a=action, r=reward, i=info)
        count += 1
        acc_reward += reward
        if done:
            print acc_reward, count
            raw_input()

df = env.sim.inspect()

df.head()
df.tail()

buyhold = lambda x, y: 2
df = env.run_strat(buyhold)

df10 = env.run_strats(buyhold, episodes)
