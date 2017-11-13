# import gym
import unittest
import timeit

import fast_trading_env


class FastTradingEnvTestCase(unittest.TestCase):
    def setUp(self):
        self.env = fast_trading_env.FastTradingEnv(name='000333.SZ', days=200)

    def test_normal_run(self):
        self.assertTrue(self.env)
        episodes = 10
        for _ in range(episodes):
            self.env.reset()
            done = False
            count = 0
            while not done:
                action = self.env.action_space.sample()  # random
                observation, reward, done, info = self.env.step(action)
                self.assertTrue(reward >= 0.0)
                count += 1
        self.assertTrue(count)

    def test_run_strat(self):
        self.assertTrue(self.env)
        episodes = 10
        buyhold = lambda x, y: 1
        df = self.env.run_strats(buyhold, episodes)
        self.assertTrue(df.shape)

    def test_performance(self):
        setup = """import fast_trading_env; env = fast_trading_env.FastTradingEnv(name='000333.SZ', days=200)"""
        code = """env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
"""
        count = 100
        total = timeit.timeit(code, setup=setup, number=count)
        print 'avg: {t} seconds'.format(t=total/count)


if __name__ == '__main__':
    unittest.main()
