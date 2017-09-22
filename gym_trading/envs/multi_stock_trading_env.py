# coding: utf-8
import logging
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import prng

from data_loader import data_loader

log = logging.getLogger(__name__)
log.info('%s logger started.', __name__)

ACTION_SIZE = 3


class YahooMarketEnvSrc(object):
    '''
    Yahoo-based implementation of a TradingEnv's data source.

    Pulls data from Yahoo data source, preps for use by TradingEnv and then
    acts as data provider for each new episode.
    '''

    STOCK_NAMES = [
        '000333.SZ', '300356.SZ', '603789.SS', '603688.SS', '000034.SZ', '603766.SS'
    ]

    def __init__(self, days=252, names=STOCK_NAMES, scale=True):
        assert(isinstance(names, list))
        self.names = names
        self.capacity = len(self.names)
        self.days = days + 1
        stock_data = dict()
        for name in self.names:
            log.info('getting data for %s from yahoo...', name)
            df = data_loader(self.name)
            log.info('got data for %s from yahoo...', name)

            df = df[~np.isnan(df.Volume)][['Close', 'Volume']]
            # we calculate returns then kill nans
            df = df[['Close', 'Volume']]
            df.Volume.replace(0, 1, inplace=True)  # days shouldn't have zero volume..
            df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
            df.dropna(axis=0, inplace=True)
            R = df.Return
            if scale:
                mean_values = df.mean(axis=0)
                std_values = df.std(axis=0)
                df = (df - np.array(mean_values)) / np.array(std_values)
            df['Return'] = R  # we don't want our returns scaled
            stock_data[name] = df
        self.data = pd.Panel(stock_data)
        # get max and min values for all attributes
        self.attributes = ['Close', 'Volume', 'Return']
        self.attributes_min = [
            self.data.minor_xs(attr).min().min()/2.0 for attr in self.attributes
        ]
        self.attributes_max = [
            self.data.minor_xs(attr).max().max()*2.0 for attr in self.attributes
        ]
        self.reset()

    def stock_name(self, s_idx):
        return self.names[s_idx]

    def tradable_stocks(self):
        obs = self.data.major_xs(self.data.major_axis[self.idx])
        return [idx for idx, name in enumerate(self.names) if not np.isnan(obs[name]['Close'])]

    def reset(self):
        self.idx = prng.np_random.randint(low=0, high=len(self.data.major_axis)-self.days)
        self.step = 0

    def _step(self):
        """obs is a DataFrame for Attributes * Stocks per day"""
        obs = self.data.major_xs(self.data.major_axis[self.idx])
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done


class MultiStockTradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.current_stock_idx = None
        self.actions = np.zeros((self.steps, 2))  # pairs of (stock_idx, operation)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.strat_retrns = np.ones(self.steps)
        self.posns = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.current_stock_idx = None
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strat_retrns.fill(0)
        self.posns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_retrns.fill(0)

    def _step(self, action, retrn):
        """ Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a summary of the day's activity. """
        self.current_stock_idx = action[0] if action[1] != 1 else None

        if self.step == 0:
            bod_posn, bod_nav, mkt_nav = 0.0, 1.0, 1.0
        else:
            bod_posn, bod_nav, mkt_nav = \
                self.posns[self.step-1], self.navs[self.step-1], self.mkt_nav[self.step-1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action

        self.posns[self.step] = action[1] - 1
        self.trades[self.step] = self.posns[self.step] - bod_posn

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((bod_posn * retrn) - self.costs[self.step])
        self.strat_retrns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strat_retrns[self.step-1])
            self.mkt_nav[self.step] = mkt_nav * (1 + self.mkt_retrns[self.step-1])

        info = {
            'reward': reward,
            'nav': self.navs[self.step],
            'mkt_nav': self.mkt_nav[self.step],
            'costs': self.costs[self.step]
        }

        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = [
            'action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return', 'position', 'costs', 'trade'
        ]
        # pdb.set_trace()
        df = pd.DataFrame({
            'action': self.actions,  # today's action (from agent)
            'bod_nav': self.navs,  # BOD Net Asset Value (NAV)
            'mkt_nav': self.mkt_nav,
            'mkt_return': self.mkt_retrns,
            'sim_return': self.strat_retrns,
            'position': self.posns,     # EOD position
            'costs': self.costs,     # eod costs
            'trade': self.trades,  # eod trade
        }, columns=cols)
        return df


class MultiStockTradingEnv(gym.Env):
    """
        This gym implements a multi-stock trading environment for reinforcement learning.

        At the beginning of your episode, you are allocated 1 unit of
        cash. This is your starting Net Asset Value (NAV). If your NAV drops
        to 0, your episode is over and you lose. If your NAV hits 2.0, then
        you win.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.days = 252
        self.src = YahooMarketEnvSrc()
        self.sim = MultiStockTradingSim(steps=self.days)
        # stock (capacity) * actions (3)
        self.action_sapce = spaces.MultiDiscrete([
            [0, self.src.capacity - 1], [0, ACTION_SIZE-1]
        ])
        # {Close, Volume, Return} for each stock
        self.observation_space = spaces.Dict([
            spaces.Box(
                low=np.array(self.src.attributes_min),
                high=np.array(self.src.attributes_max)
            ) for i in range(self.src.capacity)
        ])
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        obs, done = self.src._step()
        stock_idx = action[0]
        # Close, Volume, Return
        ret = obs[self.src.stock_name(stock_idx)][2]
        reward, info = self.sim._step(action, ret)
        if info['nav'] >= 2.0 or info['nav'] <= 0.0:
            done = True
        return obs, reward, done, info

    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _render(self, mode='human', close=False):
        # TODO
        pass

    def action_pool(self):
        actions = []
        stock_idx = self.sim.current_stock_idx
        if self.sim.posns[self.sim.step] == 1.0:
            assert(stock_idx is not None)
            # we are now in long position, only can stick in short(0) or flat(1)
            actions.extend([[stock_idx, 0], [stock_idx, 1]])
        elif self.sim.posns[self.sim.step] == -1.0:
            assert(stock_idx is not None)
            # we are now in short position, only can stick in long(2) or flat(1)
            actions.extend([[stock_idx, 1], [stock_idx, 2]])
        else:
            # we don't have any open position
            assert(stock_idx is None)
            for s in self.src.tradable_stocks():
                actions.extend([[s, 0], [s, 1], [s, 2]])
        return actions
