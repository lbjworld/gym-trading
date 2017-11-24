# coding: utf-8
import gym
from gym import spaces
from gym.utils import seeding

import logging
import numpy as np
import pandas as pd

from data_loader import data_loader
from mixin import RunMixin

logger = logging.getLogger(__name__)


class YahooEnvSrc(object):
    '''
    Yahoo-based implementation of a TradingEnv's data source.

    Pulls data from Yahoo data source, preps for use by TradingEnv and then
    acts as data provider for each new episode.
    '''

    VOLUME_SCALE_FACTOR = 1000000.0

    def __init__(self, name, days, use_adjust_close=True):
        self.name = name
        self.days = days

        data_df = data_loader(self.name)
        if data_df.empty:
            raise Exception('load data error')
        logger.debug('[{name}] data loaded'.format(name=self.name))

        close_column = 'Adj Close' if use_adjust_close else 'Close'
        data_df = data_df[['Open', 'High', 'Low', close_column, 'Volume']]
        data_df.columns = ['open', 'high', 'low', 'close', 'volume']
        data_df = data_df[~np.isnan(data_df.volume)]  # 跳过所有停牌日
        self.pct_change = data_df.pct_change().fillna(0.0) + 1.0  # 计算变化量
        data_df.volume = data_df.volume / YahooEnvSrc.VOLUME_SCALE_FACTOR
        self.data = data_df
        self.reset()

    def reset(self):
        # we want contiguous data
        high = len(self.data.index)-self.days
        if high <= 1:
            raise Exception('data too short')
        self.idx = np.random.randint(low=1, high=high)
        self.step = 0

    def _step(self):
        current_idx = self.idx + self.step
        if current_idx:
            visible_data = self.data[self.idx:current_idx].as_matrix()
            if self.days <= self.step:
                # last one
                history = visible_data
            else:
                empty_data = np.zeros((self.days-self.step, len(self.data.columns)))
                history = np.concatenate((visible_data, empty_data), axis=0)
        else:
            # first one
            history = np.zeros((self.days-self.step, len(self.data.columns)))
        assert(history.shape == (self.days, len(self.data.columns)))
        obs = {
            'pct_change': self.pct_change.iloc[current_idx].as_matrix(),
            'history': history,
        }
        self.step += 1
        done = self.step > self.days
        return obs, done

    def to_df(self):
        return self.data[self.idx:self.idx+self.step+1]

    def snapshot(self):
        return {
            'step': self.step,
            'idx': self.idx,
            'name': self.name,
            'days': self.days,
        }

    def recover(self, snapshot):
        assert('step' in snapshot and 'idx' in snapshot and 'name' in snapshot)
        assert(snapshot['name'] == self.name)
        self.step = snapshot['step']
        self.idx = snapshot['idx']


class TradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3):
        # invariant for object life
        self.trading_cost_pct_change = 1.0 - trading_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)

    def _step(self, action, nav_pct_change, done=False):
        reward = 0.0
        # record action
        self.actions[self.step] = action
        last_nav = 1.0 if self.step == 0 else self.navs[self.step-1]
        # record nav (只在LONG position情况下累积nav)
        self.navs[self.step] = last_nav * (nav_pct_change if action == 1 else 1.0)

        if not self.step == 0:
            # trading fee for changing trade position
            if abs(self.actions[self.step-1] - action) > 0:
                reward = self.navs[self.step] * self.trading_cost_pct_change
        if done:
            # episode finished, force sold
            reward = self.navs[self.step] * self.trading_cost_pct_change
        info = {
            'step': self.step,
            'reward': reward,
            'nav': self.navs[self.step],
        }

        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = [
            'action', 'nav',
        ]
        df = pd.DataFrame({
            'action': self.actions,  # today's action (from agent)
            'nav': self.navs,  # BOD Net Asset Value (NAV)
        }, columns=cols)[:self.step+1]
        return df

    def snapshot(self):
        return {
            'step': self.step,
            'actions': np.array(self.actions, copy=True),
            'navs': np.array(self.navs, copy=True),
        }

    def recover(self, snapshot, copy=True):
        assert('step' in snapshot and 'actions' in snapshot and 'navs' in snapshot)
        self.step = snapshot['step']
        self.actions = np.array(snapshot['actions'], copy=True) if copy else snapshot['actions']
        self.navs = np.array(snapshot['navs'], copy=True) if copy else snapshot['navs']


class FastTradingEnv(gym.Env, RunMixin):
    """This gym implements a simple trading environment for reinforcement learning.

    The gym provides daily observations based on real market data pulled
    from Yahoo on, by default, the SPY etf. An episode is defined as 252
    contiguous days sampled from the overall dataset. Each day is one
    'step' within the gym and for each step, the algo has a choice:

    FLAT (0)
    LONG (1)

    If you trade, you will be charged, by default, 1e-3 BPS of the size of
    your trade.

    At the beginning of your episode, you are allocated 1 unit of
    cash. This is your starting Net Asset Value (NAV). If your NAV drops
    to 0, your episode is over and you lose.

    *NOT IMPLEMENT* The trading env will track a buy-and-hold strategy which will act as
    the benchmark for the game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, name, days):
        self.name = name
        self.days = days
        self.src = YahooEnvSrc(name=self.name, days=self.days)
        self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-3)
        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(self.src.min_values, self.src.max_values)
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        obs, done = self.src._step()
        close_pct_change = obs['pct_change'][3]
        reward, info = self.sim._step(action, close_pct_change, done=done)
        if info['nav'] <= 0.0:
            done = True
        return obs['history'], reward, done, info

    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _render(self, mode='human', close=False):
        pass
        # src_df = self.src.to_df()  # open, high, low, close, volume
        # sim_df = self.sim.to_df()  # action, nav
        # src_df.close.plot.line()
        # sim_df.plot.bar()

    def action_options(self):
        return range(2)

    def snapshot(self):
        return {
            'src': self.src.snapshot(),
            'sim': self.sim.snapshot(),
        }

    def recover(self, snapshot):
        assert('src' in snapshot and 'sim' in snapshot)
        self.src.recover(snapshot['src'])
        self.sim.recover(snapshot['sim'])
