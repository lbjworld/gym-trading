# coding: utf-8
import gym
from gym import spaces
from gym.utils import seeding

import logging
import numpy as np
import pandas as pd

from data_loader import data_loader
from mixin import RunMixin

log = logging.getLogger(__name__)
log.info('%s logger started.', __name__)


def _sharpe(Returns, freq=252):
    """Given a set of returns, calculates naive (rfr=0) sharpe """
    return (np.sqrt(freq) * np.mean(Returns))/np.std(Returns)


def _prices2returns(prices):
    px = pd.DataFrame(prices)
    nl = px.shift().fillna(0)
    R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
    R = np.append(R[0].values, 0)
    return R


class YahooEnvSrc(object):
    '''
    Yahoo-based implementation of a TradingEnv's data source.

    Pulls data from Yahoo data source, preps for use by TradingEnv and then
    acts as data provider for each new episode.
    '''

    MIN_PERCENTILE_DAYS = 100
    STOCK_NAME = '000333.SZ'

    def __init__(self, days=252, name=STOCK_NAME, scale=True):
        self.name = name
        self.days = days + 1
        self.data = dict()
        log.info('getting data for %s from yahoo...', self.name)
        df = data_loader(self.name)
        log.info('got data for %s from yahoo...', self.name)

        df = df[~np.isnan(df.Volume)][['Close', 'Volume']]
        # we calculate returns and percentiles, then kill nans
        df = df[['Close', 'Volume']]
        df.Volume.replace(0, 1, inplace=True)  # days shouldn't have zero volume..
        df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
        pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        df['ClosePctl'] = df.Close.expanding(self.MIN_PERCENTILE_DAYS).apply(pctrank)
        df['VolumePctl'] = df.Volume.expanding(self.MIN_PERCENTILE_DAYS).apply(pctrank)
        df.dropna(axis=0, inplace=True)
        R = df.Return
        if scale:
            mean_values = df.mean(axis=0)
            std_values = df.std(axis=0)
            df = (df - np.array(mean_values)) / np.array(std_values)
        df['Return'] = R  # we don't want our returns scaled
        self.data = df
        self.min_values = df.min(axis=0)
        self.max_values = df.max(axis=0)
        self.reset()

    def reset(self):
        # we want contiguous data
        self.idx = np.random.randint(low=0, high=len(self.data.index)-self.days)
        self.step = 0

    def _step(self):
        obs = self.data.iloc[self.idx].as_matrix()
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done


class TradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.strat_retrns = np.ones(self.steps)
        self.posns = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
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

        if self.step == 0:
            bod_posn, bod_nav, mkt_nav = 0.0, 1.0, 1.0
        else:
            bod_posn, bod_nav, mkt_nav = \
                self.posns[self.step-1], self.navs[self.step-1], self.mkt_nav[self.step-1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action

        self.posns[self.step] = action - 1
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
            'costs': self.costs[self.step]
        }

        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = [
            'action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return', 'position', 'costs', 'trade'
        ]
        # rets = _prices2returns(self.navs)
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


class TradingEnv(gym.Env, RunMixin):
    """This gym implements a simple trading environment for reinforcement learning.

    The gym provides daily observations based on real market data pulled
    from Yahoo on, by default, the SPY etf. An episode is defined as 252
    contiguous days sampled from the overall dataset. Each day is one
    'step' within the gym and for each step, the algo has a choice:

    SHORT (0)
    FLAT (1)
    LONG (2)

    If you trade, you will be charged, by default, 10 BPS of the size of
    your trade. Thus, going from short to long costs twice as much as
    going from short to/from flat. Not trading also has a default cost of
    1 BPS per step. Nobody said it would be easy!

    At the beginning of your episode, you are allocated 1 unit of
    cash. This is your starting Net Asset Value (NAV). If your NAV drops
    to 0, your episode is over and you lose. If your NAV hits 2.0, then
    you win.

    The trading env will track a buy-and-hold strategy which will act as
    the benchmark for the game.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.days = 252
        self.src = YahooEnvSrc(days=self.days)
        self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-3, time_cost_bps=1e-4)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.src.min_values, self.src.max_values)
        self.reset()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        observation, done = self.src._step()
        # Close Volume Return ClosePctl VolumePctl
        yret = observation[2]
        reward, info = self.sim._step(action, yret)
        if info['nav'] >= 2.0 or info['nav'] <= 0.0:
            done = True
        return observation, reward, done, info

    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _render(self, mode='human', close=False):
        # TODO
        pass

    def action_options(self):
        return range(3)
