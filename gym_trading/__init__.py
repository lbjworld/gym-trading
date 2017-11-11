from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='gym_trading.envs:TradingEnv',
    timestep_limit=1000,
)
register(
    id='fast-trading-v0',
    entry_point='gym_trading.envs:FastTradingEnv',
    timestep_limit=1000,
)
register(
    id='complex-trading-v0',
    entry_point='gym_trading.envs:MultiStockTradingEnv',
    timestep_limit=1000,
)
