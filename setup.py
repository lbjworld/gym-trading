from setuptools import find_packages, setup

setup(name='gym_trading',
      version='0.0.3',
      install_requires=['gym', 'pandas', 'pandas_datareader'],
      dependency_links=['git+https://github.com/lbjworld/tushare.git@master'],
      packages=find_packages(),
)
