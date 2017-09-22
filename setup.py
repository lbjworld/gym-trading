from setuptools import find_packages, setup

setup(name='gym_trading',
      version='0.0.2',
      install_requires=['gym', 'pandas', 'pandas_datareader'],
      packages=find_packages(),
)
