from distutils.core import setup

setup(name='gym_trading',
      version='0.0.2',
      install_requires=['gym', 'pandas', 'pandas_datareader'],
      packages=['gym_trading'],
      package_dir={'gym_trading': ''},
)
