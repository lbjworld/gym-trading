# coding: utf-8
import logging
import tempfile
import pandas as pd

log = logging.getLogger(__name__)


class RunMixin(object):
    def run_strat(self, strategy, return_df=True):
        """run provided strategy, returns dataframe with all steps"""
        observation = self.reset()
        done = False
        while not done:
            action = strategy(observation, self)  # call strategy
            observation, reward, done, info = self.step(action)

        return self.sim.to_df() if return_df else None

    def run_strats(self, strategy, episodes=1, write_log=True, return_df=True):
        """ run provided strategy the specified # of times, possibly
          writing a log and possibly returning a dataframe summarizing activity.

          Note that writing the log is expensive and returning the df is moreso.
          For training purposes, you might not want to set both.
        """
        logfile = None
        if write_log:
            logfile = tempfile.NamedTemporaryFile(delete=False)
            log.info('writing log to %s', logfile.name)
            need_df = write_log or return_df

        alldf = None

        for i in range(episodes):
            df = self.run_strat(strategy, return_df=need_df)
            if write_log:
                df.to_csv(logfile, mode='a')
                if return_df:
                    alldf = df if alldf is None else pd.concat([alldf, df], axis=0)

        return alldf
