import pandas as pd
from .tags import list_to_tag
from ..active_learning.base import train
from ..metrics import MetricStorage


class Showdown(object):
    def __init__(self, times, initial_sampler):
        self.times = times
        self.initial_sampler = initial_sampler

    def get_average_performance(self, data, user, active_learner):
        storage = MetricStorage()

        # train learner for several iterations
        for _ in range(self.times):
            storage.persist(train(data, user, active_learner, self.initial_sampler))

        # compute average performance
        return storage.average_performance()

    def run(self, datasets_list, active_learners_list):
        datasets, active_learners = list_to_tag(datasets_list, active_learners_list)

        return pd.concat(
            [
                pd.concat(
                    [
                        self.get_average_performance(data, user, al) for al in active_learners.learners
                    ],
                    axis=1,
                    keys=active_learners.tags
                )
                for data, user in zip(datasets.datasets, datasets.users)
            ],
            axis=1,
            keys=datasets.tags
        ).swaplevel(2, 3, axis=1).swaplevel(1, 2, axis=1).sort_index(level=0, axis=1)
