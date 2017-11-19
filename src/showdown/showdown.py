import pandas as pd
from .tags import list_to_tag
from ..config.task import Task

class Showdown(object):
    def run(self, datasets_list, active_learners_list, times):
        datasets, active_learners = list_to_tag(datasets_list, active_learners_list)

        return pd.concat(
            [
                pd.concat(
                    [
                        Task(data, user, al).get_average_performance(times) for al in active_learners.learners
                    ],
                    axis=1,
                    keys=active_learners.tags
                )
                for data, user in zip(datasets.datasets, datasets.users)
            ],
            axis=1,
            keys=datasets.tags
        ).swaplevel(2, 3, axis=1).swaplevel(1, 2, axis=1).sort_index(level=0, axis=1)
