from collections import defaultdict

import pandas as pd

from .task import Task


class Showdown(object):
    def run(self, datasets_list, active_learners_list, times, sampler):
        results = defaultdict(list)

        for data_tag, data, user in datasets_list:
            for i in range(times):
                user.clear()
                _, sample = sampler(data, user)

                for al_tag, al in active_learners_list:
                    task = Task(data, user, al)
                    results[(data_tag, al_tag)].append(task.train(sample))

        for k, v in results.items():
            results[k] = self.average_results(v)

        keys = results.keys()
        final = pd.concat([results[k] for k in keys], axis=1, keys=keys)
        return final.swaplevel(2, 3, axis=1).swaplevel(1, 2, axis=1).sort_index(level=0, axis=1)

    def average_results(self, metrics_list):
        storage = [metric.to_dataframe() for metric in metrics_list]

        # concatenate all stored metrics
        concat_metrics = pd.concat(storage, axis=0)

        # group metrics by index
        grouped_metrics = concat_metrics.groupby(level=0)

        # return mean and standard deviation
        return pd.concat(
            [
                grouped_metrics.mean(),
                grouped_metrics.std().fillna(0),
                grouped_metrics.min(),
                grouped_metrics.max()
            ],
            axis=1,
            keys=['mean', 'std', 'min', 'max']
        )


