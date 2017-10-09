class MetricTracker(object):

    metrics_list = ['precision', 'recall', 'accuracy', 'fscore', 'version_space_volume', 'version_space_ratio', 'version_space_ratio_estimate']

    def __init__(self,  skip=0):
        self.skip = skip
        self.metrics = []

    def __len__(self):
        return len(self.metrics)

    def to_dataframe(self):
        from pandas import DataFrame

        return DataFrame(
            data=self.metrics,
            columns=self.metrics_list,
            index=range(self.skip + 1, len(self) + self.skip + 1)
        )

    def add_measurement(self, measurement):
        """ Compute and append new scores """
        if not measurement.keys() <= set(self.metrics_list):
            raise ValueError("Invalid keys in measurement.")
        self.metrics.append(measurement)


class MetricStorage(object):
    def __init__(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def clean(self):
        self.storage = []

    def persist(self, tracker):
        """ Append tracker to storage """
        self.storage.append(tracker.to_dataframe())

    def average_performance(self):
        """ Average all metrics in storage """
        if len(self) > 0:
            from pandas import concat

            # concatenate all stored metrics
            concat_metrics = concat(self.storage, axis=0)

            # group metrics by index
            grouped_metrics = concat_metrics.groupby(level=0)

            # return mean and standard deviation
            return concat(
                [
                    grouped_metrics.mean(),
                    grouped_metrics.std().fillna(0),
                    grouped_metrics.min(),
                    grouped_metrics.max()
                ],
                axis=1,
                keys=['mean', 'std', 'min', 'max']
            )
