from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, f1_score

supported_metrics = {
    'precision': lambda y_pred, y_true: precision_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1]),
    'recall': lambda y_pred, y_true: recall_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1]),
    'accuracy': lambda y_pred, y_true: accuracy_score(y_pred=y_pred, y_true=y_true),
    'f1': lambda y_pred, y_true: f1_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1])
}


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


class MetricTracker(object):

    def __init__(self, metrics_list, skip=0):
        if not set(metrics_list) <= supported_metrics.keys():
            raise KeyError("Unsupported metric found. Only {0} are available.".format(supported_metrics.keys()))

        self.skip = skip
        self.metrics_list = metrics_list
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

    def add_measurement(self, y_true, y_pred):
        """ Compute and append new scores """
        self.metrics.append([supported_metrics[name](y_pred=y_pred, y_true=y_true) for name in self.metrics_list])
