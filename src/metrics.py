from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, f1_score

supported_metrics = {
    'precision': lambda y_pred, y_true: precision_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1]),
    'recall': lambda y_pred, y_true: recall_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1]),
    'f1': lambda y_pred, y_true: accuracy_score(y_pred=y_pred, y_true=y_true),
    'accuracy': lambda y_pred, y_true: f1_score(y_pred=y_pred, y_true=y_true, labels=[-1, 1])
}


class MetricTracker(object):

    def __init__(self, metrics_list=None, skip=0):
        if not set(metrics_list) <= supported_metrics.keys():
            raise KeyError("Unsupported metric found. Only {0} are available.".format(supported_metrics.keys()))

        self.skip = skip
        self.metrics_list = metrics_list
        self.metrics = []
        self.storage = []

    @property
    def size(self):
        return len(self.metrics)

    @property
    def storage_size(self):
        return len(self.storage)

    def clear(self):
        self.storage = []
        self.metrics = []

    def persist(self):
        """ Put all metrics into storage """
        if len(self.metrics) > 0:
            from pandas import DataFrame

            # append current metrics to storage
            self.storage.append(
                DataFrame(
                    data=self.metrics,
                    columns=self.metrics_list,
                    index=range(self.skip + 1, self.size + self.skip + 1)
                )
            )

            # reset metrics and size
            self.metrics = []

    def add_measurement(self, y_true, y_pred):
        """ Compute and append new scores """
        self.metrics.append([supported_metrics[name](y_pred=y_pred, y_true=y_true) for name in self.metrics_list])

    def average_performance(self):
        """ Average all metrics in storage """
        if self.storage_size > 0:
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
