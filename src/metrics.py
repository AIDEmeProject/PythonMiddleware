class MetricTracker(object):

    def __init__(self,  skip=0):
        self.skip = skip
        self.metrics = []

    def __len__(self):
        return len(self.metrics)

    def to_dataframe(self):
        from pandas import DataFrame

        return DataFrame(
            data=self.metrics,
            index=range(self.skip + 1, len(self) + self.skip + 1)
        )

    def add_measurement(self, measurement):
        """ Compute and append new scores """
        self.metrics.append(measurement)
