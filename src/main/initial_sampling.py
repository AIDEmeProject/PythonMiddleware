from pandas import Series


class StratifiedSampler:
    """
    Binary stratified sampling method. Randomly selects a given number of positive and negative points from a list of labels.
    """
    def __init__(self, pos, neg):
        """

        :param pos: Number of positive points to sample. Must be non-negative.
        :param neg: Number of negative points to sample. Must be non-negative.
        """
        self.pos = int(pos)
        self.neg = int(neg)

        if self.pos < 0 or self.neg < 0:
            raise ValueError("Found negative sample size. Only non-negative values allowed.")

    def __call__(self, y, true_class=1.0):
        """
        Call the sampling procedure over the given labeled collection.

        :param y: label collection. Should be a numpy array or pandas Series
        :param true_class: class to be considered positive. Default to 1.0.
        :return: position or index of samples in the array
        """
        y = Series(y)

        pos_samples = y[y == true_class].sample(self.pos, replace=False).index
        neg_samples = y[y != true_class].sample(self.neg, replace=False).index

        return list(pos_samples.append(neg_samples))
