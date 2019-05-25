from numpy import arange
from sklearn.utils import check_random_state, check_array


class StratifiedSampler:
    """
        Binary stratified sampling method. Randomly selects a given number of positive and negative points from an array
        of labels.
    """
    def __init__(self, pos, neg, random_state=None):
        """

        :param pos: Number of positive points to sample. Must be non-negative.
        :param neg: Number of negative points to sample. Must be non-negative.
        """
        self.pos = int(pos)
        self.neg = int(neg)

        if self.pos < 0 or self.neg < 0:
            raise ValueError("Found negative sample size. Only non-negative values allowed.")

        if self.pos == self.neg == 0:
            raise ValueError("Either 'pos' or 'neg' must be positive.")

        self.__random_state = check_random_state(random_state)

    def __call__(self, y, true_class=1, neg_class=0):
        """
        Call the sampling procedure over the input array.

        :param y: array-like collection of labels
        :param true_class: class to be considered positive. Default to 1.0.
        :return: index of samples in the array
        """
        y = check_array(y, ensure_2d=False, allow_nd=False)
        if y.ndim == 2:
            y = y.mean(axis=1)

        idx = arange(len(y))
        pos_samples = self.__random_state.choice(idx[y == true_class], size=self.pos, replace=False)
        neg_samples = self.__random_state.choice(idx[y == neg_class], size=self.neg, replace=False)

        return list(pos_samples) + list(neg_samples)


class FixedSampler:
    """
        Dummy sampler which returns a specified selection of indexes.
    """
    def __init__(self, indexes):
        self.indexes = indexes.copy()

    def __call__(self, y):
        return self.indexes
