import numpy as np
from math import ceil
from .base import InitialSampler, FixedSize
from ..datapool import Point
from ..utils import label_all


def sample_index(mask, size):
    n = len(mask)

    idx = np.arange(n)[mask]
    sampled_idx = np.random.choice(idx, replace=False, size=size)

    return sampled_idx


class StratifiedSamplerBase(InitialSampler, FixedSize):
    """
        Returns "size" points proportionally divided between positive and negative samples
    """
    def _fraction_of_positive_samples(self, mask):
        raise NotImplementedError

    def __check_values(self, pos, neg, mask):
        if pos == self.sample_size or neg == self.sample_size:
            raise RuntimeError("All samples are either positive or negative!")

        if pos >= np.sum(mask) or neg >= np.sum(~mask):
            raise RuntimeError("Training set too small: all positive or negative samples will be sampled!")

    def __get_sample_size(self, mask):
        frac = self._fraction_of_positive_samples(mask)

        pos = int(ceil(self.sample_size * frac))
        neg = self.sample_size - pos

        self.__check_values(pos, neg, mask)

        return pos, neg

    def __get_mask(self, data, user):
        y_true = label_all(data, user)
        user.labeled_samples = self.sample_size
        return y_true == 1

    def sample(self, data, user):
        if self.sample_size > len(data):
            raise AttributeError("Training set size is smaller than sample size.")

        mask = self.__get_mask(data, user)

        pos, neg = self.__get_sample_size(mask)

        pos_index = sample_index(mask, pos)
        neg_index = sample_index(~mask, neg)

        index = list(pos_index) + list(neg_index)
        return Point(index=index, data=data[index]), [1]*pos + [-1]*neg


class StratifiedSampler(StratifiedSamplerBase):
    """
        Returns "size" points proportionally divided between positive and negative samples
    """
    def _fraction_of_positive_samples(self, mask):
        return 1.0 * np.sum(mask) / len(mask)


class FixedSizeStratifiedSampler(StratifiedSamplerBase):
    """
        Samples the same number of positive and negative points
    """
    def _fraction_of_positive_samples(self, mask):
        return 0.5

