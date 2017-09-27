import numpy as np

from .base import InitialSampler, FixedSize
from ..datapool import Point


class RandomSampler(InitialSampler):
    """
        Randomly samples unlabeled points until a positive and a negative samples have been found
    """
    def sample(self, data, user):
        indices = []
        labels = []

        for idx in np.random.permutation(np.arange(len(data))):
            point = Point(index=[idx], data=data[idx])

            indices.append(idx)
            labels.append(int(user.get_label(point)))

            if len(set(labels)) == 2:
                return Point(index=indices, data=data[indices]), labels

        raise RuntimeError("All points have the same label!")


class FixedSizeRandomSampler(InitialSampler, FixedSize):
    """
        Repeatedly samples a fixed-size batch of unlabeled points until it contains a positive and a negative samples
    """
    def sample(self, data, user):
        n = len(data)

        while True:
            user.clear()

            idx = np.random.choice(range(n), replace=False, size=self.sample_size)
            points = Point(index=idx, data=data[idx])
            labels = user.get_label(points)

            if len(set(labels)) == 2:
                return points, labels
