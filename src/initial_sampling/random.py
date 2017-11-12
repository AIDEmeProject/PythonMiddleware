import numpy as np

from .base import InitialSampler, FixedSize


class RandomSampler(InitialSampler):
    """
        Randomly samples unlabeled points until a positive and a negative samples have been found
    """
    def sample(self, data, user):
        indices = []
        labels = []

        for i in np.random.permutation(np.arange(len(data))):
            idx = point.index[i]
            point = point.loc[idx]

            indices.append(idx)
            labels.append(int(user.get_label(point)))

            if len(set(labels)) == 2:
                return data.loc[indices], labels

        raise RuntimeError("All points have the same label!")


class FixedSizeRandomSampler(InitialSampler, FixedSize):
    """
        Repeatedly samples a fixed-size batch of unlabeled points until it contains a positive and a negative samples
    """
    def sample(self, data, user):
        n = len(data)

        while True:
            user.clear()

            i = np.random.choice(range(n), replace=False, size=self.sample_size)
            idx = data.index[i]
            points = data.loc[idx]
            labels = user.get_label(points)

            if len(set(labels)) == 2:
                return points, labels
