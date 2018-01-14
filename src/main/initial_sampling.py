from numpy.random import RandomState
from pandas import Series

class StratifiedSampler:
    def __init__(self, pos, neg, pos_mask=True, neg_mask=True):
        self.pos = int(pos)
        self.neg = int(neg)

        if self.pos < 0 or self.neg < 0:
            raise ValueError("Found negative sample size. Please provide a positive number.")

        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

        self.new_random_state()

    def __call__(self, data, user):
        return self._sample(data, user)

    def _sample(self, data, user):
        user.clear()
        y_true = user.get_label(data, update_counter=False, use_noise=False)

        positive = y_true[(y_true == 1) & self.pos_mask]
        pos_samples = positive.sample(self.pos, replace=False, random_state=self.random_state)

        negative = y_true[(y_true == -1) & self.neg_mask]
        neg_samples = negative.sample(self.neg, replace=False, random_state=self.random_state)

        labels = pos_samples.append(neg_samples)
        return labels

    def reset_random_state(self):
        self.random_state.set_state(self.state)

    def new_random_state(self):
        self.random_state = RandomState()
        self.state = self.random_state.get_state()


class EmptySampler:
    def __call__(self, data, user):
        return Series()
