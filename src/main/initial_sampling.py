class StratifiedSampler:
    def __init__(self, pos, neg, pos_mask=True, neg_mask=True):
        self.pos = int(pos)
        self.neg = int(neg)

        if self.pos < 0 or self.neg < 0:
            raise ValueError("Found negative sample size. Please provide a positive number.")

        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

    def __call__(self, data, user):
        return self._sample(data, user)

    def _sample(self, data, user):
        y_true = user.get_label(data, update_counter=False)

        positive = y_true[(y_true == 1) & self.pos_mask]
        pos_samples = positive.sample(self.pos, replace=False)

        negative = y_true[(y_true == -1) & self.neg_mask]
        neg_samples = negative.sample(self.neg, replace=False)

        labels = pos_samples.append(neg_samples)
        return data.loc[labels.index], labels


class EmptySampler:
    def __call__(self, data, user):
        return [],[]
