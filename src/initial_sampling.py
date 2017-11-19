class StratifiedSampler:
    def __init__(self, pos, neg):
        self.pos = int(pos)
        self.neg = int(neg)

        if self.pos < 0 or self.neg < 0:
            raise ValueError("Found negative sample size. Please provide a positive number.")

    def __call__(self, data, user):
        return self.sample(data, user)

    def sample(self, data, user):
        y_true = user.get_label(data, update_counter=False)

        positive = y_true[y_true == 1]
        pos_samples = positive.sample(self.pos, replace=False)

        negative = y_true[y_true == -1]
        neg_samples = negative.sample(self.neg, replace=False)

        return pos_samples.append(neg_samples)