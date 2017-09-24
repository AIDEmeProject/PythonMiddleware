from src.datapool import Point


class InitialSampler(object):
    def __call__(self, pool, user):
        return self.sample(pool, user)

    def sample(self, data, user):
        return Point(index=[], data=[]), []


class FixedSize(object):
    def __init__(self, sample_size):
        if sample_size < 2:
            raise AttributeError("Sample size must >= 2")
        self.sample_size = sample_size


class DeterministicSampler(InitialSampler):
    """
        Always return the same samples
    """
    def __init__(self, indices):
        self.indices = indices

    def sample(self, data, user):
        points = Point(index=self.indices, data=data[self.indices])
        return points, user.get_label(points)

