import numpy as np

from .line_intersection import get_line_intersector


class MarkovSampler(object):
    def __init__(self, chain_length):
        self.chain_length = chain_length

    def step(self, point):
        raise NotImplementedError

    def sample_chain(self, x0):
        samples = np.empty((self.chain_length+1, len(x0)))
        samples[0] = x0

        for i in range(self.chain_length):
            samples[i+1] = self.step(samples[i])

        return samples

    def sample(self, x0):
        sample = np.array(x0)
        for i in range(self.chain_length):
            sample = self.step(sample)

        return sample

    def uniform(self, x0, n_samples):
        samples = np.empty((n_samples, len(x0)))
        for i in range(n_samples):
            samples[i] = self.sample(x0)

        return samples


class HitAndRunSampler(MarkovSampler):

    def __init__(self, chain_length, K, enclosing=None):
        super().__init__(chain_length)
        self.K = K
        self.proj = getattr(K, "proj", None)
        self.intersector = get_line_intersector(self.K, enclosing)

    def sample_direction(self):
        u = np.random.normal(size=self.K.n)
        if self.proj is not None:
            u -= self.proj.dot(u)
        return u / np.linalg.norm(u)

    def step(self, point):
        assert self.K.is_inside(point), "Initial point x must be inside K: {0}".format(np.sum(point))
        direction = self.sample_direction()
        return point + direction * self.intersector(point, direction)
