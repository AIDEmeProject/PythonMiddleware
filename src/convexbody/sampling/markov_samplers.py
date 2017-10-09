import numpy as np

from .line import Line


class MarkovSampler(object):
    def __init__(self, chain_length):
        self.chain_length = chain_length

    def _check_initial_point(self, convexbody, initial_point):
        if not convexbody.is_inside(initial_point):
            raise ValueError("Initial point x must be inside convexbody.")

    def _step(self, convexbody, point):
        raise NotImplementedError

    def sample_chain(self, convexbody, initial_point):
        self._check_initial_point(convexbody, initial_point)

        samples = np.empty((self.chain_length+1, len(initial_point)))
        samples[0] = initial_point

        for i in range(self.chain_length):
            samples[i+1] = self._step(convexbody, samples[i])

        return samples

    def sample(self, convexbody, initial_point):
        self._check_initial_point(convexbody, initial_point)

        sample = np.array(initial_point)

        for _ in range(self.chain_length):
            sample = self._step(convexbody, sample)

        return sample

    def uniform(self, convexbody, initial_point, n_samples):
        self._check_initial_point(convexbody, initial_point)

        samples = np.empty((n_samples, len(initial_point)))

        for i in range(n_samples):
            samples[i] = self.sample(convexbody, initial_point)

        return samples


class HitAndRunSampler(MarkovSampler):

    def __init__(self, chain_length):
        super().__init__(chain_length)

    def _step(self, convexbody, point):
        line = Line.sample_line(point, convexbody.projection_matrix)
        segment = convexbody.intersection(line)
        return segment.sample(1)[0]
