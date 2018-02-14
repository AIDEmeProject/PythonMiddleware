import numpy as np
from .line import Line


class MarkovSampler:
    @staticmethod
    def _check_initial_point(convexbody, initial_point):
        if not convexbody.is_inside(initial_point):
            raise ValueError("Initial point must be inside convexbody.")

    @staticmethod
    def _check_parameters(chain_length, n_samples=1):
        if chain_length <= 0 or n_samples <= 0:
            raise ValueError("chain_length and n_samples must be positive!")

    @staticmethod
    def _step(convexbody, point):
        raise NotImplementedError

    @classmethod
    def sample_chain(cls, convexbody, initial_point, chain_length):
        cls._check_initial_point(convexbody, initial_point)
        cls._check_parameters(chain_length)

        samples = np.empty((chain_length+1, len(initial_point)))
        samples[0] = initial_point

        for i in range(chain_length):
            samples[i+1] = cls._step(convexbody, samples[i])

        return samples

    @classmethod
    def sample(cls, convexbody, initial_point, chain_length):
        cls._check_initial_point(convexbody, initial_point)
        cls._check_parameters(chain_length)

        sample = np.array(initial_point)

        for _ in range(chain_length):
            sample = cls._step(convexbody, sample)

        return sample

    @classmethod
    def uniform(cls, convexbody, initial_point, chain_length, n_samples):
        cls._check_initial_point(convexbody, initial_point)
        cls._check_parameters(chain_length, n_samples)

        samples = np.empty((n_samples, len(initial_point)))

        for i in range(n_samples):
            samples[i] = cls.sample(convexbody, initial_point, chain_length)

        return samples


class HitAndRunSampler(MarkovSampler):
    @staticmethod
    def _step(convexbody, point):
        line = Line.sample_line(point, convexbody.projection_matrix)
        segment = convexbody.intersection(line)
        return segment.sample(1)[0]


class SphericalHitAndRunSampler(MarkovSampler):
    @staticmethod
    def _step(convexbody, point):
        # normalize point
        assert convexbody.is_inside(point)
        point = point / np.linalg.norm(point)

        # sample random direction orthogonal to point
        direction = np.random.normal(size=len(point))
        direction = direction - direction.dot(point) * point
        direction = direction / np.linalg.norm(direction)

        # find intersection
        # c(t) = cos(t) * point + sin(t) * direction
        a = np.dot(convexbody.inequality_constrain._matrix, point)
        b = np.dot(convexbody.inequality_constrain._matrix, direction)
        assert np.all(a < 0)
        c = -b / a
        t_max = np.min(np.pi/2 - np.arctan(c))
        t_min = np.max(-np.pi/2 - np.arctan(c))

        # compute point
        t_sample = np.random.uniform(low=t_min, high=t_max, size=(1,))
        return np.cos(t_sample) * point + np.sin(t_sample) * direction
