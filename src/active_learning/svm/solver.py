import numpy as np
from scipy.optimize import minimize
from .base import SVMBase


class BoundingPool(object):
    def __init__(self, pool_size):
        self.__pool = None
        self.__size = pool_size
        self.__count = -1

    def build_pool(self, data):
        """ Generate random sampling on bounding box of data """
        N = data.shape[1]

        # find bounding box for data
        min_vec = np.min(data, axis=0)
        max_vec = np.max(data, axis=0)

        self.pool = np.random.uniform(low=min_vec, high=max_vec, size=(self.size, N))

        # sample from faces
        for j in range(self.__size):
            k = (j // 2) % N  # current dimension
            if j % 2 == 0:
                self.pool[j, k] = min_vec[k]
            else:
                self.pool[j, k] = max_vec[k]

    def get_next_negative_point(self):
        self.__count = (self.__count + 1) % len(self.__size)
        return self.pool[self.__count]


class SolverMethod(SVMBase):

    def __init__(self, kind='kernel', pool_size=20, C=1000, kernel='rbf', degree=3, gamma='auto', max_iter=None, fit_intercept=True):
        super().__init__(kind, C, kernel, degree, gamma, max_iter, fit_intercept)
        self.bounding_pool = BoundingPool(pool_size=pool_size)

    def initialize(self, data):
        super().initialize(data)
        self.bounding_pool.build_pool(data)

    def get_fake_point(self, positive_sample, negative_sample):
        positive_sample, negative_sample = np.atleast_2d(positive_sample, negative_sample)

        func = lambda t: abs(float(
            self.clf.decision_function(t * positive_sample + (1 - t) * negative_sample)
        ))

        # find fake point on boundary (or at least the closest)
        res = minimize(func, 0.5, bounds=[(0, 1)])
        return res.x * positive_sample + (1 - res.x) * negative_sample

    def get_next(self, pool):
        """ Get closest point to SVM's boundary given current  """
        positive_sample = pool.get_positive_points()[-1]
        negative_sample = self.bounding_pool.get_next_negative_point()
        fake_point = self.get_fake_point(positive_sample, negative_sample)

        # find closest unlabeled point to fake point
        ranker = lambda data: np.linalg.norm(data - fake_point, axis=-1)
        return pool.get_minimizer_over_unlabeled_data(ranker)
