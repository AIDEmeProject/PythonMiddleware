import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
from src.active_learning.base import ActiveLearner


class SVMBase(ActiveLearner):
    def __init__(self, C=1000, kernel='rbf', degree=3, gamma='auto'):
        super().__init__()
        self.clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape='ovr', max_iter=100000)


class SimpleMargin(SVMBase):
    def get_next(self, pool):
        return pool.find_minimizer(lambda x: np.abs(self.clf.decision_function(x)))


class BoundingPool(object):
    def __init__(self, pool_size):
        self.pool = None
        self.size = pool_size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.pool[item]

    def is_empty(self):
        return self.pool is None

    def build_pool(self, data):
        """ Generate random sampling on bounding box of data """
        N = data.shape[1]

        # find bounding box for data
        min_vec = np.min(data, axis=0)
        max_vec = np.max(data, axis=0)

        self.pool = np.random.uniform(low=min_vec, high=max_vec, size=(self.size, N))

        # sample from faces
        for j in range(self.size):
            k = (j // 2) % N  # current dimension
            if j % 2 == 0:
                self.pool[j, k] = min_vec[k]
            else:
                self.pool[j, k] = max_vec[k]


class SolverMethod(SVMBase):

    def __init__(self, pool_size=20, C=1000, degree=3, kernel='rbf', gamma='auto'):
        super().__init__(C=C, degree=degree, kernel=kernel, gamma=gamma)
        self.bounding_pool = BoundingPool(pool_size=pool_size)

    def initialize(self, X):
        self.bounding_pool.build_pool(X)

    def get_fake_point(self, positive_sample, negative_sample):
        func = lambda t: abs(float(
            self.clf.decision_function(t * positive_sample.reshape(1, -1) + (1 - t) * negative_sample.reshape(1, -1))))

        # find fake point on boundary (or at least the closest)
        res = minimize(func, 0.5, bounds=[(0, 1)])
        return res.x * positive_sample + (1 - res.x) * negative_sample

    def get_next(self, pool):
        """ Get next point to label """
        points, labels = pool.get_labeled_data()

        # decision function
        pos = [x for x, y in zip(points, labels) if y == 1]

        negative_sample = self.bounding_pool[len(points) % len(self.bounding_pool)]
        positive_sample = pos[-1]

        Xfake = self.get_fake_point(positive_sample, negative_sample)

        # find closest unlabeled point to boundary
        return pool.find_minimizer(lambda x: np.linalg.norm(x - Xfake, axis=-1))
