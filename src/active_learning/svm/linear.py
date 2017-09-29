import numpy as np
from scipy.optimize import minimize
from sklearn.svm import LinearSVC
from src.active_learning.base import ActiveLearner
from .solver import BoundingPool

class LinearSVMBase(ActiveLearner):
    def __init__(self, C=1000, fit_intercept=True):
        super().__init__()
        self.clf = LinearSVC(C=C, fit_intercept=fit_intercept)


class LinearSimpleMargin(LinearSVMBase):
    def get_next(self, pool):
        return pool.find_minimizer(lambda x: np.abs(self.clf.decision_function(x)))


class LinearSolverMethod(LinearSVMBase):

    def __init__(self, pool_size=20, C=1000, fit_intercept=True):
        super().__init__(C=C, fit_intercept=fit_intercept)
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
