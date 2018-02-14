import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel

from ..convexbody.objects import ConvexBody, UnitBall
from ..convexbody.sampling.line import LineSegment
from .appendable_constrain import AppendableInequalityConstrain
from .base import VersionSpace

from ..convexbody.sampling import HitAndRunSampler, SphericalHitAndRunSampler
from ..experiments.utils import get_generator_average
from ..convexbody.sampling.ellipsoid import Ellipsoid


class LinearVersionSpace(ConvexBody, VersionSpace):
    def __init__(self, dim, rounding=False):
        ConvexBody.__init__(self, dim+1)
        VersionSpace.__init__(self)
        self.inequality_constrain = AppendableInequalityConstrain(dim+1)  # add one to account for bias

        self.rounding=rounding

    def clear(self):
        self.inequality_constrain.clear()

    def _update_single(self, point, label):
        point = np.hstack([1, point])  # add bias component
        constrain_vector = -label * point  # constrain = -y_i (1, x_i)
        self.inequality_constrain.append(constrain_vector, 0)

    def get_point(self):
        return self.inequality_constrain.get_point()

    def sample(self, chain_length, sample_size=-1, initial_point=None):
        # if initial_point is not given, fall back to linprog
        if initial_point is None or len(initial_point) == 0:
            print('fall back to linprog')
            initial_point = self.get_point()

        if sample_size > 0:
            return HitAndRunSampler.uniform(self, initial_point, chain_length, sample_size)
        return HitAndRunSampler.sample_chain(self, initial_point, chain_length)

    def get_separating_oracle(self, point):
        for a in self.inequality_constrain.matrix:
            if np.dot(a, point) > 0:
                return a

        if np.dot(point, point) > 1:
            return point

        return None

    def is_inside(self, points):
        points = np.atleast_2d(points)
        return np.logical_and(self.inequality_constrain.check(points), np.linalg.norm(points, axis=1) <= 1)

    def intersection(self, line):
        segment1 = self.inequality_constrain.intersection(line)
        segment2 = UnitBall(self.dim).intersection(line)
        return LineSegment.intersection(segment1, segment2)

    def _compute_projection_matrix(self):
        if not self.rounding:
            return None
        elp = Ellipsoid(x=np.zeros(self.dim), P = np.eye(self.dim))
        elp.weak_ellipsoid(self)
        return np.linalg.cholesky(elp.P)


class KernelVersionSpace(VersionSpace):
    __kernel_functions = {
        'rbf': rbf_kernel,
        'linear': linear_kernel,
        'poly': polynomial_kernel,
        'sigmoid': sigmoid_kernel
    }

    def __init__(self, kernel='rbf', cholesky=False, rounding=False):
        super().__init__()

        self.__linear_version_space = None
        self.kernel_name = kernel
        self.kernel = KernelVersionSpace.__kernel_functions[kernel]

        self.__labeled_points = []
        self.__labels = []

        self.cholesky = cholesky
        self.K = None

        self.rounding = rounding

    def clear(self):
        if self.__linear_version_space is not None:
            self.__linear_version_space.clear()

        self.__labeled_points = []
        self.__labels = []

        self.K = None

    def _update_single(self, point, label):
        self.__labeled_points.append(point)
        self.__labels.append(label)

        self.K = self.kernel(self.__labeled_points)
        if self.cholesky:
            self.K = np.linalg.cholesky(self.K + 1e-8 * np.eye(len(self.K)))

        # create version space
        self.__linear_version_space = LinearVersionSpace(self.K.shape[1], self.rounding)
        self.__linear_version_space.update(self.K, self.__labels)

    def get_initial_point(self, previous_sample=None):
        if previous_sample is None:
            return None
        initial_points = filter(self.is_inside, map(lambda x: np.hstack([x, [0]]), previous_sample.get_params()))
        return get_generator_average(initial_points)

    def sample(self, chain_length, sample_size=-1, previous_sample=None):
        return self.__linear_version_space.sample(chain_length, sample_size, self.get_initial_point(previous_sample))

    def sample_classifier(self, chain_length, sample_size=-1, previous_sample=None):
        samples = self.__linear_version_space.sample(chain_length, sample_size, self.get_initial_point(previous_sample))
        b,w = samples[:,0], samples[:,1:]
        if self.cholesky:
            w = w.dot(np.linalg.inv(self.K))
        return SVMClassifier(b, w, self.__labeled_points, self.kernel_name)

    def is_inside(self, points):
        if self.__linear_version_space is None:
            size = len(points) if len(points.shape) > 1 else 1
            return [True] * size
        if self.cholesky:
            points[1:] = points[1:].dot(self.K)
        return self.__linear_version_space.is_inside(points)


class SVMClassifier:
    __kernel_functions = {
        'rbf': rbf_kernel,
        'linear': linear_kernel,
        'poly': polynomial_kernel,
        'sigmoid': sigmoid_kernel
    }

    def __init__(self, bias, alpha, support_vectors, kernel):
        self.bias = np.atleast_1d(bias).reshape(-1,1)
        self.alpha = np.atleast_2d(alpha)
        self.support_vectors = np.atleast_2d(support_vectors)
        self.kernel = SVMClassifier.__kernel_functions[kernel]

    def get_params(self):
        return np.hstack([self.bias, self.alpha])

    def predict(self, data):
        K = self.kernel(data, self.support_vectors)
        return np.sign(self.bias + self.alpha.dot(K.T))