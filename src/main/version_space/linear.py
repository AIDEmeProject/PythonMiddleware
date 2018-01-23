import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel

from .appendable_constrain import AppendableInequalityConstrain
from .base import VersionSpace

from ..convexbody.objects import Polytope
from ..convexbody.sampling import HitAndRunSampler


class LinearVersionSpace(Polytope, VersionSpace):
    def __init__(self, dim):
        Polytope.__init__(self, l=[-1]*(dim+1), h=[1]*(dim+1))
        VersionSpace.__init__(self)
        self.inequality_constrain = AppendableInequalityConstrain(dim+1)  # add one to account for bias

    def clear(self):
        self.inequality_constrain.clear()

    def _update_single(self, point, label):
        point = np.hstack([1, point])  # add bias component
        constrain_vector = -label * point  # constrain = -y_i (1, x_i)
        self.inequality_constrain.append(constrain_vector, 0)

    def sample(self, chain_length, sample_size=-1, initial_point=None):
        # if initial_point is not given, fall back to linprog
        if initial_point is None or len(initial_point) == 0:
            print('fall back to linprog')
            initial_point = self.get_point()

        if sample_size > 0:
            return HitAndRunSampler.uniform(self, initial_point, chain_length, sample_size)
        return HitAndRunSampler.sample_chain(self, initial_point, chain_length)

    def get_point(self):
        return self.inequality_constrain.get_point()


class KernelVersionSpace(VersionSpace):
    __kernel_functions = {
        'rbf': rbf_kernel,
        'linear': linear_kernel,
        'poly': polynomial_kernel,
        'sigmoid': sigmoid_kernel
    }

    def __init__(self, kernel='rbf', cholesky=False):
        super().__init__()

        self.__linear_version_space = None
        self.kernel_name = kernel
        self.kernel = KernelVersionSpace.__kernel_functions[kernel]

        self.__labeled_points = []
        self.__labels = []

        self.cholesky = cholesky
        self._L = None

    def clear(self):
        if self.__linear_version_space is not None:
            self.__linear_version_space.clear()

        self.__labeled_points = []
        self.__labels = []

        self._L = None

    def _update_single(self, point, label):
        self.__labeled_points.append(point)
        self.__labels.append(label)

        K = self.kernel(self.__labeled_points)
        if self.cholesky:
            K = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
            self._L = K

        # create version space
        self.__linear_version_space = LinearVersionSpace(K.shape[1])
        self.__linear_version_space.update(K, self.__labels)

    def sample(self, chain_length, sample_size=-1, initial_point=None):
        return self.__linear_version_space.sample(chain_length, sample_size, initial_point)

    def sample_classifiers(self, chain_length, sample_size=-1, initial_point=None):
        samples = self.__linear_version_space.sample(chain_length, sample_size, initial_point)
        return SVMClassifier(samples[:,0], samples[:,1:], self.__labeled_points, self.kernel_name)
        #[
        #    SVMClassifier(bias, alpha, self.__labeled_points, self.kernel_name)
        #    for bias, alpha in zip(samples[:,0], samples[:, 1:])
        #]

    def compute_kernel_against_data(self, data):
        return self.kernel(data, self.__labeled_points)

    def is_inside(self, points):
        if self.__linear_version_space is None:
            size = len(points) if len(points.shape) > 1 else 1
            return [True] * size
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
        self.alpha = np.atleast_2d(alpha) #.ravel()
        self.support_vectors = np.atleast_2d(support_vectors)
        self.kernel = SVMClassifier.__kernel_functions[kernel]

    def get_params(self):
        return np.hstack([self.bias, self.alpha])

    def predict(self, data):
        K = self.kernel(data, self.support_vectors)
        return np.sign(self.bias + self.alpha.dot(K.T))