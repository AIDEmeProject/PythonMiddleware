import numpy as np
from scipy.optimize import linprog

from .ellipsoid import Ellipsoid


class LinearVersionSpace:
    """
    This class represents an instance of a centered linear classifiers Version Space. Basically, given a collection
    of labeled data points (x_i, y_i), a vector "w" belongs to the Version Space if:

        y_i x_i^T w > 0   AND   |w| < 1

    By defining a_i = -y_i x_i, we see that it defines a polytope:

        a_i^T w < 0   AND   |w| < 1
    """
    def __init__(self, X, y):
        y = np.where(y == 1, 1, -1).reshape(-1, 1)
        self.A = -X * y

    @property
    def dim(self):
        return self.A.shape[1]

    @staticmethod
    def __solve_second_degree_equation(a, b, c):
        delta = b ** 2 - a * c

        if delta <= 0:
            raise RuntimeError("Second degree equation has 1 or 0 solutions!")

        sq_delta = np.sqrt(delta)
        return (-b - sq_delta) / a, (-b + sq_delta) / a

    def is_inside(self, X):
        return np.all(np.dot(X, self.A.T) < 0, axis=-1)

    def intersection(self, center, direction):
        """
        Finds the intersection between the version space and a straight line.

        :param center: point on the line
        :param direction: director vector of line. Does not need to be normalized.
        :return: t1 and t2 such that center + t * direction are extremes of the line segment determined by the intersection
        """
        lower, upper = [], []

        # polytope intersection
        num = self.A.dot(center)
        den = self.A.dot(direction)
        extremes = -num / den
        lower.extend(extremes[den < 0])
        upper.extend(extremes[den > 0])

        if np.any(num[den == 0] < 0):
            raise RuntimeError("Line does not intersect polytope.")

        # ball intersection
        a, b, c = (
            np.sum(direction ** 2),
            center.dot(direction),
            np.sum(center ** 2) - 1
        )
        lower_ball, upper_ball = self.__solve_second_degree_equation(a, b, c)
        lower.append(lower_ball)
        upper.append(upper_ball)

        # get extremes
        lower_extreme = max(lower)
        upper_extreme = min(upper)

        if lower_extreme >= upper_extreme:
            raise RuntimeError("Line does not intersect polytope.")

        return lower_extreme, upper_extreme

    def get_interior_point(self):
        """
        Finds an interior point to the version space by solving the following Linear Programming problem:

            minimize s,  s.t.  |w_i| < 1  AND a_i^T w < s

        Raises an error in case the polytope is degenerate (only 0 vector).

        :return: point inside search space
        """
        n, dim = self.A.shape

        res = linprog(
            c=np.array([1.0] + [0.0] * dim),
            A_ub=np.hstack([-np.ones(shape=(n, 1)), self.A]),
            b_ub=np.zeros(n),
            bounds=[(None, None)] + [(-1, 1)] * dim
        )

        # if optimization failed, raise error
        if not res.success or res.x[0] >= 0:
            print(res)
            raise RuntimeError("Linear programming failed! Check constrains for degeneracy of Version Space.")

        # return normalized point
        point = res.x[1:]
        return point / np.linalg.norm(point)

    def get_separating_oracle(self, point):
        """
        For any given point, find a hyperplane separating it from the polytope. Basically, we check whether any constrain
        is not satisfied by the point. This method is used during the rounding procedure in Hit-and-Run sampler.

        :param point: data point
        :return: normal vector to separating hyperplane. If no such plane exists, returns None
        """
        if np.dot(point, point) > 1:
            return 1, point / np.linalg.norm(point)

        for a in self.A:
            if np.dot(a, point) > 0:
                return 0, a

        return None


class HitAndRunSampler:
    """
    Hit-and-run is a MCMC sampling technique for sampling uniformly from a Convex Polytope. In our case, we restrict
    this technique for sampling from the Linear Version Space body defined above.

    Reference: https://link.springer.com/content/pdf/10.1007%2Fs101070050099.pdf
    """

    def __init__(self, warmup=100, thin=1, rounding=True, cache=False):
        """
        :param warmup: number of initial samples to ignore
        :param thin: number of samples to skip
        :param rounding: whether to apply the rounding preprocessing step. Mixing time considerably improves, but so does
        the running time.
        """
        self.warmup = warmup
        self.thin = thin

        self.rounding = rounding
        self.elp = None

        self.cache = cache
        self.samples = None

    def sample(self, X, y, n_samples):
        """
        Compute a MCMC Sample from the version space.

        :param X: data matrix
        :param y: labels (positive label should be 1)
        :param n_samples: number of samples
        :return: samples in a numpy array (one per line)
        """
        version_space = LinearVersionSpace(X, y)

        # rounding
        rounding_matrix = None
        if self.rounding:
            self.elp = Ellipsoid(version_space)
            rounding_matrix = self.elp.L * np.sqrt(self.elp.D).reshape(1, -1)

        center = self.__get_center(version_space)
        samples = np.empty((n_samples, version_space.dim))

        # skip samples
        self.__advance(self.warmup, center, rounding_matrix, version_space)
        samples[0] = center.copy()

        # thin samples
        for i in range(1, n_samples):
            self.__advance(self.thin, center, rounding_matrix, version_space)
            samples[i] = center.copy()

        # cache samples
        if self.cache:
            self.samples = samples

        return samples

    def __advance(self, n_iter, center, rounding_matrix, version_space):
        for _ in range(n_iter):
            # sample random direction
            direction = self.__sample_direction(version_space.dim, rounding_matrix)

            # get extremes of line segment determined by intersection
            t1, t2 = version_space.intersection(center, direction)

            # get random point on line segment
            t_rand = np.random.uniform(t1, t2)
            center += t_rand * direction

    def __sample_direction(self, dim, rounding_matrix):
        direction = np.random.normal(size=(dim,))

        if self.rounding:
            direction = rounding_matrix.dot(direction)

        return direction

    def __get_center(self, version_space):
        if self.rounding:
            return self.elp.x

        if self.cache and self.samples is not None:
            samples = self.__truncate_samples(version_space.dim)
            for sample in samples:
                if version_space.is_inside(sample):
                    return sample

        print('Falling back to linprog in Hit-and-Run sampler.')
        return version_space.get_interior_point()

    def __truncate_samples(self, new_dim):
        N, dim = self.samples.shape

        if new_dim <= dim:
            return self.samples[:, :new_dim]

        return np.hstack([self.samples, np.zeros((N, new_dim - dim))])
