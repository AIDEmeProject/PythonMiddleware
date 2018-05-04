import numpy as np
from scipy.optimize import linprog
from sklearn.utils import check_array


class LinearVersionSpace:
    def __init__(self, A):
        self.A = check_array(A)

    def is_inside(self, X):
        ineq = np.all(np.dot(X, self.A.T) < 0, axis=-1)
        ball = np.linalg.norm(X, axis=-1) < 1
        return np.logical_and(ineq, ball)

    @staticmethod
    def __solve_second_degree_equation(a, b, c):
        delta = b ** 2 - a * c

        if delta <= 0:
            raise RuntimeError("Second degree equation has 1 or 0 solutions!")

        sq_delta = np.sqrt(delta)
        return (-b - sq_delta) / a, (-b + sq_delta) / a

    def intersection(self, center, direction):
        """
        Finds the intersection between the version space and a straight line.
        :param center: point on the line
        :param direction: director vector of line. Does not need to be normalized.
        :return: t1 and t2 such that center + t * direction are extremes of the line segment determined by the intersection
        """
        lower, upper = [], []

        # polytope
        den = self.A.dot(direction)
        extremes = -self.A.dot(center) / den
        lower.extend(extremes[den < 0])
        upper.extend(extremes[den > 0])

        # ball
        a, b, c = (
            np.sum(direction ** 2),
            center.dot(direction),
            np.sum(center ** 2) - 1
        )
        lower_ball, upper_ball = self.__solve_second_degree_equation(a, b, c)
        lower.append(lower_ball)
        upper.append(upper_ball)

        # get extremes
        t1 = max(lower)
        t2 = min(upper)

        if t1 >= t2:
            raise RuntimeError("Line does not intersect polytope.")

        return t1, t2

    def get_point(self):
        """
            Finds an interior point to the version space through an optimization routine.
            :return: point inside search space
        """
        n, dim = self.A.shape

        res = linprog(
            c=np.array([1.0] + [0.0] * dim),
            A_ub=np.hstack([-np.ones(shape=(n, 1)), self.A]),
            b_ub=np.zeros(n),
            bounds=[(None, None)] + [(-1, 1)] * dim
        )

        if not res.success:# or res.x[0] >= 0:
            print(res)
            raise RuntimeError("Linear programming failed! Check constrains for degeneracy of Version Space.")

        point = res.x[1:].ravel()
        return point / np.linalg.norm(point)


class HitAndRunSampler:
    def __init__(self, warmup=100, thin=1):
        self.warmup = warmup
        self.thin = thin

    def sample(self, X, y, n_samples):
        n, dim = X.shape
        prod = -X * (2*y - 1).reshape(-1, 1)

        version_space = LinearVersionSpace(prod)
        center = version_space.get_point()

        samples = []
        for i in range(self.warmup + self.thin * n_samples):
            # sample random direction
            direction = np.random.normal(size=dim)

            # get extremes of line segment determined by intersection
            t1, t2 = version_space.intersection(center, direction)

            # get random point on line segment
            t_rand = np.random.uniform(t1, t2)
            center = center + t_rand * direction

            if i >= self.warmup and i % self.thin == 0:
                samples.append(center)

        return np.array(samples)
