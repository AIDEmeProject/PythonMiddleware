from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.utils import check_X_y
from sklearn.utils.validation import column_or_1d

from .base import ConvexBody


class Polytope(ConvexBody):
    """ A polytope is a convex set defined by three set of equations:

        Linear Constrains: Ax = b
        Inequality Constrains: Mx <= q
        Bounds: l <= x <= h
    """

    def __init__(self, A=None, b=None, M=None, q=None, l=None, h=None):
        """
        All matrices are supposed to be full-rank (that is, no ambiguous constrains are introduced)

        :param A: matrix of equality constrains 
        :param b: right-hand side of linear constrains
        :param M: matrix of inequality constrains
        :param q: right-hand side of inequality constrains
        :param l: lower-bound on x
        :param h: upper-bound on x
        """

        self.has_equality = A is not None and b is not None
        if self.has_equality:
            self.A, self.b = check_X_y(A, b, y_numeric=True)
            self.proj = np.dot(np.dot(self.A.T, np.linalg.inv(self.A.dot(self.A.T))), self.A)
            self.n = self.A.shape[1]

        self.has_inequality = M is not None and q is not None
        if self.has_inequality:
            self.M, self.q = check_X_y(M, q, y_numeric=True)
            self.n = self.M.shape[1]

        self.has_lower = l is not None
        if self.has_lower:
            self.l = column_or_1d(l)

        self.has_upper = h is not None
        if self.has_upper:
            self.h = column_or_1d(h)

    def is_inside(self, x):
        lower = True if not self.has_lower else np.all(x >= self.l, axis=-1)
        upper = True if not self.has_upper else np.all(x <= self.h, axis=-1)
        equality = True if not self.has_equality else np.allclose(np.dot(x, self.A.T), self.b)
        inequality = True if not self.has_inequality else np.all(np.dot(x, self.M.T) <= self.q, axis=-1)

        return np.logical_and(np.logical_and(lower, upper), np.logical_and(equality, inequality))


class ActBoostPolytope(ConvexBody):
    """
    ActBoost algorithm search space. At every new sampled point, a new constrain is added to the polytope   
    """

    def __init__(self, n):
        self.n = int(n)
        self.constrains = []
        self.A = np.ones(shape=(1, n))
        self.proj = np.dot(np.dot(self.A.T, np.linalg.inv(self.A.dot(self.A.T))), self.A)
        self.__cons = self.__create_constrains()
        self.__minimizer = partial(minimize, fun=lambda x: x[0], jac=lambda x: np.array([1.0] + [0.0] * self.n),
                                   method="SLSQP")

    def is_inside(self, x):
        lower = np.all(x >= 0, axis=-1)
        equality = np.allclose(np.sum(x, axis=-1), 1)
        inequality = np.all(np.dot(x, self.M.T) <= 0)

        return np.logical_and(lower, np.logical_and(equality, inequality))

    def __create_constrains(self):
        cons = [
            {'type': 'eq',
             'fun': lambda x: np.sum(x[1:]) - 1.0,
             'jac': lambda x: np.hstack([0.0, np.ones(self.n)])
             }
        ]
        cons += [
            {'type': 'ineq',
             'fun': lambda x, a=i: x[0] + x[a + 1],
             'jac': lambda x, a=i: np.hstack([1.0, np.eye(self.n)[a]])
             } for i in range(self.n)
        ]
        return cons

    @property
    def M(self):
        return np.array(self.constrains)

    def get_point(self):
        """
        Finds an interior point to the current search space through an optimization routine.
        :return: q inside search space
        """
        q0 = np.ones(self.n, dtype=np.float64) / self.n
        if not self.constrains:
            return q0

        s0 = np.max(np.dot(self.M, q0)) + 1
        x0 = np.hstack([s0, q0])

        res = self.__minimizer(x0=x0, constraints=self.__cons)

        assert self.is_inside(res.x[1:]), "Point outside search space!"
        return res.x[1:]

    def append(self, x, y):
        """
        Append new linear inequality y * <x, q> >= 0 to search space.
        :param x: new feature vector
        :param y: label
        """
        m = -y * x
        self.constrains.append(m)
        self.__cons.append({'type': 'ineq',
                            'fun': lambda s: s[0] - np.dot(m, s[1:]),
                            'jac': lambda s: np.hstack([1.0, -m])
                            })

    def clear(self):
        """
        Returns search space to initial configuration. 
        """
        self.constrains = []
        self.__cons = self.__create_constrains()


