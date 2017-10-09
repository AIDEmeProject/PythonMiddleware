from functools import partial

import numpy as np
import scipy.optimize

from .base import VersionSpace
from ..convexbody.objects import Polytope
from ..convexbody.objects.constrain import InequalityConstrain


class AppendableInequalityConstrain(InequalityConstrain):
    def __init__(self):
        self._vector = []
        self._matrix = []

    def __len__(self):
        return len(self._vector)

    @property
    def vector(self):
        return np.array(self._vector)

    @property
    def matrix(self):
        return np.array(self._matrix)

    @property
    def shape(self):
        if self.is_empty():
            return (0,)
        else:
            return (len(self._matrix), len(self._matrix[0]))

    def _check_sizes(self, point):
        if not self.is_empty() and point.shape != (self.shape[1], ):
            raise ValueError("Bad point dimension: obtained {0}, expected {1}".format(point.shape, self.shape))

    def clear(self):
        self._vector = []
        self._matrix = []

    def append(self, point, value):
        value = float(value)
        point = np.asarray(point).ravel()
        self._check_sizes(point)

        self._vector.append(value)
        self._matrix.append(point)

    def check(self, points):
        return True if self.is_empty() else super().check(points)


class Minimizer:
    def __init__(self, dim):
        self.__dim = int(dim)
        self.__constrains_list = self.get_initial_constrains()

    def __call__(self, x0):
        result = scipy.optimize.minimize(
            x0=x0,
            fun=lambda x: x[0],
            jac=lambda x: np.array([1.0] + [0.0] * self.dim),
            constraints=self.__constrains_list,
            method="SLSQP"
        )

        if result.x[0] >= 0:
            raise RuntimeError("Optimization failed! Result = {0}".format(result.x))
        return result.x[1:]

    @property
    def dim(self):
        return self.__dim

    def clear(self):
        self.__constrains_list = self.get_initial_constrains()

    def get_initial_constrains(self):
        constrains = [
            {
                'type': 'ineq',
                'fun': lambda x, a=i: x[0] + x[a + 1],
                'jac': lambda x, a=i: np.hstack([1.0, np.eye(self.dim)[a]])
            } for i in range(self.dim)
        ]

        constrains.append(
            {
                'type': 'eq',
                'fun': lambda x: np.sum(x[1:]) - 1.0,
                'jac': lambda x: np.array([0] + [1] * self.dim)
            }
        )

        return constrains

    def append(self, vector):
        self.__constrains_list.append(
            {
                'type': 'ineq',
                'fun': lambda s: s[0] - np.dot(vector, s[1:]),
                'jac': lambda s: np.hstack([1.0, -vector])
            }
        )


class ActboostPolytope(Polytope, VersionSpace):
    def __init__(self, dim):
        if int(dim) <= 0:
            raise ValueError("Dimension must be positive integer.")

        VersionSpace.__init__(self)
        Polytope.__init__(self, A=[1]*dim, b=1, l=[0]*dim)

        self.inequality_constrain = AppendableInequalityConstrain()
        self._minimizer = Minimizer(dim)

    def get_point(self):
        """
        Finds an interior point to the current search space through an optimization routine.
        :return: point inside search space
        """
        q0 = np.ones(self.dim, dtype=np.float64) / self.dim
        if self.inequality_constrain.is_empty():
            return q0

        s0 = np.max(self.inequality_constrain.matrix * q0) + 1
        x0 = np.hstack([s0, q0])

        point = self._minimizer(x0)

        if not self.is_inside(point):
            raise RuntimeError(
                'Point outside search space!\n'
                'Equality constrain: {0}\n'
                'Inequality constrain: {1}\n'
                'Lower constrain: {2}'
                .format(self.equality_constrain.check(point), 
                        self.inequality_constrain.check(point), 
                        self.lower_constrain.check(point))
            )
        return point

    def clear(self):
        super().clear()
        self.inequality_constrain.clear()
        self._minimizer.clear()

    def update(self, point, label):
        constrain_vector = -label * point
        self.inequality_constrain.append(constrain_vector, 0.0)
        self._minimizer.append(constrain_vector)
