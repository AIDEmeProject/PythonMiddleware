import numpy as np

from .base import VersionSpace
from .minimizer import Minimizer
from ..convexbody.objects import Polytope


class BoostingMinimizer(Minimizer):
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


class ActboostPolytope(Polytope, VersionSpace):
    def __init__(self, dim):
        self._dim = int(dim)
        VersionSpace.__init__(self, BoostingMinimizer(self._dim))
        Polytope.__init__(self, A=[1] * self._dim, b=1, l=[0] * self._dim)

    def get_point(self):
        """
        Finds an interior point to the current search space through an optimization routine.
        :return: point inside search space
        """
        q0 = np.ones(self._dim, dtype=np.float64) / self._dim
        if self.inequality_constrain.is_empty():
            return q0

        s0 = max(np.max(self.inequality_constrain.matrix.dot(q0)), 1. / self._dim) + 1.
        x0 = np.hstack([s0, q0])

        point = self.minimizer(x0)

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

