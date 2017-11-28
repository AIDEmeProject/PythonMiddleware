import numpy as np

from .base import VersionSpace
from ..convexbody.objects import Polytope


class ActboostPolytope(Polytope, VersionSpace):
    def __init__(self, dim):
        self._dim = int(dim)
        VersionSpace.__init__(self)
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

