import numpy as np

from .base import VersionSpace
from .minimizer import Minimizer
from ..convexbody.objects import ConvexBody, UnitBall
import scipy.optimize

class SVMMinimizer(Minimizer):
    def get_initial_constrains(self):
        return []
        #return [
        #    {
        #        'type': 'eq',
        #        'fun': lambda x: 1 + x[0] - np.sum(x[1:] ** 2),
        #        'jac': lambda x: np.hstack([[1], -2 * x[1:]])
        #    }
        #]


class SVMVersionSpace(ConvexBody, VersionSpace):
    def __init__(self, dim):
        ConvexBody.__init__(self)
        VersionSpace.__init__(self, SVMMinimizer(dim))

        self._dim = int(dim)
        self.__ball = UnitBall(self._dim)

    def is_inside(self, points):
        return np.logical_and(self.inequality_constrain.check(points), self.__ball.is_inside(points))

    def intersection(self, line):
        r1, r2 = [], []
        if not self.inequality_constrain.is_empty():
            matrix = np.asarray(self.inequality_constrain.matrix)
            den = matrix.dot(line.direction)
            r = (self.inequality_constrain.vector - matrix.dot(line.center)) / den
            r1.append(r[den < 0])
            r2.append(r[den > 0])

        segment = self.__ball.intersection(line)
        r1.append(segment.left_limit)
        r2.append(segment.right_limit)

        return line.get_segment(np.max(np.hstack(r1)), np.min(np.hstack(r2)))

    def get_point(self):
        """
        Finds an interior point to the current search space through an optimization routine.
        :return: point inside search space
        """
        q0 = np.zeros(self.dim)
        if self.inequality_constrain.is_empty():
            return q0

        #x0 = np.hstack([-1., q0])
        #point = self.minimizer(x0)
        m = self.inequality_constrain.matrix.shape[0]
        res = scipy.optimize.linprog(
            c=np.array([1.0] + [0.0] * self.dim),
            A_ub=np.hstack([-np.ones(m).reshape(-1,1), self.inequality_constrain.matrix]),
            b_ub=np.zeros(m),
            bounds=[(-1, 1)] + [(-1, 1)]*self.dim,
            options={"disp": False}
        )

        point = res.x[1:]
        point = 0.5*point/np.linalg.norm(point)
        if not self.is_inside(point):
            # print(res)
            raise RuntimeError(
                'Point outside search space!\n'
                'Inequality constrain: {0}\n'
                'Ball constrain: {1}\n'
                .format(self.inequality_constrain.check(point),
                        self.__ball.is_inside(point))
            )

        return 0.5*point/np.linalg.norm(point)


