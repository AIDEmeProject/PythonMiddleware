import numpy as np
from .appendable_constrain import AppendableInequalityConstrain
from ..convexbody.objects import ConvexBody, UnitBall


class SVMVersionSpace(ConvexBody):
    def __init__(self, dim):
        ConvexBody.__init__(self)
        self.__inequality_constrain = AppendableInequalityConstrain(dim+1)  # add one to account for bias
        self.__ball = UnitBall(dim+1)

    def is_inside(self, points):
        return np.logical_and(self.__inequality_constrain.check(points), self.__ball.is_inside(points))

    def intersection(self, line):
        r1, r2 = [], []
        if not self.__inequality_constrain.is_empty():
            matrix = self.__inequality_constrain.matrix
            den = matrix.dot(line.direction)
            r = (self.__inequality_constrain.vector - matrix.dot(line.center)) / den
            r1.append(r[den < 0])
            r2.append(r[den > 0])

        segment = self.__ball.intersection(line)
        r1.append(segment.left_limit)
        r2.append(segment.right_limit)

        return line.get_segment(np.max(np.hstack(r1)), np.min(np.hstack(r2)))

    def get_point(self):
        return self.__inequality_constrain.get_point()

    def clear(self):
        self.__inequality_constrain.clear()

    def update(self, point, label):
        point = np.hstack([1, point.ravel()])  # add bias component
        constrain_vector = -label * point  # constrain = -y_i (1, x_i)
        self.__inequality_constrain.append(constrain_vector, 0)

