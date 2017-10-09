import numpy as np

from .base import ConvexBody
from .constrain import get_constrain


class Polytope(ConvexBody):
    """
    A polytope is a convex set defined by three sets of equations:

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
        super().__init__()

        self.equality_constrain = get_constrain('equality', b, A)
        self.inequality_constrain = get_constrain('inequality', q, M)
        self.lower_constrain = get_constrain('lower', l)
        self.upper_constrain = get_constrain('upper', h)

        self._dim = self.__get_dimension()

    def __get_all_constrains(self):
        return [
            self.equality_constrain,
            self.inequality_constrain,
            self.lower_constrain,
            self.upper_constrain
        ]

    def __get_dimension(self):
        all_constrains = self.__get_all_constrains()

        dimension_list = [constrain.shape[-1] for constrain in all_constrains if not constrain.is_empty()]

        number_of_unique_dimensions = len(set(dimension_list))

        if number_of_unique_dimensions > 1:
            raise ValueError("Incompatible dimensions.")  # all constrains must have the same dimension
        elif number_of_unique_dimensions == 1:
            return int(dimension_list[0])  # return dimension
        else:
            return 0  # all constrains are empty

    def _compute_projection_matrix(self):
        if self.equality_constrain.is_empty():
            return None

        matrix = self.equality_constrain.matrix
        return np.eye(self.dim) - matrix.T.dot(np.linalg.inv(matrix.dot(matrix.T))).dot(matrix)

    def is_inside(self, points):
        """
        Check whether each point satisfy all constrains.
        :param points:
        :return:
        """
        check_equality = self.equality_constrain.check(points)
        check_inequality = self.inequality_constrain.check(points)
        check_lower = self.lower_constrain.check(points)
        check_upper = self.upper_constrain.check(points)

        check_equality_and_inequality = np.logical_and(check_equality, check_inequality)
        check_lower_and_upper = np.logical_and(check_lower, check_upper)

        return np.logical_and(check_equality_and_inequality, check_lower_and_upper)

    def __check_line(self, line):
        """
        Checks whether a given line intersects the polytope or not. In order to verify this claim, we use the following result:

        Result: Let x + t * u be a line going through x and with direction u. Then, the line is contained in the polytope's
        subspace iff Ax = b and Au = 0.

        Proof sketch: A(x + tu) = b for all t  <=>  Ax = b (setting t = 0) and Au = 0 (setting t != 0)
        """
        if self.equality_constrain.is_empty():
            return

        check_center = self.equality_constrain.check(line.center)
        check_direction = np.allclose(self.equality_constrain.matrix.dot(line.direction), 0)
        if not (check_center and check_direction):
            raise ValueError("Line is not contained in polytope.")

    def intersection(self, line):
        self.__check_line(line)

        r1, r2 = [], []
        if not self.lower_constrain.is_empty():
            lbounds = (self.lower_constrain.vector - line.center) / line.direction
            r1.append(lbounds[line.direction > 0])
            r2.append(lbounds[line.direction < 0])

        if not self.upper_constrain.is_empty():
            rbounds = (self.upper_constrain.vector - line.center) / line.direction
            r1.append(rbounds[line.direction < 0])
            r2.append(rbounds[line.direction > 0])

        if not self.inequality_constrain.is_empty():
            matrix = np.asarray(self.inequality_constrain.matrix)
            den = matrix.dot(line.direction)
            r = (self.inequality_constrain.vector - matrix.dot(line.center)) / den
            r1.append(r[den < 0])
            r2.append(r[den > 0])

        if len(r1) == 0 or len(r2) == 0:
            raise RuntimeError("Line does not intersect polytope.")

        t1 = np.max(np.hstack(r1))
        t2 = np.min(np.hstack(r2))

        if t1 >= t2:
            raise RuntimeError("Line does not intersect polytope.")

        return line.get_segment(t1, t2)
