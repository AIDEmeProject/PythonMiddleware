#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import scipy.optimize
import version_space_helper

if TYPE_CHECKING:
    from aideme.utils import HyperPlane


class BoundedPolyhedralCone:
    """
    This class represents the convex set of vector "w" satisfying:

                    A w <= 0   AND   ||w|| < =1

    i.e., it is the intersection of a polyhedral cone and the unit ball B(0, 1)
    """
    def __init__(self, A: np.ndarray, use_cython: bool = True):
        self.A = A
        self.intersection = self._get_extremes_cython if use_cython else self._get_extremes_python

    @property
    def dim(self) -> int:
        """
        :return: version space dimension
        """
        return self.A.shape[1]

    def is_inside_polytope(self, X: np.ndarray) -> bool:
        """
        :param X: data point (also works for a matrix of points)
        :return: whether X satisfies the polytope equations
        """
        return np.dot(X, self.A.T).max(axis=-1) < 0

    def get_interior_point(self) -> np.ndarray:
        """
        Finds an interior point to the version space by solving the following Linear Programming problem:

            minimize s,  s.t.  |w_i| < 1  AND a_i^T w < s

        Raises an error in case the polytope is degenerate (only 0 vector).

        :return: point inside version space
        """
        n, dim = self.A.shape

        res = scipy.optimize.linprog(
            c=np.array([1.0] + [0.0] * dim),
            A_ub=np.hstack([-np.ones(shape=(n, 1)), self.A]),
            b_ub=np.zeros(n),
            bounds=[(None, None)] + [(-1, 1)] * dim,  # type: ignore
            method='revised simplex'
        )

        # if optimization failed, raise error
        if not res.success or res.x[0] >= 0:
            print(res)
            raise RuntimeError("Linear programming failed! Check constrains for degeneracy of Version Space.")

        # return normalized point
        point = res.x[1:]
        return (0.99 / np.linalg.norm(point)) * point

    def get_separating_oracle(self, point: np.ndarray) -> Optional[HyperPlane]:
        """
        For any given point, find a half-space H(b, g) = {x: g^T x < b} separating the point from the version_space, i.e.:

                version_space contained in H(b, g)   AND   point not in H(b, g)

        :param point: data point
        :return: 'b' and 'g' values above if they exist; None otherwise
        """
        if point.dot(point) >= 1:
            return 1, point / np.linalg.norm(point)

        for a in self.A:
            if np.dot(a, point) >= 0:
                return 0, a

        return None

    def _get_extremes_cython(self, center, direction):
        """
        Finds the intersection between the version space and a straight line. Cython implementation.

        :param center: point on the line
        :param direction: director vector of line. Does not need to be normalized.
        :return: t1 and t2 such that 'center + t * direction' are extremes of the line segment determined by the intersection
        """
        return version_space_helper.compute_version_space_intersection(self.A, center, direction)

    def _get_extremes_python(self, center, direction):
        """
        Finds the intersection between the version space and a straight line. Python + numpy implementation.

        :param center: point on the line
        :param direction: director vector of line. Does not need to be normalized.
        :return: t1 and t2 such that 'center + t * direction' are extremes of the line segment determined by the intersection
        """
        lower_pol, upper_pol = self.__get_polytope_extremes(center, direction)
        lower_ball, upper_ball = self.__get_ball_extremes(center, direction)
        lower_extreme, upper_extreme = max(lower_pol, lower_ball), min(upper_pol, upper_ball)

        if lower_extreme >= upper_extreme:
            raise RuntimeError("Line does not intersect convex body.")

        return lower_extreme, upper_extreme

    def __get_polytope_extremes(self, center, direction):
        num = self.A.dot(center)
        den = self.A.dot(direction)
        extremes = -num / den

        lower = extremes[den < 0]
        lower_extreme = lower.max() if len(lower) > 0 else -np.inf

        upper = extremes[den > 0]
        upper_extreme = upper.min() if len(upper) > 0 else np.inf

        return lower_extreme, upper_extreme

    @staticmethod
    def __get_ball_extremes(center, direction):
        a, b, c = direction.dot(direction), center.dot(direction), center.dot(center) - 1

        delta = b * b - a * c

        if delta <= 0:
            raise RuntimeError("Center outside unit ball.")

        sq_delta = np.sqrt(delta)
        return (-b - sq_delta) / a, (-b + sq_delta) / a
