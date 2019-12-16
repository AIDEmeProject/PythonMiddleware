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

import numpy as np
from sklearn.utils import column_or_1d

from .convex import ConvexError


def assert_float_or_column_or_1d(points):
    if isinstance(points, (int, float)):
        return points

    return column_or_1d(points, warn=False)


class OneDimensionalConvexHull:
    """
    This class represents the convex hull of a finite number of points in 1D space.
    """
    def __init__(self, points):
        self.left = np.min(points)
        self.right = np.max(points)

    @property
    def vertices(self):
        """
        :return: numpy array containing all the vertices of the convex hull
        """
        return np.array([self.left, self.right]).reshape(-1, 1)

    def add_points(self, points):
        """
        Adds the points to the convex hull
        :param points: list of points to be added to the convex hull
        """
        points = assert_float_or_column_or_1d(points)

        self.left = min(self.left, np.min(points))
        self.right = max(self.right, np.max(points))

    def is_inside(self, points):
        """
        Computes whether each data point is inside the convex hull or not
        """
        points = assert_float_or_column_or_1d(points)
        return np.logical_and(points >= self.left, points <= self.right)


class OneDimensionalConvexCone:
    """
    One dimensional version of ConvexCone class.
    """
    def __init__(self, vertices, vertex):
        self.vertex = float(vertex)
        self.hull = OneDimensionalConvexHull(vertices)

        if self.hull.is_inside(self.vertex):
            raise ConvexError

        self.right_cone = self.vertex > self.hull.right

    def add_points_to_hull(self, points):
        """
        Add new points to the positive region, updating the negative cone equations
        """
        self.hull.add_points(points)  # TODO: can we avoid modifying the entire convex hull every time?

        if self.hull.is_inside(self.vertex):
            raise ConvexError

    def is_inside(self, points):
        """
        Computes whether a point is inside the negative cone or not
        """
        points = assert_float_or_column_or_1d(points)
        return points >= self.vertex if self.right_cone else points <= self.vertex
