import numpy as np
import scipy.spatial

from explore_by_example.utils import assert_positive


class ConvexHull:
    """
    This class represents the convex hull of a finite number of points.
    """
    def __init__(self, points, tol=1e-12):
        assert_positive(tol, 'tol')

        points = np.atleast_2d(points)
        self.hull = scipy.spatial.ConvexHull(points, incremental=True)
        self.tol = tol

    @property
    def npoints(self):
        """
        :return: number of points added to the convex hull
        """
        return self.hull.npoints

    @property
    def vertices(self):
        """
        :return: numpy array containing all the vertices of the convex hull
        """
        return self.hull.points[self.hull.vertices]

    def equations_defining_vertex(self, vertex_id):
        """
        :return: equations of all hyperplanes going through a vertex. If point is not a vertex, [] is returned.
        """
        return self.hull.equations[[vertex_id in s for s in self.hull.simplices]]

    def add_points(self, points):
        """
        Adds the points to the convex hull
        :param points: list of points to be added to the convex hull
        """
        points = np.atleast_2d(points)
        self.hull.add_points(points)

    def is_inside(self, points):
        """
        Computes whether each data point is inside the convex hull or not
        """
        points = np.atleast_2d(points)
        return (np.max(points.dot(self.hull.equations[:, :-1].T) + self.hull.equations[:, -1], axis=1) <= self.tol)


class ConvexCone:
    """
    Let H be a convex hull and V any point outside H. The convex cone C = cone(H, V) is a conical collection of points
    which are guaranteed to be outside H.
    """
    def __init__(self, points, vertex, tol=1e-12):
        assert_positive(tol, 'tol')

        self.vertex = np.asarray(vertex)
        self.convex_hull = ConvexHull(np.vstack([points, self.vertex]), tol)

        self.vertex_id = self.convex_hull.npoints - 1
        self.equations = self.convex_hull.equations_defining_vertex(self.vertex_id)

        self.tol = -tol

    def add_points_to_hull(self, points):
        """
        Add new points to the positive region, updating the negative cone equations
        """
        self.convex_hull.add_points(points)  # TODO: can we avoid modifying the entire convex hull every time?
        self.equations = self.convex_hull.equations_defining_vertex(self.vertex_id)

        if len(self.equations) == 0:
            raise RuntimeError("Found a positive point inside a negative cone!")

    def is_inside(self, points):
        """
        Computes whether a point is inside the negative cone or not
        """
        points = np.atleast_2d(points)

        return (np.min(points.dot(self.equations[:, :-1].T) + self.equations[:, -1], axis=1) >= self.tol)

