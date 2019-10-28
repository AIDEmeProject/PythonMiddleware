import numpy as np


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
        return [self.left, self.right]

    def add_points(self, points):
        """
        Adds the points to the convex hull
        :param points: list of points to be added to the convex hull
        """
        self.left = min(self.left, np.min(points))
        self.right = max(self.right, np.max(points))

    def is_inside(self, points):
        """
        Computes whether each data point is inside the convex hull or not
        """
        points = np.asarray(points).reshape(-1)
        return np.logical_and(points >= self.left, points <= self.right)


class OneDimensionalConvexCone:
    """
    One dimensional version of ConvexCone class.
    """
    def __init__(self, vertices, vertex):
        self.vertex = float(vertex)
        self.hull = OneDimensionalConvexHull(vertices)

        if self.hull.is_inside(self.vertex):
            raise RuntimeError("Vertex of negative cone cannot be inside positive region.")

        self.right_cone = self.vertex > self.hull.right

    def add_points_to_hull(self, points):
        """
        Add new points to the positive region, updating the negative cone equations
        """
        self.hull.add_points(points)  # TODO: can we avoid modifying the entire convex hull every time?

        if self.hull.is_inside(self.vertex):
            raise RuntimeError("Found negative point inside positive region.")

    def is_inside(self, points):
        """
        Computes whether a point is inside the negative cone or not
        """
        points = np.asarray(points).reshape(-1)
        return points >= self.vertex if self.right_cone else points <= self.vertex
