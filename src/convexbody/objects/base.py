class ConvexBody(object):
    """
    This class represents an abstract definition of a convex subset of euclidean space.
    """
    def __init__(self):
        self._dim = None
        self._volume = None
        self.__projection_matrix = None

    @property
    def dim(self):
        return self._dim

    @property
    def encloser(self):
        return None

    def _compute_projection_matrix(self):
        return None

    @property
    def projection_matrix(self):
        if self.__projection_matrix is None:
            self.__projection_matrix = self._compute_projection_matrix()
        return self.__projection_matrix

    def _compute_volume(self):
        raise NotImplementedError

    @property
    def volume(self):
        # returns the volume of K
        if self._volume is None:
            self._volume = self._compute_volume()
        return self._volume

    def is_inside(self, points):
        """
        Tells if a point (or an array of points) x is inside or not the convex body
        """
        raise NotImplementedError

    def sample(self, n):
        # samples n points uniformly (or at least approximately) from K
        raise NotImplementedError

    def intersection(self, line):
        if not self.is_inside(line.center):
            raise RuntimeError("Initial point must be inside convex body.")

        segment = self.encloser.intersection(line)
        l1, r2 = segment.left_limit, segment.right_limit
        r1 = l2 = 0.0

        while (r2 - l1) / (r1 - l2) > 1.1:
            mid1 = (l1 + r1) / 2.0
            if self.is_inside(line.center + mid1 * line.direction):
                r1 = mid1
            else:
                l1 = mid1

            mid2 = (l2 + r2) / 2.0
            if self.is_inside(line.center + mid2 * line.direction):
                r2 = mid2
            else:
                l2 = mid2

        return line.get_segment(r1, l2)
