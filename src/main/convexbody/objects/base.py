class ConvexBody:
    """
    This class represents an abstract definition of a convex subset of euclidean space.
    """
    def __init__(self, dim: int):
        if(dim <= 0):
            raise ValueError("Dimension must be a positive number.")
        self._dim = dim
        self._volume = None
        self.__projection_matrix = None

    @property
    def dim(self):
        return self._dim

    def _compute_projection_matrix(self):
        return None

    @property
    def projection_matrix(self):
        if self.__projection_matrix is None:
            self.__projection_matrix = self._compute_projection_matrix()
        return self.__projection_matrix

    def is_inside(self, points):
        """
        Tells if a point (or an array of points) x is inside or not the convex body
        """
        raise NotImplementedError

    def intersection(self, line):
        raise NotImplementedError


