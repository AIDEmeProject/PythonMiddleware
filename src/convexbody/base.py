class ConvexBody(object):
    """
    This class represents an abstract definition of a convex subset of euclidean space.
    """

    def is_inside(self, x):
        """
        Tells if a point (or an array of points) x is inside or not the convex body
        """
        raise NotImplementedError
        
    def volume(self):
        # returns the volume of K
        raise NotImplementedError
        
    def sample(self, n):
        # samples n points uniformly (or at least approximately) from K
        raise NotImplementedError
