import numpy as np

class Ellipsoid:
    """
    Ellipsoid equation: {z : (z - x)^T P^-1 (z - x) <= 1}
    where P is SPD and z is the center.
    """

    def __init__(self, x, P):
        """
        Inputs:
            center: ellipsoid's center x
            matrix: SPD matrix P^-1
        """
        self.x = np.atleast_1d(x).ravel()
        self.P = np.atleast_2d(P)
        self.n = len(self.x)
        assert self.P.shape == (self.n, self.n)

    def __repr__(self):
        return str(self.x) + "\n" + str(self.P)

    def get_endpoints(self, factor=1.0):
        eig, P = np.linalg.eigh(self.P)
        for i in range(self.n):
            a = P[:, i] * np.sqrt(eig[i])
            yield self.x + factor * a
            yield self.x - factor * a

    def update(self, g):
        Pg = self.P.dot(g)
        Pg_norm = Pg / np.sqrt(g.dot(Pg))

        n = self.n
        self.x = self.x - Pg_norm / (n + 1)
        self.P = (n ** 2 / (n ** 2 - 1)) * (self.P - (2.0 / (n + 1)) * Pg_norm.reshape(-1, 1).dot(Pg_norm.reshape(1, -1)))

    def weak_ellipsoid(self, pol):
        converge = False
        while not converge:
            converge = True

            bad_constrain = pol.get_separating_oracle(self.x)
            if bad_constrain is not None:
                converge = False
                self.update(bad_constrain)
            else:
                for end_point in self.get_endpoints(factor=1.0 / (self.n + 1)):
                    bad_constrain = pol.get_separating_oracle(end_point)
                    if bad_constrain is not None:
                        converge = False
                        self.update(bad_constrain)
                        break
