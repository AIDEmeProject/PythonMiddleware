import numpy as np


class Ellipsoid:
    """
    Ellipsoid equation: {z : (z - x)^T P^-1 (z - x) <= 1}
    where P is SPD and z is the center.
    """

    def __init__(self, dim):
        """
        Inputs:
            center: ellipsoid's center x
            matrix: SPD matrix P^-1
        """
        if dim <= 0:
            raise ValueError("dim must be a positive integer, but received " + dim)

        self.x = np.zeros(dim)
        self.n = dim

        self.L = np.eye(self.n)
        self.D = np.ones(self.n)
        self.P = np.eye(dim)

    def __repr__(self):
        return str(self.x) + "\n" + str(self.P)


    def get_endpoints(self, factor=1.0):
        yield self.x

        eig, P = np.linalg.eigh(self.P)

        for i in range(self.n):
            a = P[:, i] * np.sqrt(eig[i])
            yield self.x + factor * a
            yield self.x - factor * a

    def update(self, b, g):
        n = self.n

        a_hat = self.L.T.dot(g)  # a_hat = L^T g
        gamma = np.sqrt(np.square(a_hat).dot(self.D))  # gamma = sqrt(a_hat^T D a_hat)
        alpha = (g.dot(self.x) - b) / gamma

        p = self.D * a_hat / gamma
        Pg = self.L.dot(p)
        # update center
        tau = (1 + n * alpha) / (n + 1)
        self.x -= tau * Pg

        # update LDL^T decomposition
        sigma = 2 * tau / (alpha + 1)
        delta = (1 - alpha * alpha) * (n * n / (n * n - 1.))
        beta = self.update_diagonal(p, sigma, delta)
        self.update_cholesky_factor(p, beta)

        # update P
        self.P -= sigma * Pg.reshape(-1, 1).dot(Pg.reshape(1, -1))
        self.P *= delta

    def update_diagonal(self, p, sigma, delta):
        """ LDL^T for D - sigma pp^T """
        t = np.empty(len(self.D) + 1)
        t[-1] = 1 - sigma * p.dot(p / self.D)
        t[:-1] = sigma * np.square(p) / self.D
        t = np.cumsum(t[::-1])[::-1]

        self.D *= t[1:]
        beta = -sigma * p / self.D
        self.D /= t[:-1]
        self.D *= delta

        return beta

    def update_cholesky_factor(self, p, b):
        """ Computes the product L K, with L and K lower triangular with unit diagonal, and Kij = pi * bj for i > j """
        v = self.L * p.reshape(1, -1)
        v = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]

        self.L[:, :-1] += v[:, 1:] * b[:-1].reshape(1, -1)

    def fit(self, pol):
        converge = False
        while not converge:
            converge = True

            for end_point in self.get_endpoints(factor=1.0 / (self.n + 1)):
                bad_constrain = pol.get_separating_oracle(end_point)
                if bad_constrain is not None:
                    converge = False
                    b, g = bad_constrain
                    self.update(b,g)
                    break
