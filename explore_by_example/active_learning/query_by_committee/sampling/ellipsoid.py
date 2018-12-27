import numpy as np


class Ellipsoid:
    """
    Ellipsoid equation: {z : (z - x)^T P^-1 (z - x) <= 1}
    where P is SPD and z is the center.
    """

    def __init__(self, body):
        self.n = body.dim
        self.body = body

        self.x = np.zeros(self.n)

        self.L = np.eye(self.n)
        self.D = np.ones(self.n)
        self.P = np.eye(self.n)

        self.__fit()

    def __repr__(self):
        return str(self.x) + "\n" + str(self.P)


    def get_endpoints(self, factor=1.0):
        yield self.x

        eig, P = np.linalg.eigh(self.P)

        for i in range(self.n):
            a = P[:, i] * np.sqrt(eig[i])
            yield self.x + factor * a
            yield self.x - factor * a

    def __update(self, b, g):
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
        beta = self.__update_diagonal(p, sigma, delta)
        self.__update_cholesky_factor(p, beta)

        # update P
        self.P -= sigma * Pg.reshape(-1, 1).dot(Pg.reshape(1, -1))
        self.P *= delta

    def __update_diagonal(self, p, sigma, delta):
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

    def __update_cholesky_factor(self, p, b):
        """ Computes the product L K, with L and K lower triangular with unit diagonal, and Kij = pi * bj for i > j """
        v = self.L * p.reshape(1, -1)
        v = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]

        self.L[:, :-1] += v[:, 1:] * b[:-1].reshape(1, -1)

    def __fit(self):
        converge = False

        while not converge:
            converge = True

            for end_point in self.get_endpoints(factor=1.0 / (self.n + 1)):
                bad_constrain = self.body.get_separating_oracle(end_point)
                if bad_constrain is not None:
                    converge = False
                    b, g = bad_constrain
                    self.__update(b, g)
                    break
