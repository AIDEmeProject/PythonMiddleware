#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.

from typing import Generator

import numpy as np
import scipy.linalg

from aideme.utils import assert_positive_integer


class Ellipsoid:
    def __init__(self, dim: int, compute_scale_matrix: bool = False):
        assert_positive_integer(dim, 'dim')

        self.dim = dim

        self.center = np.zeros(self.dim)
        self.scale = np.eye(self.dim) if compute_scale_matrix else None

        self.L = np.eye(self.dim)
        self.D = np.ones(self.dim)

    def extremes(self) -> Generator[np.ndarray, None, None]:
        yield self.center

        scale = self.scale if self.scale is not None else self.L @ np.diag(self.D) @ self.L.T
        eig, P = scipy.linalg.eigh(scale + 1e-12 * np.eye(self.dim))  # add small perturbation to diagonal to counter numerical errors

        for i in range(len(eig)):
            if eig[i] <= 0:
                raise RuntimeError("Found non-positive eigenvalue: {}".format(eig[i]))

            factor = np.sqrt(eig[i]) / (self.dim + 1)
            direction = factor * P[:, i]

            yield self.center + direction
            yield self.center - direction

    def compute_alpha(self, G: np.ndarray) -> np.ndarray:
        a_hat = G @ self.L
        gamma = np.sqrt(np.square(a_hat).dot(self.D))
        return G.dot(self.center) / gamma

    def compute_alpha_single(self, bias: float, g: np.ndarray) -> float:
        a_hat = self.L.T.dot(g)
        gamma = np.sqrt(np.square(a_hat).dot(self.D))
        return (g.dot(self.center) - bias) / gamma

    def cut(self, bias: float, g: np.ndarray) -> bool:
        a_hat = self.L.T.dot(g)
        gamma = np.sqrt(np.square(a_hat).dot(self.D))
        alpha = (g.dot(self.center) - bias) / gamma

        if alpha >= 1:
            raise RuntimeError("Invalid hyperplane: ellipsoid is contained in its positive semi-space (expected the negative one)")

        # shallow cut
        if alpha <= -1.0 / self.dim:
            return False

        p = self.D * a_hat / gamma
        Pg = self.L.dot(p)

        # update center
        tau = (1 + self.dim * alpha) / (self.dim + 1)
        self.center -= tau * Pg

        # update LDL^T
        sigma = 2 * tau / (alpha + 1)
        delta = (1. - alpha * alpha) * ((self.dim * self.dim) / (self.dim * self.dim - 1.))

        beta = self._update_diagonal(p, sigma, delta)
        self._update_cholesky_factor(p, beta)

        # update P
        if self.scale is not None:
            self.scale -= sigma * (Pg.reshape(-1, 1) @ Pg.reshape(1, -1))
            self.scale *= delta

        return True

    def _update_diagonal(self, p: np.ndarray, sigma: float, delta: float) -> np.ndarray:
        t = np.empty(self.dim + 1)
        t[-1] = 1 - sigma * p.dot(p / self.D)
        t[:-1] = sigma * np.square(p) / self.D
        t = np.cumsum(t[::-1])[::-1]

        self.D *= t[1:]
        beta = -sigma * p / self.D
        self.D /= t[:-1]
        self.D *= delta

        return beta

    def _update_cholesky_factor(self, p: np.ndarray, beta: np.ndarray) -> None:
        v = self.L * p.reshape(1, -1)
        v = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]

        self.L[:, :-1] += v[:, 1:] * beta[:-1].reshape(1, -1)
