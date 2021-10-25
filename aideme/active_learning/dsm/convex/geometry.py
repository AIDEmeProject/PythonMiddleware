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
import numpy as np

from aideme.utils import assert_positive
from .convex import ConvexHull, ConvexCone
from .one_dim_convex import OneDimensionalConvexHull, OneDimensionalConvexCone


class PositiveRegion:
    def __init__(self, tol=1e-12):
        assert_positive(tol, 'tol')

        self._hull = None
        self._cache = Cache()
        self._tol = tol

    @property
    def is_built(self):
        return self._hull is not None

    @property
    def vertices(self):
        return self._cache.data if self._hull is None else self._hull.vertices

    def clear(self):
        self._hull = None
        self._cache.clear()

    def is_inside(self, X):
        if not self.is_built:
            return np.full(len(X), False)

        return self._hull.is_inside(X)

    def update(self, pos_points):
        if self.is_built:
            self._hull.add_points(pos_points)
            return

        self._cache.update(pos_points)

        if self._cache.size > self._cache.dim:
            self._hull = self.__build_convex_hull()
            self._cache.clear()

    def __build_convex_hull(self):
        if self._cache.dim > 1:
            return ConvexHull(self._cache.data, self._tol)

        return OneDimensionalConvexHull(self._cache.data)


class NegativeCone:
    def __init__(self, vertex, tol=1e-12):
        assert_positive(tol, 'tol')

        self._vertex = np.asarray(vertex).reshape(-1)
        self._cache = Cache()
        self._cone = None
        self._tol = tol

    @property
    def is_built(self):
        return self._cone is not None

    def clear(self):
        self._cone = None
        self._cache.clear()

    def is_inside(self, X):
        if not self.is_built:
            return np.full(len(X), False)

        return self._cone.is_inside(X)

    def add_points_to_hull(self, points):
        points = np.atleast_2d(points)

        if points.shape[1] != len(self._vertex):
            raise ValueError("Bad input dimension: expected {0}, got {1}".format(len(self._vertex), points.shape[1]))

        if self._cone is not None:
            self._cone.add_points_to_hull(points)
            return

        self._cache.update(points)

        if self._cache.size >= self._cache.dim:
            self._cone = self.__build_convex_cone()
            self._cache.clear()

    def __build_convex_cone(self):
        if len(self._vertex) > 1:
            return ConvexCone(self._cache.data, self._vertex, tol=self._tol)
        else:
            return OneDimensionalConvexCone(self._cache.data, self._vertex)


class Cache:
    def __init__(self):
        self.data = None

    @property
    def size(self):
        return 0 if self.data is None else self.data.shape[0]

    @property
    def dim(self):
        return -1 if self.data is None else self.data.shape[1]

    def clear(self):
        self.data = None

    def update(self, X):
        X = np.atleast_2d(X)

        if self.data is None:
            self.data = X.copy()
        else:
            self.data = np.vstack([self.data, X])
