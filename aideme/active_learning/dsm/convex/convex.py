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
import scipy.spatial

from aideme.utils import assert_positive


class ConvexHull:
    """
    This class represents the convex hull of a finite number of points.
    """
    def __init__(self, points, tol=1e-12):
        assert_positive(tol, 'tol')
        points = np.atleast_2d(points)
        self.__set_params(points, tol)

    def __getstate__(self):
        """
        Since scipy.spatial.ConvexHull objects are not picklable, we choose to serialize the minimum amount of data
        necessary for properly reconstructing the object. More precisely, we pickle both the "tol" parameter and the set
        of "vertices" composing the convex hull.
        """
        return {
            'vertices': self.vertices,
            'tol': self.tol
        }

    def __setstate__(self, state):
        """
        When unpickling, the convex hull must be rebuilt from scratch since self.hull is not pickable.
        """
        self.__set_params(state['vertices'], state['tol'])

    def __set_params(self, points: np.ndarray, tol: float):
        self.hull = scipy.spatial.ConvexHull(points, incremental=True)
        self.tol = tol

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
        return np.max(points.dot(self.hull.equations[:, :-1].T) + self.hull.equations[:, -1], axis=1) <= self.tol


class ConvexCone:
    """
    Let H be a convex hull and V any point outside H. The convex cone C = cone(H, V) is a conical collection of points
    which are guaranteed to be outside H.
    """
    def __init__(self, points, vertex, tol=1e-12):
        assert_positive(tol, 'tol')
        points = np.atleast_2d(points)

        self.vertex = np.asarray(vertex)
        self.convex_hull = ConvexHull(np.vstack([self.vertex, points]), tol)

        self.__update_cone_equations()

        self.tol = -tol

    def add_points_to_hull(self, points):
        """
        Add new points to the positive region, updating the negative cone equations
        """
        self.convex_hull.add_points(points)  # TODO: can we avoid modifying the entire convex hull every time?
        self.__update_cone_equations()

    def __update_cone_equations(self):
        self.equations = self.convex_hull.equations_defining_vertex(0)

        if len(self.equations) == 0:
            raise ConvexError

    def is_inside(self, points):
        """
        Computes whether a point is inside the negative cone or not
        """
        points = np.atleast_2d(points)
        return np.min(points.dot(self.equations[:, :-1].T) + self.equations[:, -1], axis=1) >= self.tol


class ConvexError(Exception):
    def __init__(self, message=None):
        if not message:
            message = "Vertex of negative cone cannot be inside positive region."
        super().__init__(message)
