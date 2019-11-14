import pytest

from scipy.spatial.qhull import QhullError

from aideme.active_learning.dsm.convex import *


def assert_arrays_have_same_rows(arr1, arr2):
    from numpy.testing import assert_array_equal

    arr1 = [[x for x in row] for row in arr1]
    arr2 = [[x for x in row] for row in arr2]

    assert_array_equal(sorted(arr1), sorted(arr2))


class TestConvexHull:

    def test_less_points_than_dimension_raises_exception(self):
        points = [[1, 2, 3], [4, 5, 6]]

        with pytest.raises(QhullError):
            assert ConvexHull(points)

    def test_number_of_points_equals_dimension_raises_exception(self):
        points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        with pytest.raises(QhullError):
            assert ConvexHull(points)

    def test_flat_initial_polytope(self):
        points = [[1, 2], [3, 4], [5, 6]]  # points in a line

        with pytest.raises(QhullError):
            assert ConvexHull(points)

    def test_negative_tol_raises_exception(self):
        points = [[-1, 0], [1, 0], [0, 1]]

        with pytest.raises(ValueError):
            assert ConvexHull(points, tol=-1)

    def test_zero_tol_raises_exception(self):
        points = [[-1, 0], [1, 0], [0, 1]]

        with pytest.raises(ValueError):
            assert ConvexHull(points, tol=0)

    def test_vertices_do_not_include_interior_points(self):
        vertices = [[0,  1], [-1,  0], [0, -1], [1,  0]]
        interior = [[0, 0], [0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]]  # interior points

        hull = ConvexHull(vertices + interior)

        assert_arrays_have_same_rows(hull.vertices, vertices)

    def test_add_point_of_wrong_dimension_raises_exception(self):
        points = [[-1, 0], [1, 0], [0, 1]]
        hull = ConvexHull(points)

        with pytest.raises(ValueError):
            assert hull.add_points([1, 2, 3])

    def test_add_point_adds_a_new_vertex(self):
        points = [[-1, 0], [0, -1], [0, 1]]
        new_vertex = [[1, 0]]

        hull = ConvexHull(points)
        hull.add_points(new_vertex)

        assert_arrays_have_same_rows(hull.vertices, points + new_vertex)

    def test_new_point_can_exclude_old_vertex(self):
        points = [[-1, 0], [1, 0], [0, 1]]
        new_vertex = [[0, 2]]

        hull = ConvexHull(points)
        hull.add_points(new_vertex)

        assert_arrays_have_same_rows(hull.vertices, [[-1, 0], [1, 0], [0, 2]])

    def test_adding_an_interior_point_does_not_change_vertices(self):
        vertices = [[-1, 0], [1, 0], [0, 1]]
        new_vertex = [[0, 0.5]]

        hull = ConvexHull(vertices)
        hull.add_points(new_vertex)

        assert_arrays_have_same_rows(hull.vertices, vertices)

    def test_is_inside(self):
        vertices = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        hull = ConvexHull(vertices)

        assert hull.is_inside([0, 0])  # interior point
        assert all(hull.is_inside(vertices))  # boundary points
        assert not any(hull.is_inside([[-10, 0], [10, 0], [0, 10], [0, -10]]))  # exterior points

    def test_is_inside_with_incompatible_dimension_throws_exception(self):
        vertices = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        hull = ConvexHull(vertices)

        with pytest.raises(ValueError):
            assert hull.is_inside([1, 2, 3])
            assert hull.is_inside([[1, 2, 3], [4, 5, 6]])

    def test_equations_defining_vertex(self):
        vertices = [[0, 0], [1, 0], [0, 1], [1, 1]]

        hull = ConvexHull(vertices)

        assert_arrays_have_same_rows(hull.equations_defining_vertex(0), [[-1, 0, 0], [0, -1, 0]])  # -x <= 0, -y <= 0
        assert_arrays_have_same_rows(hull.equations_defining_vertex(1), [[0, -1, 0], [1, 0, -1]])  # -y <= 0,  x <= 1
        assert_arrays_have_same_rows(hull.equations_defining_vertex(2), [[-1, 0, 0], [0, 1, -1]])  # -x <= 0,  y <= 1
        assert_arrays_have_same_rows(hull.equations_defining_vertex(3), [[1, 0, -1], [0, 1, -1]])  #  x <= 1,  y <= 1


class TestConvexCone:
    def test_vertex_and_points_have_incompatible_dimensions(self):
        vertex = [0]
        points = [[2, 3], [4, 5]]

        with pytest.raises(ValueError):
            assert ConvexCone(points, vertex)

    def test_total_number_of_less_than_dimension_raises_exception(self):
        vertex = [0, 0, 0]
        points = [[1, 2, 3]]

        with pytest.raises(QhullError):
            assert ConvexCone(points, vertex)

    def test_total_number_of_points_equals_dimension_raises_exception(self):
        vertex = [0, 0, 0]
        points = [[1, 2, 3], [4, 5, 6]]

        with pytest.raises(QhullError):
            assert ConvexCone(points, vertex)

    def test_flat_initial_polytope(self):
        vertex = [1, 2]
        points = [[3, 4], [5, 6]]  # points and vertex are in a line

        with pytest.raises(QhullError):
            assert ConvexCone(points, vertex)

    def test_negative_tol_raises_exception(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        with pytest.raises(ValueError):
            assert ConvexCone(points, vertex, tol=-1)

    def test_zero_tol_raises_exception(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        with pytest.raises(ValueError):
            assert ConvexCone(points, vertex, tol=0)

    def test_vertex_inside_polytope_throws_exception(self):
        vertex = [0, 0]
        points = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        with pytest.raises(ConvexError):
            assert ConvexCone(points, vertex)

    def test_is_inside(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        cone = ConvexCone(points, vertex)

        assert all(cone.is_inside([[0, 2], [1, 2], [-1, 2]]))  # interior point
        assert cone.is_inside(vertex)  # vertex
        assert not any(cone.is_inside([[0, 0], [-1, 0], [1, 0], [10, 10], [-10, -10]]))  # exterior points

    def test_is_inside_with_incompatible_dimension_throws_exception(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        cone = ConvexCone(points, vertex)

        with pytest.raises(ValueError):
            assert cone.is_inside([1, 2, 3])
            assert cone.is_inside([[1, 2, 3], [4, 5, 6]])

    def test_adding_conflicting_positive_point_throws_exception(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        cone = ConvexCone(points, vertex)

        with pytest.raises(ConvexError):
            assert cone.add_points_to_hull([0, 2])

    def test_add_point_to_hull_with_incompatible_dimension_throws_exception(self):
        vertex = [0, 1]
        points = [[-1, 0], [1, 0]]

        cone = ConvexCone(points, vertex)

        with pytest.raises(ValueError):
            assert cone.add_points_to_hull([1, 2, 3])
            assert cone.add_points_to_hull([[1, 2, 3], [4, 5, 6]])
