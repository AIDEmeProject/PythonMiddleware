import pytest

from aideme.active_learning.dsm.one_dim_convex import *


def assert_lists_have_same_elements(ls1, ls2):
    ls1, ls2 = sorted(ls1), sorted(ls2)
    assert all(x == y for x, y in zip(ls1, ls2))


class TestOneDimensionalConvexHull:
    def test_empty_points_raises_exception(self):
        with pytest.raises(ValueError):
            assert OneDimensionalConvexHull([])

    def test_vertices_do_not_include_interior_points(self):
        vertices = [-1, 1]
        interior = [-0.5, 0, 0.5]

        hull = OneDimensionalConvexHull(vertices + interior)

        assert_lists_have_same_elements(hull.vertices, vertices)

    def test_add_point_outside_hull(self):
        points = [-1, 1]
        new_vertex = [-2, 3, 5, -4]

        hull = OneDimensionalConvexHull(points)
        hull.add_points(new_vertex)

        assert_lists_have_same_elements(hull.vertices, [-4, 5])

    def test_adding_an_interior_point_does_not_change_vertices(self):
        vertices = [-1, 1]
        new_vertex = [-0.5, 0, 0.5]

        hull = OneDimensionalConvexHull(vertices)
        hull.add_points(new_vertex)

        assert_lists_have_same_elements(hull.vertices, vertices)

    def test_add_point_with_incompatible_dimension_throws_exception(self):
        vertices = [-1, 1]

        hull = OneDimensionalConvexHull(vertices)

        with pytest.raises(ValueError):
            assert hull.add_points([[1, 2]])

    def test_is_inside(self):
        vertices = [-1, 1]

        hull = OneDimensionalConvexHull(vertices)

        assert all(hull.is_inside([0, 0.5, -0.5]))  # interior point
        assert all(hull.is_inside(vertices))  # boundary points
        assert not any(hull.is_inside([-3, -2, 2, 3]))  # exterior points

    def test_is_inside_when_hull_contains_single_point(self):
        vertices = [0]

        hull = OneDimensionalConvexHull(vertices)

        assert all(hull.is_inside([0]))  # interior point
        assert not any(hull.is_inside([-2, -1, 1, 2]))  # exterior points

    def test_is_inside_with_incompatible_dimension_throws_exception(self):
        points = [-1, 1]

        cone = OneDimensionalConvexHull(points)

        with pytest.raises(ValueError):
            assert cone.is_inside([[1, 2]])


class TestOneDimensionalConvexCone:
    def test_empty_points_throws_exception(self):
        with pytest.raises(ValueError):
            assert OneDimensionalConvexCone([], 0)

    def test_vertex_inside_polytope_throws_exception(self):
        points = [-1, 1]
        vertex = 0

        with pytest.raises(ConvexError):
            assert OneDimensionalConvexCone(points, vertex)

    def test_is_inside(self):
        points = [-1, 1]
        vertex = 2

        cone = OneDimensionalConvexCone(points, vertex)

        assert all(cone.is_inside([3, 4, 5]))  # interior point
        assert cone.is_inside(2)  # vertex
        assert not any(cone.is_inside([1, 0, -1, -2]))  # exterior points

    def test_is_inside_with_incompatible_dimension_throws_exception(self):
        vertex = 2
        points = [-1, 1]

        cone = OneDimensionalConvexCone(points, vertex)

        with pytest.raises(ValueError):
            assert cone.is_inside([[1, 2]])

    def test_adding_conflicting_positive_point_throws_exception(self):
        vertex = 2
        points = [-1, 1]

        cone = OneDimensionalConvexCone(points, vertex)

        with pytest.raises(ConvexError):
            assert cone.add_points_to_hull(3)

    def test_add_point_to_hull_with_incompatible_dimension_throws_exception(self):
        vertex = 2
        points = [-1, 1]

        cone = OneDimensionalConvexCone(points, vertex)

        with pytest.raises(ValueError):
            assert cone.add_points_to_hull([[1, 2]])
