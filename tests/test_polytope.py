#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.

import pytest

from aideme.active_learning.dsm.polytope import *


class TestPolytopeModel:
    def setup(self):
        self.pol = Polytope()

    def test_is_valid_is_the_beginning(self):
        assert self.pol.is_valid

    def test_is_valid_if_no_conflicting_points_are_given(self):
        self.pol.update([[-1, 0], [1, 0], [0, 1], [0, 2]], [1, 1, 1, 0])

        assert self.pol.is_valid

    def test_is_not_valid_when_negative_conflicting_point_is_given(self):
        self.pol.update([[-1, 0], [1, 0], [0, 1], [0, 0.5]], [1, 1, 1, 0])

        assert not self.pol.is_valid

    def test_is_not_valid_when_positive_conflicting_point_is_given(self):
        self.pol.update([[-1, 0], [1, 0], [0, 1], [0, 2]], [1, 1, 0, 1])

        assert not self.pol.is_valid

    def test_update_returns_False_when_conflict_is_found(self):
        assert self.pol.update([[-1, 0], [1, 0], [0, 2]], [1, 1, 1])
        assert not self.pol.update([0, 1], [0])

    def test_update_throws_exception_when_polytope_is_invalid(self):
        assert not self.pol.update([[-1, 0], [1, 0], [0, 1], [0, 0.5]], [1, 1, 1, 0])

        with pytest.raises(RuntimeError):
            assert self.pol.update([10, 10], [1])

    def test_predict(self):
        X = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        y = [1, 1, 1, 0]

        self.pol.update(X, y)

        np.testing.assert_array_equal(self.pol.predict(X), y)

    def test_predict_always_returns_uncertain_when_polytope_is_empty(self):
        X = np.random.uniform(-1, 1, (10, 2))

        np.testing.assert_array_equal(self.pol.predict(X), [0.5]*10)

    def test_predict_always_returns_uncertain_when_polytope_is_invalid(self):
        assert not self.pol.update([[-1, 0], [1, 0], [0, 1], [0, 0.5]], [1, 1, 1, 0])

        X = np.random.uniform(-1, 1, (10, 2))
        np.testing.assert_array_equal(self.pol.predict(X), [0.5] * 10)


class TestPositiveRegion:
    def setup(self):
        self.pos = PositiveRegion()
    
    def test_negative_tol_raises_exception(self):
        with pytest.raises(ValueError):
            assert PositiveRegion(tol=-1)

    def test_zero_tol_raises_exception(self):
        with pytest.raises(ValueError):
            assert PositiveRegion(tol=0)

    def test_not_build_until_more_than_dim_points(self):
        assert not self.pos.is_built

        self.pos.update([0, 0])
        assert not self.pos.is_built

        self.pos.update([1, 1])
        assert not self.pos.is_built

        self.pos.update([0, 1])
        assert self.pos.is_built

    def test_vertices_are_all_cached_points_until_polytope_is_build(self):
        assert self.pos.vertices is None

        vertices = [[0, 0], [1, 0]]
        self.pos.update(vertices)
        np.testing.assert_array_equal(self.pos.vertices, vertices)

        new_vertex = [[0, 1]]
        self.pos.update(new_vertex)
        np.testing.assert_array_equal(self.pos.vertices, vertices + new_vertex)

    def test_is_inside_always_returns_false_before_hull_is_built(self):
        vertices = [[0, 0], [2, 0], [0, 2]]
        test_points = vertices + [[0.5, 0.5], [1, 0], [0, 1], [1, 1]]

        assert not any(self.pos.is_inside(test_points))

        self.pos.update(vertices[:2])
        assert not any(self.pos.is_inside(test_points))

        self.pos.update(vertices[2])
        assert all(self.pos.is_inside(test_points))

    def test_update_throws_exception_if_polytope_is_flat(self):
        # TODO: is this the expected behavior? Or should we keep caching points until polytope can be built?
        vertices = [[0, 0], [1, 0], [2, 0]]

        with pytest.raises(Exception):
            assert self.pos.update(vertices)


class TestNegativeCone:
    def test_negative_tol_raises_exception(self):
        vertex = [0, 1]

        with pytest.raises(ValueError):
            assert NegativeCone(vertex, tol=-1)

    def test_zero_tol_raises_exception(self):
        vertex = [0, 1]

        with pytest.raises(ValueError):
            assert NegativeCone(vertex, tol=0)

    def test_not_built_if_no_positive_points(self):
        cone = NegativeCone([1, 0])
        assert not cone.is_built

    def test_not_built_if_less_than_dim_positive_points(self):
        cone = NegativeCone([0, 1])

        cone.add_points_to_hull([-1, 0])

        assert not cone.is_built

    def test_is_built_after_dim_positive_points(self):
        cone = NegativeCone([0, 1])

        cone.add_points_to_hull([[-1, 0], [1, 0]])

        assert cone.is_built

    def test_is_inside_always_returns_false_before_hull_is_built(self):
        cone = NegativeCone([0, 1])
        test_points = [[0, 1], [0, 0], [-1, 0], [1, 0]]

        assert not any(cone.is_inside(test_points))

    def test_is_inside(self):
        cone = NegativeCone([0, 1])
        cone.add_points_to_hull([[-1, 0], [1, 0]])

        assert all(cone.is_inside([[0, 2], [1, 2], [-1, 2]]))  # interior point
        assert cone.is_inside([0, 1])  # vertex
        assert not any(cone.is_inside([[0, 0], [-1, 0], [1, 0], [10, 10], [-10, -10]]))  # exterior points

    def test_is_inside_with_incompatible_dimension_throws_exception(self):
        cone = NegativeCone([0, 1])
        cone.add_points_to_hull([[-1, 0], [1, 0]])

        with pytest.raises(ValueError):
            assert cone.is_inside([1, 2, 3])
            assert cone.is_inside([[1, 2, 3], [4, 5, 6]])

    def test_adding_conflicting_point_throws_exception(self):
        cone = NegativeCone([0, 1])
        cone.add_points_to_hull([[-1, 0], [1, 0]])

        with pytest.raises(ConvexError):
            cone.add_points_to_hull([[0, 2]])

    def test_add_point_to_cone_with_incompatible_dimension_throws_exception(self):
        cone = NegativeCone([0, 1])

        with pytest.raises(ValueError):
            assert cone.add_points_to_hull([1, 2, 3])
            assert cone.add_points_to_hull([[1, 2, 3], [4, 5, 6]])

    def test_add_points_throws_exception_if_polytope_is_flat(self):
        # TODO: is this the expected behavior? Or should we keep caching points until polytope can be built?
        cone = NegativeCone([0, 0])

        with pytest.raises(Exception):
            assert cone.add_points_to_hull([[-1, 0], [1, 0]])
