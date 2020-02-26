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

import numpy as np
import pytest

from aideme.active_learning.version_space.sampling.polyhedral_cone import BoundedPolyhedralCone


class TestVersionSpace:

    def setup(self):
        A = np.array([[-1, 0], [-1, 2]], dtype='float')
        self.vs = BoundedPolyhedralCone(A)

    def test_dim_returns_expected_value(self):
        A = -np.eye(4, 5)
        vs = BoundedPolyhedralCone(A)

        assert vs.dim == A.shape[1]

    def test_is_inside_checks_for_polytope_equations(self):
        pts = np.array([
            [0, 0],   # vertex
            [2, 1],   # upper border
            [0, -1],  # lower border
            [4, 1],   # inside
            [-1, -1], # violates first constraint
            [1, 1],   # violates second constraint
        ])
        expected = np.array([False, False, False, True, False, False])

        assert np.all(self.vs.is_inside_polytope(pts) == expected)

    def test_is_inside_one_point_at_a_time(self):
        pts = np.array([
            [0, 0],   # vertex
            [2, 1],   # upper border
            [0, -1],  # lower border
            [4, 1],   # inside
            [-1, -1], # violates first constraint
            [1, 1],   # violates second constraint
        ])
        expected = np.array([False, False, False, True, False, False])

        for pt, res in zip(pts, expected):
            assert self.vs.is_inside_polytope(pt) == res

    def test_get_interior_point_returns_valid_point(self):
        interior_point = self.vs.get_interior_point()

        assert self.vs.is_inside_polytope(interior_point) and 0 < np.linalg.norm(interior_point) < 1

    def test_separating_oracle_returns_None_for_interior_point(self):
        pt = np.array([0.3, 0.1])
        assert self.vs.get_separating_oracle(pt) is None

    def test_separating_oracle_returns_expected_hyperplanes_for_outside_points(self):
        pts = np.array([
            [3, 4],         # violates norm constraint
            [0, -0.1],      # violates x > 0 constraint (on border)
            [-0.1, -0.1],   # violates x > 0 constraint (not on border)
            [0.2, 0.1],     # violates x > 2y constraint (on border)
            [0.2, 0.2],     # violates x > 2y constraint (not on border)
        ])

        expected_b = [1, 0, 0, 0, 0]
        expected_g = np.array([
            [0.6, 0.8],
            self.vs.A[0],
            self.vs.A[0],
            self.vs.A[1],
            self.vs.A[1],
        ])

        for pt, exp_b, exp_g in zip(pts, expected_b, expected_g):
            b, g = self.vs.get_separating_oracle(pt)
            assert b == exp_b and np.all(g == exp_g)

    def test_intersection_center_outside_version_space_throws_exception(self):
        centers = np.array([
            [0, 0],       # vertex
            [10, 4],      # violates norm constraint
            [-0.1, -0.1], # violates x > 0 constraint (not on border)
            [0.2, 0.2],   # violates x > 2y constraint (not on border)
        ])

        direction = np.array([1., 1.])

        for center in centers:
            with pytest.raises(RuntimeError):
                assert self.vs.intersection(center, direction)

    def test_intersection_center_inside_version_space(self):
        center = np.array([0.4, -0.1])
        directions = np.array([
            [1, 0],
            [0, 1]
        ], dtype='float')

        expected = np.array([
            [-0.4, 0.5949874371066199],
            [-0.816515138991168, 0.3],
        ])

        for direction, exp in zip(directions, expected):
            assert np.allclose(self.vs.intersection(center, direction), exp)
