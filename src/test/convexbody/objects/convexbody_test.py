import unittest
import numpy as np
from src.main.convexbody.objects import ConvexBody, Ellipsoid, Box, Polytope


class ConvexBodyTest(unittest.TestCase):
    dim = 2

    def setUp(self):
        self.body = ConvexBody(self.dim)
        self.interior_point = None
        self.exterior_point = None
        self.boundary_point = None
        self.line = None

    def test_get_dim(self):
        self.assertEqual(self.dim, self.body.dim)

    def test_is_inside_on_interior(self):
        self.assertTrue(self.body.is_inside(self.interior_point))

    def test_is_inside_on_exterior(self):
        self.assertFalse(self.body.is_inside(self.exterior_point))

    def test_is_inside_on_boundary(self):
        self.assertFalse(self.body.is_inside(self.boundary_point))

    def test_is_inside_on_interior_near_boundary(self):
        self.assertTrue(self.body.is_inside(self.boundary_point - 1e-10))

    def test_is_inside_on_exterior_near_boundary(self):
        self.assertFalse(self.body.is_inside(self.boundary_point + 1e-10))
        

class EllipsoidTest(ConvexBodyTest):

    def setUp(self):
        self.body = Ellipsoid(center=[0,0], half_axis_length=[1,2])
        self.interior_point = np.array([0,0])
        self.exterior_point = np.array([10,20])
        self.boundary_point = np.array([1,0])

class BoxTest(ConvexBodyTest):

    def setUp(self):
        self.body = Box(low=[0,0], high=[2,4])
        self.interior_point = np.array([1,2])
        self.exterior_point = np.array([3,5])
        self.boundary_point = np.array([2,2])

class PolytopeTest(ConvexBodyTest):

    def setUp(self):
        self.body = Polytope(M=[[1,1]], q=[1], l=[0,0])
        self.interior_point = np.array([0.25,0.25])
        self.exterior_point = np.array([2,4])
        self.boundary_point = np.array([0.5,0.5])


del(ConvexBodyTest)  # hack to avoid running base test class

if __name__ == '__main__':
    unittest.main()
