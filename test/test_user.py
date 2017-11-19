import unittest
from pandas import Series, DataFrame
from src.user import DummyUser, IndexUser


class TestUser(unittest.TestCase):
    def setUp(self):
        self.data = DataFrame({'x': [2,4,6,8,10], 'y': [-3,0,4,-8,20]})
        self.y_true = Series([1,-1,1,1,-1], dtype='float64')
        self.dummy_user = DummyUser(self.y_true, 3)
        self.index_user = IndexUser([0,2,3], 3)

    def test_dummy_user(self):
        index = [1,2,3]
        labels = self.dummy_user.get_label(self.data.loc[index])
        expected = self.y_true.loc[index]
        self.assertTrue(labels.equals(expected))

    def test_dummy_user_wrong_labels(self):
        with self.assertRaises(ValueError) as context:
            DummyUser(y_true=[-1,1,10], max_iter=10)
            self.assertTrue('Only {-1,1} labels are supported.' in context.exception)

    def test_index_user(self):
        index = [1,2,3]
        labels = self.index_user.get_label(self.data.loc[index])
        expected = self.y_true.loc[index]
        self.assertTrue(labels.equals(expected))

    def test_index_user_with_single_point(self):
        index = [1]
        labels = self.index_user.get_label(self.data.loc[index])
        expected = self.y_true.loc[index]
        self.assertTrue(labels.equals(expected))


if __name__ == '__main__':
    unittest.main()