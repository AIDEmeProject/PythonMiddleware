import unittest

from pandas import DataFrame, Series

from src.main.datapool import DataPool


class TestDataPool(unittest.TestCase):
    """ Tests DataPool class  """

    def setUp(self):
        self.data = DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
        self.y_true = Series([1,1,-1,1,-1])
        self.pool = DataPool(self.data)

    def test_update(self):
        # update the pool with two operations
        self.pool.update(self.y_true.loc[[1,2,3]])

        # assert update worked
        self.assertListEqual(self.pool.labeled_rows, [1,2,3])
        self.assertTrue(self.pool.labels.equals(self.y_true.loc[[1,2,3]]))

    def test_successive_updates(self):
        # first update
        self.pool.update(self.y_true.loc[[1]])

        # assert first update worked
        self.assertListEqual(self.pool.labeled_rows, [1])
        self.assertTrue(self.pool.labels.equals(self.y_true.loc[[1]]))

        # second update
        self.pool.update(self.y_true.loc[[2,3]])

        # assert second update worked
        self.assertListEqual(self.pool.labeled_rows, [1,2,3])
        self.assertTrue(self.pool.labels.equals(self.y_true.loc[[1,2,3]]))


    def test_update_same_point_twice(self):
        # update the same point twice
        self.pool.update(self.y_true.loc[[1]])

        with self.assertRaises(RuntimeError) as context:
            self.pool.update(self.y_true.loc[[1,2]])
            self.assertTrue('Trying to label the same point twice!' in context.exception)


    def test_clear(self):
        # update then clear results
        self.pool.update(self.y_true)
        self.assertTrue(self.pool.labels.equals(self.y_true))
        self.assertListEqual(self.pool.labeled_rows, [0,1,2,3,4])

        # assert labeled_rows and labels were cleared
        self.pool.clear()
        self.assertTrue(self.pool.labels.empty)
        self.assertListEqual(self.pool.labeled_rows, [])

    def test_get_positive_points(self):
        # add data to it and then clear results
        self.pool.update(Series([-1,1,-1,1,-1]))

        # get positive points
        res = DataFrame({'x': [2,4], 'y': [4,2]}, index=[1,3])
        self.assertTrue(res.equals(self.pool.get_positive_points()))

    def test_has_labeled_all(self):
        # must not be true in the beginning
        self.assertFalse(self.pool.has_labeled_all())

        # must not be true after udpating all points but one
        self.pool.update(self.y_true.loc[[0,1,2,3]])
        self.assertFalse(self.pool.has_labeled_all())

        # must be true after updating all points
        self.pool.update(self.y_true.loc[[4]])
        self.assertTrue(self.pool.has_labeled_all())

    def test_minimizer(self):
        ranker = lambda X: X[:,1]

        res = self.pool.get_minimizer_over_unlabeled_data(ranker)
        self.assertTrue(res.equals(self.data.loc[[4]]))

    def test_minimizer_after_updating(self):
        self.pool.update(self.y_true[[3,4]])
        ranker = lambda X: X[:, 1]

        res = self.pool.get_minimizer_over_unlabeled_data(ranker)
        self.assertTrue(res.equals(self.data.loc[[2]]))

    def test_minimizer_after_labeling_all(self):
        self.pool.update(self.y_true)
        ranker = lambda X: X[:, 1]

        with self.assertRaises(ValueError) as context:
            self.pool.get_minimizer_over_unlabeled_data(ranker)
            self.assertTrue("Size larger than unlabeled set size!" in context.exception)


if __name__ == '__main__':
    unittest.main()