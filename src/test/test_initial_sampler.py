import unittest

from pandas import DataFrame, Series
from src.user import DummyUser

from src.main.initial_sampling import StratifiedSampler


class TestInitialSampler(unittest.TestCase):
    def setUp(self):
        self.data = DataFrame({'x': [0,1,2,3,4,5], 'y': [5,4,3,2,1,0]})
        self.user = DummyUser(Series([1,1,1,-1,-1,-1]), 10)

    def test_statified_sampling_sample_size(self):
        sampler = StratifiedSampler(1,3)
        sample = sampler(self.data, self.user)
        self.assertTrue(sum(sample == 1) == 1)
        self.assertTrue(sum(sample == -1) == 3)

    def test_zero_sized_sample(self):
        sampler = StratifiedSampler(0, 3)
        sample = sampler(self.data, self.user)
        self.assertTrue(sum(sample == 1) == 0)

        sampler = StratifiedSampler(3, 0)
        sample = sampler(self.data, self.user)
        self.assertTrue(sum(sample == -1) == 0)

    def test_negative_sample_size(self):
        with self.assertRaises(ValueError) as context:
            StratifiedSampler(-1, 3)
            self.assertTrue("Found negative sample size. Please provide a positive number.'" in context.exception)

        with self.assertRaises(ValueError) as context:
            StratifiedSampler(1, -3)
            self.assertTrue("Found negative sample size. Please provide a positive number.'" in context.exception)

    def test_sample_larger_than_population(self):
        with self.assertRaises(ValueError) as context:
            sampler = StratifiedSampler(4, 3)
            sampler(self.data, self.user)
            self.assertTrue("Cannot take a larger sample than population when 'replace=False'" in context.exception)

        with self.assertRaises(ValueError) as context:
            sampler = StratifiedSampler(3, 4)
            sampler(self.data, self.user)
            self.assertTrue("Cannot take a larger sample than population when 'replace=False'" in context.exception)


if __name__ == '__main__':
    unittest.main()