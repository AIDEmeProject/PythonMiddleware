from pandas import Series
from pandasql import sqldf
from random import random

def bool_to_sign(ls):
    return [1.0 if x else -1.0 for x in ls]


class User:
    """
        This class represents a bridge between the real user and an Active Learning algorithm.

        For testing the algorithms, users are represented either by an 'oracle query' that labels data points
        or by directly inputting all true labels. In the future, a real user can be queried for labels.
    """

    def __init__(self, max_iter, noise=0.0):
        """
            :param max_iter: max number of points the user is willing to classify
        """
        if(max_iter <= 0 or noise < 0.0 or noise >= 1.0):
            raise ValueError("Invalid arguments in user.")

        self.max_iter = int(max_iter)
        self.noise = noise
        self.labeled_samples = 0
        self._true_index = None

    def clear(self):
        """
            Resets labeled samples count, so we can utilize the same user multiple times.
        """
        self.labeled_samples = 0

    def is_willing(self):
        """
            Returns whether the user is willing to classify more points
            :return: True or False
        """
        return self.labeled_samples < self.max_iter

    def flip_bool(self, ls):
        return [x if random() > self.noise else not x for x in ls]

    def get_label(self, points, update_counter=True, use_noise=True):
        """
            Labels user provided points
            :param points: collection of points to label
            :param update_counter: whether to update internal counter of labeled points
        """
        if not self.is_willing():
            raise RuntimeError("User has already stopped labeling.")

        if update_counter:
            self.labeled_samples += len(points)

        labels = points.index.isin(self._true_index)
        if use_noise:
            labels = self.flip_bool(labels)

        return Series(data=bool_to_sign(labels), index=points.index)


class DummyUser(User):
    """
        The dummy user represents an 'user' who knows the classification to all point in the database.
        It's just a proxy to the cases where the true labeling is given and no user is actually being queried.
    """

    def __init__(self, y_true, max_iter, true_class=1.0, noise=0.0):
        """
        :param y_true:  true labeling of points (must be -1, 1 format)
        """
        super().__init__(max_iter, noise)
        self._true_index = y_true[y_true == true_class].index


class FakeUser(User):
    def __init__(self, data, true_predicate, max_iter, noise=0.0):
        super().__init__(max_iter, noise)

        if not true_predicate:
            raise ValueError("Received empty true predicate!")

        if data.index.name is None:
            raise RuntimeError("Cannot create FakeUser from unnamed index column.")

        query = "SELECT {0} FROM data WHERE {1}".format(data.index.name, true_predicate)
        query_result = sqldf(query, {'data': data})
        self._true_index = set(query_result[data.index.name])

        if len(self._true_index) == len(data):
            raise RuntimeError("All labels are identical!")
