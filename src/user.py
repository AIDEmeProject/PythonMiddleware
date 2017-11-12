from numpy import array, atleast_1d


class User(object):
    """ 
        This class represents a bridge between the real user and an Active Learning algorithm. 

        For testing the algorithms, users are represented either by an 'oracle query' that labels data points 
        or by directly inputting all true labels. In the future, a real user can be queried for labels.
    """

    def __init__(self, max_iter):
        """
            :param max_iter: max number of points the user is willing to classify
        """
        self.max_iter = max_iter
        self.labeled_samples = 0

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

    def get_label(self, points, update_counter=True):
        """ 
            Labels user provided points
            :param points: collection of points to label
            :param update_counter: whether to update internal counter of labeled points
        """
        if update_counter:
            self.labeled_samples += len(points.index)
        return self._get_label(points)

    def _get_label(self, points):
        """
        :param points: Point instance 
        :return: the label for each point
        """
        raise NotImplementedError


class DummyUser(User):
    """
        The dummy user represents an 'user' who knows the classification to all point in the database.
        It's just a proxy to the cases where the true labeling is given and no user is actually being queried. 
    """

    def __init__(self, y_true, max_iter):
        """
        :param y_true:   true labeling of points (must be -1, 1 format)
        """
        super().__init__(max_iter)
        self.__y_true = atleast_1d(y_true).ravel()

        self._check_labels()

    def _get_label(self, points):
        return self.__y_true[points.index]

    def _check_labels(self):
        if not set(self.__y_true) <= {-1, 1}:
            raise ValueError("Only {-1,1} labels are supported.")

class IndexUser(User):
    def __init__(self, index, max_iter):
        super().__init__(max_iter)
        self.__index = set(index)

    def _get_label(self, points):
        return array([1 if idx in self.__index else -1 for idx in points.index])
