class User:
    def __init__(self, labels, max_iter):
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer, got {0}".format(max_iter))

        self.labels = labels
        self.__max_iter = max_iter
        self.__num_labeled_points = 0

    def __repr__(self):
        return 'User num_labeled_points={0} max_iter={1}'.format(self.__num_labeled_points, self.__max_iter)

    @property
    def num_labeled_points(self):
        return self.__num_labeled_points

    @property
    def max_iter(self):
        return self.__max_iter

    @property
    def is_willing(self):
        return self.__num_labeled_points <= self.__max_iter

    def label(self, idx, X):
        self.__num_labeled_points += len(idx)
        return self.labels[idx]
