class User:
    def __init__(self, labels, max_iter):
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer, got {0}".format(max_iter))

        self.labels = labels
        self.polytope_model = None
        self.__max_iter = max_iter
        self.__num_labeled_points = 0

    def __repr__(self):
        return 'User n={0} max={1}'.format(self.__num_labeled_points, self.__max_iter)

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
        flag = True
        labels = self.labels[idx]
        unknown_labels = len(labels)

        if self.polytope_model:
            unknown_labels = (self.polytope_model.predict(X) == -1).sum()
            flag = (unknown_labels > 0)

        self.__num_labeled_points += unknown_labels
        return flag, labels
