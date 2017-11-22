import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import rbf_kernel
from ..base import ActiveLearner
from src.version_space.linear import LinearVersionSpace


class SVMBase(ActiveLearner):
    def __init__(self, top=-1, kind='linear', C=1000, kernel='linear', fit_intercept=True, class_weight=None):
        super().__init__(top)

        if kind == 'kernel':
            self.clf = SVC(C=C, kernel=kernel, decision_function_shape='ovr', class_weight=class_weight)
        elif kind == 'linear':
            self.clf = LinearSVC(C=C, fit_intercept=fit_intercept, class_weight=class_weight)
        else:
            raise ValueError("Non supported kind. Only 'linear' and 'kernel' options available.")

        self.kind = kind


class SimpleMargin(SVMBase):
    """
        Picks the closest point to the decision boundary to feed the user. After Tong-and-Koller, this approximately
        cuts the version space in half at every iteration
    """
    def ranker(self, data):
        return np.abs(self.clf.decision_function(data))


class OptimalMargin(SVMBase):
    def __init__(self, chain_length=50, sample_size=8,
                 top=-1, kind='linear', C=1000, kernel='linear', fit_intercept=True, class_weight=None):
        super().__init__(top, kind, C, kernel, fit_intercept, class_weight)
        self.sample_size = sample_size
        self.chain_length = chain_length
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []

    def clear(self):
        self.__data = None
        self.__labeled_indexes = []
        self.__labels = []

    def initialize(self, data):
        self.__data = data.values

    def get_kernel_matrix(self, X, Y=None):
        if self.kind == 'linear':
            return X
        return rbf_kernel(X, Y)

    def update(self, points, labels):
        # udpate labels and indexes
        self.__labels.extend(labels.values)
        self.__labeled_indexes.extend(points.index)

        # create new version space
        K = self.get_kernel_matrix(self.__data[self.__labeled_indexes])
        self.version_space = LinearVersionSpace(K.shape[1])
        for point, label in zip(K, self.__labels):
            self.version_space.update(point, label)

    def ranker(self, data):
        samples = self.version_space.sample(self.chain_length, self.sample_size)
        bias, weight = samples[:, 0].reshape(-1, 1), samples[:, 1:]

        K = self.get_kernel_matrix(data, data[self.__labeled_indexes])
        predictions = np.sign(bias + weight.dot(K.T))
        return np.abs(np.sum(predictions, axis=0))
