import sklearn.metrics


class Metric:
    def __init__(self, print_value=False):
        self.print_value = print_value

    def __call__(self, iter, X, y, active_learner):
        scores = self.compute(iter, X, y, active_learner)

        if self.print_value:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in scores.items()))
            print('iter: {0}, {1}'.format(iter, scores_str))

        return scores

    def compute(self, iter, X, y, active_learner):
        raise NotImplementedError


class ClassificationMetric(Metric):
    __classification_metrics = {
        'true_positive': lambda cm: cm[1, 1],
        'false_positive': lambda cm: cm[0, 1],
        'false_negative': lambda cm: cm[1, 0],
        'true_negative': lambda cm: cm[0, 0],
        'accuracy': lambda cm: true_divide(cm[0, 0] + cm[1, 1], cm.sum()),
        'precision': lambda cm: true_divide(cm[1, 1], cm[1, 1] + cm[0, 1]),
        'recall': lambda cm: true_divide(cm[1, 1], cm[1, 1] + cm[1, 0]),
        'fscore': lambda cm: true_divide(2 * cm[1, 1], 2 * cm[1, 1] + cm[0, 1] + cm[1, 0]),
    }

    def __init__(self, *score_functions, print_value=False):
        super().__init__(print_value)
        self.score_functions = score_functions

    def compute(self, iter, X, y, active_learner):
        # when using factorization, we have to condense all partial labels into a single one
        # if len(y.shape) > 1:
        #    y = (np.mean(y, axis=1) == 1).astype('float')

        y_pred = active_learner.predict(X)
        cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[0, 1])
        return {score : self.__classification_metrics[score](cm) for score in self.score_functions}


class ThreeSetMetric(Metric):
    def compute(self, iter, X, y, active_learner):
        pred = active_learner.polytope_model.predict(X)

        pos = (pred == 1).sum()
        unknown = (pred == -1).sum()
        tsm = pos / (pos + unknown)

        return {'tsm': tsm}


def true_divide(x, y):
    return 0 if y == 0 else x / y
