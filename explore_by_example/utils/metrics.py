import sklearn.metrics


def three_set_metric(X, y, active_learner):
    pred = active_learner.polytope_model.predict(X)

    pos = (pred == 1).sum()
    unknown = (pred == -1).sum()
    tsm = pos / (pos + unknown)

    return {'tsm': tsm}


def classification_metrics(*score_functions):
    diff = set(score_functions) - __classification_metrics.keys()
    if len(diff) > 0:
        raise ValueError("Unknown classification metrics: {0}. Supported values are: {1}".format(sorted(diff), sorted(__classification_metrics.keys())))

    def compute(X, y, active_learner):
        # when using factorization, we have to condense all partial labels into a single one
        # if len(y.shape) > 1:
        #    y = (np.mean(y, axis=1) == 1).astype('float')

        y_pred = active_learner.predict(X)
        cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[0, 1])
        return {score : __classification_metrics[score](cm) for score in score_functions}

    return compute


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


def true_divide(x, y):
    return 0 if y == 0 else x / y
