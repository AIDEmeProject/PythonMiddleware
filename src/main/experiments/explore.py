from numpy import argsort
from pandas import Series, MultiIndex
from sklearn.metrics import f1_score


def get_minimizer_over_unlabeled_data(data, labeled_indexes, ranker, sample_size=-1):
    # if unlabeled pool is too large, restrict search to sample
    data = data if sample_size <= 0 else data.sample(sample_size)
    thresholds = ranker(data.values)

    for i in argsort(thresholds):
        idx = data.index[i]
        if idx not in labeled_indexes:
            return data.loc[[idx]]


def explore(data, user, learner, initial_samples):
    user.clear()
    learner.clear()

    # initial sampling
    labels = initial_samples
    multi_index = [(0, idx) for idx in labels.index]

    # train classifier
    learner.initialize(data)
    learner.fit_classifier(data.loc[labels.index], labels)
    learner.update(data.loc[labels.index], labels)

    # main loop
    iteration = 1
    while user.is_willing() and len(labels) < len(data):
        new_points = get_minimizer_over_unlabeled_data(data, labels.index, learner.ranker)
        new_labels = user.get_label(new_points)

        # update labeled points
        labels = labels.append(new_labels)
        multi_index.extend([(iteration, idx) for idx in new_labels.index])

        # retrain active learner
        learner.fit_classifier(data.loc[labels.index], labels)
        learner.update(new_points, new_labels)

        iteration += 1

    multi_index = MultiIndex.from_tuples(multi_index, names=['iter', 'index'])
    user.clear()
    y_true = user.get_label(data.loc[labels.index], update_counter=False, use_noise=False)
    return data.loc[labels.index].set_index(multi_index), Series(data=labels.values, index=multi_index), Series(data=y_true.values, index=multi_index)


def data_generator(run):
    for iteration in run.index.levels[0]:
        next_iter = run[run.index.get_level_values('iter') == iteration]
        X = next_iter.drop('labels', axis=1).drop('true_labels', axis=1)
        y = next_iter['labels']
        yield X, y

def compute_fscore(data, y_true, learner, run):
    f_scores = []
    learner.clear()
    Xs, ys = [], []

    for X, y in data_generator(run):
        Xs.extend(X.values)
        ys.extend(y.values)

        learner.fit_classifier(Xs, ys)

        # compute f-score over entire dataset
        y_pred = learner.predict(data)
        f_scores.append(f1_score(y_true, y_pred))

    return Series(data=f_scores)

def compute_cut_ratio(data, learner, run):
    cut_ratios = []
    version_space = learner.version_space
    version_space.clear()

    for X, y in data_generator(run):
        # sample from current version space
        samples = version_space.sample(100,100)

        # check how many samples satisfy the new constrains

        # update version space
        version_space.update(X, y)



