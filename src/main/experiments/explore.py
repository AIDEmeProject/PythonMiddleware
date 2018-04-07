import numpy as np
from pandas import Series, MultiIndex
from sklearn.metrics import f1_score
from time import perf_counter

def get_minimizer_over_unlabeled_data(data, labeled_indexes, ranker, sample_size=-1):
    # if unlabeled pool is too large, restrict search to sample
    data = data if sample_size <= 0 else data.sample(sample_size)
    thresholds = ranker(data.values)

    for i in np.argsort(thresholds):
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
    learner.update(data.loc[labels.index], labels)
    learner.fit_classifier(data.loc[labels.index], labels)

    times = []

    # main loop
    iteration = 1
    while user.is_willing() and len(labels) < len(data):
        print('iter: ' + str(iteration))
        t_init = perf_counter()
        new_points = get_minimizer_over_unlabeled_data(data, labels.index, learner.ranker)
        new_labels = user.get_label(new_points)

        # update labeled points
        labels = labels.append(new_labels)
        multi_index.extend([(iteration, idx) for idx in new_labels.index])

        # retrain active learner
        learner.update(new_points, new_labels)
        learner.fit_classifier(data.loc[labels.index], labels)

        iteration += 1
        times.append(perf_counter() - t_init)

    multi_index = MultiIndex.from_tuples(multi_index, names=['iter', 'index'])
    user.clear()
    y_true = user.get_label(data.loc[labels.index], update_counter=False, use_noise=False)
    return (data.loc[labels.index].set_index(multi_index),
            Series(data=labels.values, index=multi_index),
            Series(data=y_true.values, index=multi_index),
            Series(data=times, index=multi_index.get_level_values('iter')[2:]))


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
        print(f1_score(y_true=ys, y_pred=learner.predict(Xs)))
        # compute f-score over entire dataset
        y_pred = learner.predict(data)
        f_scores.append(f1_score(y_true, y_pred))

    return Series(data=f_scores)

def compute_cut_ratio(run, limit=100):
    cut_ratios = []
    from src.main.version_space.linear import KernelVersionSpace
    version_space = KernelVersionSpace(rounding=True)

    i=0
    sample = None
    for X, y in data_generator(run):
        if i > 0:
            # sample classifiers from current version space
            sample = version_space.sample_classifier(chain_length=64, sample_size=100, previous_sample=sample)

            # check how many samples satisfy the new constrains
            pred = np.all(sample.predict(X) == y.values, axis=1)
            prop = 1.0 - np.sum(pred) / 100.0
            cut_ratios.append(prop)

        # update version space
        version_space.update(X, y)
        i+=1
        print(i)
        if i > limit:
            break

    return Series(data=cut_ratios)



