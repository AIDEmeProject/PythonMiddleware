import timeit

from src.main.datapool import DataPool
from src.main.metrics import MetricTracker


class Task:
    def __init__(self, data, user, learner):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.pool = DataPool(self.__data)

    def clear(self):
        self.__user.clear()
        self.__learner.clear()
        self.pool.clear()

    def initialize(self):
        self.__learner.initialize(self.__data)

        # train active_learner
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X, y)

    def get_score(self, y_true):
        scores = self.__learner.score(self.__data, y_true)

        X,y = self.pool.get_labeled_set()
        scores_labeled = self.__learner.score(X, y)

        scores['Labeled Set F-Score'] = scores_labeled['F-Score']
        scores['Imbalance'] = 100.0*((y == 1).sum())/len(y)

        return scores

    def update_learner(self):
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X.iloc[[-1]], y.iloc[[-1]])

    def train(self, initial_sample):
        # clear any previous state
        self.clear()

        self.pool.update(initial_sample)
        self.initialize()

        # initialize tracker
        tracker = MetricTracker()
        #y_true = label_all(self.__data, self.__user)
        #tracker.add_measurement(self.get_score(y_true))

        while self.__user.is_willing() and (not self.pool.has_labeled_all()):
            # get next point
            t0 = timeit.default_timer()
            points = self.__learner.get_next(self.pool)
            get_next_time = timeit.default_timer() - t0

            # label point
            labels = self.__user.get_label(points)
            # update labeled/unlabeled sets
            t1 = timeit.default_timer()
            self.pool.update(labels)
            update_time = timeit.default_timer() - t1

            # retrain active learner
            t2 = timeit.default_timer()
            self.update_learner()
            retrain_time = timeit.default_timer() - t2

            iteration_time = timeit.default_timer() - t0
            # append new metrics

            scores = {
                'Get Next Time': get_next_time,
                'Update Time': update_time,
                'Retrain Time': retrain_time,
                'Iteration Time': iteration_time
            }
            #scores.update(self.get_score(y_true))
            tracker.add_measurement(scores)

        return tracker.to_dataframe(), self.pool.get_labeled_set()

from numpy import argsort
def get_minimizer_over_unlabeled_data(data, labeled_indexes, ranker, sample_size=-1):
    # if unlabeled pool is too large, restrict search to sample
    data = data if sample_size <= 0 else data.sample(sample_size)

    thresholds = ranker(data.values)
    sorted_index = argsort(thresholds)

    for i in sorted_index:
        idx = data.index[i]
        if idx not in labeled_indexes:
            return data.loc[[idx]]

def explore(data, user, learner, initial_sampler):
    user.clear()
    learner.clear()

    # initial sampling
    labels = initial_sampler(data, user)
    multi_index = [(0, idx) for idx in labels.index]

    # train classifier
    learner.initialize(data)
    learner.fit_classifier(data.loc[labels.index], labels)
    learner.update(data.loc[labels.index], labels)

    # main loop
    iteration = 1
    while user.is_willing() and len(labels) < len(data):
        new_points = get_minimizer_over_unlabeled_data(data, labels.index, learner.ranker) # learner.get_next(data)
        new_labels = user.get_label(new_points)

        # update labeled points
        labels = labels.append(new_labels)
        multi_index.extend([(iteration, idx) for idx in new_labels.index])

        # retrain active learner
        learner.fit_classifier(data.loc[labels.index], labels)
        learner.update(new_points, new_labels)

        iteration += 1
    multi_index = MultiIndex.from_tuples(multi_index, names=['iteration', 'index'])
    return data.loc[labels.index].set_index(multi_index), Series(data=labels.values, index=multi_index)


from pandas import Series, MultiIndex
from sklearn.metrics import f1_score
def compute_fscore(data, y_true, learner, run):
    f_scores = []
    labels = Series()
    learner.clear()

    for iteration in run.index.levels[0]:
        # data from run
        new_labels = run.loc[iteration]
        new_points = data.loc[new_labels.index]

        labels = labels.append(new_labels)

        learner.fit_classifier(data.loc[labels.index], labels)
        #learner.update(new_points, new_labels)

        # compute f-score over entire dataset
        y_pred = learner.predict(data)
        f_scores.append(f1_score(y_true, y_pred))

    return Series(data=f_scores)

if __name__ == '__main__':
    from src.main.config.task import get_dataset_and_user
    from src.main.initial_sampling import StratifiedSampler
    from src.main.active_learning.svm import SimpleMargin, OptimalMargin
    task = 'user_study_query12'

    data, user = get_dataset_and_user(task, keep_duplicates=False)  # sdss_Q1.1
    #learner = SimpleMargin(kind='linear', C=1024)
    learner = OptimalMargin(kind='kernel', kernel='rbf', C=100000)
    X, y = explore(data, user, learner, StratifiedSampler(1,1))

    data, user = get_dataset_and_user(task, keep_duplicates=True)
    y_true = user.get_label(data, update_counter=False)
    print(compute_fscore(data, y_true, learner, y))