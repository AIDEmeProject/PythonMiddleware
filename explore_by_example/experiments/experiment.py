from time import perf_counter

import pandas as pd
import sklearn

from .directory_manager import ExperimentDirManager
from .logger import ExperimentLogger
from .plot import ExperimentPlotter
from ..io import read_task


class Experiment:
    def __init__(self):
        self.skip_list = []
        self.logger = ExperimentLogger()
        self.dir_manager = ExperimentDirManager()
        self.plotter = ExperimentPlotter(self.dir_manager)

    @classmethod
    def __check_tags(cls, ls):
        tags = [x[0] for x in ls]
        if len(tags) != len(set(tags)):
            raise ValueError("All tags must be distinct!")

    def run(self, datasets, learners, times, explore):
        # check tags
        Experiment.__check_tags(datasets)
        Experiment.__check_tags(learners)

        # add new experiments folder
        self.dir_manager.set_new_experiment_folder()

        # set logging path
        self.logger.set_folder(self.dir_manager.experiment_folder)

        for data_tag, task_tag in datasets:
            # get data and user
            X, y = read_task(task_tag, distinct=False)

            for learner_tag, learner in learners:
                # if learners failed previously, skip it
                if learner_tag in self.skip_list:
                    self.logger.skip()
                    continue

                # add learner folder
                data_folder = self.dir_manager.get_data_folder(data_tag, learner_tag)

                # create new task and try to run it
                try:
                    for i in range(times):
                        # log task begin
                        self.logger.begin(data_tag, learner_tag, i+1)

                        # run task
                        metrics = explore.run(X, y, learner)

                        # persist run
                        filename = "run{0}_raw.tsv".format(i+1)
                        df = pd.DataFrame.from_dict({i: metric for i, metric in enumerate(metrics)}, orient='index')
                        data_folder.write(df, filename, index=True)

                    # self.dir_manager.compute_folder_average(data_tag, learner_tag)

                except Exception as e:
                    # if error occurred, log error and add learner to skip list
                    self.logger.error(e)
                    self.skip_list.append(learner_tag)

                finally:
                    pass  # continue to next tasks

        # log experiment end
        self.logger.end()


class PoolBasedExploration:
    def __init__(self, iter, initial_sampler, metric=None, plot=None):
        """
        :param iter: number of iterations of the active learning algorithm to run
        :param initial_sampler: InitialSampler object
        :param metric: accuracy metric to be computed every iteration
        :param plot: plotting function for each iteration
        """
        self.iter = iter
        self.initial_sampler = initial_sampler
        self.metric = metric
        self.plot = plot

    def _run_iter(self, iter, X, y, learner, labeled_indexes, metrics):
        t0 = perf_counter()
        learner.fit(X[labeled_indexes], y[labeled_indexes])
        metrics[iter]['fit_time'] = perf_counter() - t0

        if self.metric:
            t0 = perf_counter()
            y_pred = learner.predict(X)
            metrics[iter]['predict_time'] = perf_counter() - t0
            metrics[iter]['accuracy'] = self.metric(y_true=y, y_pred=y_pred)

        if self.plot:
            self.plot(X, y, learner, labeled_indexes)

    def run(self, X, y, learner):
        """
        Run Active Learning model over data, for a given number of iterations.

        :param X: data matrix
        :param y: labels array
        :param learner: Active Learning algorithm to run
        :return: labeled points chosen by the algorithm
        """
        X, y = sklearn.utils.check_X_y(X, y)

        metrics = [dict() for _ in range(self.iter+1)]

        # fit model over initial sample
        t0 = perf_counter()
        labeled_indexes = self.initial_sampler(y)
        metrics[0]['index'] = labeled_indexes.copy()

        self._run_iter(0, X, y, learner, labeled_indexes, metrics)
        metrics[0]['iter_time'] = perf_counter() - t0

        # run iterations
        for i in range(self.iter):
            # get next point to label
            t0 = perf_counter()
            idx = learner.get_next(X, labeled_indexes)
            metrics[i+1]['get_next_time'] = perf_counter() - t0
            metrics[i+1]['index'] = idx
            labeled_indexes.append(idx)

            self._run_iter(i+1, X, y, learner, labeled_indexes, metrics)
            metrics[i+1]['iter_time'] = perf_counter() - t0

        return metrics
