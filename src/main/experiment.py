from collections import defaultdict

import pandas as pd

from .task import Task


class Experiment:
    def run(self, datasets_list, active_learners_list, times, sampler):
        results = defaultdict(list)

        for data_tag, data, user in datasets_list:
            for i in range(times):
                user.clear()
                sample = sampler(data, user)

                for al_tag, al in active_learners_list:
                    task = Task(data, user, al)
                    results[(data_tag, al_tag)].append(task.train(sample))

        for k, v in results.items():
            results[k] = self.average_results(v)

        keys = results.keys()
        final = pd.concat([results[k] for k in keys], axis=1, keys=keys)
        return final.swaplevel(2, 3, axis=1).swaplevel(1, 2, axis=1).sort_index(level=0, axis=1)

    def average_results(self, metrics_list):
        storage = metrics_list #[metric.to_dataframe() for metric in metrics_list]

        # concatenate all stored metrics
        concat_metrics = pd.concat(storage, axis=0)

        # group metrics by index
        grouped_metrics = concat_metrics.groupby(level=0)

        # return mean and standard deviation
        return pd.concat(
            [
                grouped_metrics.mean(),
                grouped_metrics.std().fillna(0),
                grouped_metrics.min(),
                grouped_metrics.max()
            ],
            axis=1,
            keys=['mean', 'std', 'min', 'max']
        )


from numpy.random import RandomState
import os
from datetime import datetime
from definitions import ROOT_DIR
from src.main.task import Task
import logging

EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
logger = logging.getLogger('experiment')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


class Exp:
    def __init__(self, times, sampler):
        self.times = times
        self.sampler = sampler

        self.skip_list = []
        self.total = 0
        self.skipped = 0
        self.errors = 0

    def get_root_folder(self):
        # get current datetime
        now = datetime.now()

        # create new folder for current experiments
        folder_path = os.path.join(EXPERIMENTS_DIR, 'tmp', str(now))
        os.makedirs(folder_path)

        return folder_path

    def add_folder(self, path, name):
        new_folder = os.path.join(path, name)
        os.makedirs(new_folder)
        return new_folder

    def set_logger(self, folder_path):
        handler = logging.FileHandler(os.path.join(folder_path, 'experiment.log'))
        handler.setFormatter(formatter)
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logger.addHandler(handler)


    def run(self, datasets, learners):
        folder_path = self.get_root_folder()
        self.set_logger(folder_path)

        for data_tag, data, user in datasets:
            self.sampler.new_random_state()

            dataset_folder = self.add_folder(folder_path, data_tag)

            for learner_tag, learner in learners:
                # if AL has failed before, skip it
                if learner_tag in self.skip_list:
                    self.skipped += 1
                    self.total += 1
                    logger.info("Skipping experiment #{0}".format(self.total))
                    continue

                learner_folder = self.add_folder(dataset_folder, learner_tag)
                task = Task(data, user, learner)

                # get average results
                try:
                    for i in range(self.times):
                        # get initial sample
                        sample = self.sampler(data, user)

                        # train
                        self.total += 1
                        logger.info("Starting experiment #{4}: TASK = {0}, LEARNER = {1}, RUN = {2}, SAMPLE = {3}".format(
                            data_tag,
                            learner_tag,
                            i + 1,
                            list(sample.index),
                            self.total
                        ))
                        result = task.train(sample)

                        filename = "run{0}.tsv".format(i + 1)
                        result_path = os.path.join(learner_folder, filename)
                        result.to_csv(result_path, sep='\t', header=True, index=True, index_label='iter', float_format='%.6f')
                except Exception as e:
                    logger.error(e, exc_info=1)
                    self.skip_list.append(learner_tag)  # avoid rerunning failed algorithm
                    self.errors += 1
                finally:
                    pass  # continue to next tasks

                self.sampler.reset_random_state()

        logger.info("Finished all experiments! Total: {0}, Success: {1}, Fail: {2}, Skipped: {3}".format(self.total,
                                                                                                    self.total - self.errors - self.skipped,
                                                                                                    self.errors,
                                                                                                    self.skipped))


