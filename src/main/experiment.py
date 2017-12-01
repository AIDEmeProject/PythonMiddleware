import os
import logging
from datetime import datetime

from definitions import ROOT_DIR
from .task import Task


EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')


class ExperimentLogger:
    def __init__(self):
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.total = 0
        self.skips = 0
        self.errors = 0

    def remove_handlers(self):
        for handler in self.logger.handlers[:]:  # loop through a copy to avoid simultaneously looping and deleting
            self.logger.removeHandler(handler)

    def set_folder(self, folder_path):
        self.remove_handlers()

        handler = logging.FileHandler(os.path.join(folder_path, 'experiment.log'))
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def begin(self, data_tag, learner_tag, run_id, sample_index):
        self.total += 1

        self.logger.info("Starting experiment #{0}: TASK = {1}, LEARNER = {2}, RUN = {3}, SAMPLE = {4}".format(
            self.total,
            data_tag,
            learner_tag,
            run_id,
            list(sample_index)
        ))

    def skip(self):
        self.skips += 1
        self.total += 1
        self.logger.info("Skipping experiment #{0}".format(self.total))

    def error(self, exception):
        self.errors += 1

        self.logger.error(exception, exc_info=1)

    def end(self):
        self.logger.info("Finished all experiments! Total: {0}, Success: {1}, Fail: {2}, Skipped: {3}".format(
            self.total,
            self.total - self.errors - self.skips,
            self.errors,
            self.skips
        ))


class ExperimentDirManager:
    def __init__(self):
        self.root_folder = os.path.join(ROOT_DIR, 'experiments')
        self.experiment_folder = None

    def add_experiment_folder(self):
        now = datetime.now()
        self.experiment_folder = os.path.join(self.root_folder, 'tmp', str(now))
        self.create_dir(self.experiment_folder)

    def add_data_folder(self, data_tag):
        path = os.path.join(self.experiment_folder, data_tag)
        self.create_dir(path)

    def add_learner_folder(self, data_tag, learner_tag):
        path = self.get_learner_folder_path(data_tag, learner_tag)
        self.create_dir(path)

    def get_learner_folder_path(self, data_tag, learner_tag):
        return os.path.join(self.experiment_folder, data_tag, learner_tag)

    def create_dir(self, path):
        os.makedirs(path)

    def persist_results(self, results, data_tag, learner_tag, run_id):
        filename = "run{0}.tsv".format(run_id)
        folder = self.get_learner_folder_path(data_tag, learner_tag)
        result_path = os.path.join(folder, filename)

        results.to_csv(result_path, sep='\t', header=True, index=True, index_label='iter', float_format='%.6f')


class Experiment:
    def __init__(self, times, sampler):
        self.times = times
        self.initial_sampler = sampler

        self.skip_list = []

        self.logger = ExperimentLogger()
        self.dir_manager = ExperimentDirManager()

    def __check_tags(self, ls):
        tags = [x[0] for x in ls]
        if len(tags) != len(set(tags)):
            raise ValueError("All tags must be distinct!")

    def run(self, datasets, learners):
        # check tags
        self.__check_tags(datasets)
        self.__check_tags(learners)

        # add new experiments folder
        self.dir_manager.add_experiment_folder()

        # set logging path
        self.logger.set_folder(self.dir_manager.experiment_folder)

        for data_tag, data, user in datasets:
            # create dataset folder
            self.dir_manager.add_data_folder(data_tag)

            # get new random state
            self.initial_sampler.new_random_state()

            for learner_tag, learner in learners:
                # if learners failed previously, skip it
                if learner_tag in self.skip_list:
                    self.logger.skip()
                    continue

                # add learner folder
                self.dir_manager.add_learner_folder(data_tag, learner_tag)

                # create new task and try to run it
                task = Task(data, user, learner)

                try:
                    for i in range(self.times):
                        # get initial sample
                        sample = self.initial_sampler(data, user)

                        # log task begin
                        self.logger.begin(data_tag, learner_tag, i+1, list(sample.index))

                        # run task
                        result = task.train(sample)

                        # persist results
                        self.dir_manager.persist_results(result, data_tag, learner_tag, i+1)

                except Exception as e:
                    # if error occurred, log error and add learner to skip list
                    self.logger.error(e)
                    self.skip_list.append(learner_tag)
                finally:
                    # continue to next tasks
                    pass

                # reset random state, so all learners are run over the set of initial samples
                self.initial_sampler.reset_random_state()

        # log experiment end
        self.logger.end()
