from pandas import concat

from .directory_manager import ExperimentDirManager
from .explore import explore, compute_fscore
from .logger import ExperimentLogger
from .plot import ExperimentPlotter
from .utils import get_generator_average

from ..config import get_dataset_and_user


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

    def run(self, datasets, learners, times, initial_sampler):
        # check tags
        Experiment.__check_tags(datasets)
        Experiment.__check_tags(learners)

        # add new experiments folder
        self.dir_manager.set_new_experiment_folder()

        # set logging path
        self.logger.set_folder(self.dir_manager.experiment_folder)

        for data_tag, task_tag in datasets:
            # get data and user
            data, user = get_dataset_and_user(task_tag)

            # get new random state
            initial_sampler.new_random_state()

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
                        sample = initial_sampler(data, user)

                        # log task begin
                        self.logger.begin(data_tag, learner_tag, i+1, list(sample.index))

                        # run task
                        X, y = explore(data, user, learner, sample)

                        # persist metrics
                        # filename = "run{0}_time.tsv".format(i+1)
                        # self.dir_manager.persist(metrics, data_tag, learner_tag, filename)

                        # persist run
                        filename = "run{0}_raw.tsv".format(i+1)
                        X['labels'] = y

                        data_folder.write(X, filename, index=True)

                    # self.dir_manager.compute_folder_average(data_tag, learner_tag)

                except Exception as e:
                    # if error occurred, log error and add learner to skip list
                    self.logger.error(e)
                    self.skip_list.append(learner_tag)

                finally:
                    pass  # continue to next tasks

                # reset random state, so all learners are run over the set of initial samples
                initial_sampler.reset_random_state()

        # log experiment end
        self.logger.end()

    def get_average_fscores(self, datasets, learners):
        for data_tag, task_tag in datasets:
            data, user = get_dataset_and_user(task_tag, keep_duplicates=True)
            y_true = user.get_label(data, update_counter=False)

            for learner_tag, learner in learners:
                # get runs
                data_folder = self.dir_manager.get_data_folder(data_tag, learner_tag)
                runs = data_folder.get_raw_runs()

                # compute average
                scores = [compute_fscore(data, y_true, learner, run) for run in runs]
                final = concat(scores, axis=1)
                final.columns = ['run{0}'.format(i+1) for i in range(len(scores))]
                final['average'] = final.mean(axis=1)

                data_folder.write(final, "average_fscore.tsv", index=False)

    def make_plot(self, datasets, learners, iter_lim=None):
        self.plotter.plot_comparisons(datasets, learners, iter_lim)