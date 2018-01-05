from .directory_manager import ExperimentDirManager
from .logger import ExperimentLogger
from ..config import get_dataset_and_user
from .task import explore, compute_fscore
from pandas import concat

class Experiment:
    def __init__(self):
        self.skip_list = []
        self.logger = ExperimentLogger()
        self.dir_manager = ExperimentDirManager()

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
        self.dir_manager.set_experiment_folder()

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
                self.dir_manager.add_folder(data_tag, learner_tag)

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

                        self.dir_manager.persist(X, data_tag, learner_tag, filename)

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
                final_scores = [compute_fscore(data, y_true, learner, run) for run in self.dir_manager.get_raw_runs(data_tag, learner_tag)]
                avg = sum(final_scores)/len(final_scores)
                final = concat(final_scores + [avg], axis=1)
                final.columns = ['run{0}'.format(i) for i in range(len(final_scores))] + ['average']
                self.dir_manager.persist(final, data_tag, learner_tag, "average_fscore.tsv")
