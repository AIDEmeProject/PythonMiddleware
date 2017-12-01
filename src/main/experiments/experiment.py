from .directory_manager import ExperimentDirManager
from .logger import ExperimentLogger
from .task import Task


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
        self.dir_manager.new_experiment_folder()

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
                try:
                    task = Task(data, user, learner)
                    for i in range(self.times):
                        # get initial sample
                        sample = self.initial_sampler(data, user)

                        # log task begin
                        self.logger.begin(data_tag, learner_tag, i+1, list(sample.index))

                        # run task
                        metrics = task.train(sample)

                        # persist experiments
                        filename = "run{0}.tsv".format(i+1)
                        self.dir_manager.persist(metrics, data_tag, learner_tag, filename)

                    self.dir_manager.compute_folder_average(data_tag, learner_tag)

                except Exception as e:
                    # if error occurred, log error and add learner to skip list
                    self.logger.error(e)
                    self.skip_list.append(learner_tag)

                finally:
                    pass  # continue to next tasks

                # reset random state, so all learners are run over the set of initial samples
                self.initial_sampler.reset_random_state()

        # log experiment end
        self.logger.end()
