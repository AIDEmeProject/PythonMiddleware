from .directory_manager import ExperimentDirManager
from .logger import ExperimentLogger
from .task import Task
from ..config import get_dataset_and_user
from ..metrics import MetricTracker

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
        self.dir_manager.set_experiment_folder()

        # set logging path
        self.logger.set_folder(self.dir_manager.experiment_folder)

        for data_tag, task_tag in datasets:
            # get data and user
            data, user = get_dataset_and_user(task_tag)

            # get new random state
            self.initial_sampler.new_random_state()

            for learner_tag, learner in learners:
                # if learners failed previously, skip it
                if learner_tag in self.skip_list:
                    self.logger.skip()
                    continue

                # add learner folder
                self.dir_manager.add_folder(data_tag, learner_tag)

                # create new task and try to run it
                try:
                    task = Task(data, user, learner)
                    for i in range(self.times):
                        # get initial sample
                        sample = self.initial_sampler(data, user)

                        # log task begin
                        self.logger.begin(data_tag, learner_tag, i+1, list(sample.index))

                        # run task
                        metrics, (X,y) = task.train(sample)

                        # persist metrics
                        filename = "run{0}_metrics.tsv".format(i+1)
                        self.dir_manager.persist(metrics, data_tag, learner_tag, filename)

                        # persist run
                        filename = "raw_run{0}.tsv".format(i+1)
                        X['labels'] = y
                        #X.index = [0,0] + list(metrics.index)
                        self.dir_manager.persist(X, data_tag, learner_tag, filename)

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

    def get_average_fscores(self, datasets, learners):
        for data_tag, task_tag in datasets:
            # get data and user
            data, user = get_dataset_and_user(task_tag)
            y_true = user.get_label(data, update_counter=False)

            for learner_tag, learner in learners:
                final_scores = []
                for run in self.dir_manager.get_raw_runs(data_tag, learner_tag):
                    tracker = MetricTracker()
                    X_run = run.drop('labels', axis=1)
                    y_run = run['labels']

                    learner.clear()
                    learner.initialize(data)

                    for i in range(2, len(X_run)):
                        X, y = X_run.iloc[:i], y_run.iloc[:i]
                        learner.fit_classifier(X, y)
                        if i == 2:
                            learner.update(X, y)
                        else:
                            learner.update(X.iloc[[-1]], y.iloc[[-1]])
                        tracker.add_measurement(learner.score(data, y_true))

                    final_scores.append(tracker.to_dataframe())
                final = sum(final_scores)/len(final_scores)
                self.dir_manager.persist(final, data_tag, learner_tag, "average_fscore.tsv")
