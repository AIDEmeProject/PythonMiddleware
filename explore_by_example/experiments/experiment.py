import pandas as pd

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
