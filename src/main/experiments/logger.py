import logging
from os.path import join


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

    def set_folder(self, folder_path, filename='experiment.log'):
        self.remove_handlers()

        handler = logging.FileHandler(join(folder_path, filename))
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def clear(self):
        self.total = 0
        self.skips = 0
        self.errors = 0

    def begin(self, data_tag, learner_tag, run_id, sample_index):
        self.total += 1

        self.logger.info("Starting experiment #{0}: TASK = {1}, LEARNER = {2}, RUN = {3}, SAMPLE = {4}".format(
            self.total,
            data_tag,
            learner_tag,
            run_id,
            list(sample_index)
        ))

    def averaging(self, data_tag, learner_tag):
        self.total += 1
        self.logger.info("Starting F-score computation: TASK = {0}, LEARNER = {1}".format(data_tag, learner_tag))

    def skip(self):
        self.skips += 1
        self.total += 1
        self.logger.info("Skipping experiment #{0}".format(self.total))

    def error(self, exception):
        self.errors += 1

        self.logger.error(exception, exc_info=1)

    def end(self):
        msg = "Finished all experiments! Total: {0}, Success: {1}, Fail: {2}, Skipped: {3}".format(
            self.total,
            self.total - self.errors - self.skips,
            self.errors,
            self.skips
        )
        self.logger.info(msg)
        return msg

