import os
from datetime import datetime

from pandas import read_csv

from .utils import get_generator_average
from definitions import ROOT_DIR


class ExperimentDirManager:
    def __init__(self):
        self.root = os.path.join(ROOT_DIR, 'experiments')
        self.experiment_folder = None

    def set_experiment_folder(self):
        now = datetime.now()
        self.experiment_folder = os.path.join(self.root, 'tmp', str(now))
        os.makedirs(self.experiment_folder)

    def persist(self, data, data_tag, learner_tag, filename):
        folder = self.get_folder_path(data_tag, learner_tag)
        path = os.path.join(folder, filename)
        data.to_csv(path, sep='\t', header=True, index=True, index_label=data.index.names, float_format='%.6f')

    def compute_folder_average(self, data_tag, learner_tag):
        run_files = self.get_run_files(data_tag, learner_tag)

        if not run_files:  # if empty, do nothing
            return

        runs = (read_csv(f, sep='\t', index_col=['iter', 'index']) for f in run_files)
        average = get_generator_average(runs)
        self.persist(average, data_tag, learner_tag, 'average.tsv')

    def get_run_files(self, data_tag, learner_tag):
        folder = self.get_folder_path(data_tag, learner_tag)
        return [os.path.join(folder, f) for f in os.listdir(folder) if 'time' in f]

    def add_folder(self, data_tag, learner_tag):
        path = self.get_folder_path(data_tag, learner_tag)
        os.makedirs(path)

    def get_folder_path(self, data_tag, learner_tag):
        return os.path.join(self.experiment_folder, data_tag, learner_tag)

    def read_average(self, data_tag, learner_tag, metrics=None):
        folder = self.get_folder_path(data_tag, learner_tag)
        path = os.path.join(folder, 'average.tsv')
        if metrics is not None and 'iter' not in metrics:
            metrics.append('iter')
        return read_csv(path, sep='\t', index_col='iter', usecols=metrics)

    def get_raw_run_files(self, data_tag, learner_tag):
        folder = self.get_folder_path(data_tag, learner_tag)
        return [os.path.join(folder, f) for f in os.listdir(folder) if 'raw' in f]

    def get_raw_runs(self, data_tag, learner_tag):
        files = self.get_raw_run_files(data_tag, learner_tag)
        return (read_csv(file, sep='\t', index_col=['iter', 'index']) for file in files)
