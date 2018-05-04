import os
from datetime import datetime

from pandas import read_csv

from definitions import ROOT_DIR


class ExperimentDirManager:
    def __init__(self):
        self.root = os.path.join(ROOT_DIR, 'experiments')
        self.experiment_folder = None

    def set_new_experiment_folder(self):
        now = datetime.now()
        self.experiment_folder = os.path.join(self.root, 'tmp', str(now))
        os.makedirs(self.experiment_folder)

    def get_data_folder(self, data_tag, learner_tag):
        return DataFolder(self.experiment_folder, data_tag, learner_tag)


class DataFolder:
    def __init__(self, experiment_folder, data_tag, learner_tag):
        self.path = os.path.join(experiment_folder, data_tag, learner_tag)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_full_path(self, file):
        return os.path.join(self.path, file)

    def get_files(self, filter_str=''):
        return map(self.get_full_path,  filter(lambda name: filter_str in name, os.listdir(self.path)))

    def get_raw_runs(self):
        raw_files = self.get_files('raw')
        return (read_csv(file, sep='\t', index_col=['iter', 'index']) for file in raw_files)

    def read_average(self):
        return read_csv(self.get_full_path('average_fscore.tsv'), sep='\t')

    def write(self, data, filename, index=False):
        path = os.path.join(self.path, filename)
        data.to_csv(path, sep='\t', header=True, index=index, index_label=data.index.names, float_format='%.6f')
