from functools import partial
from pandas import get_dummies

class Preprocessor:
    def __init__(self):
        self.__preprocessing_list = []
        self.mapper = {
            'onehotencoding': partial(get_dummies, drop_first=True),
            'standardize': lambda df: (df - df.mean()) / df.std(),
            'minmax': lambda df: (df - df.min()) / (df.max() - df.min()),
            'frombinarytosign': lambda df: 2.0 * df - 1.0
        }

    def get_map(self, name):
        name = name.strip().replace(' ', '').lower()
        return self.mapper[name]

    def set(self, preprocessors):
        self.__preprocessing_list = map(self.get_map, preprocessors)

    def transform(self, data):
        for transformation in self.__preprocessing_list:
            data = transformation(data)
        return data
