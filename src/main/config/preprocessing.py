from pandas import get_dummies
from functools import partial

class PreprocessingList:
    __mapper = {
        'onehotencoding': partial(get_dummies, drop_first=False),
        'standardize': lambda df: (df - df.mean()) / df.std(),
        'minmax': lambda df: (df - df.min()) / (df.max() - df.min()),
        'frombinarytosign': lambda df: 2.0 * df - 1.0
    }

    def __init__(self, preprocessing=None):
        self._preprocessing_list = [self.parse_function(func) for func in preprocessing] if preprocessing is not None else []

    def parse_function(self, func):
        if callable(func):
            return func
        elif isinstance(func, str):
            func = ''.join([c for c in func if c.isalnum()])
            return PreprocessingList.__mapper[func]
        else:
            raise ValueError("Unrecognized function.")

    def append(self, func):
        self._preprocessing_list.append(self.parse_function(func))

    def preprocess(self, data):
        for func in self._preprocessing_list:
            data = func(data)
        return data
