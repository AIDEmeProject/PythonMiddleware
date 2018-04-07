from .dataset import read_dataset
from .preprocessing import PreprocessingList
from .user import get_user
from .utils import read_task_config
from ..datasets import *

def get_dataset_and_user(task, keep_duplicates=False, noise=0.0):
    if '=' in task:
        return parse_dataset(task)

    config = read_task_config(task)

    dataset_config = config.get('dataset')
    keep_duplicates = config.get('keep_duplicates', keep_duplicates)
    data = read_dataset(dataset_config.get('name'), dataset_config.get('columns', None), keep_duplicates)

    user_config = config.get('user')
    user_config['dataset_name'] = dataset_config['name']
    user_config['noise'] = noise
    user = get_user(data, **user_config)

    preprocessor = PreprocessingList(config.get('preprocessing', None))
    data = preprocessor.preprocess(data)

    return data, user


def parse_dataset(task_string):
    elems = task_string.split('_')

    params = {}
    for elem in elems[1:]:
        key,val = elem.split('=')
        params[key] = float(val)

    if elems[0] == 'circle':
        return circle_query(N=10000, center=[0]*int(params['dim']), sel=params['sel'], sep=params['sep'])
    elif elems[0] == 'xor':
        return xor_query(N=10000, sel=params['sel'])

    raise ValueError("Unknown query type.")