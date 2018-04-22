from ..io import read_task
from ..user import DummyUser

def get_dataset_and_user(task, keep_duplicates=False, noise=0.0):
    # config = read_task_config(task)
    #
    # dataset_config = config.get('dataset')
    # data = read_dataset(dataset_config.get('name'), dataset_config.get('columns', None), keep_duplicates)
    #
    # user_config = config.get('user')
    # user_config['dataset_name'] = dataset_config['name']
    # user_config['noise'] = noise
    # user = get_user(data, **user_config)
    #
    # preprocessor = PreprocessingList(config.get('preprocessing', None))
    # data = preprocessor.preprocess(data)
    #
    # return data, user
    data, labels = read_task(task, keep_duplicates)
    return data, DummyUser(labels, 100)

