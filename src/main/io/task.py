from .dataset import read_dataset
from .label import read_labels
from .preprocessing import preprocess
from .utils import get_config_from_resources


def read_task(task, distinct=False, get_raw=False):
    """
    Get data and labels for a given task.

    :param task: task to retrieve data
    :param distinct: whether to remove duplicates from data
    :param get_raw: whether to recover raw data. If False, preprocessing is applied
    :return: data and labels
    """
    # get task config
    task_config = get_config_from_resources('tasks', task)

    # read data
    dataset_config = task_config['dataset']
    data = read_dataset(distinct=distinct,  **dataset_config)

    # read labels
    labels_config = task_config.get('user', {})
    labels = read_labels(data=data, **labels_config)

    if not get_raw:
        preprocess_list = task_config.get('preprocessing', [])
        data = preprocess(data, preprocess_list)

    return data, labels
