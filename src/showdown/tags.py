from collections import namedtuple

DataTag = namedtuple('DataTag', ['tags', 'datasets', 'users'])
LearnerTag = namedtuple('LearnerTag', ['tags', 'learners', 'samplers'])


def list_to_tag(datasets_list, active_learners_list):
    datasets = DataTag(
        tags=[x[0] for x in datasets_list],
        datasets=[x[1] for x in datasets_list],
        users=[x[2] for x in datasets_list]
    )

    active_learners = LearnerTag(
        tags=[x[0] for x in active_learners_list],
        learners=[x[1] for x in active_learners_list],
        samplers=[x[2] for x in active_learners_list]
    )

    return datasets, active_learners
