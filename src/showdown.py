from pandas import concat
from .active_learning.base import train
from .metrics import MetricStorage


def compare_learners(active_learners, times, datasets):

    dataset_names = []
    dataset_output = []

    # train over all databases
    for data_tag, data, user in datasets:

        # train all active learners for this database
        al_names = []
        al_output = []
        for active_learner_tag, initial_sampler, active_learner in active_learners:
            storage = MetricStorage()

            # train learner for several iterations
            for _ in range(times):
                user.clear()
                storage.persist(train(data, user, active_learner, initial_sampler))
                active_learner.clear()

            # compute average performance
            al_output.append(storage.average_performance())
            al_names.append(active_learner_tag)

        dataset_output.append(concat(al_output, axis=1, keys=al_names))
        dataset_names.append(data_tag)

    dataset_output = concat(dataset_output, axis=1, keys=dataset_names)
    dataset_output = dataset_output.swaplevel(2, 3, axis=1).swaplevel(1, 2, axis=1).sortlevel(0, axis=1)
    return dataset_output
