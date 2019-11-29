"""
All functions in the module are possible criteria for stopping the exploration process. A valid "convergence criteria"
is any function with the following signature:

    def convergence_criteria(data, metrics):
        return True if exploration can stop, False otherwise

Here, 'data' is a PartitionedDataset instance containing the current data partition, and 'metrics' is the dictionary
of all callback_metrics which have been computed in this iteration.
"""

def all_points_are_known(data, metrics):
    """
    :return: whether no points remain in the unknown partition
    """
    return data.unknown_size == 0


def metric_reached_threshold(metric, threshold):
    """
    Computes a convergence criteria which stops exploration when a given callback metric reaches a certain threshold.
    Note that exploration may take a few iterations more to stop depending on the 'callback_skip' exploration parameter.

    :param metric: metric name
    :param threshold:
    :return: a convergence criteria
    """
    def converged(data, metrics):
        if metric not in metrics:
            return False

        return metrics[metric] >= threshold

    return converged
