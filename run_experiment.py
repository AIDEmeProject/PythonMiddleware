from explore_by_example.main.experiments import Experiment
from explore_by_example.main.initial_sampling import StratifiedSampler

from active_learning.svm import *


def get_user_study(ls):
    datasets_list = []
    for i in ls:
        num = str(i)
        if i < 10:
            num = '0' + num

        datasets_list.append(('User Study Query ' + num, 'user_study_query' + str(i)))
    return datasets_list

def get_sdss():
    query_list = ['1.1', '2.1', '3.1']
    return [('SDSS Query ' + q, 'sdss_Q' + q) for q in query_list]

#datasets_list = get_user_study([11, 12])  # range(1,13)
datasets_list = get_sdss()[:1]

# set learners
active_learners_list = [
    ("Simple Margin", SimpleMargin(kernel='rbf', C=1024)),
]

# run experiment
experiment = Experiment()
experiment.run(datasets_list, active_learners_list, times=1, initial_sampler=StratifiedSampler(1, 1), noise=0.1)
experiment.get_average_fscores(datasets_list, active_learners_list)
experiment.make_plot([x[0] for x in datasets_list], [x[0] for x in active_learners_list])