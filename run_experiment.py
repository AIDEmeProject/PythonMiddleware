from src.main.active_learning.svm import *
from src.main.experiments import Experiment
from src.main.initial_sampling import StratifiedSampler


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
    ("Simple Margin", SimpleMargin(top=-1, kind='kernel', kernel='rbf', C=1024)),
    #("Optimal Margin", OptimalMargin(top=-1, cholesky=False, chain_length=64, sample_size=8, kind='kernel', kernel='rbf', C=100000)),
    #("Cholesky Optimal Margin", OptimalMargin(top=500, cholesky=True, chain_length=64, sample_size=8, kind='kernel', kernel='rbf', C=1024)),
    #("Simple Margin C=100000", SimpleMargin(top=-1, kind='kernel', kernel='rbf', C=100000)),
    #("Majority Vote", MajorityVote(top=500, chain_length=64, sample_size=8)),
    #("Majority Vote + Cholesky", MajorityVote(top=500, cholesky=True, chain_length=64, sample_size=8))
]

# run experiment
experiment = Experiment(times=2, sampler=StratifiedSampler(1, 1))
experiment.run(datasets_list, active_learners_list)
experiment.get_average_fscores(datasets_list, active_learners_list)