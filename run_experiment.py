from src.main.config import get_dataset_and_user
from src.main.active_learning.svm import *
from src.main.experiments import Experiment
from src.main.initial_sampling import StratifiedSampler


def get_user_study(ls):
    datasets_list = []
    for i in ls:
        num = str(i)
        if i < 10:
            num = '0' + num

        X, user = get_dataset_and_user('user_study_query' + str(i))
        datasets_list.append(('query ' + num, X, user))
    return datasets_list

def get_sdss():
    datasets_list = []
    for query in ['1.1', '1.2', '1.3', '2.1', '2.2', '2.3', '3.1', '3.2', '3.3']:
        name = 'sdss_Q' + query
        X, user = get_dataset_and_user(name)
        datasets_list.append(('SDSS Query ' + query, X, user))
    return datasets_list


#datasets_list = get_user_study([11, 12])  # range(1,13)
datasets_list = get_sdss()


# set learners
active_learners_list = [
    ("Simple Margin", SimpleMargin(top=500, kind='kernel', kernel='rbf', C=1024)),
    ("Optimal Margin", OptimalMargin(top=500, cholesky=False, chain_length=64, sample_size=8, kind='kernel', kernel='rbf', C=1024)),
    ("Cholesky Optimal Margin", OptimalMargin(top=500, cholesky=True, chain_length=64, sample_size=8, kind='kernel', kernel='rbf', C=1024)),
    #("Simple Margin C=100000", SimpleMargin(top=-1, kind='kernel', kernel='rbf', C=100000)),
    ("Majority Vote", MajorityVote(top=500, chain_length=64, sample_size=8)),
    ("Majority Vote + Cholesky", MajorityVote(top=500, cholesky=True, chain_length=64, sample_size=8))
]

# run experiment
experiment = Experiment(times=1, sampler=StratifiedSampler(1, 1))
experiment.run(datasets_list, active_learners_list)
