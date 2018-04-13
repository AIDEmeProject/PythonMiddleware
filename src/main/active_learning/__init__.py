from .svm import SimpleMargin, SolverMethod, MajorityVote
from .svm import *
from .agnostic import RandomLearner
from .boosting import QueryByBoosting, ActBoost
from .bayesian import BayesianLogisticActiveLearner, KernelBayesianActiveLearner

__all__=['RandomLearner', 'QueryByBoosting', 'ActBoost', 'SimpleMargin', 'SolverMethod', 'OptimalMargin', 'MajorityVote',
         'BayesianLogisticActiveLearner', 'KernelBayesianActiveLearner']

learner_configs = {
    'simplemargin': SimpleMargin,
    'solvermethod': SolverMethod,
    'optimalmargin': OptimalMargin,
    'majorityvote': MajorityVote,
    'random': RandomLearner,
    'actboost': ActBoost,
    'querybyboosting': QueryByBoosting
}

def get_active_learner(name, params):
    name = name.replace(" ", "").lower()
    learner = learner_configs[name]
    return learner(**params)
