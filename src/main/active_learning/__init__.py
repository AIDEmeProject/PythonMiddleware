from .svm import SimpleMargin, SolverMethod, OptimalMargin, MajorityVote
from .agnostic import RandomLearner
from .boosting import QueryByBoosting, ActBoost

__all__=['RandomLearner', 'QueryByBoosting', 'ActBoost', 'SimpleMargin', 'SolverMethod', 'OptimalMargin']

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
