from .svm import SimpleMargin, SolverMethod, OptimalMargin
from .agnostic import RandomLearner
from .boosting import QueryByBoosting, ActBoost

__all__=['RandomLearner', 'QueryByBoosting', 'ActBoost', 'SimpleMargin', 'SolverMethod', 'OptimalMargin']

learner_configs = {
    'simplemargin': SimpleMargin,
    'solvermethod': SolverMethod,
    'optimalmargin': OptimalMargin,
    'random': RandomLearner,
    'actboost': ActBoost,
    'querybyboosting': QueryByBoosting
}