import os
from src.main.metrics import MetricTracker
from src.main.active_learning.svm import OptimalMargin

def get_average_fscores(dir_manager, datasets, learners):
    for data_tag, task_tag in datasets:
        # get data and user
        data, user = get_dataset_and_user(task_tag)
        y_true = user.get_label(data, update_counter=False)

        for learner_tag, learner in learners:
            final_scores = []
            for run in dir_manager.get_raw_runs(data_tag, learner_tag):
                tracker = MetricTracker()
                X_run = run.drop('labels', axis=1)
                y_run = run['labels']

                learner.clear()
                learner.initialize(data)

                for i in range(2, len(X_run)):
                    X, y = X_run.iloc[:i], y_run.iloc[:i]
                    learner.fit_classifier(X, y)
                    if i == 2:
                        learner.update(X, y)
                    else:
                        learner.update(X.iloc[[-1]], y.iloc[[-1]])
                    tracker.add_measurement(learner.score(data, y_true))

                final_scores.append(tracker.to_dataframe())
            final = sum(final_scores)/len(final_scores)
            dir_manager.persist(final, data_tag, learner_tag, "average_fscore.tsv")


if __name__ == '__main__':
    dir_manager = ExperimentDirManager() 
    dir_manager.experiment_folder = os.path.join(dir_manager.root, 'tmp', '2017-12-13 21:45:13.696469')
    get_average_fscores(
        dir_manager, 
        [('SDSS Query 1.1', 'sdss_Q1.1')], 
        [('Optimal Margin top=50000', OptimalMargin(kind='kernel', kernel='rbf', C=1024)),
         #('Optimal Margin top=50000', OptimalMargin(kind='kernel', kernel='rbf', C=100000))
        ]
    )