import pandas as pd
from mafese import Data
from mafese.wrapper.sequential import SequentialSelector

# load dataset
dataset = pd.read_csv('0.9_5subjectslabelled_data.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)

data.split_train_test(test_size=0.2, inplace=True)

# define problem, estimator, and direction values
problems = ["classification", "regression"]
estimators = ['knn', 'svm', 'rf', 'adaboost', 'xgb', 'tree', 'ann']
directions = ["forward", "backward"]

# iterate over all combinations
for problem in problems:
    for estimator in estimators:
        for direction in directions:
            # initialize feature selector
            feat_selector = SequentialSelector(problem=problem, estimator=estimator, n_features=3, direction=direction)

            # perform feature selection
            feat_selector.fit(data.X_train, data.y_train)

            # print selected feature indexes
            print(f"Problem: {problem}, Estimator: {estimator}, Direction: {direction}")
            print(feat_selector.selected_feature_indexes)
            print("------------------------")
