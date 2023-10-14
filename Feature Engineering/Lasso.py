import pandas as pd
from mafese import Data

from mafese.embedded.lasso import LassoSelector
from mafese import get_dataset, evaluator
from sklearn.svm import SVC

# load X and y
# NOTE mafese accepts numpy arrays only, hence the .values attribute
dataset = pd.read_csv('0.9_5subjectslabelled_data.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]
data = Data(X, y)

data.split_train_test(test_size=0.2, inplace=True)
print(data.X_train[:2].shape)
print(data.y_train[:2].shape)

# define mafese feature selection method
feat_selector = LassoSelector(problem="classification", estimator="lasso", estimator_paras={"alpha": 0.1})
# find all relevant features
feat_selector.fit(data.X_train, data.y_train)

# check selected features - True (or 1) is selected, False (or 0) is not selected
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)

# check the index of selected features
print(feat_selector.selected_feature_indexes)

# call transform() on X to filter it down to selected features
X_train_selected = feat_selector.transform(data.X_train)
X_test_selected = feat_selector.transform(data.X_test)

# Evaluate final dataset with different estimator with multiple performance metrics (classification)
results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS"])
print(results)

# Change problem to regression
feat_selector.problem = "regression"

# Evaluate final dataset with different estimator with multiple performance metrics (regression)
results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["MSE", "RMSE"])
print(results)
