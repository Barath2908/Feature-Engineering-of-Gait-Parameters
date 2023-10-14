import pandas as pd
from mafese import Data
from mafese import get_dataset, evaluator
from mafese.wrapper.recursive import RecursiveSelector
from sklearn.svm import SVC, SVR


# Load dataset
dataset = pd.read_csv('0.9_5subjectslabelled_data.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]  # Assumption that the last column is label column

data = Data(X, y)

data.split_train_test(test_size=0.2, inplace=True)
print(data.X_train[:2].shape)
print(data.y_train[:2].shape)

# Define mafese feature selection methods
methods_classification = ['svm', 'rf', 'adaboost', 'xgb', 'tree']
methods_regression = ['svm', 'rf', 'adaboost', 'xgb', 'tree']

# Feature selection for classification problem
for method in methods_classification:
    # Define feature selector for classification
    feat_selector = RecursiveSelector(problem="classification", estimator=method, n_features=5)
    # Find all relevant features
    feat_selector.fit(data.X_train, data.y_train)
    # Check selected features
    print(f"Selected features for {method} (classification):")
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)
    # Call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    print(X_filtered)
    print("\n")
    results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["AS"])
    print(results)

# Feature selection for regression problem
for method in methods_regression:
    # Define feature selector for regression
    feat_selector = RecursiveSelector(problem="regression", estimator=method, n_features=5)
    # Find all relevant features
    feat_selector.fit(data.X_train, data.y_train)
    # Check selected features
    print(f"Selected features for {method} (regression):")
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    print(feat_selector.selected_feature_indexes)
    # Call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    print(X_filtered)
    print("\n")
    results = evaluator.evaluate(feat_selector, estimator=SVC(), data=data, metrics=["MSE", "RMSE"])
    print(results)