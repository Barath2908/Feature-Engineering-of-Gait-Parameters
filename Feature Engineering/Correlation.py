import pandas as pd
from mafese.filter import FilterSelector
from mafese import get_dataset, evaluator
from sklearn.svm import SVC

# load dataset
dataset = pd.read_csv('0.9_5subjectslabelled_data.csv', index_col=0).values
X, y = dataset[:, 0:-1], dataset[:, -1]     # Assumption that the last column is label column

# Define the feature selection methods for classification
classification_methods = ["ANOVA", "MI", "KENDALL", "SPEARMAN", "POINT"]

# Iterate over the classification methods
for method in classification_methods:
    print(f"--- Classification: {method} ---")
    # Define mafese feature selection method
    feat_selector = FilterSelector(problem='classification', method=method, n_features=5)
    # Find all relevant features
    feat_selector.fit(X, y)

    # Check selected features - True (or 1) is selected, False (or 0) is not selected
    print("Selected feature masks:")
    print(feat_selector.selected_feature_masks)

    # Check the scores/ranks of selected features
    print("Selected feature solution:")
    print(feat_selector.selected_feature_solution)

    # Check the index of selected features
    print("Selected feature indexes:")
    print(feat_selector.selected_feature_indexes)

    # Call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    print("Shape of filtered X:")
    print(X_filtered.shape)


# Define the feature selection methods for regression
regression_methods = ["PEARSON", "ANOVA", "MI", "KENDALL", "SPEARMAN", "POINT"]

# Iterate over the regression methods
for method in regression_methods:
    print(f"--- Regression: {method} ---")
    # Define mafese feature selection method
    feat_selector = FilterSelector(problem='regression', method=method, n_features=5)
    # Find all relevant features
    feat_selector.fit(X, y)

    # Check selected features - True (or 1) is selected, False (or 0) is not selected
    print("Selected feature masks:")
    print(feat_selector.selected_feature_masks)

    # Check the scores/ranks of selected features
    print("Selected feature solution:")
    print(feat_selector.selected_feature_solution)

    # Check the index of selected features
    print("Selected feature indexes:")
    print(feat_selector.selected_feature_indexes)

    # Call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    print("Shape of filtered X:")
    print(X_filtered.shape)
