import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from featurewiz import featurewiz
np.random.seed(1234)

data = pd.read_csv('0.9_5subjectslabelled_data.csv')

X = data.drop(['Gait Cycle Phase'], axis=1)
y = data['Gait Cycle Phase'].values

X_scaled = StandardScaler().fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size = 0.2, stratify=y, random_state=1)

classifier = SVC()

classifier.fit(X_train, y_train)

# make prediction
preds = classifier.predict(X_valid)
# check performance
accuracy_score(preds,y_valid)

# automatic feature selection by using featurewiz package
target = 'Gait Cycle Phase'

features, train = featurewiz(data, target, corr_limit=0.7, verbose=2, sep=",",
                             header=0, test_data="", feature_engg="", category_encoders="")
print(features)
# split data into feature and target
X_new = train.drop(['Gait Cycle Phase'], axis=1)

y = train['Gait Cycle Phase'].values
# preprocessing the features
X_scaled =  StandardScaler().fit_transform(X_new)

#split data into train and validate

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size = 0.2, stratify=y, random_state=1)
classifier = SVC()
classifier.fit(X_train, y_train)

# make prediction

preds = classifier.predict(X_valid)

# check performance

accuracy_score(preds, y_valid)
