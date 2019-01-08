import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ss
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv("creditcard.csv")
#print(data.columns)
#print(data.shape)
#print(data.describe())
#data.hist(figsize = (20,20))
#plt.show()
#determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)
print("fraud cases: {}".format(len(fraud)))
print ("valid cases: {}".format(len(valid)))
#correlation matrix
#cormat = data.corr()
#fig = plt.figure(figsize = (12, 9))
#ss.heatmap(cormat, vmax = .8, square = True)
#plt.show()
#Get all the columns from the DataFrame
columns = data.columns.tolist()

#filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

#store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

#print the shapes of X an Y
print(X.shape)
print(Y.shape)

#define a random state
state = 1

#define the outlier detection methods
classifiers = {
        "Isolation Forest": IsolationForest(max_samples=len(X),
            contamination = outlier_fraction,random_state = state),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors = 20,
            contamination = outlier_fraction)
        }
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            score_pred = clf.negative_outlier_factor_
        else:
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)

        #Reshape the prediction values to 0 for valid, 1 for fraud
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1

        n_errors = (y_pred != Y).sum()

        #Run classification metrics
        print('{}: {}'.format(clf_name, n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y, y_pred))

