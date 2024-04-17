import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Assuming the files are in CSV format. Adjust the read function if they are in a different format.
dataset_train = pd.read_csv('dataset_train.csv')
dataset_test = pd.read_csv('dataset_test.csv')
label_train = pd.read_csv('label_train.csv')
label_test = pd.read_csv('label_test.csv')

# If the labels are not in a single column, adjust accordingly.
# For example, if they are one-hot encoded, you might need to convert them to a single label column.

# Create a pipeline that includes scaling and the classifier
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

# Fit the model
svm_pipeline.fit(dataset_train, label_train.values.ravel())

# Set the parameters by cross-validation
tuned_parameters = [{'svc__C': [0.1, 1, 10, 100],
                     'svc__gamma': [1, 0.1, 0.01, 0.001],
                     'svc__kernel': ['rbf']}]

clf = GridSearchCV(svm_pipeline, tuned_parameters, scoring='accuracy')
clf.fit(dataset_train, label_train.values.ravel())

print("Best parameters set found on development set:")
print(clf.best_params_)

# Predict on the test data
label_pred = clf.predict(dataset_test)

# Print the classification report and accuracy
print(classification_report(label_test, label_pred))
print("Accuracy:", accuracy_score(label_test, label_pred))