import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv('dataset_train.csv')
y_train = pd.read_csv('label_train.csv')
X_test = pd.read_csv('dataset_test.csv')
y_test = pd.read_csv('label_test.csv')

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(classification_report(y_test, y_pred))

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     # Add other parameters here
# }

# # Initialize the Grid Search model
# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train.values.ravel())

# # Print the best parameters
# print("Best parameters found: ", grid_search.best_params_)

# # Use the best estimator for making predictions
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)

# # Evaluate the best model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy of the best model: {accuracy * 100:.2f}%")

