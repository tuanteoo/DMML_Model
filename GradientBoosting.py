import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load your datasets
dataset_train = pd.read_csv('dataset_trainT.csv')
dataset_test = pd.read_csv('dataset_testT.csv')
label_train = pd.read_csv('label_trainT.csv')
label_test = pd.read_csv('label_testT.csv')

# Preprocess your data here (scaling, encoding, etc.)

# Initialize the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=0)

# Train the model
gb_clf.fit(dataset_train, label_train.values.ravel())

# Predict on the test set
predictions = gb_clf.predict(dataset_test)

# Evaluate the model
accuracy = accuracy_score(label_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")