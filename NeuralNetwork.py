import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Assuming the files are in CSV format. Adjust the read function if they are in a different format.
dataset_train = pd.read_csv('dataset_train.csv')
dataset_test = pd.read_csv('dataset_test.csv')
label_train = pd.read_csv('label_train.csv')
label_test = pd.read_csv('label_test.csv')

scaler = StandardScaler()
dataset_train_scaled = scaler.fit_transform(dataset_train)
dataset_test_scaled = scaler.transform(dataset_test)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=dataset_train_scaled.shape[1], activation='relu'))  # Adjust the input_dim based on features
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Use 'categorical_crossentropy' for multi-class

# Fit the model
history = model.fit(dataset_train_scaled, label_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(dataset_test_scaled, label_test, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Predict on the test data
label_pred = model.predict(dataset_test_scaled)
label_pred = (label_pred > 0.5).astype("int32")  # Threshold the predictions for binary classification

# Print the classification report and accuracy
print(classification_report(label_test, label_pred))
print("Accuracy:", accuracy_score(label_test, label_pred))