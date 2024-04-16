import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ các tập tin CSV
X_train = pd.read_csv('dataset_train.csv')
y_train = pd.read_csv('label_train.csv')
X_test = pd.read_csv('dataset_test.csv')
y_test = pd.read_csv('label_test.csv')

# Xây dựng mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Dự đoán giá trị trên tập kiểm tra
predictions = model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, predictions)

print(f"Độ chính xác của mô hình trên tập kiểm tra: {accuracy}")

