from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv') 

# df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
# # Tất cả các đặc trưng ngoại trừ 'Heart Attack Risk'
# X = df.drop(['Heart Attack Risk','Blood Pressure','Triglycerides','BMI','Previous Heart Problems'], axis=1)
# # Đặc trưng mục tiêu là 'Heart Attack Risk'
# y = df['Heart Attack Risk']

X = df.drop(['target'], axis=1)
# Đặc trưng mục tiêu là 'Heart Attack Risk'
y = df['target']

# Chia dữ liệu với tỉ lệ 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lưu tập huấn luyện và tập kiểm thử vào file CSV
X_train.to_csv('dataset_train.csv', index=False)
y_train.to_csv('label_train.csv', index=False)
X_test.to_csv('dataset_test.csv', index=False)
y_test.to_csv('label_test.csv', index=False)
